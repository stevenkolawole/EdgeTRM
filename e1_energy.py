#!/usr/bin/env python3
"""E1: measured DATACENTER-GPU energy (NOT edge energy; Babel L40S, one exclusive card per job; an early partial
Sudoku-MLP run on a p4d A100 is kept apart) for looped reasoners,
the PyTorch-vs-TensorRT framework tax at batch 1, and TensorRT FP16/INT8 as a toolchain correctness test.

Models (one supervision step = the carry step `stepc`: z_H, z_L, tokens, puzzle row -> z_H', z_L', logits,
q_halt[, q_cont]); default depth H (TRM 3, HRM 2) and 16 supervision steps, as in M5:
  sudoku_mlp   TRM MLP mixer, Sudoku-Extreme          exit rule: two consecutive halts (M5 R1, fp32-chosen)
  sudoku_attn  TRM attention, Sudoku-Extreme          exit rule: native q_halt > 0
  maze         TRM attention, Maze-30x30-Hard         exit rule: min 2 steps + native
  hrm_sudoku   HRM official checkpoint, Sudoku-Extreme exit rule: HRM's own q_halt > q_continue

Runtimes: eager PyTorch (fp32 with TF32 off, fp16), eager + CUDA graph (the same kernels, dispatch removed),
TensorRT 10.13 engines (fp32 with TF32 off, fp16, int8 = INT8+FP16 fallback, int8f32 = INT8+FP32 fallback;
implicit INT8 with entropy calibration on REAL carries from fp32 rollouts on calibration puzzles disjoint
from the evaluation puzzles). Batch-1 TRM engines are built from X3's ONNX exports
(~/aihub_artifacts/h1/<task>/stepc_H{1,3}.onnx, batch 1, static); batched engines and every HRM engine come
from a dynamic-batch export made here with the same wrapper and checked against X3's file.

Stages (resumable; everything under ~/e1/work/<task> and ~/e1/results/<task>):
    python e1_energy.py --stage prep   --task sudoku_mlp     exports + equivalence check
    python e1_energy.py --stage build  --task sudoku_mlp     TensorRT engines (+ layer precision summary)
    python e1_energy.py --stage record --task sudoku_mlp     correctness: per-step exact, cell agreement
                                                             and z_H cosine vs the eager fp32 reference
    python e1_energy.py --stage energy --task sudoku_mlp --nvml_index 5   NVML windows (reserved card only)
Energy protocol (reports/edgetrm_exp/ENERGY_PROTOCOL.md): NVML total-energy counter, windows >= 60 s of a
looped workload, median of 5 windows per configuration, idle power of the same card in the same session
(process alive, no work) subtracted, clocks NOT pinned (default DVFS) but sampled at 10 Hz and reported,
accuracy taken from the outputs of the very windows that were measured.
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import threading
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
HOME = pathlib.Path.home()
WORK = pathlib.Path(os.environ.get("E1_WORK", HOME / "e1" / "work"))          # engines (regenerable; /data on Babel)
RES = pathlib.Path(os.environ.get("E1_RES", HOME / "e1" / "results"))           # json records ($HOME)
X3ART = pathlib.Path(os.environ.get("E1_X3ART", HOME / "aihub_artifacts" / "h1"))  # X3's exports (copied from p4d)
ONNX = os.environ.get("E1_ONNX")          # dir with <task>/stepc_dyn_H<H>.onnx made by --stage prep on p4d
NSUP = 16

TASKS = {
    "sudoku_mlp": dict(kind="trm", x3="sudoku_mlp", H=3, data="sudoku", B=256, rule="patience2", n=1000, K=100,
                       cal=(1000, 1128), calB=64),
    "sudoku_attn": dict(kind="trm", x3="sudoku_attn", H=3, data="sudoku", B=256, rule="native", n=1000, K=100,
                        cal=(1000, 1128), calB=64),
    "maze": dict(kind="trm", x3="maze", H=3, data="maze", B=32, rule="min2native", n=900, K=32,
                 cal=(900, 932), calB=16),
    "hrm_sudoku": dict(kind="hrm", H=2, data="sudoku", B=256, rule="hrm_own", n=1000, K=100,
                       cal=(1000, 1128), calB=64),
}
OUTS5 = ["z_H_out", "z_L_out", "logits", "q_halt", "q_cont"]
INS = ["z_H", "z_L", "inputs", "puzzle_embedding"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def wdir(task):
    d = WORK / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def rdir(task):
    d = RES / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump(path, obj):
    tmp = pathlib.Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.rename(path)


# ----------------------------------------------------------------------------- models and data
def load_rows(kind, lo, hi):
    import breadth_sweep as BS
    q, a = BS.load_qa(kind, hi)
    return q[lo:hi].astype(np.int32), a[lo:hi].astype(np.int64)


def build_inner(task, H):
    """fp32 CPU inner model, its puzzle-embedding row 0 (every Sudoku/Maze puzzle uses id 0), and init carry."""
    t = TASKS[task]
    if t["kind"] == "trm":
        import aihub_h1 as A
        m, cfg, sd = A.build(t["x3"], H)
        pe = sd["inner.puzzle_emb.weights"][:1].float().clone()
    else:
        import breadth_sweep as BS
        m, _ = BS.build_hrm("hrm_sudoku", H, NSUP, 1)
        pe = m.inner.puzzle_emb.weights[:1].float().clone()
    inner = m.inner.eval()
    return inner, pe


def make_step(inner, dtype="fp32"):
    """Step module with the same interface as X3's stepc export (plus q_cont). fp16: weights, buffers and the
    embedding cast all in half; the model's own RMSNorm still upcasts to fp32 internally."""
    import torch

    class Emb(torch.nn.Module):
        def __init__(self, holder):
            super().__init__()
            self.h = [holder]

        def forward(self, idx):
            return self.h[0]._pe

    class Step(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self._pe = None
            self.Carry = type(inner.empty_carry(1))
            self.inner.puzzle_emb = Emb(self)
            self.register_buffer("pid", torch.zeros(1, dtype=torch.int32), persistent=False)

        def forward(self, z_H, z_L, inputs, puzzle_embedding):
            self._pe = puzzle_embedding
            nc, logits, (qh, qc) = self.inner(self.Carry(z_H=z_H, z_L=z_L),
                                              {"inputs": inputs, "puzzle_identifiers": self.pid})
            return nc.z_H, nc.z_L, logits, qh, qc

    s = Step(inner).eval()
    s.requires_grad_(False)
    if dtype == "fp16":
        s = s.half()
        inner.forward_dtype = torch.float16
        for mod in inner.modules():
            if hasattr(mod, "cast_to"):
                mod.cast_to = torch.float16
    return s


def init_carry(inner):
    return inner.H_init.detach().float().clone(), inner.L_init.detach().float().clone()


# ----------------------------------------------------------------------------- runners (common interface)
class EagerRunner:
    """z_H/z_L/logits/qh/qc after each step(); load(x) sets tokens (B, seq) and resets the carry."""

    def __init__(self, task, H, B, dtype="fp32", graph=False):
        import torch
        self.torch = torch
        inner, pe = build_inner(task, H)
        self.dt = torch.float16 if dtype == "fp16" else torch.float32
        zh0, zl0 = init_carry(inner)
        self.step_mod = make_step(inner, dtype).cuda()
        self.S = inner.config.seq_len + inner.puzzle_emb_len
        self.B, self.graph = B, graph
        self.zh0 = zh0.cuda().to(self.dt)
        self.zl0 = zl0.cuda().to(self.dt)
        self.pe = pe.cuda().to(self.dt).expand(B, -1).contiguous()
        self.x = torch.zeros(B, inner.config.seq_len, dtype=torch.int32, device="cuda")
        self.zH = self.zh0.expand(B, self.S, -1).contiguous()
        self.zL = self.zl0.expand(B, self.S, -1).contiguous()
        self.logits = self.qh = self.qc = None
        if graph:
            self._capture()

    def _capture(self):
        torch = self.torch
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), torch.cuda.stream(s):          # no_grad, not inference_mode: replay writes
            for _ in range(3):                                # the static (normal) carry tensors in place
                o = self.step_mod(self.zH, self.zL, self.x, self.pe)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.g):
            o = self.step_mod(self.zH, self.zL, self.x, self.pe)
            self.zH.copy_(o[0])
            self.zL.copy_(o[1])
            self.logits, self.qh, self.qc = o[2], o[3], o[4]

    def load(self, x):
        self.x.copy_(x)
        if self.graph:                       # static buffers the captured graph reads and advances in place
            self.zH.copy_(self.zh0.expand_as(self.zH))
            self.zL.copy_(self.zl0.expand_as(self.zL))
        else:                                # outputs of the last step are inference tensors: reallocate
            self.zH = self.zh0.expand(self.B, self.S, -1).clone()
            self.zL = self.zl0.expand(self.B, self.S, -1).clone()

    def step(self):
        if self.graph:
            self.g.replay()
            return
        with self.torch.inference_mode():
            o = self.step_mod(self.zH, self.zL, self.x, self.pe)
        self.zH, self.zL, self.logits, self.qh, self.qc = o

    def cur_zH(self):
        return self.zH


class TRTRunner:
    def __init__(self, engine_path, task, H, B):
        import torch
        import tensorrt as trt
        self.torch, self.trt = torch, trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.rt = trt.Runtime(self.logger)
        self.engine = self.rt.deserialize_cuda_engine(pathlib.Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        inner, pe = build_inner(task, H)
        zh0, zl0 = init_carry(inner)
        self.S = inner.config.seq_len + inner.puzzle_emb_len
        seq = inner.config.seq_len
        self.B = B
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.names = names
        tdt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
               trt.int64: torch.int64}
        shapes = {"z_H": (B, self.S, 512), "z_L": (B, self.S, 512), "inputs": (B, seq), "puzzle_embedding": (B, 512)}
        for n in INS:
            if -1 in tuple(self.engine.get_tensor_shape(n)):
                self.ctx.set_input_shape(n, shapes[n])
        self.buf = {}
        for n in names:
            shp = tuple(self.ctx.get_tensor_shape(n))
            self.buf[n] = torch.zeros(shp, dtype=tdt[self.engine.get_tensor_dtype(n)], device="cuda")
        self.zH = [self.buf["z_H"], torch.zeros_like(self.buf["z_H"])]
        self.zL = [self.buf["z_L"], torch.zeros_like(self.buf["z_L"])]
        self.zh0 = zh0.cuda().to(self.zH[0].dtype)
        self.zl0 = zl0.cuda().to(self.zL[0].dtype)
        self.buf["puzzle_embedding"].copy_(pe.cuda().to(self.buf["puzzle_embedding"].dtype).expand(B, -1))
        self.cur = 0
        self.has_qc = "q_cont" in names
        for n in names:
            if n not in ("z_H", "z_L", "z_H_out", "z_L_out"):
                self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
        self.logits = self.buf["logits"]
        self.qh = self.buf["q_halt"]
        self.qc = self.buf["q_cont"] if self.has_qc else None

    def load(self, x):
        self.buf["inputs"].copy_(x.to(self.buf["inputs"].dtype))
        self.cur = 0
        self.zH[0].copy_(self.zh0.expand_as(self.zH[0]))
        self.zL[0].copy_(self.zl0.expand_as(self.zL[0]))

    def step(self):
        c, o = self.cur, 1 - self.cur
        ctx = self.ctx
        ctx.set_tensor_address("z_H", self.zH[c].data_ptr())
        ctx.set_tensor_address("z_L", self.zL[c].data_ptr())
        ctx.set_tensor_address("z_H_out", self.zH[o].data_ptr())
        ctx.set_tensor_address("z_L_out", self.zL[o].data_ptr())
        ok = ctx.execute_async_v3(self.torch.cuda.current_stream().cuda_stream)
        assert ok
        self.cur = o

    def cur_zH(self):
        return self.zH[self.cur]


# ----------------------------------------------------------------------------- exit rules (online, per puzzle)
def rule_fires(rule, s, qh, qc, prev_halt):
    """s = 1-based step just completed; qh/qc floats; prev_halt = did the head fire at step s-1."""
    halt = (qh > qc) if rule == "hrm_own" else (qh > 0)
    if rule == "native" or rule == "hrm_own":
        return halt, halt
    if rule == "patience2":
        return halt and prev_halt, halt
    if rule == "min2native":
        return halt and s >= 2, halt
    raise ValueError(rule)


def offline_exit(rule, qh, qc):
    """qh, qc: (n, NSUP) arrays -> exit step (1-based) per puzzle, capped at NSUP."""
    n = qh.shape[0]
    out = np.full(n, NSUP)
    for i in range(n):
        prev = False
        for s in range(1, NSUP + 1):
            fire, prev = rule_fires(rule, s, float(qh[i, s - 1]), float(qc[i, s - 1]), prev)
            if fire:
                out[i] = s
                break
    return out


# ----------------------------------------------------------------------------- stage: prep (exports)
def dyn_onnx(task):
    return wdir(task) / f"stepc_dyn_H{TASKS[task]['H']}.onnx"


def dyn_onnx_H(task, H):
    if ONNX:
        return pathlib.Path(ONNX) / task / f"stepc_dyn_H{H}.onnx"
    return wdir(task) / f"stepc_dyn_H{H}.onnx"


def stage_prep(task):
    import torch
    import onnxruntime as ort
    t = TASKS[task]
    Hs = [t["H"]] + ([1] if t["kind"] == "trm" else [])
    rec = {}
    for H in Hs:
        path = dyn_onnx_H(task, H)
        inner, pe = build_inner(task, H)
        zh0, zl0 = init_carry(inner)
        step = make_step(inner)
        S = inner.config.seq_len + inner.puzzle_emb_len
        x, _ = load_rows(t["data"], 0, 4)
        B = 2
        # real carries: run 3 steps from the init carry on 2 puzzles
        with torch.no_grad():
            zH = zh0.expand(B, S, -1).contiguous()
            zL = zl0.expand(B, S, -1).contiguous()
            xx = torch.as_tensor(x[:B])
            pp = pe.expand(B, -1).contiguous()
            for _ in range(3):
                zH, zL, *_r = step(zH, zL, xx, pp)
            ref = step(zH, zL, xx, pp)
        args = (zH.clone(), zL.clone(), xx, pp)
        if not path.exists():
            torch.onnx.export(step, args, str(path), input_names=INS, output_names=OUTS5, opset_version=17,
                              dynamo=False, dynamic_axes={n: {0: "B"} for n in INS + OUTS5})
        sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        got = sess.run(OUTS5, {n: a.numpy() for n, a in zip(INS, args)})
        errs = [float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref)]
        r = {"dyn_vs_torch_maxabs": errs, "size_MB": path.stat().st_size / 1e6}
        if t["kind"] == "trm":
            xp = X3ART / t["x3"] / f"stepc_H{H}.onnx"
            s3 = ort.InferenceSession(str(xp), providers=["CPUExecutionProvider"])
            outs3 = [o.name for o in s3.get_outputs()]
            g3 = s3.run(outs3, {n: a[:1].numpy() for n, a in zip(INS, args)})
            r["x3_file"] = str(xp)
            r["x3_outputs"] = outs3
            r["dyn_vs_x3_maxabs_b1"] = [float(np.abs(g3[i].reshape(-1) - got[i][:1].reshape(-1)).max())
                                        for i in range(len(outs3))]
        rec[f"H{H}"] = r
        log(task, f"H{H}", r)
    dump(rdir(task) / "prep.json", rec)


# ----------------------------------------------------------------------------- stage: build (TensorRT)
def calib_samples(task, H, max_n=None):
    """Real carries: fp32 eager rollouts (GPU, batched) on calibration rows; two distinct random steps per
    puzzle (carry entering the step). Returns dict of CPU arrays + the (row, step) list."""
    import torch
    t = TASKS[task]
    lo, hi = t["cal"]
    x, _ = load_rows(t["data"], lo, hi)
    r = EagerRunner(task, H, B=len(x))
    r.load(torch.as_tensor(x, device="cuda"))
    rng = np.random.default_rng(0)
    pick = np.stack([rng.choice(NSUP, 2, replace=False) for _ in range(len(x))])
    zh, zl, meta = [], [], []
    for s in range(NSUP):
        sel = np.where((pick == s).any(1))[0]
        if len(sel):
            zh.append(r.zH[sel].float().cpu().numpy())
            zl.append(r.zL[sel].float().cpu().numpy())
            meta += [(int(lo + i), s) for i in sel]
        r.step()
    rows = np.array([m[0] - lo for m in meta])
    out = dict(z_H=np.concatenate(zh), z_L=np.concatenate(zl), inputs=x[rows],
               puzzle_embedding=np.repeat(r.pe[:1].float().cpu().numpy(), len(rows), 0))
    del r
    torch.cuda.empty_cache()
    return out, meta


def make_calibrator(data, batch, cache_path):
    import torch
    import tensorrt as trt

    class Calib(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.i = 0
            self.n = data["z_H"].shape[0] // batch * batch
            self.dev = {k: torch.as_tensor(v).cuda() for k, v in data.items()}
            self.cur = None

        def get_batch_size(self):
            return batch

        def get_batch(self, names):
            if self.i >= self.n:
                return None
            sl = slice(self.i, self.i + batch)
            self.cur = {k: self.dev[k][sl].contiguous() for k in names}
            self.i += batch
            return [int(self.cur[k].data_ptr()) for k in names]

        def read_calibration_cache(self):
            return cache_path.read_bytes() if cache_path.exists() else None

        def write_calibration_cache(self, cache):
            cache_path.write_bytes(bytes(cache))

    return Calib()


def engine_list(task):
    """(name, onnx, H, B, prec, cache). All main engines come from the dynamic-batch re-export of X3's step graph
    (outputs equal to X3's file to <= 6e-5, exactly 0 for Sudoku-MLP; results/<task>/prep.json). Per H, the first
    engine is the calibration seed, calibrated at batch calB on 2 x (#calibration puzzles) real carries; every
    other INT8 engine at that H reads the seed's cache. (TensorRT's entropy calibration costs ~25 s per
    get_batch call on these graphs whatever the batch size, and the seed's cache does not transfer to X3's
    static files: 516 of the tensor names differ.) Provenance check on X3's own file (TRM, H=1 only): FP16 and
    INT8-GEMM engines built from ~/aihub_artifacts/h1/<task>/stepc_H1.onnx, calibrated directly on the first
    32 of the same carries at batch 1 (X3/AIMET used 32 Sudoku / 16 Maze carries)."""
    t = TASKS[task]
    H0 = t["H"]
    out = []
    Hs = [H0] + ([1] if t["kind"] == "trm" else [])
    for H in Hs:
        src, cache = dyn_onnx_H(task, H), f"calib_dyn_H{H}.cache"
        out.append((f"dyn_H{H}_b1_int8", src, H, 1, "int8", cache))
        for prec in ("fp32", "fp16", "int8f32", "int8mm", "int8all"):
            out.append((f"dyn_H{H}_b1_{prec}", src, H, 1, prec, cache))
        out.append((f"dyn_H{H}_b1_int8proj", src, H, 1, "int8proj", cache))
        if t["kind"] == "trm" and H == 1:
            x3src = X3ART / t["x3"] / f"stepc_H{H}.onnx"
            for prec in ("fp16", "int8mm"):
                out.append((f"x3_H{H}_b1_{prec}", x3src, H, 1, prec, f"calib_x3_H{H}.cache"))
    for prec in ("fp32", "fp16", "int8", "int8mm", "int8proj"):
        out.append((f"dyn_H{H0}_b{t['B']}_{prec}", dyn_onnx_H(task, H0), H0, t["B"], prec, f"calib_dyn_H{H0}.cache"))
    # build order: the H0 calibration seed, then the engines the energy job times (it waits for them), then the rest
    en = {c.get("engine") for c in energy_configs(task)}
    seeds = [e for e in out if e[0] == f"dyn_H{H0}_b1_int8"]
    return seeds + [e for e in out if e[0] in en] + [e for e in out if e[0] not in en and e not in seeds]


def b1_engine(task, H, prec):
    return f"dyn_H{H}_b1_{prec}"


def layer_precisions(engine_bytes):
    import tensorrt as trt
    rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    eng = rt.deserialize_cuda_engine(engine_bytes)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    import re
    pat = re.compile(r"(Int8|INT8|FP16|Half|FP32|Float|Int32|INT32|Int64|INT64|Bool|BOOL|FP8)")

    def dts(lst):
        got = set()
        for o in lst:
            m = pat.search(str(o.get("Format/Datatype", "")))
            got.add({"INT8": "Int8", "Half": "FP16", "Float": "FP32", "INT32": "Int32", "INT64": "Int64",
                     "BOOL": "Bool"}.get(m.group(1), m.group(1)) if m else "?")
        return ",".join(sorted(got)) or "-"
    counts, norm = {}, {}
    for L in info.get("Layers", []):
        if isinstance(L, str):
            continue
        dt, idt = dts(L.get("Outputs", [])), dts(L.get("Inputs", []))
        key = f"in:{idt}|out:{dt}"
        counts[key] = counts.get(key, 0) + 1
        nm = L.get("Name", "")
        if any(k in nm for k in ("Pow", "ReduceMean", "Sqrt", "Reciprocal", "Div", "rms", "norm")):
            norm[key] = norm.get(key, 0) + 1
    return {"n_layers": sum(counts.values()), "by_io_dtype": counts, "norm_like_layers_by_io_dtype": norm}, info


def stage_build(task, only=None):
    import tensorrt as trt

    class CountLogger(trt.ILogger):
        def __init__(self):
            super().__init__()
            self.msgs = []

        def log(self, sev, msg):
            if sev <= trt.ILogger.Severity.WARNING:
                self.msgs.append(str(msg))
                if sev <= trt.ILogger.Severity.ERROR:
                    print("TRT", sev, msg, flush=True)

    t = TASKS[task]
    summ_path = rdir(task) / "engines.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    calib_cache = {}
    tc_path = wdir(task) / "timing.cache"
    for name, src, H, B, prec, cache_name in engine_list(task):
        if only and name not in only:
            continue
        ep = wdir(task) / f"{name}.engine"
        if (ep.exists() or summ.get(name, {}).get("BUILD_FAILED")) and name in summ:
            continue
        logger = CountLogger()
        builder = trt.Builder(logger)
        net = builder.create_network(0)
        parser = trt.OnnxParser(net, logger)
        assert parser.parse_from_file(str(src)), [parser.get_error(i) for i in range(parser.num_errors)]
        cfg = builder.create_builder_config()
        cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
        cfg.clear_flag(trt.BuilderFlag.TF32)
        cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        # engines timed in the energy stage keep TensorRT's default optimization level (3); correctness-only
        # engines (H=1, int8f32, int8all, X3-file checks) use level 1 to keep build times bearable
        energy_names = {c.get("engine") for c in energy_configs(task)}
        opt_level = 3 if name in energy_names else 1
        cfg.builder_optimization_level = opt_level
        tcache = cfg.create_timing_cache(tc_path.read_bytes() if tc_path.exists() else b"")
        cfg.set_timing_cache(tcache, ignore_mismatch=False)
        if prec in ("fp16", "int8", "int8mm", "int8all", "int8proj"):
            cfg.set_flag(trt.BuilderFlag.FP16)
        # int8 = TensorRT's own choice per layer (INT8 only where its timing says faster); int8f32 = the same
        # without FP16 kernels; int8mm = INT8 REQUIRED on every MatMul (TRT's INT8 GEMMs, the closest analogue
        # of a w8a8 GEMM path), rest FP16; int8all = INT8 PREFERRED on every layer (the closest TRT gets to
        # AIMET's all-activations-8-bit graph), FP16 wherever TRT has no INT8 implementation.
        forced = []
        if prec == "int8mm":
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for i in range(net.num_layers):
                L = net.get_layer(i)
                if L.type == trt.LayerType.MATRIX_MULTIPLY:
                    L.precision = trt.int8
                    forced.append(L.name)
        n_score = 0
        if prec == "int8proj":
            # INT8 required on every weight MatMul (projections, MLP, heads); the attention score matmuls
            # (q.k^T and attn.v: both operands are activations, no constant weight upstream) forced to FP16
            cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            prod = {}
            for i in range(net.num_layers):
                L = net.get_layer(i)
                for k in range(L.num_outputs):
                    prod[L.get_output(k).name] = L
            passthru = {trt.LayerType.SHUFFLE, trt.LayerType.IDENTITY, trt.LayerType.CAST, trt.LayerType.SLICE,
                        trt.LayerType.CONCATENATION, trt.LayerType.UNARY}

            def from_const(tensor, depth=0):
                L = prod.get(tensor.name)
                if L is None or depth > 8:
                    return False
                if L.type == trt.LayerType.CONSTANT:
                    return True
                if L.type in passthru:
                    return all(from_const(L.get_input(j), depth + 1) for j in range(L.num_inputs)
                               if L.get_input(j) is not None)
                return False
            for i in range(net.num_layers):
                L = net.get_layer(i)
                if L.type != trt.LayerType.MATRIX_MULTIPLY:
                    continue
                if any(from_const(L.get_input(j)) for j in range(2)):
                    L.precision = trt.int8
                    forced.append(L.name)
                else:
                    L.precision = trt.float16
                    L.set_output_type(0, trt.float16)
                    n_score += 1
        if prec == "int8all":
            cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
            skip = {trt.LayerType.SHAPE, trt.LayerType.CONSTANT, trt.LayerType.IDENTITY, trt.LayerType.CAST,
                    trt.LayerType.SHUFFLE, trt.LayerType.SLICE, trt.LayerType.CONCATENATION}
            for i in range(net.num_layers):
                L = net.get_layer(i)
                if L.type in skip or any(L.get_output(k).dtype in (trt.int32, trt.int64, trt.bool)
                                         for k in range(L.num_outputs)):
                    continue
                L.precision = trt.int8
                forced.append(L.name)
        dyn = any(-1 in tuple(net.get_input(i).shape) for i in range(net.num_inputs))
        if dyn:
            prof = builder.create_optimization_profile()
            for i in range(net.num_inputs):
                inp = net.get_input(i)
                shp = (B,) + tuple(inp.shape)[1:]
                prof.set_shape(inp.name, shp, shp, shp)
            cfg.add_optimization_profile(prof)
        cache = wdir(task) / cache_name
        cb = t["calB"] if dyn else 1
        calibrated_here = False
        if prec.startswith("int8"):
            cfg.set_flag(trt.BuilderFlag.INT8)
            if not cache.exists():
                if H not in calib_cache:
                    calib_cache[H] = calib_samples(task, H)
                data = calib_cache[H][0]
                if not dyn:                                         # X3 static file: 32 carries at batch 1
                    data = {k: v[:32] for k, v in data.items()}
                calibrated_here = True
            else:
                data = {"z_H": np.zeros((1, 1, 1), np.float32)}          # cache present: never called
            if dyn:
                cprof = builder.create_optimization_profile()
                for i in range(net.num_inputs):
                    inp = net.get_input(i)
                    shp = (cb,) + tuple(inp.shape)[1:]
                    cprof.set_shape(inp.name, shp, shp, shp)
                cfg.set_calibration_profile(cprof)
            cfg.int8_calibrator = make_calibrator(data, cb, cache)
        t0 = time.time()
        ser = builder.build_serialized_network(net, cfg)
        if ser is None:                                      # a toolchain failure is a result: record it
            errs = [m for m in logger.msgs if "Error" in m or "error" in m]
            summ[name] = {"onnx": str(src), "H": H, "B": B, "prec": prec, "BUILD_FAILED": True,
                          "build_s": round(time.time() - t0, 1), "trt": trt.__version__,
                          "n_layers_precision_constrained": len(forced), "builder_optimization_level": opt_level,
                          "errors": [e[:400] for e in errs[-4:]]}
            dump(summ_path, summ)
            log(task, name, "BUILD FAILED", errs[-1][:200] if errs else "")
            continue
        b = bytes(ser)
        ep.write_bytes(b)
        tc_path.write_bytes(bytes(tcache.serialize()))
        lp, info = layer_precisions(b)
        (wdir(task) / f"{name}.layers.json").write_text(json.dumps(info))
        miss = [m for m in logger.msgs if "Missing scale" in m]
        summ[name] = {"onnx": str(src), "H": H, "B": B, "prec": prec, "build_s": round(time.time() - t0, 1),
                      "size_MB": round(len(b) / 1e6, 2), "trt": trt.__version__, **lp,
                      "n_layers_precision_constrained": len(forced), "builder_optimization_level": opt_level,
                      "n_score_matmuls_fp16": n_score,
                      "warnings": {"n": len(logger.msgs), "missing_int8_scale": len(miss),
                                   "missing_examples": [m[:160] for m in miss[:6]],
                                   "other": sorted({m[:200] for m in logger.msgs if "Missing scale" not in m})[:12]}}
        if prec.startswith("int8"):
            summ[name]["calib"] = {"cache": cache.name, "calibrated_in_this_build": calibrated_here,
                                   "batch": cb, "rows": list(t["cal"]), "algo": "IInt8EntropyCalibrator2",
                                   "samples": "2 distinct random steps (of 16) per calibration puzzle, "
                                              "carry entering the step, fp32 eager rollout"}
            if calibrated_here:
                summ[name]["calib"]["n"] = int(data["z_H"].shape[0])
                summ[name]["calib"]["samples_row_step_head"] = calib_cache[H][1][:8]
        dump(summ_path, summ)
        log(task, name, summ[name]["build_s"], "s", summ[name]["size_MB"], "MB", lp["by_io_dtype"],
            "missing scales", len(miss))


# ----------------------------------------------------------------------------- stage: record (correctness)
def make_runner(task, cfg, B):
    """Every runner works on a fresh non-default CUDA stream: TensorRT's enqueueV3 on the legacy default stream
    adds cudaStreamSynchronize calls (its own warning), which would tax the TensorRT side of the comparison."""
    import torch
    torch.cuda.set_stream(torch.cuda.Stream())
    rt, H, prec = cfg["rt"], cfg["H"], cfg["prec"]
    if rt == "eager":
        return EagerRunner(task, H, B, prec)
    if rt == "graph":
        return EagerRunner(task, H, B, prec, graph=True)
    return TRTRunner(wdir(task) / f"{cfg['engine']}.engine", task, H, B)


def rollout_record(task, runner, x, lab, B, K, ref=None):
    """Fixed NSUP steps on all puzzles in chunks of B (the runner's batch). Per-step preds (uint8), qh, qc,
    z_H of the first K puzzles (kept when ref is None) or their cosine to ref."""
    import torch
    n = len(x)
    preds = np.zeros((n, NSUP, x.shape[1]), np.uint8)
    qh = np.zeros((n, NSUP), np.float32)
    qc = np.zeros((n, NSUP), np.float32)
    zkeep = [] if ref is None else None
    cos = np.full((K, NSUP), np.nan, np.float32)
    xg = torch.as_tensor(x, device="cuda")
    for s0 in range(0, n, B):
        idx = np.arange(s0, s0 + B) % n
        m = min(B, n - s0)
        runner.load(xg[idx])
        zsteps = []
        for s in range(NSUP):
            runner.step()
            preds[s0:s0 + m, s] = runner.logits[:m].argmax(-1).to(torch.uint8).cpu().numpy()
            qh[s0:s0 + m, s] = runner.qh.float().reshape(-1)[:m].cpu().numpy()
            if runner.qc is not None:
                qc[s0:s0 + m, s] = runner.qc.float().reshape(-1)[:m].cpu().numpy()
            if s0 < K:
                kk = min(K - s0, m)
                z = runner.cur_zH()[:kk].float()
                if ref is None:
                    zsteps.append(z.clone())
                else:
                    r = ref[s0:s0 + kk, s]
                    cos[s0:s0 + kk, s] = torch.nn.functional.cosine_similarity(
                        z.flatten(1), r.flatten(1), dim=-1).cpu().numpy()
        if ref is None and zsteps:
            zkeep.append(torch.stack(zsteps, 1))
    out = {"preds": preds, "qh": qh, "qc": qc, "cos": cos}
    if ref is None:
        out["zref"] = torch.cat(zkeep)[:K]
    return out


def summarize_record(task, r, lab, refpreds, x3host=None):
    t = TASKS[task]
    ex = (r["preds"] == lab[:, None, :]).all(-1)                   # (n, NSUP)
    cell = (r["preds"] == lab[:, None, :]).mean(-1)
    agree = (r["preds"] == refpreds).mean(-1)                      # cell agreement with the fp32 reference
    ex_step = offline_exit(t["rule"], r["qh"], r["qc"])
    at_exit = ex[np.arange(len(ex)), ex_step - 1]
    s = {"n": int(len(ex)), "exact_per_step": [round(float(v), 4) for v in ex.mean(0)],
         "cell_per_step": [round(float(v), 4) for v in cell.mean(0)],
         "ref_cell_agreement_per_step": [round(float(v), 4) for v in agree.mean(0)],
         "ref_puzzle_agreement_final": round(float((r["preds"][:, -1] == refpreds[:, -1]).all(-1).mean()), 4),
         "zH_cos_per_step_meanK": [round(float(v), 5) for v in np.nanmean(r["cos"], 0)],
         "zH_cos_min_final": round(float(np.nanmin(r["cos"][:, -1])), 5),
         "K": int(r["cos"].shape[0]),
         "exit_rule": t["rule"], "exit_exact": round(float(at_exit.mean()), 4),
         "exit_mean_steps": round(float(ex_step.mean()), 3),
         "first100_exact_per_step": [int(v) for v in ex[:100].sum(0)]}
    if x3host is not None:
        hp = x3host["preds"]                                        # (steps, N, seq) from aihub_infer host_loop
        m = hp.shape[1]
        ks = min(hp.shape[0], NSUP)
        s["x3_ort_host_fp32"] = {"n": int(m), "steps": ks,
                                 "cell_agreement_per_step": [round(float((r["preds"][:m, k] == hp[k]).mean()), 4)
                                                             for k in range(ks)],
                                 "ort_exact_per_step": [int((hp[k] == lab[:m]).all(-1).sum()) for k in range(ks)]}
    return s


def record_configs(task):
    t = TASKS[task]
    H0, B = t["H"], t["B"]
    cfgs = []
    Hs = [H0] + ([1] if t["kind"] == "trm" else [])
    for H in Hs:
        cfgs.append(dict(name=f"eager_fp32_H{H}_b1", rt="eager", H=H, prec="fp32", B=1))
        cfgs.append(dict(name=f"eager_fp16_H{H}_b1", rt="eager", H=H, prec="fp16", B=1))
        for prec in ("fp32", "fp16", "int8", "int8f32", "int8mm", "int8all", "int8proj"):
            cfgs.append(dict(name=f"trt_{prec}_H{H}_b1", rt="trt", H=H, prec=prec, B=1,
                             engine=b1_engine(task, H, prec)))
        if t["kind"] == "trm" and H == 1:
            for prec in ("fp16", "int8mm"):
                cfgs.append(dict(name=f"trt_x3{prec}_H{H}_b1", rt="trt", H=H, prec=prec, B=1,
                                 engine=f"x3_H{H}_b1_{prec}"))
    cfgs.append(dict(name=f"graph_fp16_H{H0}_b1", rt="graph", H=H0, prec="fp16", B=1, nmax=50))
    cfgs.append(dict(name=f"eager_fp16_H{H0}_b{B}", rt="eager", H=H0, prec="fp16", B=B))
    for prec in ("fp32", "fp16", "int8", "int8mm", "int8proj"):
        cfgs.append(dict(name=f"trt_{prec}_H{H0}_b{B}", rt="trt", H=H0, prec=prec, B=B,
                         engine=f"dyn_H{H0}_b{B}_{prec}"))
    return cfgs


def stage_record(task, only=None, nlimit=None):
    import torch
    t = TASKS[task]
    n = nlimit or t["n"]
    K = min(t["K"], n)
    x, lab = load_rows(t["data"], 0, n)
    lab8 = lab.astype(np.uint8)
    out_path = rdir(task) / "record.json"
    rec = json.loads(out_path.read_text()) if out_path.exists() else {}
    Hs = [t["H"]] + ([1] if t["kind"] == "trm" else [])
    refs = {}
    for H in Hs:
        name = f"ref_eager_fp32_H{H}_b{t['B']}"
        rr = EagerRunner(task, H, t["B"], "fp32")
        r = rollout_record(task, rr, x, lab8, t["B"], K)
        del rr
        refs[H] = r
        x3host = None
        if t["kind"] == "trm" and H == 1:
            hp = X3ART / t["x3"] / "infer" / "host_fp32.npz"
            x3host = dict(np.load(hp)) if hp.exists() else None
        rec[name] = summarize_record(task, {**r, "cos": np.ones((K, NSUP), np.float32)}, lab8, r["preds"], x3host)
        np.savez_compressed(wdir(task) / f"{name}.npz", preds=r["preds"], qh=r["qh"], qc=r["qc"])
        log(task, name, "exact@16", rec[name]["exact_per_step"][-1], "exit", rec[name]["exit_exact"],
            rec[name]["exit_mean_steps"])
    dump(out_path, rec)
    for cfg in record_configs(task):
        if only and cfg["name"] not in only:
            continue
        if cfg["name"] in rec and not only:
            continue
        if cfg["rt"] == "trt" and not (wdir(task) / f"{cfg['engine']}.engine").exists():
            log("missing engine", cfg["engine"])
            continue
        nn_ = min(n, cfg.get("nmax", n))
        ref = refs[cfg["H"]]
        runner = make_runner(task, cfg, cfg["B"])
        t0 = time.time()
        r = rollout_record(task, runner, x[:nn_], lab8[:nn_], cfg["B"], min(K, nn_), ref["zref"])
        el = time.time() - t0
        del runner
        torch.cuda.empty_cache()
        x3host = None
        if t["kind"] == "trm" and cfg["H"] == 1:
            hp = X3ART / t["x3"] / "infer" / "host_fp32.npz"
            x3host = dict(np.load(hp)) if hp.exists() else None
        s = summarize_record(task, r, lab8[:nn_], ref["preds"][:nn_], x3host)
        s.update({k: v for k, v in cfg.items() if k != "name"}, wall_s=round(el, 1),
                 note="record pass wall time includes per-step host copies; not a latency measurement")
        rec[cfg["name"]] = s
        np.savez_compressed(wdir(task) / f"{cfg['name']}.npz", preds=r["preds"], qh=r["qh"], qc=r["qc"],
                            cos=r["cos"])
        dump(out_path, rec)
        log(task, cfg["name"], "exact@8/16", s["exact_per_step"][7], s["exact_per_step"][-1], "agree@16",
            s["ref_cell_agreement_per_step"][-1], "cos@16", s["zH_cos_per_step_meanK"][-1], "exit",
            s["exit_exact"], s["exit_mean_steps"])


# ----------------------------------------------------------------------------- stage: energy (NVML windows)
class Sampler(threading.Thread):
    def __init__(self, nv, h):
        super().__init__(daemon=True)
        self.nv, self.h = nv, h
        self.on = threading.Event()
        self.stop_ = threading.Event()
        self.rows = []

    def run(self):
        nv, h = self.nv, self.h
        while not self.stop_.is_set():
            if self.on.is_set():
                try:
                    u = nv.nvmlDeviceGetUtilizationRates(h)
                    self.rows.append((time.time(), nv.nvmlDeviceGetPowerUsage(h),
                                      nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
                                      nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
                                      nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU), u.gpu, u.memory))
                except Exception:
                    pass
            time.sleep(0.1)

    def take(self):
        r, self.rows = self.rows, []
        if not r:
            return {}
        a = np.array(r, dtype=float)
        return {"n_samples": len(r), "power_W_mean": round(a[:, 1].mean() / 1e3, 2),
                "power_W_max": round(a[:, 1].max() / 1e3, 2), "sm_MHz_mean": round(a[:, 2].mean(), 1),
                "sm_MHz_min": int(a[:, 2].min()), "sm_MHz_max": int(a[:, 2].max()),
                "mem_MHz_mean": round(a[:, 3].mean(), 1), "temp_C_start": int(a[0, 4]), "temp_C_end": int(a[-1, 4]),
                "util_gpu_mean": round(a[:, 5].mean(), 1), "util_mem_mean": round(a[:, 6].mean(), 1)}


def card_clean(nv, h):
    procs = nv.nvmlDeviceGetComputeRunningProcesses(h)
    others = [p.pid for p in procs if p.pid != os.getpid()]
    return not others, others


def energy_configs(task):
    t = TASKS[task]
    H, B = t["H"], t["B"]
    c = []
    # TRT's own INT8 choice ("int8") is not timed: on these graphs it runs no layer in INT8 (engines.json), so it is
    # the FP16 engine under another name; int8mm (INT8 required on every MatMul) is the INT8 configuration.
    for rt, prec in (("eager", "fp32"), ("eager", "fp16"), ("graph", "fp16"), ("trt", "fp32"), ("trt", "fp16"),
                     ("trt", "int8mm")):
        d = dict(name=f"{rt}_{prec}_b1_fixed", rt=rt, prec=prec, H=H, B=1, mode="fixed")
        if rt == "trt":
            d["engine"] = b1_engine(task, H, prec)
        c.append(d)
    for rt, prec in (("eager", "fp32"), ("eager", "fp16"), ("graph", "fp16"), ("trt", "fp16"), ("trt", "int8mm")):
        d = dict(name=f"{rt}_{prec}_b1_exit", rt=rt, prec=prec, H=H, B=1, mode="exit")
        if rt == "trt":
            d["engine"] = b1_engine(task, H, prec)
        c.append(d)
    for rt, prec in (("eager", "fp32"), ("eager", "fp16"), ("trt", "fp16"), ("trt", "int8mm")):
        d = dict(name=f"{rt}_{prec}_b{B}_fixed", rt=rt, prec=prec, H=H, B=B, mode="fixed")
        if rt == "trt":
            d["engine"] = f"dyn_H{H}_b{B}_{prec}"
        c.append(d)
    return c


def stage_energy(task, nvml_index, win_s=60.0, nwin=5, idle_s=30.0, nidle=5, only=None, warm_s=10.0,
                 max_wait_s=900):
    import torch
    import pynvml as nv
    assert torch.cuda.is_available(), "CUDA not available: abort"
    t = TASKS[task]
    nv.nvmlInit()
    props = torch.cuda.get_device_properties(0)
    uuid_t = "GPU-" + str(props.uuid)
    h = None
    for i in range(nv.nvmlDeviceGetCount()):                  # the NVML handle of the CUDA device, by UUID
        hh = nv.nvmlDeviceGetHandleByIndex(i)
        u = nv.nvmlDeviceGetUUID(hh)
        u = u.decode() if isinstance(u, bytes) else u
        if u == uuid_t:
            h, nvml_index, uuid_nv = hh, i, u
    assert h is not None, (uuid_t, "no NVML device with the CUDA device's UUID")
    torch.zeros(1, device="cuda")
    out_path = rdir(task) / "energy.json"
    rec = json.loads(out_path.read_text()) if out_path.exists() else {"windows": {}, "idle": []}
    samp = Sampler(nv, h)
    samp.start()
    x_all, lab_all = load_rows(t["data"], 0, t["n"])
    xg = torch.as_tensor(x_all, device="cuda")
    labg = torch.as_tensor(lab_all, device="cuda")
    n = len(x_all)
    rec["meta"] = {"gpu": nv.nvmlDeviceGetName(h) if isinstance(nv.nvmlDeviceGetName(h), str)
                   else nv.nvmlDeviceGetName(h).decode(), "uuid": uuid_nv, "nvml_index": nvml_index,
                   "driver": nv.nvmlSystemGetDriverVersion(), "torch": torch.__version__,
                   "power_limit_W": nv.nvmlDeviceGetPowerManagementLimit(h) / 1e3,
                   "clocks": "not pinned (default DVFS); sampled at 10 Hz per window", "win_s": win_s,
                   "nwin": nwin, "n_puzzles": n, "H": t["H"], "nsup": NSUP, "rule": t["rule"],
                   "host": __import__("socket").gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                   "platform": "datacenter GPU (not edge): " + torch.cuda.get_device_name(0)}
    try:
        import tensorrt as trt
        rec["meta"]["tensorrt"] = trt.__version__
    except Exception:
        pass

    def wait_clean():
        t0 = time.time()
        while True:
            ok, others = card_clean(nv, h)
            if ok:
                return True, []
            if time.time() - t0 > max_wait_s:
                return False, others
            log("card busy with", others, "waiting")
            time.sleep(20)

    def idle_windows(k, label):
        res = []
        for _ in range(k):
            ok, others = wait_clean()
            torch.cuda.synchronize()
            samp.take()
            samp.on.set()
            e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            time.sleep(idle_s)
            e1, t1 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            samp.on.clear()
            w = {"label": label, "E_J": (e1 - e0) / 1e3, "T_s": t1 - t0, "P_W": (e1 - e0) / 1e3 / (t1 - t0),
                 "clean": ok, "others": others, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "loadavg": os.getloadavg(), **samp.take()}
            res.append(w)
            log("idle", label, round(w["P_W"], 2), "W", w.get("sm_MHz_mean"), "MHz", w.get("temp_C_end"), "C")
        return res

    rec["idle"] += idle_windows(nidle, "start")
    dump(out_path, rec)
    idle_P = statistics.median([w["P_W"] for w in rec["idle"] if w["clean"]])

    for cfg in energy_configs(task):
        if only and cfg["name"] not in only:
            continue
        if cfg["name"] in rec["windows"] and len(rec["windows"][cfg["name"]]["windows"]) >= nwin:
            continue
        if cfg["rt"] == "trt":
            waited, lim = 0, float(os.environ.get("E1_ENGINE_WAIT", 0))
            summ_p = rdir(task) / "engines.json"
            while True:
                summ = json.loads(summ_p.read_text()) if summ_p.exists() else {}
                if cfg["engine"] in summ or waited >= lim:
                    break
                if waited % 600 == 0:
                    log("waiting for engine", cfg["engine"])
                time.sleep(30)
                waited += 30
            if summ.get(cfg["engine"], {}).get("BUILD_FAILED") or not (wdir(task) / f"{cfg['engine']}.engine").exists():
                log("missing engine", cfg["engine"])
                continue
        B = cfg["B"]
        runner = make_runner(task, cfg, B)
        ptr = [0]
        first = {}
        rule = t["rule"]

        def unit():
            if B == 1:
                i = ptr[0] % n
                ptr[0] += 1
                runner.load(xg[i:i + 1])
                if cfg["mode"] == "fixed":
                    for _ in range(NSUP):
                        runner.step()
                    steps = NSUP
                else:
                    prev = False
                    steps = NSUP
                    for s in range(1, NSUP + 1):
                        runner.step()
                        if runner.qc is not None and rule == "hrm_own":
                            d = (runner.qh.float() - runner.qc.float()).item()
                            qh, qc = d, 0.0
                        else:
                            qh, qc = runner.qh.float().reshape(-1)[0].item(), 0.0
                        fire, prev = rule_fires(rule if rule != "hrm_own" else "native", s, qh, qc, prev)
                        if fire:
                            steps = s
                            break
                ok = bool((runner.logits.argmax(-1)[0] == labg[i]).all().item())
                if i not in first:
                    first[i] = (steps, ok)
                return steps, 1, int(ok)
            idx = torch.arange(ptr[0], ptr[0] + B, device="cuda") % n
            ptr[0] += B
            runner.load(xg[idx])
            for _ in range(NSUP):
                runner.step()
            okv = (runner.logits.argmax(-1) == labg[idx]).all(-1)
            okc = okv.cpu().numpy()
            for j, i in enumerate(idx.cpu().numpy().tolist()):
                if i not in first:
                    first[i] = (NSUP, bool(okc[j]))
            return NSUP * B, B, int(okc.sum())

        # warm-up (also JIT/cudnn autotune, clocks up)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < warm_s:
            unit()
        torch.cuda.synchronize()
        wins = []
        for k in range(nwin):
            ok_clean, others = wait_clean()
            samp.take()
            samp.on.set()
            steps = puz = solved = units = 0
            torch.cuda.synchronize()
            e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            while True:
                s_, p_, c_ = unit()
                steps += s_
                puz += p_
                solved += c_
                units += 1
                if time.perf_counter() - t0 >= win_s:
                    break
            torch.cuda.synchronize()
            e1, t1 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            samp.on.clear()
            clean_after, others2 = card_clean(nv, h)
            E, T = (e1 - e0) / 1e3, t1 - t0
            Ed = E - idle_P * T
            w = {"E_J": E, "T_s": T, "P_W": E / T, "E_dyn_J": Ed, "idle_P_W": idle_P, "steps": steps,
                 "puzzles": puz, "solved": solved, "units": units,
                 "mJ_per_step": 1e3 * E / steps, "mJ_per_step_dyn": 1e3 * Ed / steps,
                 "J_per_puzzle": E / puz, "J_per_solved": E / max(solved, 1),
                 "J_per_solved_dyn": Ed / max(solved, 1), "ms_per_step": 1e3 * T / steps,
                 "ms_per_puzzle": 1e3 * T / puz, "ms_per_solved": 1e3 * T / max(solved, 1),
                 "acc_window": solved / puz, "clean": bool(ok_clean and clean_after),
                 "others": others + others2, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                 "loadavg": os.getloadavg(), **samp.take()}
            wins.append(w)
            log(task, cfg["name"], f"w{k}", f"{w['P_W']:.1f} W", f"{w['mJ_per_step']:.2f} mJ/step",
                f"{w['ms_per_step']:.3f} ms/step", f"{w['J_per_solved']:.3f} J/solved", f"acc {w['acc_window']:.3f}",
                f"{w.get('sm_MHz_mean')} MHz", "clean" if w["clean"] else "DIRTY")
        # complete the first pass (untimed) so accuracy and mean steps cover all n puzzles of this run
        t_c = time.perf_counter()
        if B == 1:
            for i in range(n):
                if i not in first:
                    ptr[0] = i
                    unit()
        else:
            while len(first) < n:
                unit()
        t_c = time.perf_counter() - t_c
        cov = len(first)
        fa = float(np.mean([v[1] for v in first.values()])) if first else None
        fs = float(np.mean([v[0] for v in first.values()])) if first else None
        good = [w for w in wins if w["clean"]]
        med = {k: statistics.median([w[k] for w in good]) for k in
               ("P_W", "mJ_per_step", "mJ_per_step_dyn", "J_per_puzzle", "J_per_solved", "J_per_solved_dyn",
                "ms_per_step", "ms_per_puzzle", "ms_per_solved", "acc_window")} if good else {}
        rec["windows"][cfg["name"]] = {**{k: v for k, v in cfg.items() if k != "name"}, "windows": wins,
                                       "median_clean": med, "n_clean": len(good),
                                       "first_pass": {"puzzles_covered": cov, "exact": fa, "mean_steps": fs,
                                                      "completion_untimed_s": round(t_c, 1),
                                                      "per_puzzle": {int(i): v for i, v in sorted(first.items())}}}
        dump(out_path, rec)
        del runner
        torch.cuda.empty_cache()
    rec["idle"] += idle_windows(2, "end")
    dump(out_path, rec)
    samp.stop_.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "build", "record", "energy"])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--only", default=None, help="comma-separated engine/config names")
    ap.add_argument("--nvml_index", type=int, default=0, help="ignored: the NVML device is found by UUID")
    ap.add_argument("--win_s", type=float, default=60.0)
    ap.add_argument("--nwin", type=int, default=5)
    ap.add_argument("--idle_s", type=float, default=30.0)
    ap.add_argument("--nidle", type=int, default=5)
    ap.add_argument("--nlimit", type=int, default=None)
    a = ap.parse_args()
    import torch
    assert a.stage == "prep" or torch.cuda.is_available(), "CUDA not available: abort"   # prep is CPU-only
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    log("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "CUDA_VISIBLE_DEVICES",
        os.environ.get("CUDA_VISIBLE_DEVICES"))
    only = a.only.split(",") if a.only else None
    if a.stage == "prep":
        stage_prep(a.task)
    elif a.stage == "build":
        stage_build(a.task, only)
    elif a.stage == "record":
        stage_record(a.task, only, a.nlimit)
    else:
        stage_energy(a.task, a.nvml_index, a.win_s, a.nwin, a.idle_s, a.nidle, only)


if __name__ == "__main__":
    main()
