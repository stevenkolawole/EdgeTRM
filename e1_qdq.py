#!/usr/bin/env python3
"""E1 follow-up (2): TensorRT's EXPLICIT (QDQ) INT8 path on the same QDQ graphs ONNX Runtime and QNN executed.

Graphs
  TRM (sudoku_mlp, sudoku_attn, maze): X3's AI Hub / AIMET w8a8 stepc graph at H=1 (p4d
    ~/aihub_artifacts/h1/<task>/infer/qdq_w8a8/model.onnx, copied to $E1_X3ART/<task>/qdq_w8a8/), the very graph whose
    ORT execution gave 0/100 on Sudoku-MLP and whose QNN compile ran on the S24.
  HRM Sudoku: no AIMET graph exists (X3 never sent HRM through AI Hub). A QDQ graph in the same scheme is made here
    with ONNX Runtime's static quantizer (uint8 asymmetric per-tensor activations on every quantizable op, int8
    symmetric per-channel weights, MinMax calibration on 64 real carries from Sudoku test rows 1000+) from the
    dynamic-batch H=2 export. Labelled "ORT-quantizer graph", never "AIMET".

TensorRT supports only symmetric INT8 (zero point 0) in Q/DQ, so every graph is rewritten in two steps:
  fold  Q->DQ pairs on constant initializers that are not INT8 weights (AIMET's uint8 embedding table, the int32
        Gemm bias) are replaced by their exact fake-quantized float values (verified equal under ORT);
  sym   every uint8 asymmetric activation Q/DQ (scale s, zero point z) becomes int8 symmetric with scale
        s * max(z, 255 - z) / 127 (the smallest symmetric grid that covers the same range: never clips, up to 2x
        coarser for one-sided tensors such as x*x, sigmoid or the RMSNorm variance).
Each graph is then executed by
  ORT reference QDQ (CPU, ORT_DISABLE_ALL: every Q/DQ as written) on the original asymmetric graph and on the
      symmetric graph (separates the effect of symmetrisation from the effect of TensorRT), and
  TensorRT 10.13 explicit quantization of the symmetric graph: INT8 + FP16 elsewhere (qdq_fp16) and INT8 + FP32
      elsewhere (qdq_fp32),
as a 16-step carry chain on the same puzzles (Sudoku 0-99 / Maze 0-49 for ORT; TensorRT also on 0-999 / 0-899),
scored like e1_energy.py record: exact per step, cell agreement and z_H cosine per step vs the eager fp32 reference.

    python e1_qdq.py --task sudoku_mlp       (all stages, resumable; results $E1_RES/<task>/qdq.json)
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e1_energy as E  # noqa: E402

N_ORT = {"sudoku_mlp": 100, "sudoku_attn": 100, "maze": 50, "hrm_sudoku": 100}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ----------------------------------------------------------------------------- graph rewriting
def _vals(g):
    from onnx import numpy_helper
    init = {i.name: i for i in g.initializer}
    cons = {n.output[0]: n for n in g.node if n.op_type == "Constant"}

    def get(name):
        if name in init:
            return numpy_helper.to_array(init[name])
        if name in cons:
            return numpy_helper.to_array(cons[name].attribute[0].t)
        return None
    return init, cons, get


def fold_and_sym(src, fold_dst, sym_dst):
    import onnx
    from onnx import numpy_helper, TensorProto
    m = onnx.load(str(src))
    g = m.graph
    init, cons, get = _vals(g)
    consumers = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    st = {"folded_q_dq_pairs": 0, "folded_dq": 0, "sym_q": 0, "sym_dq": 0, "int8_weight_pairs_kept": 0,
          "coarsening_factor_max": 1.0, "coarsening_factor_median": None}
    drop, new_init = set(), []
    # 1. fold
    for n in list(g.node):
        if n.op_type == "QuantizeLinear" and n.input[0] in init:
            zp = get(n.input[2]) if len(n.input) > 2 else None
            if zp is not None and zp.dtype == np.int8:
                st["int8_weight_pairs_kept"] += 1
                continue
            x = numpy_helper.to_array(init[n.input[0]]).astype(np.float32)
            s = get(n.input[1]).astype(np.float32)
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            if s.ndim == 1 and s.size > 1:
                shp = [1] * x.ndim
                shp[axis] = -1
                s = s.reshape(shp)
                zp = zp.reshape(shp)
            lo, hi = (0, 255) if zp.dtype == np.uint8 else ((-128, 127) if zp.dtype == np.int8 else (0, 65535))
            fq = ((np.clip(np.round(x / s) + zp.astype(np.float32), lo, hi) - zp.astype(np.float32)) * s).astype(np.float32)
            for d in consumers.get(n.output[0], []):
                assert d.op_type == "DequantizeLinear", d.op_type
                new_init.append(numpy_helper.from_array(fq, d.output[0]))
                drop.add(id(d))
            drop.add(id(n))
            st["folded_q_dq_pairs"] += 1
        elif n.op_type == "DequantizeLinear" and n.input[0] in init and \
                numpy_helper.to_array(init[n.input[0]]).dtype == np.int32:
            q = numpy_helper.to_array(init[n.input[0]]).astype(np.float64)
            s = get(n.input[1]).astype(np.float64)
            zp = get(n.input[2]).astype(np.float64) if len(n.input) > 2 else 0.0
            new_init.append(numpy_helper.from_array(((q - zp) * s).astype(np.float32), n.output[0]))
            drop.add(id(n))
            st["folded_dq"] += 1
    keep = [n for n in g.node if id(n) not in drop]
    del g.node[:]
    g.node.extend(keep)
    g.initializer.extend(new_init)
    onnx.save(m, str(fold_dst), save_as_external_data=True, location=pathlib.Path(fold_dst).name + ".data")
    # 2. symmetrise uint8 activation Q/DQ
    init, cons, get = _vals(g)
    made, factors = {}, []
    for n in g.node:
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear") or len(n.input) < 3:
            continue
        zp = get(n.input[2])
        if zp is None or zp.dtype != np.uint8:
            continue
        s = get(n.input[1]).astype(np.float64)
        key = (n.input[1], n.input[2])
        if key not in made:
            z = zp.astype(np.float64)
            f = np.maximum(z, 255.0 - z) / 127.0
            factors.append(float(np.max(f)))
            sn, zn = f"{n.input[1]}__sym{len(made)}", f"{n.input[2]}__sym{len(made)}"
            g.initializer.extend([numpy_helper.from_array((s * f).astype(np.float32), sn),
                                  numpy_helper.from_array(np.zeros_like(zp, dtype=np.int8), zn)])
            made[key] = (sn, zn)
        n.input[1], n.input[2] = made[key]
        st["sym_q" if n.op_type == "QuantizeLinear" else "sym_dq"] += 1
    st["coarsening_factor_max"] = round(max(factors), 3) if factors else 1.0
    st["coarsening_factor_median"] = round(float(np.median(factors)), 3) if factors else None
    st["n_sym_encodings"] = len(made)
    onnx.save(m, str(sym_dst), save_as_external_data=True, location=pathlib.Path(sym_dst).name + ".data")
    return st


def hrm_ort_quantize(task, H, dst):
    """ONNX Runtime static quantizer, AIMET-like scheme, on real carries (HRM only: no AIMET graph exists)."""
    import onnx
    from onnxruntime.quantization import (quantize_static, CalibrationDataReader, QuantFormat, QuantType,
                                          CalibrationMethod)
    src = E.dyn_onnx_H(task, H)
    data, meta = E.calib_samples(task, H)
    n = min(64, data["z_H"].shape[0])

    class R(CalibrationDataReader):
        def __init__(self):
            self.i = 0

        def get_next(self):
            if self.i >= n:
                return None
            d = {k: data[k][self.i:self.i + 1] for k in E.INS}
            self.i += 1
            return d
    ops = sorted({nd.op_type for nd in onnx.load(str(src), load_external_data=False).graph.node} -
                 {"Constant", "Shape", "ConstantOfShape", "Gather", "Unsqueeze", "Squeeze", "Range", "Cast",
                  "Equal", "Where", "Expand"})
    quantize_static(str(src), str(dst), R(), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                    weight_type=QuantType.QInt8, per_channel=True, op_types_to_quantize=ops,
                    calibrate_method=CalibrationMethod.MinMax, use_external_data_format=True,
                    extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    g = onnx.load(str(dst), load_external_data=False).graph
    from collections import Counter
    return {"source": str(src), "calib_carries": n, "op_types_requested": ops,
            "q_nodes": sum(1 for x in g.node if x.op_type == "QuantizeLinear"),
            "dq_nodes": sum(1 for x in g.node if x.op_type == "DequantizeLinear"),
            "quantized_consumer_ops": dict(Counter(c.op_type for x in g.node if x.op_type == "DequantizeLinear"
                                                    for c in g.node if x.output[0] in c.input))}


# ----------------------------------------------------------------------------- runners
class ORTRunner:
    """Batch-1 ORT chain with the E.* runner interface (outputs as CUDA tensors for the shared scorer)."""

    def __init__(self, path, task, H, level="disable"):
        import onnxruntime as ort
        import torch
        self.torch = torch
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.environ.get("E1_ORT_THREADS", 4))
        so.graph_optimization_level = {"disable": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
                                       "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL}[level]
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.outs = [o.name for o in self.sess.get_outputs()]
        inner, pe = E.build_inner(task, H)
        zh0, zl0 = E.init_carry(inner)
        self.S = inner.config.seq_len + inner.puzzle_emb_len
        self.zh0, self.zl0 = zh0.numpy(), zl0.numpy()
        self.pe = pe.numpy().astype(np.float32)
        self.qc = None

    def load(self, x):
        assert x.shape[0] == 1
        self.x = x.cpu().numpy().astype(np.int32)
        self.zH = np.broadcast_to(self.zh0, (1, self.S, 512)).astype(np.float32).copy()
        self.zL = np.broadcast_to(self.zl0, (1, self.S, 512)).astype(np.float32).copy()

    def step(self):
        o = dict(zip(self.outs, self.sess.run(self.outs, {"z_H": self.zH, "z_L": self.zL, "inputs": self.x,
                                                           "puzzle_embedding": self.pe})))
        self.zH, self.zL = o["z_H_out"], o["z_L_out"]
        t = self.torch
        self.logits = t.as_tensor(o["logits"]).cuda()
        self.qh = t.as_tensor(o["q_halt"]).cuda()
        self.qc = t.as_tensor(o["q_cont"]).cuda() if "q_cont" in o else None

    def cur_zH(self):
        return self.torch.as_tensor(self.zH).cuda()


def build_qdq_engine(onnx_path, eng_path, fp16, B=1):
    import tensorrt as trt
    msgs = []

    class Lg(trt.ILogger):
        def __init__(self):
            super().__init__()

        def log(self, sev, msg):
            if sev <= trt.ILogger.Severity.WARNING:
                msgs.append(str(msg))

    lg = Lg()
    b = trt.Builder(lg)
    net = b.create_network(0)
    p = trt.OnnxParser(net, lg)
    ok = p.parse_from_file(str(onnx_path))
    if not ok:
        return {"BUILD_FAILED": True, "stage": "parse", "errors": [str(p.get_error(i))[:400] for i in range(p.num_errors)]}
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.clear_flag(trt.BuilderFlag.TF32)
    cfg.set_flag(trt.BuilderFlag.INT8)
    if fp16:
        cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if any(-1 in tuple(net.get_input(i).shape) for i in range(net.num_inputs)):
        prof = b.create_optimization_profile()
        for i in range(net.num_inputs):
            inp = net.get_input(i)
            shp = (B,) + tuple(inp.shape)[1:]
            prof.set_shape(inp.name, shp, shp, shp)
        cfg.add_optimization_profile(prof)
    t0 = time.time()
    ser = b.build_serialized_network(net, cfg)
    if ser is None:
        return {"BUILD_FAILED": True, "stage": "build", "build_s": round(time.time() - t0, 1),
                "errors": [m[:400] for m in msgs if "rror" in m][-4:]}
    by = bytes(ser)
    pathlib.Path(eng_path).write_bytes(by)
    lp, info = E.layer_precisions(by)
    return {"build_s": round(time.time() - t0, 1), "size_MB": round(len(by) / 1e6, 2), **lp,
            "warnings_sample": sorted({m[:200] for m in msgs})[:10]}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(E.TASKS))
    a = ap.parse_args()
    import torch
    assert torch.cuda.is_available(), "CUDA not available: abort"
    torch.backends.cuda.matmul.allow_tf32 = False
    task = a.task
    t = E.TASKS[task]
    H = 1 if t["kind"] == "trm" else t["H"]
    wd = E.wdir(task) / "qdq"
    wd.mkdir(parents=True, exist_ok=True)
    outp = E.rdir(task) / "qdq.json"
    rec = json.loads(outp.read_text()) if outp.exists() else {}
    rec["meta"] = {"H": H, "gpu": torch.cuda.get_device_name(0), "host": __import__("socket").gethostname(),
                   "slurm_job": os.environ.get("SLURM_JOB_ID")}
    import onnxruntime
    import tensorrt
    rec["meta"].update(onnxruntime=onnxruntime.__version__, tensorrt=tensorrt.__version__)
    # graphs
    if t["kind"] == "trm":
        orig = E.X3ART / t["x3"] / "qdq_w8a8" / "model.onnx"
        rec["meta"]["graph"] = f"AIMET w8a8 stepc H=1 (AI Hub quantize job; X3), {orig}"
    else:
        orig = wd / "ortq_asym.onnx"
        if not orig.exists():
            rec["ortq"] = hrm_ort_quantize(task, H, orig)
            E.dump(outp, rec)
        rec["meta"]["graph"] = f"ORT-quantizer QDQ graph (uint8 asym act, int8 per-channel w), {orig}"
    fold, sym = wd / "fold.onnx", wd / "sym.onnx"
    if not sym.exists():
        rec["rewrite"] = fold_and_sym(orig, fold, sym)
        E.dump(outp, rec)
        log(task, "rewrite", rec["rewrite"])
    # fold check: identical outputs to the original under ORT on one real carry
    x, lab = E.load_rows(t["data"], 0, t["n"])
    lab8 = lab.astype(np.uint8)
    if "fold_check_maxabs" not in rec:
        r0 = ORTRunner(orig, task, H)
        r1 = ORTRunner(fold, task, H)
        xx = torch.as_tensor(x[:1])
        r0.load(xx)
        r1.load(xx)
        for _ in range(2):
            r0.step()
            r1.step()
        rec["fold_check_maxabs"] = [float((r0.cur_zH() - r1.cur_zH()).abs().max()),
                                    float((r0.logits - r1.logits).abs().max())]
        E.dump(outp, rec)
        log(task, "fold check", rec["fold_check_maxabs"])
    # reference (eager fp32, batched) with z_H of the first K
    K = 100 if t["data"] == "sudoku" else 50
    ref = E.rollout_record(task, E.EagerRunner(task, H, t["B"], "fp32"), x, lab8, t["B"], K)
    refn = {"exact_per_step": [round(float(v), 4) for v in (ref["preds"] == lab8[:, None]).all(-1).mean(0)]}
    rec["ref_eager_fp32"] = refn
    runs = [("ort_ref_asym", "ort", orig, None), ("ort_ref_sym", "ort", sym, None),
            ("trt_qdq_fp16", "trt", sym, True), ("trt_qdq_fp32", "trt", sym, False)]
    for name, kind, path, fp16 in runs:
        if name in rec and rec[name].get("n"):
            continue
        if kind == "ort":
            nn_ = N_ORT[task]
            runner = ORTRunner(path, task, H)
        else:
            ep = wd / f"{name}.engine"
            if not ep.exists():
                info = build_qdq_engine(path, ep, fp16)
                rec[f"{name}_engine"] = info
                E.dump(outp, rec)
                log(task, name, "engine", {k: v for k, v in info.items() if k != "warnings_sample"})
                if info.get("BUILD_FAILED"):
                    rec[name] = {"BUILD_FAILED": True}
                    E.dump(outp, rec)
                    continue
            nn_ = t["n"]
            torch.cuda.set_stream(torch.cuda.Stream())
            runner = E.TRTRunner(ep, task, H, 1)
        t0 = time.time()
        r = E.rollout_record(task, runner, x[:nn_], lab8[:nn_], 1, min(K, nn_), ref["zref"])
        s = E.summarize_record(task, r, lab8[:nn_], ref["preds"][:nn_])
        s["wall_s"] = round(time.time() - t0, 1)
        s["first_n_exact_per_step"] = [int(v) for v in (r["preds"][:N_ORT[task]] == lab8[:N_ORT[task], None]).all(-1).sum(0)]
        rec[name] = s
        E.dump(outp, rec)
        np.savez_compressed(wd / f"{name}.npz", preds=r["preds"], qh=r["qh"], cos=r["cos"])
        log(task, name, "exact@8/16", s["exact_per_step"][7], s["exact_per_step"][-1], "first-n @8/16",
            s["first_n_exact_per_step"][7], s["first_n_exact_per_step"][-1], "agree@16",
            s["ref_cell_agreement_per_step"][-1], "cos@16", s["zH_cos_per_step_meanK"][-1])
        del runner
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
