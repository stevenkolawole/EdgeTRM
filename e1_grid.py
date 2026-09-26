#!/usr/bin/env python3
"""E1 grid (MLSys): energy and latency per solved puzzle on ONE exclusive Babel L40S per model.

Grid per model: toolchain {eager fp32, eager fp16, TensorRT FP16, TensorRT INT8 (int8proj = INT8 on every weight
GEMM, attention score matmuls in FP16: the best correct INT8 configuration from the E1 follow-ups), ONNX Runtime
CUDA EP fp16} x batch {1, 8, 64, 256} x mode {fixed 16 steps, halting exit (M5 fp32-chosen rule per model)}, plus
head-gated polish at batch 1 (M5 R20 "E+P" transposed to a toolchain pair: TensorRT INT8 steps under the exit rule,
then TensorRT FP16 polish steps until the FP16 head's rule fires, at most 4).

Serving model: continuous batching. B slots each hold one puzzle; after every step the exit rule is evaluated per
slot on the host (one sync per step, part of the measured cost); a finished slot emits its answer and is refilled
with the next puzzle (carry reset to the init state). Fixed mode = all slots finish together after 16 steps (one
sync per 16 steps). Puzzles cycle over test rows 0..n-1 in order.

Measurement: NVML total-energy counter; 5 idle windows (30 s) at the start and 2 at the end, median subtracted
("dyn"); per configuration 10 s warm-up then 5 consecutive windows of >= 60 s on the steady-state stream; per window
energy per completed puzzle, mean latency per puzzle (slot fill -> answer on host), throughput, clocks (10 Hz
samples). Accuracy and mean steps from the same run: the first completion of every puzzle 0..n-1 (the stream is
continued untimed until all n are covered). J/solved = (J per completed puzzle) / accuracy; 95% t-intervals over the
5 windows. Output: $E1_RES/<task>/grid.json (resumable per configuration).

    python e1_grid.py --task sudoku_mlp
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e1_energy as E  # noqa: E402

BATCHES = [1, 8, 64, 256]
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}
POLISH_CAP = 4


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def eng_name(H, B, prec):
    return f"grid_H{H}_b{B}_{prec}"


# ----------------------------------------------------------------------------- engines (reuse e1_energy.stage_build)
def build_grid_engines(task):
    t = E.TASKS[task]
    H = t["H"]
    names = [(eng_name(H, B, p), E.dyn_onnx_H(task, H), H, B, p, f"calib_dyn_H{H}.cache")
             for B in BATCHES for p in ("fp16", "int8proj")]
    E.engine_list = lambda _t: names                                    # build exactly these, at opt level 3
    E.energy_configs = lambda _t: [{"engine": n[0]} for n in names]
    try:
        E.stage_build(task)
    except Exception as ex:                                              # a failure is recorded per engine below
        log("stage_build raised", repr(ex)[:300])


# ----------------------------------------------------------------------------- ORT CUDA fp16
def ort_fp16_model(task, H):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16, DEFAULT_OP_BLOCK_LIST
    dst = E.wdir(task) / f"ort_fp16_H{H}.onnx"
    if dst.exists():
        return dst
    m = onnx.load(str(E.dyn_onnx_H(task, H)))
    # keep the RMSNorm reductions in fp32, as the model's own rms_norm does (and as eager fp16 runs it)
    block = list(DEFAULT_OP_BLOCK_LIST) + ["Pow", "ReduceMean", "Sqrt", "Reciprocal"]
    m16 = convert_float_to_float16(m, keep_io_types=True, op_block_list=block, disable_shape_infer=True)
    onnx.save(m16, str(dst), save_as_external_data=True, location=dst.name + ".data")
    return dst


class ORTCudaRunner:
    def __init__(self, task, H, B, stream):
        import onnxruntime as ort
        import torch
        self.torch = torch
        path = ort_fp16_model(task, H)
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 2                  # explicit: the default pins threads and fails under SLURM cgroups
        so.inter_op_num_threads = 1
        prov = [("CUDAExecutionProvider", {"device_id": 0, "user_compute_stream": str(stream.cuda_stream)}),
                "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(str(path), so, providers=prov)
        assert self.sess.get_providers()[0] == "CUDAExecutionProvider", self.sess.get_providers()
        inner, pe = E.build_inner(task, H)
        zh0, zl0 = E.init_carry(inner)
        S, seq = inner.config.seq_len + inner.puzzle_emb_len, inner.config.seq_len
        dev = "cuda"
        self.B = B
        self.zh0, self.zl0 = zh0.to(dev), zl0.to(dev)
        self.zH = [torch.zeros(B, S, 512, device=dev), torch.zeros(B, S, 512, device=dev)]
        self.zL = [torch.zeros(B, S, 512, device=dev), torch.zeros(B, S, 512, device=dev)]
        self.x = torch.zeros(B, seq, dtype=torch.int32, device=dev)
        self.pe = pe.to(dev).float().expand(B, -1).contiguous()
        V = inner.config.vocab_size
        self.logits = torch.zeros(B, seq, V, device=dev)
        self.qh = torch.zeros(B, device=dev)
        self.qc = torch.zeros(B, device=dev)
        self.cur = 0
        self.b = [self._bind(c) for c in (0, 1)]

    def _bind(self, c):
        o = 1 - c
        b = self.sess.io_binding()

        def bi(n, t, dt):
            b.bind_input(n, "cuda", 0, dt, tuple(t.shape), t.data_ptr())

        def bo(n, t):
            b.bind_output(n, "cuda", 0, np.float32, tuple(t.shape), t.data_ptr())
        bi("z_H", self.zH[c], np.float32)
        bi("z_L", self.zL[c], np.float32)
        bi("inputs", self.x, np.int32)
        bi("puzzle_embedding", self.pe, np.float32)
        bo("z_H_out", self.zH[o])
        bo("z_L_out", self.zL[o])
        bo("logits", self.logits)
        bo("q_halt", self.qh)
        bo("q_cont", self.qc)
        return b

    def step(self):
        self.sess.run_with_iobinding(self.b[self.cur])
        self.cur = 1 - self.cur

    def reset(self, idx, xrows):
        self.x[idx] = xrows.to(self.x.dtype)
        self.zH[self.cur][idx] = self.zh0
        self.zL[self.cur][idx] = self.zl0


# ----------------------------------------------------------------------------- slot resets for the e1_energy runners
def reset_eager(r, idx, xrows):
    torch = r.torch
    with torch.inference_mode():                   # in place (the carry tensors are inference tensors)
        r.x[idx] = xrows.to(r.x.dtype)
        r.zH[idx] = r.zh0
        r.zL[idx] = r.zl0


def reset_trt(r, idx, xrows):
    r.buf["inputs"][idx] = xrows.to(r.buf["inputs"].dtype)
    r.zH[r.cur][idx] = r.zh0
    r.zL[r.cur][idx] = r.zl0


def make(task, tool, B, H, stream):
    import torch
    torch.cuda.set_stream(stream)
    if tool.startswith("eager"):
        r = E.EagerRunner(task, H, B, tool.split("_")[1])
        r.x = r.x.clone()
        return r, reset_eager
    if tool.startswith("trt"):
        prec = tool.split("_")[1]
        ep = E.wdir(task) / f"{eng_name(H, B, prec)}.engine"
        if not ep.exists():
            raise FileNotFoundError(ep)
        return E.TRTRunner(ep, task, H, B), reset_trt
    return ORTCudaRunner(task, H, B, stream), ORTCudaRunner.reset


# ----------------------------------------------------------------------------- the stream
class Stream:
    """Continuous-batching driver over one runner (or a cheap/polish pair at batch 1)."""

    def __init__(self, task, runner, reset, B, mode, xg, labg, polish=None):
        import torch
        self.torch, self.task = torch, task
        self.r, self.reset, self.B, self.mode = runner, reset, B, mode
        self.rule = E.TASKS[task]["rule"]
        self.xg, self.labg, self.n = xg, labg, xg.shape[0]
        self.polish = polish                                   # (runner, reset) for FP16 polish, batch 1 only
        self.ptr = 0
        dev = "cuda"
        self.pid = torch.zeros(B, dtype=torch.long, device=dev)
        self.steps = torch.zeros(B, dtype=torch.int32, device=dev)
        self.prev = torch.zeros(B, dtype=torch.bool, device=dev)
        self.t_fill = [0.0] * B
        self.first = {}
        self.fill(torch.arange(B, device=dev))

    def fill(self, idx):
        torch = self.torch
        k = idx.numel()
        p = (torch.arange(self.ptr, self.ptr + k, device="cuda") % self.n)
        self.ptr += k
        self.pid[idx] = p
        self.steps[idx] = 0
        self.prev[idx] = False
        self.reset(self.r, idx, self.xg[p])
        now = time.perf_counter()
        for i in idx.tolist():
            self.t_fill[i] = now

    def _fire(self, r, steps, prev):
        qh = r.qh.float().reshape(-1)
        if self.rule == "hrm_own" and r.qc is not None:
            halt = qh > r.qc.float().reshape(-1)
        else:
            halt = qh > 0
        if self.rule == "patience2":
            fire = halt & prev
        elif self.rule == "min2native":
            fire = halt & (steps >= 2)
        else:
            fire = halt
        return fire | (steps >= E.NSUP), halt

    def emit(self, idx, logits, steps_used):
        """Answers of finished slots; returns (n_done, n_correct, sum_latency)."""
        torch = self.torch
        ok = (logits[idx].argmax(-1) == self.labg[self.pid[idx]]).all(-1)
        okc = ok.cpu().numpy()
        now = time.perf_counter()
        pids = self.pid[idx].cpu().numpy()
        st = steps_used[idx].cpu().numpy() if torch.is_tensor(steps_used) else [steps_used] * len(pids)
        lat = 0.0
        for j, i in enumerate(idx.tolist()):
            lt = now - self.t_fill[i]
            lat += lt
            p = int(pids[j])
            if p not in self.first:
                self.first[p] = (int(st[j]), bool(okc[j]), lt)
        return len(pids), int(okc.sum()), lat

    def unit(self):
        """Advance the stream; returns (slot-steps executed, puzzles done, correct, latency sum)."""
        torch = self.torch
        r, B = self.r, self.B
        if self.mode == "fixed":
            for _ in range(E.NSUP):
                r.step()
            idx = torch.arange(B, device="cuda")
            d, c, lat = self.emit(idx, r.logits, E.NSUP)
            self.fill(idx)
            return E.NSUP * B, d, c, lat
        if self.mode == "exit":
            r.step()
            self.steps += 1
            fire, halt = self._fire(r, self.steps, self.prev)
            self.prev = halt
            idx = torch.nonzero(fire).flatten()
            if idx.numel():
                d, c, lat = self.emit(idx, r.logits, self.steps)
                self.fill(idx)
                return B, d, c, lat
            return B, 0, 0, 0.0
        # polish (batch 1): cheap runner under the exit rule, then FP16 polish until its rule fires (<= POLISH_CAP)
        assert B == 1
        pr, preset = self.polish
        n_cheap = 0
        prev = torch.zeros(1, dtype=torch.bool, device="cuda")
        st = torch.zeros(1, dtype=torch.int32, device="cuda")
        while True:
            r.step()
            n_cheap += 1
            st += 1
            fire, halt = self._fire(r, st, prev)
            prev = halt
            if bool(fire.item()):
                break
        # hand the carry to the FP16 engine
        pr.buf["inputs"].copy_(r.buf["inputs"])
        pr.cur = 0
        pr.zH[0].copy_(r.zH[r.cur])
        pr.zL[0].copy_(r.zL[r.cur])
        prev = torch.zeros(1, dtype=torch.bool, device="cuda")
        stp = torch.zeros(1, dtype=torch.int32, device="cuda")
        n_pol = 0
        while n_pol < POLISH_CAP:
            pr.step()
            n_pol += 1
            stp += 1
            fire, halt = self._fire(pr, stp + (1 if self.rule == "min2native" else 0), prev)
            prev = halt
            if bool(fire.item()) and not (self.rule == "patience2" and n_pol < 2):
                break
        idx = torch.zeros(1, dtype=torch.long, device="cuda")
        d, c, lat = self.emit(idx, pr.logits, n_cheap + n_pol)
        self.polish_steps.append((n_cheap, n_pol))
        self.fill(idx)
        return n_cheap + n_pol, d, c, lat


def ci95(v):
    if len(v) < 2:
        return None
    m = statistics.mean(v)
    h = T95.get(len(v), 2.0) * statistics.stdev(v) / len(v) ** 0.5
    return [m - h, m + h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(E.TASKS))
    ap.add_argument("--win_s", type=float, default=60.0)
    ap.add_argument("--nwin", type=int, default=5)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    import torch
    import pynvml as nv
    assert torch.cuda.is_available(), "CUDA not available: abort"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        import onnxruntime as ort
        ort.preload_dlls()
    except Exception as ex:
        log("ORT preload", repr(ex)[:200])
    task = a.task
    t = E.TASKS[task]
    H = t["H"]
    build_grid_engines(task)
    nv.nvmlInit()
    uuid_t = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    h = next(nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())
             if (lambda u: u.decode() if isinstance(u, bytes) else u)(nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(i))) == uuid_t)
    samp = E.Sampler(nv, h)
    samp.start()
    outp = E.rdir(task) / "grid.json"
    rec = json.loads(outp.read_text()) if outp.exists() else {"configs": {}, "idle": []}
    rec["meta"] = {"gpu": torch.cuda.get_device_name(0), "uuid": uuid_t, "host": __import__("socket").gethostname(),
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "driver": nv.nvmlSystemGetDriverVersion(),
                   "torch": torch.__version__, "H": H, "nsup": E.NSUP, "rule": t["rule"], "n": t["n"],
                   "win_s": a.win_s, "nwin": a.nwin, "polish_cap": POLISH_CAP,
                   "platform": "Babel L40S, one exclusive card (datacenter GPU, not edge); clocks not pinned"}
    try:
        import tensorrt
        rec["meta"]["tensorrt"] = tensorrt.__version__
        rec["meta"]["onnxruntime"] = ort.__version__
    except Exception:
        pass

    def clean():
        return not [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h) if p.pid != os.getpid()]

    def idle(k, label):
        for _ in range(k):
            torch.cuda.synchronize()
            samp.take()
            samp.on.set()
            e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            time.sleep(30)
            e1, t1 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            samp.on.clear()
            w = {"label": label, "P_W": (e1 - e0) / 1e3 / (t1 - t0), "clean": clean(),
                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **samp.take()}
            rec["idle"].append(w)
            log("idle", label, round(w["P_W"], 2), "W", w.get("sm_MHz_mean"))
        E.dump(outp, rec)

    idle(5, "start")
    idle_P = statistics.median([w["P_W"] for w in rec["idle"] if w["clean"]])
    x, lab = E.load_rows(t["data"], 0, t["n"])
    xg = torch.as_tensor(x, device="cuda")
    labg = torch.as_tensor(lab, device="cuda")
    tools = ["eager_fp32", "eager_fp16", "trt_fp16", "trt_int8proj", "ort_fp16"]
    cfgs = [(tool, B, mode) for B in BATCHES for tool in tools for mode in ("fixed", "exit")]
    cfgs.append(("trt_int8proj", 1, "polish"))
    only = set(a.only.split(",")) if a.only else None
    for tool, B, mode in cfgs:
        name = f"{tool}_b{B}_{mode}"
        if only and name not in only:
            continue
        if name in rec["configs"] and (rec["configs"][name].get("n_clean", 0) >= a.nwin or rec["configs"][name].get("skipped")):
            continue
        stream = torch.cuda.Stream()
        try:
            runner, reset = make(task, tool, B, H, stream)
            pol = None
            if mode == "polish":
                pol = (make(task, "trt_fp16", 1, H, stream)[0], reset_trt)
            S = Stream(task, runner, reset, B, mode, xg, labg, pol)
            S.polish_steps = []
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 10:
                S.unit()
            torch.cuda.synchronize()
        except Exception as ex:                                      # OOM, missing engine, ORT EP failure: recorded
            rec["configs"][name] = {"tool": tool, "B": B, "mode": mode, "skipped": True, "reason": repr(ex)[:300]}
            E.dump(outp, rec)
            log(name, "SKIPPED", repr(ex)[:200])
            torch.cuda.empty_cache()
            continue
        wins = []
        for k in range(a.nwin):
            ok0 = clean()
            samp.take()
            samp.on.set()
            steps = done = corr = 0
            lat = 0.0
            torch.cuda.synchronize()
            e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            while time.perf_counter() - t0 < a.win_s:
                s_, d_, c_, l_ = S.unit()
                steps += s_
                done += d_
                corr += c_
                lat += l_
            torch.cuda.synchronize()
            e1, t1 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            samp.on.clear()
            E_, T_ = (e1 - e0) / 1e3, t1 - t0
            w = {"E_J": E_, "T_s": T_, "P_W": E_ / T_, "E_dyn_J": E_ - idle_P * T_, "slot_steps": steps, "done": done,
                 "correct": corr, "J_per_puzzle": E_ / max(done, 1), "J_per_puzzle_dyn": (E_ - idle_P * T_) / max(done, 1),
                 "puzzles_per_s": done / T_, "lat_ms_mean": 1e3 * lat / max(done, 1),
                 "mJ_per_slot_step": 1e3 * E_ / max(steps, 1), "clean": bool(ok0 and clean()),
                 "loadavg": os.getloadavg(), **samp.take()}
            wins.append(w)
            log(task, name, f"w{k}", f"{w['P_W']:.1f} W", f"{w['J_per_puzzle']:.4f} J/puz",
                f"{w['lat_ms_mean']:.2f} ms", f"{w['puzzles_per_s']:.1f} puz/s", f"{w.get('sm_MHz_mean')} MHz",
                "clean" if w["clean"] else "DIRTY")
        tc = time.perf_counter()
        guard = 0
        while len(S.first) < S.n and guard < 10 ** 7:
            S.unit()
            guard += 1
        tc = time.perf_counter() - tc
        acc = float(np.mean([v[1] for v in S.first.values()]))
        msteps = float(np.mean([v[0] for v in S.first.values()]))
        good = [w for w in wins if w["clean"]]
        jp = [w["J_per_puzzle"] for w in good]
        jd = [w["J_per_puzzle_dyn"] for w in good]
        lt = [w["lat_ms_mean"] for w in good]
        tp = [w["puzzles_per_s"] for w in good]
        summ = {"acc": acc, "mean_steps": msteps, "n_clean": len(good),
                "J_solved_median": statistics.median(jp) / acc if good and acc else None,
                "J_solved_ci95": [v / acc for v in ci95(jp)] if len(jp) > 1 and acc else None,
                "J_solved_dyn_median": statistics.median(jd) / acc if good and acc else None,
                "J_solved_dyn_ci95": [v / acc for v in ci95(jd)] if len(jd) > 1 and acc else None,
                "lat_ms_median": statistics.median(lt) if good else None, "lat_ms_ci95": ci95(lt),
                "ms_per_solved_median": 1e3 / statistics.median(tp) / acc if good and acc else None,
                "puzzles_per_s_median": statistics.median(tp) if good else None,
                "P_W_median": statistics.median([w["P_W"] for w in good]) if good else None,
                "sm_MHz_median": statistics.median([w.get("sm_MHz_mean", 0) for w in good]) if good else None}
        if mode == "polish" and S.polish_steps:
            ps = np.array(S.polish_steps)
            summ["polish_mean_cheap_steps"], summ["polish_mean_polish_steps"] = float(ps[:, 0].mean()), float(ps[:, 1].mean())
        rec["configs"][name] = {"tool": tool, "B": B, "mode": mode, "idle_P_W": idle_P, "windows": wins,
                                "summary": summ, "completion_untimed_s": round(tc, 1),
                                "first_pass": {int(p): v for p, v in sorted(S.first.items())}}
        E.dump(outp, rec)
        log(task, name, "acc", round(acc, 4), "steps", round(msteps, 2), "J/solved", summ["J_solved_median"],
            "lat ms", summ["lat_ms_median"])
        del S, runner
        torch.cuda.empty_cache()
    idle(2, "end")
    samp.stop_.set()


if __name__ == "__main__":
    main()
