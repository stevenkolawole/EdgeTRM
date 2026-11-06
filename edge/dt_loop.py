#!/usr/bin/env python3
"""Board runner for the DeepThinking mazes (second looped family, no halting head): the projection graph once per
maze, then the recurrent step graph looped from the host with the state fed back (export_dt.py: dt_proj / dt_step,
size-33 mazes, 72x72 images), stopped at a fixed T or by a head-free exit rule, with edge_loop.py's power sampler,
idle window and timed windows (the same functions: the energy numbers mean the same thing as the TRM/HRM ones).

Exit rules (edge_loop.headfree_fires; tuned on the fp32 M5 record results/breadth/babel/m5/dt_maze33/fp32.npz,
mazes 50-999, cap 300, smallest mean steps within 0.5 points of fixed 300, then frozen):
  resid0.0794  dt_exit's residual ||h_t - h_{t-1}|| / ||h_t|| below 10^-1.1 (fp32: 100% on mazes 0-49, 269 steps)
  stable1      masked answer unchanged for one iteration (fp32: 100% on mazes 0-49, 103 steps)
Neither fires at iteration 1. The answer is dt_exit's: argmax of the 2-way head on open cells (walls = 0); a maze is
solved when every cell equals the target. Only the signal the rule needs is read back inside the timed step (the
logits for stable, the state for resid); fixed T reads the logits once at the end.

Passes: check (fixed --check_T iterations per maze: exact / changed cells / residual per iteration against the host
reference host_fp32.npz from export_dt.py, and the offline exit of both rules), record (every maze once in the run
mode), idle, energy (--nwin windows of --win_s looping the mazes); record layout as edge_loop.py.

    python3 dt_loop.py --backend trt --proj engines/dt_mazes/dt_proj_fold_fp32.plan \
        --step engines/dt_mazes/dt_step_fold_fp16.plan --precision fp16 --mazes assets/dt_mazes/mazes33_0_50.npz \
        --ref assets/dt_mazes/host_fp32.npz --check_only --out results/nano-a_dt_check_fp16.json
    python3 dt_loop.py ... --mode fixed --depth 100 --no_check --win_s 120 --out results/nano-a_dt_fixed100.json
    python3 dt_loop.py ... --mode exit --rule stable1 --depth 300 --no_check --win_s 120 --out ...
Laptop smoke test: --backend ort with the .onnx files and --no_power.
"""
import argparse
import os
import platform
import socket
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_loop as E  # noqa: E402

RULES = ["resid0.0794", "stable1"]


class DtOrt(object):
    backend = "ort"

    def __init__(self, proj, step, threads=None):
        import onnxruntime as ort
        self.version = ort.__version__
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.proj = ort.InferenceSession(proj, so, providers=["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(step, so, providers=["CPUExecutionProvider"])
        self.providers = self.sess.get_providers()

    def load(self, x):
        self.x = np.ascontiguousarray(x.astype(np.float32))
        self.h = self.proj.run(None, {"x": self.x})[0]

    def step(self):
        self.h, self._logits = self.sess.run(None, {"h": self.h, "x": self.x})

    def logits(self):
        return self._logits

    def state(self):
        return self.h


def _trt_engine(trt, runtime, plan):
    with open(plan, "rb") as f:
        eng = runtime.deserialize_cuda_engine(f.read())
    assert eng is not None, "engine failed to deserialize: " + plan
    ctx = eng.create_execution_context()
    v3 = hasattr(eng, "num_io_tensors") and hasattr(ctx, "execute_async_v3")
    io = []                                                   # (name, shape, dtype), engine order
    if v3:
        for i in range(eng.num_io_tensors):
            n = eng.get_tensor_name(i)
            io.append((n, tuple(eng.get_tensor_shape(n)), trt.nptype(eng.get_tensor_dtype(n))))
    else:
        for i in range(eng.num_bindings):
            io.append((eng.get_binding_name(i), tuple(eng.get_binding_shape(i)), trt.nptype(eng.get_binding_dtype(i))))
    return eng, ctx, v3, io


class DtTrt(object):
    """Two TensorRT engines sharing device buffers: the projection writes h0 straight into the step's state buffer;
    the state ping-pongs between two buffers (no copy per iteration)."""
    backend = "trt"

    def __init__(self, proj, step, cuda=None):
        import tensorrt as trt
        self.version = trt.__version__
        self.cu = cuda or E.CudaMem()
        self.mem_mode = self.cu.mode
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        rt = trt.Runtime(logger)
        self.p = _trt_engine(trt, rt, proj)
        self.s = _trt_engine(trt, rt, step)
        sio = dict((n, (shp, dt)) for n, shp, dt in self.s[3])
        for n in ("h", "x", "h_out", "logits"):
            assert n in sio, "step engine lacks %s (have %s)" % (n, list(sio))
        assert sio["h"] == sio["h_out"], sio
        self.dt = dict((n, v[1]) for n, v in sio.items())
        nb = lambda n: int(np.prod(sio[n][0])) * np.dtype(sio[n][1]).itemsize  # noqa: E731
        self.h_dev = [self.cu.alloc(nb("h")), self.cu.alloc(nb("h"))]
        self.x_dev, self.l_dev = self.cu.alloc(nb("x")), self.cu.alloc(nb("logits"))
        self.x_shape = sio["x"][0]
        self.h_host = np.zeros(sio["h"][0], sio["h"][1])
        self.l_host = np.zeros(sio["logits"][0], sio["logits"][1])
        self.cur = 0

    def _run(self, e, addr):
        eng, ctx, v3, io = e
        if v3:
            for n, _, _ in io:
                ctx.set_tensor_address(n, addr[n])
            ok = ctx.execute_async_v3(self.cu.handle)
        else:
            ok = ctx.execute_async_v2([addr[n] for n, _, _ in io], self.cu.handle)
        assert ok, "TensorRT execute failed"

    def load(self, x):
        self.cur = 0
        self.cu.htod(self.x_dev, x.astype(self.dt["x"]).reshape(self.x_shape))
        self._run(self.p, {"x": self.x_dev, "h0": self.h_dev[0]})
        self.cu.sync()

    def step(self):
        c = self.cur
        self._run(self.s, {"h": self.h_dev[c], "x": self.x_dev, "h_out": self.h_dev[1 - c], "logits": self.l_dev})
        self.cur = 1 - c
        self.cu.sync()

    def logits(self):
        self.cu.dtoh(self.l_host, self.l_dev)
        self.cu.sync()
        return self.l_host

    def state(self):
        self.cu.dtoh(self.h_host, self.h_dev[self.cur])
        self.cu.sync()
        return self.h_host


class DtLoop(object):
    """The interface edge_loop.window() drives: n, B, run_puzzle(i, mode, depth)."""
    B = 1

    def __init__(self, runner, x, lab, rule):
        self.r, self.x, self.lab, self.rule = runner, x, lab, rule
        self.mask = x.max(1) > 0                              # open cells (walls = 0), (n, H, W)
        self.n = len(x)

    def pred(self, i):
        lg = self.r.logits()[0]
        return (lg[1] > lg[0]) & self.mask[i]                 # argmax of the 2-way head, masked; ties -> 0 as torch

    def run_puzzle(self, i, mode, depth, per_step=True):
        r = self.r
        r.load(self.x[i:i + 1])
        ms, steps, streak = [], 0, 0
        stable = mode == "exit" and self.rule.startswith("stable")
        resid_rule = mode == "exit" and self.rule.startswith("resid")
        p_prev = h_prev = None
        while steps < depth:
            t0 = time.perf_counter()
            r.step()
            changed, resid = True, 1.0
            if stable:                                        # the signal is read inside the timed step: it is its cost
                p = self.pred(i)
                changed = p_prev is None or bool((p != p_prev).any())
                p_prev = p
            elif resid_rule:
                h = r.state().astype(np.float32)
                if h_prev is not None:
                    resid = float(np.linalg.norm(h - h_prev) / max(np.linalg.norm(h), 1e-12))
                h_prev = h.copy()
            ms.append((time.perf_counter() - t0) * 1e3)
            steps += 1
            if mode == "exit":
                fire, streak = E.headfree_fires(self.rule, steps, changed, resid, streak)
                if fire:
                    break
        ok = bool((self.pred(i) == self.lab[i].astype(bool)).all())
        return steps, ok, ms, [], None, []


def check_pass(loop, ref, T):
    """Fixed T iterations per maze with every signal read: exact, changed cells and residual per iteration (dt_exit's
    definitions; the residual of iteration 1 is against the projection), against the host reference."""
    r, n = loop.r, loop.n
    ex = np.zeros((n, T), bool)
    chg = np.zeros((n, T), np.int32)
    res = np.zeros((n, T), np.float64)
    ms = np.zeros((n, T), np.float64)
    keep = [int(k) for k in ref["keep"]] if ref is not None else []
    preds = np.zeros((len(keep), n) + loop.x.shape[2:], np.uint8)
    t_all = time.perf_counter()
    for i in range(n):
        r.load(loop.x[i:i + 1])
        h_prev = r.state().astype(np.float32).copy()
        p_prev = None
        for t in range(T):
            t0 = time.perf_counter()
            r.step()
            ms[i, t] = (time.perf_counter() - t0) * 1e3
            p = loop.pred(i)
            h = r.state().astype(np.float32)
            ex[i, t] = (p == loop.lab[i].astype(bool)).all()
            chg[i, t] = int(loop.mask[i].sum()) if p_prev is None else int((p != p_prev).sum())
            res[i, t] = np.linalg.norm(h - h_prev) / max(np.linalg.norm(h), 1e-8)
            if t + 1 in keep:
                preds[keep.index(t + 1), i] = p
            p_prev, h_prev = p, h.copy()
        if (i + 1) % 5 == 0 or i == n - 1:
            E.log("check %d/%d solved@%d %d step %.2f ms" % (i + 1, n, T, int(ex[:i + 1, -1].sum()), ms[:i + 1].mean()))
    out = {"n": int(n), "T": int(T), "exact_count_per_step": [int(v) for v in ex.sum(0)],
           "step_ms": E.lat_summary(ms), "wall_s": round(time.perf_counter() - t_all, 1), "exit": {},
           "per_maze": [{"i": int(i), "first_solved": int(np.argmax(ex[i])) + 1 if ex[i].any() else None,
                         "res": [round(float(v), 6) for v in res[i]], "chg": [int(v) for v in chg[i]]} for i in range(n)]}
    changed = chg > 0
    for rule in RULES:
        st = E.offline_headfree(rule, changed, res, T)
        out["exit"][rule] = {"exact": round(float(ex[np.arange(n), st - 1].mean()), 4), "mean_steps": round(float(st.mean()), 2)}
    if ref is not None:
        m, k = min(n, ref["correct"].shape[0]), min(T, ref["correct"].shape[1])
        rc = ref["correct"][:m, :k]
        out["ref"] = {"n_compared": int(m), "steps_compared": int(k),
                      "exact_flags_differing": int((rc != ex[:m, :k]).sum()),
                      "exact_count_match": [int(v) for v in rc.sum(0)] == [int(v) for v in ex[:m, :k].sum(0)],
                      "res_max_abs_diff": float(np.abs(ref["res"][:m, :k] - res[:m, :k]).max()),
                      "chg_max_abs_diff": int(np.abs(ref["chg"][:m, :k] - chg[:m, :k]).max()),
                      "pred_cells_differing_at": dict((str(kk), int((ref["preds"][j, :m] != preds[j, :m]).sum()))
                                                      for j, kk in enumerate(keep) if kk <= T)}
        for rule in RULES:
            st = E.offline_headfree(rule, ref["chg"][:m, :k] > 0, ref["res"][:m, :k], k)
            out["ref"]["exit_" + rule] = {"exact": round(float(rc[np.arange(m), st - 1].mean()), 4),
                                          "mean_steps": round(float(st.mean()), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["ort", "trt"])
    ap.add_argument("--proj", required=True, help="dt_proj graph (.onnx for ort, .plan for trt)")
    ap.add_argument("--step", required=True, help="dt_step graph (.onnx for ort, .plan for trt)")
    ap.add_argument("--mazes", required=True, help="npz from export_dt.py: inputs (n,3,H,W), labels (n,H,W) uint8")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--ref", default=None, help="host_fp32.npz from export_dt.py (check pass)")
    ap.add_argument("--board", default=socket.gethostname())
    ap.add_argument("--precision", default="fp32")
    ap.add_argument("--mode", choices=["exit", "fixed"], default="fixed")
    ap.add_argument("--depth", type=int, default=300, help="iterations in fixed mode, the cap in exit mode")
    ap.add_argument("--rule", default="resid0.0794", help="exit rule: " + ", ".join(RULES) + " (any stable<p>/resid<tau>)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--no_check", action="store_true")
    ap.add_argument("--check_only", action="store_true", help="the check pass alone, then exit")
    ap.add_argument("--check_n", type=int, default=None)
    ap.add_argument("--check_T", type=int, default=300)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--win_s", type=float, default=120.0)
    ap.add_argument("--nwin", type=int, default=5)
    ap.add_argument("--nidle", type=int, default=1)
    ap.add_argument("--power_dir", default=E.NANO_POWER_DIR)
    ap.add_argument("--power_files", default=E.NANO_POWER_FILES)
    ap.add_argument("--rail_names", default=E.NANO_RAIL_NAMES)
    ap.add_argument("--total_rail", type=int, default=0)
    ap.add_argument("--power_hz", type=float, default=50.0)
    ap.add_argument("--no_power", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    d = np.load(a.mazes)
    x, lab = d["inputs"], d["labels"]
    if a.n:
        x, lab = x[:a.n], lab[:a.n]
    ref = dict(np.load(a.ref)) if a.ref else None

    files = [os.path.join(a.power_dir, f) for f in a.power_files.split(",") if f]
    names = a.rail_names.split(",")[:len(files)]
    sampler = None
    if not a.no_power and not a.check_only:
        missing = [f for f in files if not os.access(f, os.R_OK)]
        if missing:
            E.log("power files missing or unreadable (use --no_power for a latency-only run):", missing)
            sys.exit(2)
        sampler = E.PowerSampler(files, names, a.power_hz)
        sampler.start()
        E.log("power sampler on %s at %.0f Hz, first read %s mW" % (a.power_dir, a.power_hz, sampler.read_once()))

    t_load = time.perf_counter()
    runner = DtTrt(a.proj, a.step) if a.backend == "trt" else DtOrt(a.proj, a.step, a.threads)
    t_load = time.perf_counter() - t_load
    loop = DtLoop(runner, x, lab, a.rule)
    name = "%s_%s_b1_%s%d" % (a.backend, a.precision, a.mode, a.depth) + ("_" + a.rule if a.mode == "exit" else "")
    rec = {"meta": {
        "board": a.board, "host": socket.gethostname(), "platform": "edge: " + a.board + " " + platform.machine(),
        "backend": a.backend, "runtime_version": runner.version, "python": platform.python_version(),
        "numpy": np.__version__, "task": "dt_maze33", "proj": os.path.abspath(a.proj), "model": os.path.abspath(a.step),
        "model_size_MB": round(os.path.getsize(a.step) / 1e6, 2), "precision": a.precision, "mode": a.mode,
        "depth": a.depth, "rule": a.rule if a.mode == "exit" else None, "B": 1, "threads": a.threads,
        "providers": getattr(runner, "providers", None), "cuda_mem": getattr(runner, "mem_mode", None),
        "mazes": os.path.abspath(a.mazes), "n_puzzles": int(len(x)), "ref": os.path.abspath(a.ref) if a.ref else None,
        "power": {"dir": a.power_dir, "files": files, "rails": names, "total_rail": names[a.total_rail] if names else None,
                  "hz": a.power_hz, "active": sampler is not None,
                  "note": "INA3221 sysfs mW, trapezoid over sample timestamps; idle window same length, same session"},
        "win_s": a.win_s, "nwin": a.nwin, "nidle": a.nidle, "clocks": E.read_clocks(), "thermal_start": E.read_thermal(),
        "load_s": round(t_load, 2), "at": E.now_utc(), "argv": sys.argv, "note": a.note,
        "record_layout": "edge_loop.py: meta / idle[] / windows[cfg]{windows[], median_clean, first_pass} / check / record"},
        "idle": [], "windows": {}, "check": None, "record": None}
    E.dump(a.out, rec)

    for i in range(min(a.warm, len(x))):
        loop.run_puzzle(i, a.mode, min(a.depth, 30))

    if not a.no_check or a.check_only:
        nchk = min(a.check_n or len(x), len(x))
        sub = DtLoop(runner, x[:nchk], lab[:nchk], a.rule)
        chk = check_pass(sub, ref, a.check_T)
        rec["check"] = chk
        E.log("check: solved at 30/100/%d: %s | exit %s | step ms median %s" % (
            a.check_T, [chk["exact_count_per_step"][t - 1] for t in (30, 100, a.check_T) if t <= a.check_T],
            chk["exit"], chk["step_ms"].get("median")))
        if "ref" in chk:
            rr = chk["ref"]
            E.log("check vs ref: exact flags differing %d, counts match %s, res max diff %.3g, chg max diff %d, pred cells "
                  "differing %s" % (rr["exact_flags_differing"], rr["exact_count_match"], rr["res_max_abs_diff"],
                                     rr["chg_max_abs_diff"], rr["pred_cells_differing_at"]))
        E.dump(a.out, rec)
        if a.check_only:
            E.log("wrote", a.out)
            return

    per, first, allms = [], {}, []
    t_all = time.perf_counter()
    for i in range(loop.n):
        steps, ok, ms, _, _, _ = loop.run_puzzle(i, a.mode, a.depth)
        per.append({"i": int(i), "steps": int(steps), "ok": bool(ok), "ms": [round(float(v), 3) for v in ms]})
        first[i] = (steps, ok)
        allms += ms
    fp = {"puzzles_covered": len(first), "exact": float(np.mean([v[1] for v in first.values()])),
          "mean_steps": float(np.mean([v[0] for v in first.values()])), "completion_untimed_s": 0.0,
          "wall_s": round(time.perf_counter() - t_all, 1),
          "per_puzzle": dict((str(i), [int(v[0]), bool(v[1])]) for i, v in sorted(first.items()))}
    lat = E.lat_summary(allms)
    rec["record"] = {"cfg": name, "per_puzzle": per, "step_ms": lat, "exact": fp["exact"], "mean_steps": fp["mean_steps"]}
    E.log("record: exact %.4f mean steps %.2f step ms median %s" % (fp["exact"], fp["mean_steps"], lat.get("median")))
    w_entry = {"rt": a.backend, "prec": a.precision, "B": 1, "mode": a.mode, "engine": os.path.basename(a.step),
               "windows": [], "median_clean": {}, "n_clean": 0, "first_pass": fp}
    rec["windows"][name] = w_entry
    E.dump(a.out, rec)

    idle_P = None
    if sampler and a.nwin > 0:
        for k in range(a.nidle):
            w = E.idle_window(a.win_s, sampler, names, a.total_rail, "start")
            rec["idle"].append(w)
            E.log("idle start %.3f W (%d samples, %.1f Hz)" % (w["P_W"] or -1, w["n_samples"], w["sample_hz"] or 0))
            E.dump(a.out, rec)
        ps = [w["P_W"] for w in rec["idle"] if w["P_W"] is not None]
        idle_P = statistics.median(ps) if ps else None
    for k in range(a.nwin):
        w = E.window(loop, a.mode, a.depth, a.win_s, sampler, names, a.total_rail, idle_P)
        w_entry["windows"].append(w)
        E.log("%s w%d %.1f s %d mazes %d steps %.3f ms/step acc %.3f%s%s" % (
            name, k, w["T_s"], w["puzzles"], w["steps"], w["ms_per_step"], w["acc_window"],
            (" %.3f W %.2f mJ/step %.3f J/solved" % (w["P_W"], w["mJ_per_step"], w["J_per_solved"])) if "E_J" in w else "",
            "" if w["clean"] else " THROTTLED"))
        good = [g for g in w_entry["windows"] if g["clean"]]
        keys = [kk for kk in ("P_W", "mJ_per_step", "mJ_per_step_dyn", "J_per_puzzle", "J_per_solved", "J_per_solved_dyn",
                              "ms_per_step", "ms_per_puzzle", "ms_per_solved", "acc_window") if good and kk in good[0]]
        w_entry["median_clean"] = dict((kk, statistics.median([g[kk] for g in good])) for kk in keys)
        w_entry["n_clean"] = len(good)
        E.dump(a.out, rec)
    rec["meta"]["thermal_end"] = E.read_thermal()
    rec["meta"]["clocks_end"] = E.read_clocks()
    rec["meta"]["finished_at"] = E.now_utc()
    if sampler:
        rec["meta"]["power"]["sampler_errors"] = sampler.errors
        sampler.close()
    E.dump(a.out, rec)
    E.log("wrote", a.out)


if __name__ == "__main__":
    main()
