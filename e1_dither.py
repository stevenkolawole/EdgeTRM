#!/usr/bin/env python3
"""E1 dither row (MLSys): energy per correct answer of PERTURB-EVERY-STEP (dither) on one exclusive Babel L40S.

Dither = before EVERY supervision step the low-bit weights are re-drawn by stochastic rounding from the stored 8-bit
copy (master w8c, per-channel RTN of the fp32 weights: the copy the chip ships for finishing), exactly as
m2_bestofk's `dither` arm with `--master w8c` (tab:dither arm 6). EAGER PyTorch only: a TensorRT engine freezes its
weights at build time and cannot re-round them per step, so no TensorRT row exists for this policy (recorded in the
meta of the output file). The step kernels, windows and idle subtraction are those of e1_grid.py (continuous
batching, NVML total-energy counter, 5 idle windows x 30 s at the start and 2 at the end, 10 s warm-up, 5 windows of
>= 60 s per configuration, accuracy from the first completion of every puzzle in the same run).

Arms (all eager, fp32 compute by default, TF32 off; --compute fp16 casts as e1_energy's eager fp16 path):
  fp32    unquantized weights (the in-job eager baseline; the grid's eager_fp32 rows on another day/card)
  rtn     the cell's low-bit spec (w4c / w3c) by round-to-nearest from the SAME w8c master, weights fixed:
          the non-dithered cheap loop at identical kernel cost (the control dither is measured against)
  dither  stochastic re-draw of the spec from the w8c master before every step (one draw per step, shared by the
          batch, as in the M2 exhibits)
Modes: fixed 16 steps and the M5 halting exit, at batch 1 and batch 256 (--batches); plus `polish` at batch 1 =
the combined policy: cheap steps under the exit rule (dither or rtn), then finishing steps with the w8c master
weights until the w8c head's rule fires (<= POLISH_CAP = 4, as the grid's TRT polish row). In eager the finishing
model is the same module with its weights pointed at the master copy, so the hand-over is free.

Per dither configuration the re-draw is also timed on its own (50 iterations) next to a plain step at the same
batch, so the record carries the time share the prediction was written from.

Output: $E1_RES/<task>/energy_dither.json (resumable per configuration).
    python e1_dither.py --task sudoku_mlp            # spec from CELLS (w4c), master w8c
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
import e1_grid as G  # noqa: E402

CELLS = {"sudoku_mlp": "w4c", "sudoku_attn": "w3c", "hrm_sudoku": "w3c", "maze": "w4c"}   # tab:dither cells
POLISH_CAP = G.POLISH_CAP
log = E.log


# ----------------------------------------------------------------------------- rounding (breadth_sweep grids)
def parse_spec(spec):
    assert spec.startswith("w"), spec
    body = spec[1:]
    i = 0
    while i < len(body) and body[i].isdigit():
        i += 1
    return int(body[:i]), body[i:]


def _linears(m):
    return [mod for mod in m.modules() if mod.__class__.__name__ == "CastedLinear"]


def round_grid(W, bits, gran, U=None):
    """breadth_sweep.quantize_weights_ (U None: round-to-nearest) / m2_bestofk.sr_quantize_ (U ~ U[0,1): stochastic
    rounding q = floor(W/s + U)) on one fp32 weight tensor; same scales and zero points in both cases."""
    import torch
    import torch.nn.functional as F
    rnd = (lambda x: torch.round(x)) if U is None else (lambda x: torch.floor(x + U))
    if gran == "a":
        qmax = 2 ** bits - 1
        wmin, wmax = W.amin(dim=1, keepdim=True), W.amax(dim=1, keepdim=True)
        scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
        zp = torch.round(-wmin / scale)
        return (torch.clamp(rnd(W / scale) + zp, 0, qmax) - zp) * scale
    qmax = 2 ** (bits - 1) - 1
    if gran in ("t", "c"):
        s = ((W.abs().max() if gran == "t" else W.abs().amax(dim=1, keepdim=True)) / qmax).clamp_min(1e-8)
        return rnd(W / s).clamp(-qmax, qmax) * s
    g = int(gran[1:])
    out_f, in_f = W.shape
    pad = (-in_f) % g
    Wp = F.pad(W, (0, pad)).view(out_f, -1, g)
    s = (Wp.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-8)
    if U is None:
        q = torch.round(Wp / s)
    else:
        q = torch.floor(Wp / s + F.pad(U, (0, pad)).view(out_f, -1, g))
    return (q.clamp(-qmax, qmax) * s).view(out_f, -1)[:, :in_f]


# ----------------------------------------------------------------------------- runner
class DitherRunner(E.EagerRunner):
    """EagerRunner whose linears hold (a) the unquantized weights, (b) the spec by RTN from the master, or (c) a fresh
    stochastic draw of the spec from the master before every step. `finish()` points the weights at the master
    (the 8-bit finishing model of the combined policy); `cheap()` returns to the arm's cheap weights."""

    def __init__(self, task, H, B, compute, arm, spec, master_spec, seed):
        super().__init__(task, H, B, compute)
        import torch
        self.arm = arm
        self.lin = _linears(self.step_mod)
        assert self.lin, "no CastedLinear found"
        if compute == "fp32":
            W32 = [mod.weight.data.float().clone() for mod in self.lin]
        else:                                   # master from the fp32 weights, not from the fp16-cast ones
            inner32, _ = E.build_inner(task, H)
            W32 = [mod.weight.data.float().cuda().clone() for mod in _linears(inner32)]
            assert len(W32) == len(self.lin)
        mb, mg = parse_spec(master_spec)
        self.master32 = [round_grid(W, mb, mg) for W in W32]               # stored 8-bit copy (fp32 values on the grid)
        self.master = [M.to(self.dt) for M in self.master32]
        self.bits, self.gran = parse_spec(spec)
        self.nparam = sum(W.numel() for W in W32)
        if arm == "fp32":
            self.cheap_w = [W.to(self.dt) for W in W32]
        elif arm == "rtn":
            self.cheap_w = [round_grid(M, self.bits, self.gran).to(self.dt) for M in self.master32]
        else:
            self.cheap_w = None
        self.gen = torch.Generator(device="cuda").manual_seed(int(seed))
        self.seed = int(seed)
        self.dither_on = arm == "dither"
        self.n_redraw = 0
        self.cheap()

    def _set(self, ws):
        for mod, W in zip(self.lin, ws):
            mod.weight.data = W

    def redraw(self):
        torch = self.torch
        for mod, M in zip(self.lin, self.master32):
            U = torch.rand(M.shape, generator=self.gen, device=M.device, dtype=M.dtype)
            q = round_grid(M, self.bits, self.gran, U)
            mod.weight.data = q if self.dt == torch.float32 else q.to(self.dt)
        self.n_redraw += 1

    def cheap(self):
        self.dither_on = self.arm == "dither"
        if self.arm == "dither":
            self.redraw()
        else:
            self._set(self.cheap_w)

    def finish(self):
        self.dither_on = False
        self._set(self.master)

    def step(self):
        if self.dither_on:
            self.redraw()
        super().step()

    def step_plain(self):
        super().step()


class DitherStream(G.Stream):
    """e1_grid.Stream with the polish unit for the eager pair (same runner: cheap weights -> master weights)."""

    def unit(self):
        if self.mode != "polish":
            return super().unit()
        torch = self.torch
        r = self.r
        assert self.B == 1
        n_cheap = 0
        prev = torch.zeros(1, dtype=torch.bool, device="cuda")
        st = torch.zeros(1, dtype=torch.int32, device="cuda")
        r.cheap()
        while True:
            r.step()
            n_cheap += 1
            st += 1
            fire, halt = self._fire(r, st, prev)
            prev = halt
            if bool(fire.item()):
                break
        r.finish()                                  # same carry, weights now the w8c master
        prev = torch.zeros(1, dtype=torch.bool, device="cuda")
        stp = torch.zeros(1, dtype=torch.int32, device="cuda")
        n_pol = 0
        while n_pol < POLISH_CAP:
            r.step()
            n_pol += 1
            stp += 1
            fire, halt = self._fire(r, stp + (1 if self.rule == "min2native" else 0), prev)
            prev = halt
            if bool(fire.item()) and not (self.rule == "patience2" and n_pol < 2):
                break
        idx = torch.zeros(1, dtype=torch.long, device="cuda")
        d, c, lat = self.emit(idx, r.logits, n_cheap + n_pol)
        self.polish_steps.append((n_cheap, n_pol))
        self.fill(idx)
        return n_cheap + n_pol, d, c, lat


def time_redraw(r, iters=50):
    """Re-draw alone and a plain step alone at this batch (ms), for the time share; not part of the windows."""
    torch = r.torch
    for f in (r.redraw, r.step_plain):
        for _ in range(5):
            f()
    out = {}
    for name, f in (("redraw_ms", r.redraw), ("step_ms_noredraw", r.step_plain)):
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            f()
        torch.cuda.synchronize()
        out[name] = 1e3 * (time.perf_counter() - t) / iters
    out["redraw_share"] = out["redraw_ms"] / (out["redraw_ms"] + out["step_ms_noredraw"])
    r.cheap()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(E.TASKS))
    ap.add_argument("--spec", default=None, help="low-bit cell (default per task: %s)" % CELLS)
    ap.add_argument("--master", default="w8c", help="stored copy the draws are made from (per-channel RTN)")
    ap.add_argument("--compute", default="fp32", choices=("fp32", "fp16"))
    ap.add_argument("--batches", default="1,256")
    ap.add_argument("--arms", default="fp32,rtn,dither")
    ap.add_argument("--seed", type=int, default=17, help="draw seed (m2_bestofk wseed 0 -> 7919*0+17)")
    ap.add_argument("--win_s", type=float, default=60.0)
    ap.add_argument("--nwin", type=int, default=5)
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="energy_dither.json")
    a = ap.parse_args()
    import torch
    import pynvml as nv
    assert torch.cuda.is_available(), "CUDA not available: abort"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    task = a.task
    t = E.TASKS[task]
    H = t["H"]
    spec = a.spec or CELLS[task]
    batches = [int(b) for b in a.batches.split(",")]
    arms = a.arms.split(",")
    nv.nvmlInit()
    uuid_t = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    h = next(nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())
             if (lambda u: u.decode() if isinstance(u, bytes) else u)(nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(i))) == uuid_t)
    samp = E.Sampler(nv, h)
    samp.start()
    outp = E.rdir(task) / a.out
    rec = json.loads(outp.read_text()) if outp.exists() else {"configs": {}, "idle": []}
    rec["meta"] = {"gpu": torch.cuda.get_device_name(0), "uuid": uuid_t, "host": __import__("socket").gethostname(),
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "driver": nv.nvmlSystemGetDriverVersion(),
                   "torch": torch.__version__, "H": H, "nsup": E.NSUP, "rule": t["rule"], "n": t["n"],
                   "win_s": a.win_s, "nwin": a.nwin, "polish_cap": POLISH_CAP, "spec": spec, "master": a.master,
                   "compute": a.compute, "seed": a.seed, "batches": batches,
                   "policy": "dither = stochastic re-draw of <spec> from the <master> copy before EVERY step "
                             "(m2_bestofk sr_quantize_, arm 6 of tab:dither); rtn = same spec fixed by RTN from the "
                             "same master; polish = cheap steps under the exit rule then <= polish_cap steps with the "
                             "master weights until its head fires",
                   "no_tensorrt_row": "TensorRT engines freeze weights at build time and cannot re-round them per "
                                      "step; the dither policy exists in the eager path only",
                   "platform": "Babel L40S, one exclusive card (datacenter GPU, not edge); clocks not pinned"}

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
    cfgs = [(arm, B, mode) for B in batches for arm in arms for mode in ("fixed", "exit")]
    for arm in arms:
        if arm != "fp32":
            cfgs.append((arm, 1, "polish"))
    only = set(a.only.split(",")) if a.only else None
    for arm, B, mode in cfgs:
        name = f"{arm}_b{B}_{mode}"
        if only and name not in only:
            continue
        if name in rec["configs"] and (rec["configs"][name].get("n_clean", 0) >= a.nwin or rec["configs"][name].get("skipped")):
            continue
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        try:
            runner = DitherRunner(task, H, B, a.compute, arm, spec, a.master, a.seed)
            runner.x = runner.x.clone()
            S = DitherStream(task, runner, G.reset_eager, B, mode, xg, labg, None)
            S.polish_steps = []
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 10:
                S.unit()
            torch.cuda.synchronize()
            share = time_redraw(runner) if arm == "dither" else None
            if share:
                log(name, "re-draw", f"{share['redraw_ms']:.3f} ms", "plain step", f"{share['step_ms_noredraw']:.3f} ms",
                    "share", round(share["redraw_share"], 4))
        except Exception as ex:                                      # OOM etc.: recorded, next configuration
            rec["configs"][name] = {"arm": arm, "B": B, "mode": mode, "skipped": True, "reason": repr(ex)[:300]}
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
                 "mJ_per_slot_step": 1e3 * E_ / max(steps, 1), "window_acc": corr / max(done, 1),
                 "clean": bool(ok0 and clean()), "loadavg": os.getloadavg(), **samp.take()}
            wins.append(w)
            log(task, name, f"w{k}", f"{w['P_W']:.1f} W", f"{w['J_per_puzzle']:.4f} J/puz", f"{w['lat_ms_mean']:.2f} ms",
                f"{w['puzzles_per_s']:.1f} puz/s", f"acc {w['window_acc']:.3f}", f"{w.get('sm_MHz_mean')} MHz",
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
        wacc = [w["window_acc"] for w in good]
        summ = {"acc": acc, "mean_steps": msteps, "n_clean": len(good),
                "window_acc_mean": statistics.mean(wacc) if good else None,
                "J_solved_median": statistics.median(jp) / acc if good and acc else None,
                "J_solved_ci95": [v / acc for v in G.ci95(jp)] if len(jp) > 1 and acc else None,
                "J_solved_dyn_median": statistics.median(jd) / acc if good and acc else None,
                "J_solved_dyn_ci95": [v / acc for v in G.ci95(jd)] if len(jd) > 1 and acc else None,
                "lat_ms_median": statistics.median(lt) if good else None, "lat_ms_ci95": G.ci95(lt),
                "ms_per_solved_median": 1e3 / statistics.median(tp) / acc if good and acc else None,
                "puzzles_per_s_median": statistics.median(tp) if good else None,
                "mJ_per_slot_step_median": statistics.median([w["mJ_per_slot_step"] for w in good]) if good else None,
                "P_W_median": statistics.median([w["P_W"] for w in good]) if good else None,
                "sm_MHz_median": statistics.median([w.get("sm_MHz_mean", 0) for w in good]) if good else None}
        if mode == "polish" and S.polish_steps:
            ps = np.array(S.polish_steps)
            summ["polish_mean_cheap_steps"], summ["polish_mean_polish_steps"] = float(ps[:, 0].mean()), float(ps[:, 1].mean())
        rec["configs"][name] = {"arm": arm, "B": B, "mode": mode, "spec": spec, "master": a.master, "compute": a.compute,
                                "idle_P_W": idle_P, "redraw_timing": share, "params_rerounded": runner.nparam,
                                "n_redraws": runner.n_redraw, "windows": wins, "summary": summ,
                                "completion_untimed_s": round(tc, 1),
                                "first_pass": {int(p): v for p, v in sorted(S.first.items())}}
        E.dump(outp, rec)
        log(task, name, "acc", round(acc, 4), "steps", round(msteps, 2), "J/solved", summ["J_solved_median"],
            "dyn", summ["J_solved_dyn_median"], "lat ms", summ["lat_ms_median"])
        del S, runner
        torch.cuda.empty_cache()
    idle(2, "end")
    samp.stop_.set()


if __name__ == "__main__":
    main()
