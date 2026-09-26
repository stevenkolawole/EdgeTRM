#!/usr/bin/env python3
"""X2 contraction loss vs readout margin (predictions in results/breadth/x2_margin_predictions.md, written before this
script ran). Label-free, per example:
  r_q   late contraction ratio of the compressed trajectory: median over the last 3 steps of
        ||z_{t+1} - z_t|| / ||z_t - z_{t-1}||;  res_q = ||F_q(z_T) - z_T|| / ||z_T||
  r_f, res_f   the same on the fp32 trajectory (L_0)
  t_r   readout margin: first t on TR_GRID along z* + t (z*_q - z*) at which the decoded answer (no further steps)
        differs from the fp32 answer at z* (inf if never, i.e. > TR_GRID[-1])
  disp  ||z*_q - z*|| / ||z*||
Cells as in x2_return.py (weights, +a<b>, anoise<s>). TRM / HRM / DT.
    python x2_margin.py --task sudoku_mlp --n 256 --cells noise0.05,w4t --out retm_sudoku_mlp.json
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from x2_return import parse_cell, load, save  # noqa: E402

TR_GRID = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0, 128.0]
LAST = 3


def _norm(v):
    return v.reshape(v.shape[0], -1).norm(dim=1)


def traj(F, T, torch):
    z = F.z0()
    steps = []
    prev = z
    for _ in range(T):
        z = F(z)
        steps.append(_norm(z - prev))
        prev = z
    s = torch.stack(steps)                              # (T, B): ||z_t - z_{t-1}||
    ratios = s[-LAST:] / s[-LAST - 1:-1].clamp_min(1e-30)
    Fz = F(z)
    return z, ratios.median(0).values, _norm(Fz - z) / _norm(z)


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    assert torch.cuda.is_available() or os.environ.get("X2_CPU"), "NO CUDA"
    import breadth_sweep as bs
    import contraction as C
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.ckpt:
        bs.CKPT_OVERRIDE = a.ckpt
    if a.task in C.DT_TASKS:
        family, T = "dt", a.T or C.DT_TASKS[a.task]["T"]
        batch = a.batch or C.DT_TASKS[a.task]["batch"]
        inp, lab = C.load_dt(a.task, a.n)
    else:
        family = "trm" if a.task in bs.TRM_TASKS else "hrm"
        T = a.T or 16
        t_ = bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task]
        batch = a.batch or (64 if t_["seq"] < 200 else 8)
        inp, lab = bs.load_qa(t_["data"], a.n)
    mk = (lambda net, x: C.DTRecursion(net, x)) if family == "dt" else (lambda net, x: C.Recursion(net, family, x))
    meta = {"family": family, "task": a.task, "model": a.name or a.task, "ckpt": a.ckpt, "n": int(len(inp)), "T": T,
            "tr_grid": TR_GRID, "last": LAST}
    blob = load(a.out, meta)
    done = {r["cell"] for r in blob["records"]}
    chunks = [(s, min(s + batch, len(inp))) for s in range(0, len(inp), batch)]
    m0 = C.make_model(a.task, ("fp32", 0), T, batch)[0]
    ref, t0 = [], time.time()
    with torch.no_grad(), C.sdpa_math():
        for s, e in chunks:
            x = torch.from_numpy(inp[s:e]).to(bs.DEV)
            y = torch.from_numpy(lab[s:e]).to(bs.DEV)
            Fn = mk(m0, x)
            z, rf, resf = traj(Fn, T, torch)
            ro = Fn.readout(z, y)
            ref.append({"z": z, "corr": ro["correct"], "pred": ro["pred"], "r_f": rf, "res_f": resf})
    blob["fp32_acc"] = float(torch.cat([r["corr"] for r in ref]).float().mean())
    blob["r_f"] = torch.cat([r["r_f"] for r in ref]).cpu().numpy().round(5).tolist()
    blob["res_f"] = torch.cat([r["res_f"] for r in ref]).cpu().numpy().round(6).tolist()
    blob["corr_f"] = torch.cat([r["corr"] for r in ref]).int().cpu().numpy().tolist()
    print(f"{meta['model']}: fp32 acc {blob['fp32_acc']*100:.1f} r_f med {np.median(blob['r_f']):.3f} "
          f"res_f med {np.median(blob['res_f']):.2e} (n={len(inp)}, T={T}; {time.time()-t0:.0f}s)", flush=True)
    for cell in a.cells.split(","):
        if not cell or cell in done:
            continue
        t1 = time.time()
        wspec, act, an, draw = parse_cell(cell)
        mq, _, q = C.make_model(a.task, (wspec, draw), T, batch)
        per = {k_: [] for k_ in ("corr_q", "r_q", "res_q", "t_r", "disp")}
        with torch.no_grad(), C.sdpa_math():
            for ci, (s, e) in enumerate(chunks):
                x = torch.from_numpy(inp[s:e]).to(bs.DEV)
                y = torch.from_numpy(lab[s:e]).to(bs.DEV)
                R = ref[ci]
                hooks, obs = [], []
                if an is not None:
                    torch.manual_seed(1000 + ci + 100 * draw)
                    for mod in bs._linears(mq):
                        hooks.append(mod.register_forward_pre_hook(bs._ActNoise(an)))
                if act is not None:
                    for mod in bs._linears(mq):
                        o = bs._ActQuant(act)
                        obs.append(o)
                        hooks.append(mod.register_forward_pre_hook(o))
                    Fq = mk(mq, x)
                    z = Fq.z0()
                    for _ in range(T):
                        z = Fq(z)
                    for o in obs:
                        o.freeze()
                Fq = mk(mq, x)
                zq, rq, resq = traj(Fq, T, torch)
                roq = Fq.readout(zq, y)
                for h in hooks:
                    h.remove()
                Ff = mk(m0, x)
                d = zq - R["z"]
                tr = torch.full((zq.shape[0],), float("inf"), device=zq.device)
                for t in TR_GRID:
                    ch = ~(Ff.readout(R["z"] + t * d, y)["pred"] == R["pred"]).all(-1)
                    tr = torch.where(ch & torch.isinf(tr), torch.full_like(tr, t), tr)
                per["corr_q"].append(roq["correct"].int().cpu())
                per["r_q"].append(rq.cpu())
                per["res_q"].append(resq.cpu())
                per["t_r"].append(tr.cpu())
                per["disp"].append((_norm(d) / _norm(R["z"])).cpu())
        cat = {k_: torch.cat(v).numpy() for k_, v in per.items()}
        cf = np.array(blob["corr_f"], bool)
        cq = cat["corr_q"].astype(bool)
        sel = cf if cf.any() else np.ones_like(cf)
        rec = {"cell": cell, "quant": q, "n": int(len(cq)), "acc_f": float(cf.mean()), "acc_q": float(cq.mean()),
               "retained": float(cq.mean() / max(cf.mean(), 1e-9)),
               "r_q_median": float(np.median(cat["r_q"][sel])), "res_q_median": float(np.median(cat["res_q"][sel])),
               "t_r_median": float(np.median(cat["t_r"][sel])), "disp_median": float(np.median(cat["disp"][sel])),
               "per": {"corr_q": cq.astype(int).tolist(), "r_q": cat["r_q"].round(5).tolist(),
                       "res_q": cat["res_q"].round(6).tolist(), "t_r": [None if not np.isfinite(v) else float(v) for v in cat["t_r"]],
                       "disp": cat["disp"].round(5).tolist()},
               "secs": time.time() - t1}
        blob["records"].append(rec)
        save(a.out, blob)
        print(f"  {cell}: acc {rec['acc_q']*100:.1f} (fp32 {rec['acc_f']*100:.1f}) r_q {rec['r_q_median']:.3f} "
              f"res_q {rec['res_q_median']:.2e} t_r {rec['t_r_median']:.2f} disp {rec['disp_median']:.4f} ({rec['secs']:.0f}s)",
              flush=True)
        del mq
        torch.cuda.empty_cache()
    save(a.out, blob)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
