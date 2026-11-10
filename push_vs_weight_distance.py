#!/usr/bin/env python3
"""Does GPTQ lower the push (Eq. 2) while raising the weight distance?  RTN vs GPTQ on the same grid, TRM Sudoku models.

For each model and spec: weight distance (relative Frobenius error ||Wq-W||/||W||, pooled over all CastedLinear and the
mean of the per-tensor values) and the push delta = ||F_q(z*) - F(z*)|| / ||z*|| of one supervision step at the fp32 state
z* (fp32 trajectory from the model's own initial carry, state after T steps, T in --T), per puzzle, on test rows
[--offset, --offset+--n). F is contraction.Recursion (as the paper's contraction records); the compressed map uses the
compressed model's own embedding and weights, z* comes from the fp32 model. GPTQ weights: M5_GPTQ_DIR/<task>/gq<fmt>.pt.

    python push_vs_weight_distance.py --task sudoku_mlp --specs w4t,gqw4t,w3c,gqw3c --n 256 --T 16,64 --out OUT.json
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import breadth_sweep as bs
import contraction as K
import m5_gptq as G


def bnorm(t):
    return t.flatten(1).norm(dim=1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True); ap.add_argument("--specs", required=True)
    ap.add_argument("--n", type=int, default=256); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--T", default="16,64"); ap.add_argument("--ckpt", default=""); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Ts = [int(t) for t in a.T.split(",")]
    if a.ckpt:
        bs.CKPT_OVERRIDE = a.ckpt
    kind = (bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task])["data"]
    q, y = bs.load_qa(kind, a.offset + a.n)
    x = torch.tensor(q[a.offset:a.offset + a.n]).to(bs.DEV)
    fam = "trm" if a.task in bs.TRM_TASKS else "hrm"
    bt = 64

    def build():
        m, _ = bs.build(a.task, None, 10 ** 6, bt)
        for p in m.parameters():
            p.requires_grad_(False)
        return m.to(bs.DEV).eval()

    m0 = build()
    lin0 = {n: mod.weight.detach().clone() for n, mod in m0.named_modules() if mod.__class__.__name__ == "CastedLinear"}
    F0 = K.Recursion(m0, fam, x)
    zs = {}
    z = F0.z0()
    for t in range(1, max(Ts) + 1):
        z = F0(z)
        if t in Ts:
            zs[t] = z.clone()
    f0 = {t: F0(zs[t]) for t in Ts}
    out = {"task": a.task, "n": a.n, "offset": a.offset, "T": Ts, "cells": {}}
    for spec in a.specs.split(","):
        m = build()
        if G.is_gptq(spec):
            G.load_if_gptq(m, a.task, spec, wait_s=1)
            fmt = spec[2:]
        else:
            qq = bs.parse_quant(spec)
            bs.quantize_weights_(m, qq["bits"], qq["gran"], qq.get("sigma"))
            fmt = spec
        lin = {n: mod.weight.detach() for n, mod in m.named_modules() if mod.__class__.__name__ == "CastedLinear"}
        per = [float((lin[n].float() - lin0[n].float()).norm() / lin0[n].float().norm()) for n in lin0]
        glob = float(np.sqrt(sum(float((lin[n].float() - lin0[n].float()).norm() ** 2) for n in lin0) / sum(float(lin0[n].float().norm() ** 2) for n in lin0)))
        Fq = K.Recursion(m, fam, x)
        res = {"fmt": fmt, "gptq": G.is_gptq(spec), "w_dist_pooled": glob, "w_dist_mean_tensor": float(np.mean(per)), "n_tensors": len(per), "push": {}}
        for t in Ts:
            d = (bnorm(Fq(zs[t]) - f0[t]) / bnorm(zs[t])).cpu().numpy()
            res["push"][str(t)] = {"median": float(np.median(d)), "mean": float(d.mean()), "q25": float(np.percentile(d, 25)), "q75": float(np.percentile(d, 75))}
        out["cells"][spec] = res
        print(f"{a.task:12s} {spec:10s} w-dist pooled {glob:.4f} mean-tensor {np.mean(per):.4f} | push median " + " ".join(f"T{t}={res['push'][str(t)]['median']:.4f}" for t in Ts), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
