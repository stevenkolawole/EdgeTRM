#!/usr/bin/env python3
"""M5 add-ons for ICLR, from the EXISTING p4d controller records only (results/m5/ctl/p4d; no new runs).

(1) Selective polish with honest costing. A cheap step costs 1. An fp32 polish step costs r = 32 / b_eff, the weight-
    traffic ratio of fp32 to the cheap format: batch-1 recursion is weight-bandwidth-bound, so a step's cost scales with the
    bytes of weights read. b_eff = bits, plus 16/g for the fp16 scale of group-g formats (w3g32: 3.5 bits, r = 9.1).
    Per-channel / per-tensor scales are negligible. w4: r = 8; w3: 10.7; w2: 16; w2g32: 12.8. Weight-noise cells and fp32
    cells carry fp32-sized weights (r = 1), so the honest-cost table uses the 24 quantized cells only. The 46-cell table
    (r = 1 for noise/fp32 cells) is printed too.
    SelP: the exit rule (chosen on dev fp32, frozen) stops the cheap trajectory at step t <= D0 <= 16. If the head is
    unconfident there (q_halt, or q_halt - q_continue for HRM, below theta), j fp32 polish steps follow from that state
    (exit_polish records c_pol[t, j]). D0, theta and j are chosen per budget on the dev rows, then scored on the held-out
    rows. theta is searched over dev quantiles of the stop-step confidence. The comparison methods are the m5_ctl_ho ones,
    every cost recomputed with fp32 steps x r: fixed depth, E, P (M1), E+P, and the full controller E+S+P.
(2) Head-free exit from the latent step size. resH = ||z_H(k) - z_H(k-1)|| / ||z_H(k)|| is saved at every H=1 step of the
    exit_sweep records; the value at each default-H step boundary is used. Rules, capped at 16 steps and compared with the
    halting-head exit and with fixed 16 on the held-out rows:
      res-fp32  stop at the first step with resH < eps, eps chosen on the dev rows of the model's fp32 cell (fewest mean
                steps within 0.5 pt of fixed 16, the head rule's criterion) and frozen for every cell of the model
      res-cell  the same with eps chosen on the dev rows of each cell itself (per-cell recalibration)
      plateau   untuned: stop at the first step where resH did not shrink (resH_t >= resH_{t-1}), from step 2 on
    Rows: Sudoku dev 0-999 / held out 1000-1999; mazes dev 0-499 / held out 500-999.

    python m5_addons.py      # tables + results/breadth/m5_addons.json
"""
import json
import os
import re
from pathlib import Path

import numpy as np

import analyze_m5 as A
import m5_ctl_ho as C
from m5_controller import fire_mask, first

ROOT = C.ROOT
NB = 2000
BUD = [8, 16, 24, 32, 48, 64, 96, 128]


def b_eff(spec):
    if spec == "fp32" or spec.startswith("noise") or spec.startswith("anoise"):
        return 32.0
    m = re.match(r"w(\d)([a-z])(\d+)?", spec)
    b = float(m.group(1))
    if m.group(2) == "g":
        b += 16.0 / float(m.group(3))
    return b


def load_res(path, rows=None):
    d = np.load(path)
    h = json.loads(str(d["info"]))["H_default"]
    sl = rows or slice(None)
    return d["resH"][sl, h - 1::h].astype(np.float32)


def boot(x, rng, y=None):
    x = np.asarray(x, float) - (0 if y is None else np.asarray(y, float))
    idx = rng.integers(0, len(x), (NB, len(x)))
    b = x[idx].mean(1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


# ----------------------------------------------------------------------------- (1) selective polish
def honest(split, r):
    base = split.eval

    def ev(cfg):
        a, c, p = selp(split, *cfg[1:]) if cfg[0] == "selp" else base(cfg)
        return a, c + (r - 1.0) * p, p
    return ev


def conf_at(split, t):
    q = split.dq[np.arange(split.n), t]
    return q - split.dqc[np.arange(split.n), t] if split.dqc is not None else q


def selp(split, D0, theta, j):
    td = split.td
    t = np.where((td >= 0) & (td < D0), td, D0 - 1)
    unconf = conf_at(split, t) < theta
    acc = np.where(unconf, split.pc[np.arange(split.n), t, j - 1], split.dc[np.arange(split.n), t])
    pol = unconf * float(j)
    return acc, (t + 1).astype(float) + pol, pol


def selp_configs(split):
    out = []
    for D0 in [d for d in (4, 8, 12, 16) if d <= min(split.T, split.P)]:
        td = split.td
        t = np.where((td >= 0) & (td < D0), td, D0 - 1)
        cf = conf_at(split, t)
        for qt in (0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
            th = np.inf if qt == 100 else (-np.inf if qt == 0 else float(np.percentile(cf, qt)))
            for j in [x for x in (1, 2, 4) if x <= split.J]:
                out.append(("selp", D0, th, j))
    return out


def select(dev, ho, cfgs, rd, rh):
    evd, evh = honest(dev, rd), honest(ho, rh)
    res = {c: evd(c) for c in cfgs}
    out = {}
    for b in BUD:
        feas = [(v[0].mean(), -v[1].mean(), c) for c, v in res.items() if v[1].mean() <= b + 1e-9]
        if not feas:
            continue
        c = max(feas, key=lambda z: (z[0], z[1]))[2]
        a, cost, pol = evh(c)
        out[b] = {"cfg": c, "acc_i": a, "cost": float(cost.mean()), "fp32_steps": float(pol.mean())}
    return out


# ----------------------------------------------------------------------------- (2) residual exit
def res_steps(res, eps, K=16):
    m = res[:, :K] < eps
    return np.where(m.any(1), m.argmax(1), K - 1)


def plateau_steps(res, K=16):
    r = res[:, :K]
    m = np.zeros_like(r, bool)
    m[:, 1:] = r[:, 1:] >= r[:, :-1]
    return np.where(m.any(1), m.argmax(1), K - 1)


def tune_eps(c, res, K=16, budget=0.5):
    acc_k = 100 * c[:, K - 1].mean()
    grid = np.unique(np.quantile(res[:, :K], np.linspace(0, 1, 101)))
    best = None
    for e in grid:
        t = res_steps(res, e, K)
        a = 100 * c[np.arange(len(t)), t].mean()
        if a >= acc_k - budget and (best is None or (t + 1).mean() < best[1]):
            best = (float(e), float((t + 1).mean()))
    return best[0] if best else 0.0


def main():
    rng = np.random.default_rng(0)
    out = {"selp": {}, "resexit": {}}
    rows_sel = []
    print("=== (2) head-free exit from the latent step size (held out; capped at 16; rules frozen/tuned on dev rows)")
    print(f"{'model':12s} {'spec':14s} {'fixed16':>6s} | head: acc steps d | res-fp32: acc steps d | res-cell: acc steps d | plateau: acc steps d   (d = minus fixed16 [CI])")
    for fam in C.families():
        task = fam["task"]
        sub = fam.get("sub", task)
        fp = C.cell_splits(task, "fp32", fam)
        if fp is None:
            continue
        rule, rrule = C.rules_for(fp[0][0], task)
        maze = fam["kind"] == "maze"
        rows = (slice(0, 500), slice(500, 1000)) if maze else (None, None)
        dirs = ("det_o0", "det_o0") if maze else ("det_o0", "det_o1000")
        rfp = load_res(ROOT / dirs[0] / sub / "fp32.npz", rows[0])
        eps_fp = tune_eps(fp[0][0][0], rfp)
        for spec in fam["cells"]:
            cs = C.cell_splits(task, spec, fam)
            if cs is None:
                continue
            det, m2, pol, hrm = cs
            dev, ho = [C.Split(*det[k], *m2[k], pol[k], hrm, rule, rrule) for k in (0, 1)]
            # (1) selective polish, honest costs
            r = 32.0 / b_eff(spec)
            res = {}
            for m in ("fixed", "E", "P", "E+P", "E+S", "E+S+P"):
                res[m] = select(dev, ho, dev.configs(m), r, r)
            res["SelP"] = select(dev, ho, selp_configs(dev), r, r)
            rows_sel.append((fam["name"], spec, r, res))
            # (2) residual exit on held-out rows
            rd = load_res(ROOT / dirs[0] / sub / f"{spec}.npz", rows[0])
            rh = load_res(ROOT / dirs[1] / sub / f"{spec}.npz", rows[1])
            ch = ho.dc
            fix = ch[:, 15]
            th = np.where((ho.td >= 0) & (ho.td < 16), ho.td, 15)
            rules = {"head": th, "res-fp32": res_steps(rh, eps_fp), "res-cell": res_steps(rh, tune_eps(dev.dc, rd)),
                     "plateau": plateau_steps(rh)}
            row = {}
            line = f"{fam['name']:12s} {spec:14s} {100 * fix.mean():6.1f} |"
            for nm, t in rules.items():
                a = ch[np.arange(ho.n), t]
                d = boot(a, rng, fix)
                row[nm] = {"acc": 100 * a.mean(), "steps": float((t + 1).mean()), "d_vs_fixed16": d,
                           "d_vs_head": boot(a, rng, ch[np.arange(ho.n), th]) if nm != "head" else None}
                line += f" {100 * a.mean():5.1f} {(t + 1).mean():5.2f} {d[0]:+5.1f}[{d[1]:+.1f},{d[2]:+.1f}] |"
            print(line)
            out["resexit"][f"{fam['name']}|{spec}"] = row
    # summaries for (2)
    print("\n  summary over cells (held out): mean acc minus fixed16 / minus head, mean steps, cells non-inferior to head (CI low > -1)")
    for nm in ("head", "res-fp32", "res-cell", "plateau"):
        vals = [v[nm] for v in out["resexit"].values()]
        dh = [v["d_vs_head"] for v in vals if v["d_vs_head"] is not None]
        print(f"  {nm:9s} d_fixed {np.mean([v['d_vs_fixed16'][0] for v in vals]):+5.2f}  steps {np.mean([v['steps'] for v in vals]):5.2f}"
              + (f"  d_head {np.mean([x[0] for x in dh]):+5.2f}  non-inferior to head {sum(x[1] > -1 for x in dh)}/{len(dh)}"
                 f"  worse than head {sum(x[2] < 0 for x in dh)}/{len(dh)}" if dh else ""))
    # (1) report
    for label, keep in (("24 quantized cells (honest r = 32/b_eff)", lambda s: not (s == "fp32" or "noise" in s)),
                        ("all 46 cells (r = 1 for fp32 / noise cells)", lambda s: True)):
        sel = [x for x in rows_sel if keep(x[1])]
        print(f"\n=== (1) selective polish, held out, {label}: mean acc @ mean honest cost (cheap-step equivalents), fp32 steps per puzzle")
        ms = ("fixed", "E", "P", "E+P", "SelP", "E+S", "E+S+P")
        print(f"  {'budget':>6s} | " + " | ".join(f"{m:>20s}" for m in ms) + " | SelP - fixed [CI] | SelP - full [CI]")
        for b in BUD:
            if not all(b in x[3][m] for x in sel for m in ms):
                continue
            parts = []
            for m in ms:
                acc = np.mean([x[3][m][b]["acc_i"].mean() for x in sel]) * 100
                cst = np.mean([x[3][m][b]["cost"] for x in sel])
                f32 = np.mean([x[3][m][b]["fp32_steps"] for x in sel])
                parts.append(f"{acc:5.1f} @{cst:5.1f} ({f32:4.2f})")
            dfx = [x[3]["SelP"][b]["acc_i"].astype(float) - x[3]["fixed"][b]["acc_i"] for x in sel]
            dfl = [x[3]["SelP"][b]["acc_i"].astype(float) - x[3]["E+S+P"][b]["acc_i"] for x in sel]

            def pooled(ds):
                bs = np.mean([d[rng.integers(0, len(d), (NB, len(d)))].mean(1) for d in ds], 0)
                return f"{100 * np.mean([d.mean() for d in ds]):+5.2f} [{100 * np.percentile(bs, 2.5):+.2f},{100 * np.percentile(bs, 97.5):+.2f}]"
            print(f"  {b:6d} | " + " | ".join(parts) + f" | {pooled(dfx)} | {pooled(dfl)}")
        out["selp"][label] = {f"{x[0]}|{x[1]}": {"r": x[2], **{m: {str(b): {"acc": 100 * v["acc_i"].mean(), "cost": v["cost"],
                                                                            "fp32_steps": v["fp32_steps"], "cfg": [str(z) for z in v["cfg"]]}
                                                                      for b, v in x[3][m].items()} for m in x[3]}} for x in sel}
    json.dump(out, open(A.RB / "m5_addons.json", "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("\nwrote", A.RB / "m5_addons.json")


if __name__ == "__main__":
    main()
