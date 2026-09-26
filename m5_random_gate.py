#!/usr/bin/env python3
"""Random-gate control for head-gated finishing (M5 R20 / RN133), from the existing p4d records only (no new runs).

Question: does the halting head pick WHICH puzzles to finish, or would finishing any subset of the same size do as well?
Protocol (same cells, rows, selection and honest cost as m5_addons.py):
  * cells: the 24 quantized cells of RN133's headline (every quantized cell with records; noise and fp32 cells excluded);
    with M5_POLISH=w8c the finishing model is w8c and the cells are the 20 quantized Sudoku cells of RN141.
  * a configuration ("ctl", D0, 0, jE, jF) stops each puzzle at the head's fire step t_i < D0 (or at D0), and then runs
    p_i finishing steps (jE if the head fired, jF if it did not). "m1" configurations finish every puzzle.
  * the configuration is chosen on dev rows (0-999) under an honest-cost budget (a finishing step costs r = 32/b_eff
    cheap steps for fp32, 8/b_eff for w8c), and scored on held-out rows (1000-1999), exactly as m5_addons.select.
  * head arm: the configuration as chosen. Random arm: the SAME stop steps t_i and the SAME multiset of finishing steps
    p_i, randomly permuted across puzzles (1000 permutations per cell). Cost is identical by construction, so the only
    difference is which puzzles are finished.
  * pooled over cells: for each permutation index, the mean over cells; mean and 2.5/97.5 percentiles of the 1000 means.
Two selections are reported:
  (A) E+P as in RN133 (the method may switch the gate off: it can also pick pure exit or finish-everything, where the
      random arm is identical to the head arm by construction);
  (B) gated only: configurations ("ctl", D0, 0, 0, j) with D0 <= 16, j in {1, 2, 4}: finish exactly the puzzles on which
      the head has not fired by D0. This isolates the gate.

    python m5_random_gate.py                  # fp32 finishing -> results/breadth/m5_random_gate.json
    M5_POLISH=w8c python m5_random_gate.py    # w8c finishing  -> results/breadth/m5_random_gate_w8c.json
"""
import json

import numpy as np

import analyze_m5 as A
import m5_ctl_ho as C
import m5_addons as MA

NDRAW = 1000
BUDS = (24, 64)


def stops_and_polish(split, cfg):
    """per-puzzle stop index s_i and finishing steps p_i for an E+P-type configuration."""
    n = split.n
    if cfg[0] == "m1":
        t, j = int(cfg[1]), int(cfg[2])
        return np.full(n, t - 1), np.full(n, j)
    if cfg[0] == "fixed":
        return np.full(n, int(cfg[1]) - 1), np.zeros(n, int)
    D0, K, jE, jF = (int(x) for x in cfg[1:])
    assert K == 0, cfg
    td = split.td
    fired = (td >= 0) & (td < D0)
    s = np.where(fired, td, D0 - 1)
    p = np.where(fired, jE, jF if D0 <= split.P else 0)
    return s, p.astype(int)


def acc_of(split, s, p):
    i = np.arange(split.n)
    a = split.dc[i, s].copy()
    m = p > 0
    a[m] = split.pc[i[m], s[m], p[m] - 1]
    return a


def main():
    pol = C.POL
    rng = np.random.default_rng(0)
    out = {"polish": pol, "ndraw": NDRAW, "cells": {}}
    gated = lambda sp: [("ctl", d, 0, 0, j) for d in (1, 2, 4, 8, 12, 16) if d <= sp.P for j in (1, 2, 4) if j <= sp.J]  # noqa: E731
    for fam in C.families():
        task = fam["task"]
        sub = fam.get("sub", task)
        maze = fam["kind"] == "maze"
        d0 = C.load_det(C.ROOT / "det_o0" / sub / "fp32.npz", slice(0, 500) if maze else None)
        if d0 is None:
            continue
        rule, rrule = C.rules_for(d0, task)
        for spec in fam["cells"]:
            if spec == "fp32" or "noise" in spec:
                continue
            cs = C.cell_splits(task, spec, fam)
            if cs is None:
                continue
            det, m2, pl, hrm = cs
            dev, ho = [C.Split(*det[k], *m2[k], pl[k], hrm, rule, rrule) for k in (0, 1)]
            r = (32.0 if pol == "fp32" else 8.0) / MA.b_eff(spec)
            cell = {"r": r}
            for tag, cfgs in (("A_EP", dev.configs("E+P")), ("B_gated", gated(dev))):
                sel = MA.select(dev, ho, cfgs, r, r)
                cell[tag] = {}
                for b in BUDS:
                    if b not in sel:
                        continue
                    cfg = sel[b]["cfg"]
                    s, p = stops_and_polish(ho, cfg)
                    a_head = acc_of(ho, s, p)
                    ok = np.allclose(a_head, sel[b]["acc_i"])
                    frac = float((p > 0).mean())
                    rnd = np.empty(NDRAW)
                    for k in range(NDRAW):
                        rnd[k] = acc_of(ho, s, rng.permutation(p)).mean()
                    cell[tag][b] = {"cfg": [str(x) for x in cfg], "check_matches_select": bool(ok),
                                    "frac_finished": frac, "cost": sel[b]["cost"], "head": 100 * a_head.mean(),
                                    "rand_draws": 100 * rnd, "nontrivial": bool(0 < frac < 1)}
            out["cells"][f"{fam['name']}|{spec}"] = cell
            print(f"{fam['name']:12s} {spec:7s} r={r:5.2f} " + " ".join(
                f"{t}@{b}: head {v['head']:5.1f} rand {v['rand_draws'].mean():5.1f} fin {v['frac_finished']:.2f} {'' if v['check_matches_select'] else 'MISMATCH'}"
                for t in ("A_EP", "B_gated") for b, v in cell[t].items()))
    print(f"\n=== pooled over {len(out['cells'])} cells, held out, finishing = {pol}; random = same stop steps, same finishing steps, permuted")
    summ = {}
    for tag in ("A_EP", "B_gated"):
        for b in BUDS:
            vs = [c[tag][b] for c in out["cells"].values() if b in c[tag]]
            head = np.array([v["head"] for v in vs])
            rd = np.stack([v["rand_draws"] for v in vs])            # [cells, draws]
            pooled = rd.mean(0)
            diff = head.mean() - pooled
            nt = [i for i, v in enumerate(vs) if v["nontrivial"]]
            percell = [(head[i] - rd[i].mean(), float((rd[i] >= head[i]).mean())) for i in nt]
            s = {"cells": len(vs), "nontrivial_cells": len(nt), "head": float(head.mean()),
                 "rand_mean": float(pooled.mean()), "rand_lo": float(np.percentile(pooled, 2.5)),
                 "rand_hi": float(np.percentile(pooled, 97.5)), "head_minus_rand": float(diff.mean()),
                 "head_minus_rand_lo": float(np.percentile(diff, 2.5)), "head_minus_rand_hi": float(np.percentile(diff, 97.5)),
                 "draws_rand_ge_head": int((pooled >= head.mean()).sum()),
                 "mean_frac_finished": float(np.mean([v["frac_finished"] for v in vs])),
                 "mean_cost": float(np.mean([v["cost"] for v in vs])),
                 "nontrivial_head_minus_rand": [float(x[0]) for x in percell],
                 "nontrivial_p_rand_ge_head": [x[1] for x in percell],
                 "nontrivial_names": [list(out["cells"])[i] for i in nt]}
            if nt:
                hn = head[nt].mean()
                pn = rd[nt].mean(0)
                s.update({"nt_head": float(hn), "nt_rand_mean": float(pn.mean()), "nt_rand_lo": float(np.percentile(pn, 2.5)),
                          "nt_rand_hi": float(np.percentile(pn, 97.5))})
            summ[f"{tag}@{b}"] = s
            print(f"  {tag:8s} budget {b:3d}: head {s['head']:5.1f} | random {s['rand_mean']:5.1f} [{s['rand_lo']:5.1f}, {s['rand_hi']:5.1f}]"
                  f" | head - random {s['head_minus_rand']:+5.2f} [{s['head_minus_rand_lo']:+.2f}, {s['head_minus_rand_hi']:+.2f}]"
                  f" | cost {s['mean_cost']:5.1f} | finished {s['mean_frac_finished']:.2f} | gate active in {len(nt)}/{len(vs)} cells"
                  + (f" (those: head {s['nt_head']:5.1f} vs random {s['nt_rand_mean']:5.1f} [{s['nt_rand_lo']:5.1f}, {s['nt_rand_hi']:5.1f}])" if nt else ""))
    out["summary"] = summ
    for c in out["cells"].values():
        for tag in ("A_EP", "B_gated"):
            for v in c[tag].values():
                v["rand_mean"] = float(v["rand_draws"].mean())
                v["rand_lo"], v["rand_hi"] = (float(x) for x in np.percentile(v["rand_draws"], [2.5, 97.5]))
                del v["rand_draws"]
    fn = A.RB / ("m5_random_gate.json" if pol == "fp32" else f"m5_random_gate_{pol}.json")
    json.dump(out, open(fn, "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("wrote", fn)


if __name__ == "__main__":
    main()
