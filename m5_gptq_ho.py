#!/usr/bin/env python3
"""M5 R22 read-out: the controller vs and with GPTQ (p4d records, results/m5/ctl/p4d; pre-stated in M5.md R22).

Per model x format, held-out rows 1000-1999 with configurations chosen on dev rows 0-999 (m5_ctl_ho selection):
  (1) RTN fixed depth   (2) GPTQ fixed depth   (3) RTN + full controller (w8c finishing)   (4) GPTQ + the same
  ceiling: fp32 fixed depth
Fixed depth = the best dev depth whose cost fits the budget (<= 64 steps recorded). The controller is E+S+P from m5_ctl_ho,
with w8c polish (rollout end states unpolished, K8 rollouts; existing RTN K16 records use their first 8 rollouts).
Honest cost: a w8c polish step costs 8 / b_eff cheap steps (b_eff of the format; GPTQ has the same grid as RTN).
Paired puzzle-bootstrap CIs on the held-out rows. The interaction (gain on GPTQ minus gain on RTN) is paired per puzzle.

    python m5_gptq_ho.py      # table + results/breadth/m5_gptq_ho.json
"""
import glob
import json

import numpy as np

import analyze_m5 as A
import m5_addons as AD
import m5_ctl_ho as C

NB = 2000
FMTS = {"sudoku_mlp": ["w3c", "w3g32", "w2g32", "w4t"], "sudoku_attn": ["w3c", "w3g32", "w2g32"], "hrm_sudoku": ["w3c", "w3g32", "w2g32"]}
BUD = [16, 32, 64]
KR = 8


def load_m2_k8(task, spec, sigma, K, suffix):
    for k in (8, 16, 32):
        fs = sorted(glob.glob(str(C.ROOT / "m2cells" / f"{task}__{spec}__ptrm{sigma:g}__D16_K{k}_P2{suffix}.npz")))
        if fs:
            d = np.load(fs[0])
            return d["corr"][:KR].astype(bool), d["q"][:KR].astype(np.float32), d["pol_corr"][:KR].astype(bool)
    return None


def boot(x, rng):
    x = np.asarray(x, float)
    b = x[rng.integers(0, len(x), (NB, len(x)))].mean(1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def main():
    rng = np.random.default_rng(0)
    C.load_m2 = load_m2_k8
    out = {}
    f3 = lambda t: f"{t[0]:+5.1f} [{t[1]:+.1f},{t[2]:+.1f}]"  # noqa: E731
    for task, fmts in FMTS.items():
        fam = {"name": task, "task": task, "kind": "sudoku", "K": KR, "cells": []}
        fp = C.cell_splits(task, "fp32", fam)
        rule, rrule = C.rules_for(fp[0][0], task)
        f32dev, f32ho = fp[0][0][0], fp[0][1][0]
        for fmt in fmts:
            arms = {}
            ok = True
            for kind, spec in (("RTN", fmt), ("GPTQ", "gq" + fmt)):
                C.POL = "w8c"
                try:
                    cs = C.cell_splits(task, spec, fam)
                finally:
                    C.POL = "fp32"
                if cs is None:
                    print(f"  {task} {spec}: records missing")
                    ok = False
                    break
                det, m2, pc, hrm = cs
                dev, ho = [C.Split(*det[k], *m2[k], pc[k], hrm, rule, rrule) for k in (0, 1)]
                r = 8.0 / AD.b_eff(fmt)
                arms[kind] = {m: AD.select(dev, ho, dev.configs(m), r, r) for m in ("fixed", "E+S+P")}
                arms[kind]["fixed16"] = ho.dc[:, 15]
            if not ok:
                continue
            row = {"fp32_fixed16": boot(f32ho[:, 15], rng)}
            print(f"\n=== {task} {fmt} (held out n={len(f32ho)}; fp32 fixed-16 ceiling {row['fp32_fixed16'][0]:.1f}; "
                  f"RTN fixed-16 {100 * arms['RTN']['fixed16'].mean():.1f}, GPTQ fixed-16 {100 * arms['GPTQ']['fixed16'].mean():.1f})")
            print(f"  {'budget':>6s} | (1) RTN fixed | (2) GPTQ fixed | (3) RTN+ctl | (4) GPTQ+ctl | (4)-(2) ctl on GPTQ | (3)-(1) ctl on RTN | "
                  f"(3)-(2) RTN+ctl vs GPTQ | (4)-(3) | interaction")
            for b in BUD:
                if not all(b in arms[k][m] for k in arms for m in ("fixed", "E+S+P")):
                    continue
                a1, a2 = arms["RTN"]["fixed"][b], arms["GPTQ"]["fixed"][b]
                a3, a4 = arms["RTN"]["E+S+P"][b], arms["GPTQ"]["E+S+P"][b]
                x1, x2, x3, x4 = (v["acc_i"].astype(float) for v in (a1, a2, a3, a4))
                d = {"ctl_on_gptq": boot(x4 - x2, rng), "ctl_on_rtn": boot(x3 - x1, rng), "rtnctl_vs_gptq": boot(x3 - x2, rng),
                     "gptqctl_vs_rtnctl": boot(x4 - x3, rng), "interaction": boot((x4 - x2) - (x3 - x1), rng),
                     "gptqctl_vs_fp32": boot(x4 - f32ho[:, 15], rng)}
                cell = lambda v: f"{100 * v['acc_i'].mean():5.1f}@{v['cost']:5.1f}"  # noqa: E731
                print(f"  {b:6d} | {cell(a1):>13s} | {cell(a2):>14s} | {cell(a3):>11s} | {cell(a4):>12s} | {f3(d['ctl_on_gptq']):>19s} | "
                      f"{f3(d['ctl_on_rtn']):>18s} | {f3(d['rtnctl_vs_gptq']):>23s} | {f3(d['gptqctl_vs_rtnctl'])} | {f3(d['interaction'])}")
                row[str(b)] = {"acc": {k: 100 * v["acc_i"].mean() for k, v in (("RTN_fixed", a1), ("GPTQ_fixed", a2), ("RTN_ctl", a3), ("GPTQ_ctl", a4))},
                               "cost": {k: v["cost"] for k, v in (("RTN_fixed", a1), ("GPTQ_fixed", a2), ("RTN_ctl", a3), ("GPTQ_ctl", a4))},
                               "w8c_steps": {"RTN_ctl": a3["fp32_steps"], "GPTQ_ctl": a4["fp32_steps"]},
                               "cfg": {"RTN_ctl": [str(z) for z in a3["cfg"]], "GPTQ_ctl": [str(z) for z in a4["cfg"]]}, "diffs": d}
            out[f"{task}|{fmt}"] = row
    # pooled over cells at each budget
    print("\n=== pooled over cells (mean of per-cell held-out accuracy; paired CIs pooled over cells)")
    for b in BUD:
        ks = [k for k, v in out.items() if str(b) in v]
        if not ks:
            continue
        m = lambda key: np.mean([out[k][str(b)]["acc"][key] for k in ks])  # noqa: E731
        print(f"  budget {b}: {len(ks)} cells | fp32 fixed16 {np.mean([out[k]['fp32_fixed16'][0] for k in ks]):.1f} | RTN fixed {m('RTN_fixed'):.1f} | "
              f"GPTQ fixed {m('GPTQ_fixed'):.1f} | RTN+ctl {m('RTN_ctl'):.1f} | GPTQ+ctl {m('GPTQ_ctl'):.1f} | cells with ctl-on-GPTQ CI>0: "
              f"{sum(out[k][str(b)]['diffs']['ctl_on_gptq'][1] > 0 for k in ks)} | RTN+ctl > GPTQ fixed (CI>0): "
              f"{sum(out[k][str(b)]['diffs']['rtnctl_vs_gptq'][1] > 0 for k in ks)}, < (CI<0): {sum(out[k][str(b)]['diffs']['rtnctl_vs_gptq'][2] < 0 for k in ks)}")
    json.dump(out, open(A.RB / "m5_gptq_ho.json", "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("wrote", A.RB / "m5_gptq_ho.json")


if __name__ == "__main__":
    main()
