#!/usr/bin/env python3
"""M5: does the controller's lead survive without fp32 weights? (p4d records only; results/m5/ctl/p4d)

Same cells, same dev / held-out rows, same selection as m5_ctl_ho.py / m5_addons.py. The polish model is either fp32
(records <spec>__fp32.npz; rollout end states polished with fp32, M2's P2 records) or w8c (<spec>__w8c.npz, m5w8_* lines;
rollout end states are NOT polished, because rollout polish records exist for fp32 only).
Honest cost: a cheap step costs 1. A polish step costs r = b_polish / b_eff(cheap), with b_polish = 32 (fp32) or 8 (w8c):
weight traffic of bandwidth-bound batch-1 recursion. w8c polish: w4 2, w3 2.67, w3g32 2.29, w2g32 3.2, w2a 4.
Noise cells have fp32-sized cheap weights, so r = 0.25 for w8c and 1 for fp32. They appear only in the 28-cell table.
Cells with w8c records: released TRM-MLP / TRM-attn / HRM Sudoku compressed cells (15) and trained seeds x {w4c, w3c} (12).
No maze cells (their polish lines were not run) and no fp32 cells.
Rows: dev 0-999 (configurations chosen here), held out 1000-1999 (every number reported).

Part B, cheap: the TRM-MLP head rule on the raw halting logit. The fp32-chosen pick for the released MLP is patience 2 on
the native rule (q_halt > 0), whose recovery past the edge is low. Alternative: q_halt > tau on the raw logit, tau chosen on
the dev rows of the fp32 cell (fewest mean steps within 0.5 pt of fixed 16, the pick criterion), frozen, capped at 16,
held out. Compared with pat2 and native on the MLP cells (released and mlp512 seeds).

    python m5_w8.py      # tables + results/breadth/m5_w8.json
"""
import json

import numpy as np

import analyze_m5 as A
import m5_addons as AD
import m5_ctl_ho as C

NB = 2000
BUD = [16, 24, 32, 48, 64, 96, 128]
METH = ("fixed", "E+P", "E+S", "E+S+P")


def splits(fam, spec, pol):
    C.POL = pol
    try:
        cs = C.cell_splits(fam["task"], spec, fam)
    finally:
        C.POL = "fp32"
    return cs


def pooled(ds, rng):
    bs = np.mean([d[rng.integers(0, len(d), (NB, len(d)))].mean(1) for d in ds], 0)
    return 100 * np.mean([d.mean() for d in ds]), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def thr_steps(q, tau, K=16):
    m = q[:, :K] > tau
    return np.where(m.any(1), m.argmax(1), K - 1)


def main():
    rng = np.random.default_rng(0)
    rows = []
    out = {"w8": {}, "mlp_logit": {}}
    for fam in C.families():
        if fam["kind"] == "maze":
            continue
        fp = C.cell_splits(fam["task"], "fp32", fam)
        rule, rrule = C.rules_for(fp[0][0], fam["task"])
        for spec in fam["cells"]:
            if spec == "fp32":
                continue
            a = splits(fam, spec, "fp32")
            b = splits(fam, spec, "w8c")
            if a is None or b is None:
                continue
            be = AD.b_eff(spec)
            res = {}
            for pol, cs, rp in (("fp32", a, 32.0 / be), ("w8c", b, 8.0 / be)):
                det, m2, pc, hrm = cs
                dev, ho = [C.Split(*det[k], *m2[k], pc[k], hrm, rule, rrule) for k in (0, 1)]
                for m in METH:
                    if pol == "w8c" and m in ("fixed", "E+S"):
                        continue
                    res[(m, pol)] = AD.select(dev, ho, dev.configs(m), rp, rp)
            rows.append((fam["name"], spec, be, res))
    groups = {"20 quantized Sudoku cells": [x for x in rows if "noise" not in x[1]], "all 27 compressed Sudoku cells": rows}
    keys = [("fixed", "fp32"), ("E+S", "fp32"), ("E+P", "fp32"), ("E+P", "w8c"), ("E+S+P", "fp32"), ("E+S+P", "w8c")]
    for label, sel in groups.items():
        print(f"\n=== {label}, held out: mean acc @ honest cost (fp32 or w8c polish steps per puzzle); paired pooled CIs")
        print(f"  {'budget':>6s} | " + " | ".join(f"{m + '/' + p:>22s}" for m, p in keys)
              + " | full w8c - fixed | full w8c - full fp32 | full w8c - E+S | gated w8c - fixed")
        tab = {}
        for bgt in BUD:
            if not all(bgt in x[3][k] for x in sel for k in keys):
                continue
            parts = []
            for k in keys:
                parts.append(f"{100 * np.mean([x[3][k][bgt]['acc_i'].mean() for x in sel]):5.1f} @{np.mean([x[3][k][bgt]['cost'] for x in sel]):5.1f} "
                             f"({np.mean([x[3][k][bgt]['fp32_steps'] for x in sel]):4.2f})")
            d = lambda k1, k2: pooled([x[3][k1][bgt]["acc_i"].astype(float) - x[3][k2][bgt]["acc_i"] for x in sel], rng)  # noqa: E731
            dd = {"full_w8-fixed": d(("E+S+P", "w8c"), ("fixed", "fp32")), "full_w8-full_fp32": d(("E+S+P", "w8c"), ("E+S+P", "fp32")),
                  "full_w8-ES": d(("E+S+P", "w8c"), ("E+S", "fp32")), "gated_w8-fixed": d(("E+P", "w8c"), ("fixed", "fp32"))}
            f = lambda t: f"{t[0]:+5.2f} [{t[1]:+.2f},{t[2]:+.2f}]"  # noqa: E731
            print(f"  {bgt:6d} | " + " | ".join(f"{p:>22s}" for p in parts) + " | " + " | ".join(f(v) for v in dd.values()))
            tab[bgt] = {"mean": {f"{k[0]}/{k[1]}": (100 * np.mean([x[3][k][bgt]["acc_i"].mean() for x in sel]),
                                                     float(np.mean([x[3][k][bgt]["cost"] for x in sel])),
                                                     float(np.mean([x[3][k][bgt]["fp32_steps"] for x in sel]))) for k in keys},
                        "diffs": dd}
        out["w8"][label] = {"cells": [f"{x[0]}|{x[1]}" for x in sel], "table": tab}
    # per-cell at budget 64 for the record
    print("\n  per cell, budget 64: fixed | full fp32 | full w8c | full w8c - fixed [CI]")
    for x in rows:
        if 64 in x[3][("E+S+P", "w8c")]:
            fx, f32, f8 = (x[3][k][64]["acc_i"] for k in (("fixed", "fp32"), ("E+S+P", "fp32"), ("E+S+P", "w8c")))
            dlt = AD.boot(f8, rng, fx)
            print(f"  {x[0]:12s} {x[1]:14s} {100 * fx.mean():5.1f} | {100 * f32.mean():5.1f} | {100 * f8.mean():5.1f} | {dlt[0]:+5.1f} [{dlt[1]:+.1f},{dlt[2]:+.1f}]")
    # ---------------- part B: MLP head rule on the raw logit
    print("\n=== B: TRM-MLP head rule on the raw halting logit (q_halt > tau, tau from dev fp32), held out, capped at 16")
    print(f"  {'model':12s} {'spec':14s} fixed16 | native: acc steps rec | pat2: acc steps rec | logit>tau: acc steps rec [d vs pat2 CI]")
    for fam in C.families():
        if fam["task"] != "sudoku_mlp":
            continue
        fp = C.cell_splits(fam["task"], "fp32", fam)
        dc, dq, _ = fp[0][0]
        ref = A.Cell("sudoku_mlp", "fp32", "p4d", dc[:, :16], dq[:, :16])
        pick = A.rule_name(*A.select_rule(ref))
        acc16 = 100 * dc[:, 15].mean()
        best = None
        for tau in np.round(np.arange(-6, 12.01, 0.25), 2):
            t = thr_steps(dq, tau)
            a = 100 * dc[np.arange(len(t)), t].mean()
            if a >= acc16 - 0.5 and (best is None or (t + 1).mean() < best[1]):
                best = (float(tau), float((t + 1).mean()))
        tau = best[0]
        for spec in fam["cells"]:
            cs = C.cell_splits(fam["task"], spec, fam)
            if cs is None:
                continue
            c, q, _ = cs[0][1]
            c, q = c[:, :16], q[:, :16]
            fin = c[:, 15]
            gap = c.any(1).mean() - fin.mean()
            rs = {"native": thr_steps(q, 0.0), "pat2": A.first_true(A.run_len(q > 0) >= 2), f"logit>{tau:g}": thr_steps(q, tau)}
            line = f"  {fam['name']:12s} {spec:14s} {100 * fin.mean():6.1f} |"
            row = {"fp32_pick": pick, "tau": tau}
            for nm, t in rs.items():
                ok = c[np.arange(len(t)), t]
                rec = (ok.mean() - fin.mean()) / gap if gap >= 0.02 else None
                row[nm] = {"acc": 100 * ok.mean(), "steps": float((t + 1).mean()), "recovery": rec}
                line += f" {100 * ok.mean():5.1f} {(t + 1).mean():5.2f} " + (f"{rec:+.2f}" if rec is not None else "   - ") + " |"
            d = AD.boot(c[np.arange(len(fin)), rs[f"logit>{tau:g}"]], rng, c[np.arange(len(fin)), rs["pat2"]])
            row["d_logit_vs_pat2"] = d
            print(line + f" [{d[0]:+.1f} ({d[1]:+.1f},{d[2]:+.1f})]  (fp32 pick {pick})")
            out["mlp_logit"][f"{fam['name']}|{spec}"] = row
    json.dump(out, open(A.RB / "m5_w8.json", "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("\nwrote", A.RB / "m5_w8.json")


if __name__ == "__main__":
    main()
