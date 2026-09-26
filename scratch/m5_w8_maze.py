"""Maze cells through the exact m5_w8.py pipeline (same splits, selection rule, honest cost), which m5_w8.py skips.
Rows: dev 0-499, held out 500-999 (the Maze split of m5_ctl_ho.py). Prints the same pooled table as m5_w8.py."""
import sys, json, numpy as np
sys.path.insert(0, ".")
import m5_w8 as W, m5_ctl_ho as C, m5_addons as AD
rng = np.random.default_rng(0)
rows = []
for fam in C.families():
    if fam["kind"] != "maze":
        continue
    fp = C.cell_splits(fam["task"], "fp32", fam)
    rule, rrule = C.rules_for(fp[0][0], fam["task"])
    for spec in fam["cells"]:
        if spec == "fp32" or "noise" in spec:
            continue
        a = W.splits(fam, spec, "fp32"); b = W.splits(fam, spec, "w8c")
        be = AD.b_eff(spec); res = {}
        for pol, cs, rp in (("fp32", a, 32.0 / be), ("w8c", b, 8.0 / be)):
            det, m2, pc, hrm = cs
            dev, ho = [C.Split(*det[k], *m2[k], pc[k], hrm, rule, rrule) for k in (0, 1)]
            for m in W.METH:
                if pol == "w8c" and m in ("fixed", "E+S"):
                    continue
                res[(m, pol)] = AD.select(dev, ho, dev.configs(m), rp, rp)
        rows.append((fam["name"], spec, be, res))
keys = [("fixed", "fp32"), ("E+S", "fp32"), ("E+P", "fp32"), ("E+P", "w8c"), ("E+S+P", "fp32"), ("E+S+P", "w8c")]
sel = rows
print(f"\n=== 4 quantized Maze cells (TRM Maze w2a, w4t; HRM Maze w2a, w2g32), held out rows 500-999: mean acc @ honest cost (polish steps per puzzle)")
print(f"  {'budget':>6s} | " + " | ".join(f"{m + '/' + p:>22s}" for m, p in keys) + " | full w8c - fixed | full w8c - full fp32 | gated w8c - fixed")
out = {}
for bgt in [8, 16, 24, 32, 48, 64]:
    if not all(bgt in x[3][k] for x in sel for k in keys):
        continue
    parts = [f"{100 * np.mean([x[3][k][bgt]['acc_i'].mean() for x in sel]):5.1f} @{np.mean([x[3][k][bgt]['cost'] for x in sel]):5.1f} ({np.mean([x[3][k][bgt]['fp32_steps'] for x in sel]):4.2f})" for k in keys]
    d = lambda k1, k2: W.pooled([x[3][k1][bgt]["acc_i"].astype(float) - x[3][k2][bgt]["acc_i"] for x in sel], rng)
    dd = {"full_w8-fixed": d(("E+S+P", "w8c"), ("fixed", "fp32")), "full_w8-full_fp32": d(("E+S+P", "w8c"), ("E+S+P", "fp32")), "gated_w8-fixed": d(("E+P", "w8c"), ("fixed", "fp32"))}
    f = lambda t: f"{t[0]:+5.2f} [{t[1]:+.2f},{t[2]:+.2f}]"
    print(f"  {bgt:6d} | " + " | ".join(f"{p:>22s}" for p in parts) + " | " + " | ".join(f(v) for v in dd.values()))
    out[bgt] = {"mean": {f"{k[0]}/{k[1]}": (100 * np.mean([x[3][k][bgt]["acc_i"].mean() for x in sel]), float(np.mean([x[3][k][bgt]["cost"] for x in sel])), float(np.mean([x[3][k][bgt]["fp32_steps"] for x in sel]))) for k in keys}, "diffs": {k: [float(v) for v in t] for k, t in dd.items()}}
for bgt in (32, 48):
    print(f"\n  per cell, budget {bgt}: fixed | E+P fp32 | E+P w8c | full fp32 | full w8c | full w8c - fixed [CI]   (acc @ cost)")
    for x in rows:
        if bgt in x[3][("E+S+P", "w8c")]:
            g = lambda k: x[3][k][bgt]
            fx, f32, f8 = (g(k)["acc_i"] for k in (("fixed", "fp32"), ("E+S+P", "fp32"), ("E+S+P", "w8c")))
            dlt = AD.boot(f8, rng, fx)
            print(f"  {x[0]:10s} {x[1]:6s} {100*fx.mean():5.1f}@{g(('fixed','fp32'))['cost']:4.1f} | {100*g(('E+P','fp32'))['acc_i'].mean():5.1f}@{g(('E+P','fp32'))['cost']:4.1f} | {100*g(('E+P','w8c'))['acc_i'].mean():5.1f}@{g(('E+P','w8c'))['cost']:4.1f} | {100*f32.mean():5.1f}@{g(('E+S+P','fp32'))['cost']:4.1f} | {100*f8.mean():5.1f}@{g(('E+S+P','w8c'))['cost']:4.1f} | {dlt[0]:+5.1f} [{dlt[1]:+.1f},{dlt[2]:+.1f}]")
json.dump({"cells": [f"{x[0]}|{x[1]}" for x in rows], "table": out}, open("results/breadth/m5_w8_maze.json", "w"), indent=1)
print("\nwrote results/breadth/m5_w8_maze.json")
