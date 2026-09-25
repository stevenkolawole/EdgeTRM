#!/usr/bin/env python3
"""X7 depth read-out with paired intervals: gap to fp32 (same task, H, L, nsup, platform, examples) at
depth d, and the change of that gap between the shallowest and deepest setting. Paired bootstrap over
examples (2000 resamples), 95% percentile intervals.
    python x7_depth_ci.py 'results/breadth/p4d/x7/depth_*.json' results/breadth/x7_depth_ci.json"""
import glob, json, sys
import numpy as np

rng = np.random.default_rng(0)
B = 2000


def boot(fn, n):
    idx = rng.integers(0, n, size=(B, n))
    v = np.array([fn(i) for i in idx])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


out = {}
for f in sorted(glob.glob(sys.argv[1])):
    d = json.load(open(f))
    recs = d["records"]
    task = d["task"]
    T = {}
    for r in recs:
        T[(r["cell"], r["H_cycles"], r["L_cycles"], r["nsup"])] = r
    res = {}
    # (a) nsup axis at the default config (the nsup-32 run): steps 1, 16, 32
    for (c, H, L, ns), r in T.items():
        if c == "fp32":
            continue
        fp = T[("fp32", H, L, ns)]
        steps = [s for s in ("1", "2", "4", "8", "16", "32") if s in r["correct_at"]]
        q = {s: np.array(r["correct_at"][s], float) for s in steps}
        p = {s: np.array(fp["correct_at"][s], float) for s in steps}
        n = len(q[steps[0]])
        row = {}
        for s in steps:
            g = q[s] - p[s]
            row[f"gap@{s}"] = {"gap": float(g.mean()), "ci95": boot(lambda i: g[i].mean(), n),
                               "q": float(q[s].mean()), "fp32": float(p[s].mean())}
        lo, hi = steps[0], steps[-1]
        dg = (q[hi] - p[hi]) - (q[lo] - p[lo])
        row[f"gap_change_{lo}->{hi}"] = {"delta": float(dg.mean()), "ci95": boot(lambda i: dg[i].mean(), n)}
        res[f"{c}|H{H}|L{L}|nsup{ns}"] = row
    out[task] = res
json.dump(out, open(sys.argv[2], "w"), indent=1)
for task, res in out.items():
    print(f"== {task}")
    for k, row in res.items():
        ch = [v for kk, v in row.items() if kk.startswith("gap_change")][0]
        chk = [kk for kk in row if kk.startswith("gap_change")][0]
        g16 = row.get("gap@16")
        print(f"  {k:28s} gap@1 {row['gap@1']['gap']*100:+6.1f}  gap@16 {g16['gap']*100:+6.1f} "
              f"[{g16['ci95'][0]*100:+.1f},{g16['ci95'][1]*100:+.1f}]  {chk} {ch['delta']*100:+6.1f} "
              f"[{ch['ci95'][0]*100:+.1f},{ch['ci95'][1]*100:+.1f}]")
