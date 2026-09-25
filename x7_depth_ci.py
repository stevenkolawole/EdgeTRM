#!/usr/bin/env python3
"""X7 depth read-out with paired intervals, every series listed.

A series = (model, quantizer, axis). Axes, each read from written records only:
  nsup : default H/L, gap at n_sup 1 vs n_sup 32 (same run, same examples)
  H    : n_sup 16, gap at the smallest vs the largest H (L at default)
  L    : n_sup 16, gap at the smallest vs the largest L (H at default)
gap = quantized minus fp32 exact accuracy on the same examples, same (task, H, L, n_sup), same file (each file
carries its own fp32 run on the same platform). 95% paired bootstrap over examples (2000 resamples).
A quantizer is in-basin for a model if its default-config n_sup-16 exact accuracy is at least half of fp32's.
A series "widens" if the change of the gap (deep minus shallow) has a 95% CI entirely below 0, "narrows" if
entirely above 0, else "flat". Each series also gets a trend over ALL its points: the least-squares slope of the
gap against log2(depth) (points per doubling), with a paired bootstrap CI (same resampled examples at every
point), so an intermediate widening is not hidden by the choice of endpoints.

    python x7_depth_ci.py 'results/breadth/p4d/x7/depth*_*.json' results/breadth/x7_depth_ci.json [table.md]
"""
import glob
import json
import sys

import numpy as np

TASKS = {"sudoku_mlp": (3, 6), "sudoku_attn": (3, 6), "maze_attn": (3, 4), "hrm_sudoku": (2, 2)}
B = 2000


def boot(v, seed):
    rng = np.random.default_rng(seed)
    n = len(v)
    m = np.array([v[rng.integers(0, n, n)].mean() for _ in range(B)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def slope_ci(vecs, xs, seed):
    """vecs: list of per-example gap vectors (same examples), xs: log2 depths. slope of mean gap vs x."""
    X = np.array(xs, float)
    X = X - X.mean()
    V = np.stack(vecs)                     # points x examples
    est = float((X * V.mean(1)).sum() / (X * X).sum())
    rng = np.random.default_rng(seed)
    n = V.shape[1]
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bs.append((X * V[:, i].mean(1)).sum() / (X * X).sum())
    return est, [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    files = sorted(glob.glob(sys.argv[1]))
    recs = {}          # task -> {(cell, H, L, nsup): (record, fp32 record of the same file)}
    for f in files:
        d = json.load(open(f))
        if d.get("mode") != "depth":
            continue
        fp = {(r["H_cycles"], r["L_cycles"], r["nsup"]): r for r in d["records"] if r["cell"] == "fp32"}
        for r in d["records"]:
            if r["cell"] == "fp32":
                continue
            k = (r["H_cycles"], r["L_cycles"], r["nsup"])
            if k in fp:
                recs.setdefault(d["task"], {})[(r["cell"],) + k] = (r, fp[k])
    out, rows = {}, []
    for task, R in recs.items():
        Hd, Ld = TASKS[task]
        cells = sorted({k[0] for k in R})
        out[task] = {}
        for c in cells:
            base = [v for k, v in R.items() if k[0] == c and k[1] == Hd and k[2] == Ld and k[3] >= 32]
            if not base:
                continue
            q0, f0 = base[0]
            ex16, fx16 = q0["exact"][15], f0["exact"][15]
            inb = ex16 >= 0.5 * fx16
            res = {"in_basin": bool(inb), "exact16": ex16, "fp32_16": fx16, "series": {}}

            def gapvec(q, f, s):
                return np.array(q["correct_at"][s], float) - np.array(f["correct_at"][s], float)

            # nsup axis
            g1, g32 = gapvec(q0, f0, "1"), gapvec(q0, f0, "32")
            pts = {s: {"gap": float(gapvec(q0, f0, s).mean()), "ci95": boot(gapvec(q0, f0, s), 1)}
                   for s in ("1", "2", "4", "8", "16", "32")}
            ss = ("1", "2", "4", "8", "16", "32")
            sl, slci = slope_ci([gapvec(q0, f0, x) for x in ss], [np.log2(int(x)) for x in ss], 5)
            res["series"]["nsup"] = {"from": "1", "to": "32", "points": pts,
                                     "change": float((g32 - g1).mean()), "ci95": boot(g32 - g1, 2),
                                     "slope_per_doubling": sl, "slope_ci95": slci}
            # H and L axes at nsup 16
            for ax, idx, fixed_idx, fixed in (("H", 1, 2, Ld), ("L", 2, 1, Hd)):
                sel = {k[idx]: v for k, v in R.items() if k[0] == c and k[fixed_idx] == fixed}
                if len(sel) < 2:
                    continue
                vals = sorted(sel)
                pts = {}
                for v in vals:
                    q, f = sel[v]
                    g = gapvec(q, f, "16")
                    pts[str(v)] = {"gap": float(g.mean()), "ci95": boot(g, 3 + v)}
                lo, hi = vals[0], vals[-1]
                d = gapvec(*sel[hi], "16") - gapvec(*sel[lo], "16")
                sl, slci = slope_ci([gapvec(*sel[v], "16") for v in vals], [np.log2(v) for v in vals], 6)
                res["series"][ax] = {"from": str(lo), "to": str(hi), "points": pts,
                                     "change": float(d.mean()), "ci95": boot(d, 4),
                                     "slope_per_doubling": sl, "slope_ci95": slci}
            for ax, s in res["series"].items():
                lo, hi = s["ci95"]
                s["verdict"] = "widens" if hi < 0 else ("narrows" if lo > 0 else "flat")
                lo, hi = s["slope_ci95"]
                s["trend"] = "widens" if hi < 0 else ("narrows" if lo > 0 else "flat")
                rows.append((task, c, inb, ax, s))
            out[task][c] = res
    json.dump(out, open(sys.argv[2], "w"), indent=1)

    lines = ["| model | quantizer | in basin | axis | gap at every point | change deep-shallow [95% CI] | endpoints | slope per doubling [95% CI] | trend |",
             "|---|---|---|---|---|---|---|---|---|"]
    for task, c, inb, ax, s in rows:
        p = s["points"]
        a, b = p[s["from"]], p[s["to"]]
        allp = " / ".join(f"{k}:{v['gap']*100:+.1f}" for k, v in p.items())
        lines.append(f"| {task} | {c} | {'yes' if inb else 'no'} | {ax} {s['from']}->{s['to']} | {allp} "
                     f"| {s['change']*100:+.1f} [{s['ci95'][0]*100:+.1f}, {s['ci95'][1]*100:+.1f}] | {s['verdict']} "
                     f"| {s['slope_per_doubling']*100:+.2f} [{s['slope_ci95'][0]*100:+.2f}, {s['slope_ci95'][1]*100:+.2f}] | {s['trend']} |")
    inb_rows = [r for r in rows if r[2]]
    summ = []
    for ax in ("nsup", "H", "L"):
        rr = [r for r in inb_rows if r[3] == ax]
        for key, name in (("verdict", "endpoints"), ("trend", "trend")):
            w = [f"{r[0]}/{r[1]}" for r in rr if r[4][key] == "widens"]
            n_ = [f"{r[0]}/{r[1]}" for r in rr if r[4][key] == "narrows"]
            summ.append(f"in-basin series along {ax} ({name}): {len(rr)}; widen {len(w)} ({', '.join(w) or '-'}); "
                        f"narrow {len(n_)} ({', '.join(n_) or '-'}); flat {len(rr) - len(w) - len(n_)}")
    # every per-point CI excluding 0 on the H axis (for claims like "H sweeps are flat")
    hpts = []
    for task, c, inb, ax, s in inb_rows:
        if ax == "H":
            for v, p in s["points"].items():
                if p["ci95"][1] < 0 or p["ci95"][0] > 0:
                    hpts.append(f"{task}/{c} H={v}: {p['gap']*100:+.1f} [{p['ci95'][0]*100:+.1f}, {p['ci95'][1]*100:+.1f}]")
    summ.append("in-basin H points with a nonzero offset (gap CI excludes 0; an offset, not a trend): " + ("; ".join(hpts) or "none"))
    text = "\n".join(lines + [""] + summ)
    print(text)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(text + "\n")


if __name__ == "__main__":
    main()
