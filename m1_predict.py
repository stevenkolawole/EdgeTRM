#!/usr/bin/env python3
"""M1: does a rule computed from single-format runs only predict when the hybrid helps?

For every (task, checkpoint, offset, T, cheap map) with per-input records, predictors measured WITHOUT running a
hybrid (from the cheap map alone and fp32 alone at the same depth T):
  comp   complementarity = acc(fp32@T OR cheap@T) - max(acc fp32@T, acc cheap@T)   (the two-solver headroom)
  cheap_only, fp32_only   fractions solved by one run only
  ratio  acc(cheap@T) / acc(fp32@T)                                                  (strength)
  gap    acc(cheap@T) - acc(fp32@T)
Outcome (needs the hybrid): gain = acc(best hybrid c(T-k)+INT8(k), k <= T/8) - max(acc fp32@T, acc cheap@T, acc INT8@T)
(the best single format at the same step count; stricter traffic-matched versions are in m1_frontier.py).
Prints the table, Spearman correlations, and the rule "comp >= c and ratio >= r" with (c, r) chosen on the dev
rows (offset 0, released attention + MLP + HRM) and evaluated on every other row.

    python m1_predict.py 'results/breadth/babel/m1/p_*_R.json' 'results/breadth/babel/m1/p_sudoku_*_T64_B*.json' ...
"""
import argparse
import glob
import json
import re

import numpy as np


def arr(s):
    return (np.frombuffer(s.encode(), np.uint8) - 48).astype(int)


def spearman(a, b, n_perm=5000, seed=0):
    """Spearman rho (average ranks) and a two-sided permutation p-value."""
    def rk(x):
        x = np.asarray(x, float)
        o = np.argsort(x)
        r = np.empty(len(x))
        r[o] = np.arange(len(x))
        for v in np.unique(x):                      # average ranks for ties
            m = x == v
            r[m] = r[m].mean()
        return r
    ra, rb = rk(a), rk(b)
    rho = float(np.corrcoef(ra, rb)[0, 1])
    g = np.random.default_rng(seed)
    null = np.array([np.corrcoef(ra, g.permutation(rb))[0, 1] for _ in range(n_perm)])
    return rho, float((np.abs(null) >= abs(rho)).mean())


def rows_from(files):
    out = []
    for f in files:
        b = json.load(open(f))
        T = b["T"]
        fa = b["fp32"].get("at", {})
        if str(T) not in fa:
            continue
        fT = arr(fa[str(T)])
        int8 = {}
        for r in b["records"]:
            ps = r["schedules"].get(f"P{T}")
            if r["polish"].startswith("w8") and ps and "at" in ps and str(T) in ps["at"]:
                int8[r["polish"]] = arr(ps["at"][str(T)])
        for r in b["records"]:
            if "c_at" not in r or not r["polish"].startswith("w8"):
                continue
            cT = arr(r["c_at"][T - 1])
            hyb = [v["acc"] for s, v in r["schedules"].items()
                   if (m := re.fullmatch(r"c(\d+)\.P(\d+)", s)) and int(m.group(1)) + int(m.group(2)) == T
                   and int(m.group(2)) <= max(T // 8, 1)]
            if not hyb:
                continue
            single = max(fT.mean(), cT.mean(), *(v.mean() for v in int8.values()))
            out.append(dict(file=f.split("/")[-1], task=b["task"], ckpt=(b.get("ckpt") or "").split("/")[-2] if b.get("ckpt") else "",
                            off=b.get("offset", 0), T=T, cell=f"{r['cell']}#{r['draw']}", pol=r["polish"],
                            comp=np.maximum(fT, cT).mean() - max(fT.mean(), cT.mean()),
                            cheap_only=float((cT & (1 - fT)).mean()), fp32_only=float((fT & (1 - cT)).mean()),
                            ratio=cT.mean() / max(fT.mean(), 1e-9), gap=cT.mean() - fT.mean(),
                            gain=max(hyb) - single))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    files = sorted({p for g in a.paths for p in glob.glob(g)})
    R = rows_from(files)
    seen, rows = set(), []
    for r in R:                                    # one row per (task, ckpt, off, T, cell), best polish
        key = (r["task"], r["ckpt"], r["off"], r["T"], r["cell"])
        if key in seen:
            continue
        cand = [x for x in R if (x["task"], x["ckpt"], x["off"], x["T"], x["cell"]) == key]
        rows.append(max(cand, key=lambda x: x["gain"]))
        seen.add(key)
    print("| task | ckpt | offset | T | cheap | comp | cheap-only | fp32-only | ratio | gain vs best single @T |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["task"], r["ckpt"], r["off"], r["T"], r["cell"])):
        print(f"| {r['task']} | {r['ckpt'] or 'released'} | {r['off']} | {r['T']} | {r['cell']} | {100*r['comp']:.1f} | "
              f"{100*r['cheap_only']:.1f} | {100*r['fp32_only']:.1f} | {r['ratio']:.3f} | {100*r['gain']:+.1f} |")
    g = np.array([r["gain"] for r in rows])
    for k in ("comp", "cheap_only", "ratio", "gap"):
        rho, p = spearman([r[k] for r in rows], g)
        print(f"Spearman(gain, {k}) = {rho:+.2f} (p = {p:.2g}, n = {len(rows)})")
    dev = [r for r in rows if r["off"] == 0 and not r["ckpt"]]
    rest = [r for r in rows if not (r["off"] == 0 and not r["ckpt"])]
    best = None
    for c in np.arange(0.0, 0.08, 0.005):
        for q in np.arange(0.85, 1.05, 0.01):
            pred = np.array([(r["comp"] >= c) and (r["ratio"] >= q) for r in dev])
            y = np.array([r["gain"] > 0.01 for r in dev])
            accu = (pred == y).mean()
            if best is None or accu > best[0]:
                best = (accu, c, q)
    ydev = np.array([r["gain"] > 0.01 for r in dev])
    print(f"\nbase rate on dev: {int(ydev.sum())}/{len(dev)} rows help (always-'no' agreement {100*(1-ydev.mean()):.0f}%)")
    print(f"rule fitted on {len(dev)} dev rows (released checkpoints, offset 0): helps (gain > 1 pt) iff comp >= "
          f"{100*best[1]:.1f} pts and cheap/fp32 >= {best[2]:.2f}; dev agreement {100*best[0]:.0f}%")
    if rest:
        pred = np.array([(r["comp"] >= best[1]) and (r["ratio"] >= best[2]) for r in rest])
        y = np.array([r["gain"] > 0.01 for r in rest])
        tp, fp_, fn = int((pred & y).sum()), int((pred & ~y).sum()), int((~pred & y).sum())
        print(f"out-of-sample base rate: {int(y.sum())}/{len(rest)} help (always-'no' {100*(1-y.mean()):.0f}%); "
              f"precision {tp}/{tp+fp_}, recall {tp}/{tp+fn}")
        print(f"out-of-sample rows ({len(rest)}: seeds, held-out puzzles, other families): agreement "
              f"{100*(pred == y).mean():.0f}% (predicted helps {int(pred.sum())}, actually helps {int(y.sum())}, both {int((pred & y).sum())})")


if __name__ == "__main__":
    main()
