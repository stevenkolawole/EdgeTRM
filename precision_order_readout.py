#!/usr/bin/env python3
"""Read-out of the precision-order runs (polish.py --only_extra --polish w8c, schedules c16.P16 (marks 20/24/32), Pk.c16, c(16+k), c16, f16).
Per cell and k: accuracy of LAST = c16.Pk, FIRST = Pk.c16, CONTROL = c(16+k), c16, fp32 at 16 loops and at 16+k loops, with paired
bootstrap (4000 resamples, 95%) over puzzles for LAST-FIRST, LAST-c16, FIRST-c16, FIRST-CONTROL; fraction of the gap (fp32@16 - c16) recovered.
    python precision_order_readout.py out_dir/*.json
"""
import json, sys
import numpy as np
rng = np.random.default_rng(0)
def dec(s): return np.array([int(c) for c in s], float)
def bci(d, nb=4000):
    b = d[rng.integers(0, len(d), (nb, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)
for f in sys.argv[1:]:
    B = json.load(open(f))
    fp = B["fp32"]; f16 = dec(fp["correct"])
    for r in B["records"]:
        S = r["schedules"]
        c16 = dec(S["c16"]["correct"]); gap = 100 * (f16.mean() - c16.mean())
        print(f"\n{B['task']} {r['cell']} n={B['n']}: fp32@16 {100*f16.mean():.1f}  c16 {100*c16.mean():.1f}  gap {gap:.1f}; fp32 curve@16,20,24,32 "
              f"{[round(100*fp['curve'][k-1],1) for k in (16,20,24,32) if k <= len(fp['curve'])]}")
        print("  k | LAST c16.Pk | FIRST Pk.c16 | CONTROL c(16+k) | LAST-FIRST [CI] | LAST-c16 [CI] | FIRST-c16 [CI] | FIRST-CONTROL [CI] | recovered LAST / FIRST")
        for k in (4, 8, 16):
            last = np.array([int(ch) for ch in S["c16.P16"]["at"][str(16 + k)]], float)
            first = dec(S[f"P{k}.c16"]["correct"]); ctl = dec(S[f"c{16+k}"]["correct"])
            fm = lambda t: f"{t[0]:+5.1f} [{t[1]:+5.1f},{t[2]:+5.1f}]"
            rl = (100*(last.mean()-c16.mean()))/gap if gap > 0 else float("nan"); rf = (100*(first.mean()-c16.mean()))/gap if gap > 0 else float("nan")
            print(f"  {k:2d} | {100*last.mean():5.1f} | {100*first.mean():5.1f} | {100*ctl.mean():5.1f} | {fm(bci(last-first))} | {fm(bci(last-c16))} | {fm(bci(first-c16))} | {fm(bci(first-ctl))} | {rl:.2f} / {rf:.2f}")
