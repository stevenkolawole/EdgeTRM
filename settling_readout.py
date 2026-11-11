#!/usr/bin/env python3
"""Read-out of the settling test (polish.py --polish w8c --only_extra, k=8 8-bit loops, T compressed loops, T in 16..256).
Per cell and T: LAST cT.P8, FIRST P8.cT, CONTROL c(T+8), cT alone; gains G_first = FIRST - CONTROL, G_last = LAST - CONTROL, LAST - FIRST, with paired bootstrap
(4000 resamples, 95%) over puzzles; fp32 curve at T+8 as reference.    python settling_readout.py out/*.json
"""
import json, sys
import numpy as np
rng = np.random.default_rng(0)
dec = lambda s: np.array([int(c) for c in s], float)
def bci(d, nb=4000):
    b = d[rng.integers(0, len(d), (nb, len(d)))].mean(1)
    return 100 * d.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)
fm = lambda t: f"{t[0]:+6.1f} [{t[1]:+6.1f},{t[2]:+6.1f}]"
for f in sys.argv[1:]:
    B = json.load(open(f)); fp = B["fp32"]
    for r in B["records"]:
        S = r["schedules"]
        print(f"\n{B['task']} {r['cell']} n={B['n']}: fp32@16 {100*fp['curve'][15]:.1f}; fp32@24/40/72/136/264 "
              f"{[round(100*fp['curve'][k-1],1) for k in (24,40,72,136,264) if k <= len(fp['curve'])]}")
        print("    T | cT | CONTROL c(T+8) | FIRST P8.cT | LAST cT.P8 | G_first = FIRST-CONTROL [CI] | G_last = LAST-CONTROL [CI] | LAST-FIRST [CI]")
        for T in (16, 32, 64, 128, 256):
            g = lambda n: dec(S[n]["correct"])
            c, ctl, first, last = g(f"c{T}"), g(f"c{T+8}"), g(f"P8.c{T}"), g(f"c{T}.P8")
            print(f"  {T:3d} | {100*c.mean():5.1f} | {100*ctl.mean():5.1f} | {100*first.mean():5.1f} | {100*last.mean():5.1f} | {fm(bci(first-ctl))} | {fm(bci(last-ctl))} | {fm(bci(last-first))}")
