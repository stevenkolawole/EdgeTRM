#!/usr/bin/env python3
"""M1: does the hybrid's gain sit where there are many attractors (Higham & Mary's boundary condition)?

Attractor count per puzzle (from an fp32-only run with --attractors K): the number of distinct fp32 answers reached
from the default carry plus K random initial carries at depth T. Per puzzle, from a hybrid record of the same task,
depth and puzzles: gain = hybrid correct and fp32@T wrong; loss = the reverse; switch = the hybrid's answer differs from
fp32@T's answer (from the per-input 'agree' < 1 of fp32-polish records, else hybrid-vs-fp32 correctness disagreement).
Reports gain / loss rates by attractor-count bin, and per model the fraction of multi-attractor puzzles against
the model-level hybrid gain.

    python m1_attractors.py --attr results/breadth/babel/m1/p_sudoku_attn_T128_attr_R.json \
        --hyb results/breadth/babel/m1/p_sudoku_attn_T128_div_ew_R.json --cell w3c --sched c120.P8
"""
import argparse
import json

import numpy as np


def arr(s):
    return (np.frombuffer(s.encode(), np.uint8) - 48).astype(int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr", required=True)
    ap.add_argument("--hyb", required=True)
    ap.add_argument("--cell", default="w3c")
    ap.add_argument("--polish", default="")
    ap.add_argument("--sched", required=True)
    a = ap.parse_args()
    A = json.load(open(a.attr))
    H = json.load(open(a.hyb))
    assert A["n"] == H["n"] and A.get("offset", 0) == H.get("offset", 0) and A["T"] == H["T"], "different puzzles or T"
    T = A["T"]
    nd = np.array(A["fp32"]["attractors"]["n_distinct"])
    fr = np.array(A["fp32"]["attractors"]["frac_random_init_correct"])
    f = arr(H["fp32"]["at"][str(T)])
    rec = [r for r in H["records"] if r["cell"] == a.cell and (not a.polish or r["polish"] == a.polish)][0]
    sc = rec["schedules"][a.sched]
    h = arr(sc["correct"])
    if "agree" in sc:
        sw = np.array(sc["agree"]) < 1.0
        swdef = "answer differs from fp32@T (per-cell agreement < 1)"
    else:
        sw = h != f
        swdef = "correctness differs from fp32@T"
    gain, loss = (h == 1) & (f == 0), (h == 0) & (f == 1)
    K = A["fp32"]["attractors"]["K"]
    print(f"# {H['task']} T={T} n={len(h)}: {a.cell} {a.sched} ({rec['polish']}) vs fp32@T; attractors from default + {K} random carries\n")
    print(f"hybrid {100*h.mean():.1f}, fp32@T {100*f.mean():.1f}; puzzles with > 1 attractor: {100*np.mean(nd > 1):.1f}%; "
          f"switch ({swdef}): {100*sw.mean():.1f}%\n")
    print("| distinct fp32 answers | puzzles | fp32@T acc | hybrid acc | gain (hyb right, fp32 wrong) | loss | net | switch rate |")
    print("|---|---|---|---|---|---|---|---|")
    for lo, hi in [(1, 1), (2, 2), (3, 4), (5, 99)]:
        m = (nd >= lo) & (nd <= hi)
        if not m.any():
            continue
        print(f"| {lo}{'' if lo == hi else '-' + str(hi if hi < 99 else '')} | {m.sum()} | {100*f[m].mean():.1f} | {100*h[m].mean():.1f} | "
              f"{100*gain[m].mean():.1f} | {100*loss[m].mean():.1f} | {100*(gain[m].mean()-loss[m].mean()):+.1f} | {100*sw[m].mean():.1f} |")
    # which predicts the per-puzzle gain better: attractor count or switch?
    def auroc(score, lab):
        score, lab = np.asarray(score, float), np.asarray(lab, bool)
        r = np.argsort(np.argsort(score)) + 1.0
        for v in np.unique(score):
            m = score == v
            r[m] = r[m].mean()
        return (r[lab].sum() - lab.sum() * (lab.sum() + 1) / 2) / (lab.sum() * (~lab).sum())
    print(f"\nAUROC for 'hybrid gains this puzzle' among puzzles fp32@T gets wrong (n={int((f == 0).sum())}): "
          f"attractor count {auroc(nd[f == 0], gain[f == 0]):.2f}, fraction of random carries that solve it "
          f"{auroc(fr[f == 0], gain[f == 0]):.2f}")
    print(f"gain among switched puzzles {100*gain[sw].mean():.1f}% vs unswitched {100*gain[~sw].mean():.1f}%; "
          f"loss among switched {100*loss[sw].mean():.1f}% vs unswitched {100*loss[~sw].mean():.1f}%")


if __name__ == "__main__":
    main()
