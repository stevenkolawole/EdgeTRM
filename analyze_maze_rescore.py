#!/usr/bin/env python3
"""Maze-30x30-Hard re-scored (X1; reports/edgetrm_exp/X1.md), from maze_rescore.py records
(results/breadth/babel/x1/maze_rescore_{maze_attn,hrm_maze}_*.json, Babel L40S, nsup 16).

    python analyze_maze_rescore.py acc      # accuracy per definition and quantizer, gaps to fp32
    python analyze_maze_rescore.py fails    # what exact-match failures are (valid? shortest?)
    python analyze_maze_rescore.py auroc    # halting-head AUROC for "wrong" under each definition
    python analyze_maze_rescore.py json     # -> results/breadth/x1_maze_rescore.json

Definitions (maze_rescore.py): exact = every cell equals the reference; shortest = a valid simple
S->G path exactly as long as the reference (HRM's own "valid and optimal"); simple = a valid simple
path of any length; conn = S and G connected through the marked cells, non-path cells intact.
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent / "results" / "breadth"
DEFS = ("exact", "shortest", "simple", "conn")
TASKS = (("maze_attn", "TRM Maze-attn"), ("hrm_maze", "HRM Maze"))


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def auroc(score, positive):
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    npos, nneg = positive.sum(), (~positive).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def load(task):
    recs = {}
    for f in sorted(glob.glob(str(B / "babel" / "x1" / f"maze_rescore_{task}_*.json"))):
        d = json.load(open(f))
        K = d["nsup"]
        for r in d["records"]:
            if r["spec"] in recs:
                continue
            flags = {k: np.array([[(int(h, 16) >> j) & 1 for j in range(K)] for h in v], bool)
                     for k, v in r["flags"].items()}
            recs[r["spec"]] = {"flags": flags, "qhalt": np.array(r["qhalt"], float), "K": K,
                               "wrong_cells": np.array(r["wrong_cells"]), "file": Path(f).name}
    return recs


def acc():
    for task, label in TASKS:
        recs = load(task)
        if not recs:
            print(f"\n### {label}: no records yet")
            continue
        K = recs["fp32"]["K"]
        n = recs["fp32"]["flags"]["exact"].shape[0]
        print(f"\n### {label}: accuracy % at nsup {K} under each definition (n={n}; Wilson 95%)")
        print("| quant | exact | shortest (valid+optimal) | simple valid path | connects | Δ shortest vs fp32 | Δ exact vs fp32 |")
        print("|---|---|---|---|---|---|---|")
        f = recs["fp32"]["flags"]
        for s, r in recs.items():
            cells = []
            for d in DEFS:
                k = int(r["flags"][d][:, K - 1].sum())
                p, lo, hi = wilson(k, n)
                cells.append(f"{100*p:.1f} [{100*lo:.0f},{100*hi:.0f}]")
            ds = 100 * (r["flags"]["shortest"][:, K - 1].mean() - f["shortest"][:, K - 1].mean())
            de = 100 * (r["flags"]["exact"][:, K - 1].mean() - f["exact"][:, K - 1].mean())
            print(f"| {s} | " + " | ".join(cells) + f" | {ds:+.1f} | {de:+.1f} |")
        print(f"\n{label}: shortest-path accuracy by ACT step (1/2/4/8/16), fp32 vs a few quantizers")
        for s in [x for x in ("fp32", "w4c", "w4t", "w3g32", "w2a", "w2g32", "w3t", "noise1.0", "anoise1.0") if x in recs]:
            r = recs[s]["flags"]
            print(f"  {s:10s} exact " + "/".join(f"{100*r['exact'][:, k-1].mean():.1f}" for k in (1, 2, 4, 8, 16) if k <= K)
                  + "   shortest " + "/".join(f"{100*r['shortest'][:, k-1].mean():.1f}" for k in (1, 2, 4, 8, 16) if k <= K))


def fails():
    for task, label in TASKS:
        recs = load(task)
        if not recs:
            continue
        K = recs["fp32"]["K"]
        print(f"\n### {label}: what the exact-match failures at nsup {K} are")
        print("| quant | exact failures | of which valid simple path | valid and shortest | connects only | invalid |")
        print("|---|---|---|---|---|---|")
        for s, r in recs.items():
            fl = {d: r["flags"][d][:, K - 1] for d in DEFS}
            ef = ~fl["exact"]
            m = int(ef.sum())
            if m == 0:
                print(f"| {s} | 0 | | | | |")
                continue
            vs = int((ef & fl["simple"]).sum())
            sh = int((ef & fl["shortest"]).sum())
            co = int((ef & fl["conn"] & ~fl["simple"]).sum())
            inv = int((ef & ~fl["conn"]).sum())
            print(f"| {s} | {m} | {vs} ({100*vs/m:.0f}%) | {sh} ({100*sh/m:.0f}%) | {co} | {inv} ({100*inv/m:.0f}%) |")


def auroc_table():
    for task, label in TASKS:
        recs = load(task)
        if not recs:
            continue
        K = recs["fp32"]["K"]
        print(f"\n### {label}: AUROC of -q_halt for 'answer wrong' at nsup {K}, per correctness definition")
        print("| quant | acc exact / shortest | AUROC exact | AUROC shortest | AUROC simple | AUROC conn | "
              "AUROC shortest vs invalid (exact-failures only) |")
        print("|---|---|---|---|---|---|---|")
        for s, r in recs.items():
            q = r["qhalt"][:, K - 1]
            fl = {d: r["flags"][d][:, K - 1] for d in DEFS}
            a = [auroc(-q, ~fl[d]) for d in DEFS]
            ef = ~fl["exact"]
            # among exact-match failures: does the halting head separate valid shortest answers from invalid ones?
            sub = auroc(-q[ef], ~fl["shortest"][ef]) if ef.sum() > 5 else float("nan")
            print(f"| {s} | {100*fl['exact'].mean():.1f} / {100*fl['shortest'].mean():.1f} | "
                  + " | ".join(f"{x:.2f}" for x in a) + f" | {sub:.2f} |")
        print(f"\n{label}: AUROC (shortest) by ACT step 1/2/4/8/16")
        for s in [x for x in ("fp32", "w8c", "w4c", "w4t", "w3g32", "w2a", "w2g32") if x in recs]:
            r = recs[s]
            row = []
            for k in (1, 2, 4, 8, 16):
                if k > K:
                    continue
                row.append(f"{auroc(-r['qhalt'][:, k-1], ~r['flags']['shortest'][:, k-1]):.2f}")
            print(f"  {s:8s} " + " / ".join(row))


def to_json():
    out = {}
    for task, label in TASKS:
        recs = load(task)
        if not recs:
            continue
        K = recs["fp32"]["K"]
        t = {}
        for s, r in recs.items():
            q = r["qhalt"][:, K - 1]
            t[s] = {d: {"acc_by_step": r["flags"][d].mean(0).tolist(),
                        "auroc_final": auroc(-q, ~r["flags"][d][:, K - 1])} for d in DEFS}
        out[task] = {"label": label, "nsup": K, "n": int(recs["fp32"]["flags"]["exact"].shape[0]), "quants": t,
                     "unique_shortest_mazes": 0, "note": "0 of 1000 test mazes have a unique shortest path"}
    p = B / "x1_maze_rescore.json"
    json.dump(out, open(p, "w"))
    print("wrote", p)


if __name__ == "__main__":
    {"acc": acc, "fails": fails, "auroc": auroc_table, "json": to_json}[sys.argv[1]]()
