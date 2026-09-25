#!/usr/bin/env python3
"""M2 figure: accuracy vs K per weight spec (best-Q, vote, pass@K, random), one panel per model,
from results/m2/m2_summary.json (m2_analyze.py json).  Noise draws are averaged (range as band).

    python m2_fig.py --root results/m2 --out figures/m2_bestofk.pdf
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("sudoku_mlp", "TRM Sudoku-MLP"), ("sudoku_attn", "TRM Sudoku-attn"), ("hrm_sudoku", "HRM Sudoku (H=2)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/m2")
    ap.add_argument("--out", default="figures/m2_bestofk.pdf")
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--specs", default="")
    a = ap.parse_args()
    S = json.load(open(Path(a.root) / "m2_summary.json"))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (task, title) in zip(axes, MODELS):
        g = defaultdict(list)
        for e in S:
            if e["task"] == task and e["D"] == 16 and e["K"] > 1 and abs(e["sigma"] - a.sigma) < 1e-9 and "sel" in e:
                g[e["spec"].split("@s")[0]].append(e)
        specs = [s for s in (a.specs.split(",") if a.specs else sorted(g)) if s in g]
        cmap = plt.get_cmap("tab10")
        for i, sp in enumerate(specs):
            es = g[sp]
            Ks = sorted(int(k) for k in es[0]["sel"])
            mq = np.array([[e["sel"][str(k)]["maxq"]["exact"] for k in Ks] for e in es]) * 100
            ps = np.array([[e["sel"][str(k)]["pass"]["exact"] for k in Ks] for e in es]) * 100
            ax.plot(Ks, mq.mean(0), "-o", ms=3, color=cmap(i % 10), label=sp)
            if len(es) > 1:
                ax.fill_between(Ks, mq.min(0), mq.max(0), color=cmap(i % 10), alpha=0.15)
            ax.plot(Ks, ps.mean(0), ":", color=cmap(i % 10), lw=1)
        ax.set_xscale("log", base=2)
        ax.set_title(title)
        ax.set_xlabel("K rollouts (sigma=%.1f, D=16)" % a.sigma)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("exact accuracy (%)  solid: best-Q@K, dotted: pass@K")
    axes[-1].legend(fontsize=7, loc="lower right", ncol=2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
