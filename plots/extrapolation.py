#!/usr/bin/env python3
"""Compression caps test-time scaling: deep-thinking maze nets, exact accuracy against test-time
iterations for five maze sizes and four weight formats. Data: results/breadth/p4d/dt_extrap_mazes_<size>.json.

  python plots/extrapolation.py --out figures/extrapolation.pdf
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "results" / "breadth" / "p4d"
spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
SIZES = ["9", "13", "19", "25", "33"]
STYLE = {"fp32": (fs.INK, "-", "o", "fp32"), "w4c": (fs.BLUE, "-", "s", "INT4 per-channel"),
         "w3a": (fs.AMBER, "--", "D", "INT3 asymmetric"), "w4t": (fs.RED, "-", "^", "INT4 per-tensor")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "extrapolation.pdf")
    a = ap.parse_args()
    fs.use()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(SIZES), figsize=(fs.TEXT_W, 1.7), sharey=True)
    for ax, size in zip(axes, SIZES):
        recs = json.load(open(B / f"dt_extrap_mazes_{size}.json"))["records"]
        for q, (c, ls, mk, lab) in STYLE.items():
            pts = sorted((r["iters"], 100 * r["acc"]) for r in recs if spec(r) == q)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=c, ls=ls, marker=mk, ms=2.4, lw=1.1, label=lab)
        ax.axvline(30, color=fs.GREY_LIGHT, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([10, 100])
        ax.set_xticklabels(["10", "100"])
        ax.set_xlim(8, 600)
        ax.set_ylim(0, 105)
        ax.set_title(f"maze {size}×{size}", fontsize=8)
        ax.grid(True, axis="y", color=fs.GREY_LIGHT, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("solved (%)")
    axes[2].set_xlabel("iterations at test time (trained to 30)")
    axes[-1].legend(loc="lower right", fontsize=6, handlelength=1.4, borderpad=0.25, labelspacing=0.25)
    fig.subplots_adjust(wspace=0.12)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
