#!/usr/bin/env python3
"""Figure 1 of the EdgeTRM rewrite (EDGETRM_OUTLINE.md): the two panels that carry the claim.
(a) Depth does not leave the basin: TRM Sudoku-MLP exact accuracy against supervision steps for
    fp32, an in-basin quantizer (per-channel INT4) and an out-of-basin one (per-tensor INT4).
    Data: results/breadth/sudoku_mlp_all.json (H=3, n=1..16) + sudoku_mlp_deep.json (32, 64).
(b) Compression caps test-time scaling: deep-thinking maze 33x33 exact accuracy against
    iterations for fp32, w4c, w4t, w3a. Data: results/breadth/p4d/dt_extrap_mazes_33.json.

  python plots/fig1_summary.py --out figures/fig1_summary.pdf
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "results" / "breadth"
spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]


def sudoku_curves():
    recs = [r for f in ("sudoku_mlp_all.json", "sudoku_mlp_deep.json") for r in json.load(open(B / f))["records"]
            if r.get("H_cycles") == 3]
    out = {}
    for q in ("fp32", "w4c", "w4t"):
        pts = sorted((r["nsup"], 100 * r["pexact"]) for r in recs if spec(r) == q)
        out[q] = pts
    return out


def maze_curves():
    recs = json.load(open(B / "p4d" / "dt_extrap_mazes_33.json"))["records"]
    out = {}
    for q in ("fp32", "w4c", "w4t", "w3a"):
        out[q] = sorted((r["iters"], 100 * r["acc"]) for r in recs if spec(r) == q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "fig1_summary.pdf")
    a = ap.parse_args()
    fs.use()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.1))
    style = {"fp32": (fs.INK, "-", "o", "fp32"), "w4c": (fs.BLUE, "-", "s", "INT4 per-channel"),
             "w4t": (fs.RED, "-", "^", "INT4 per-tensor"), "w3a": (fs.AMBER, "--", "D", "INT3 asymmetric")}
    where = {"w4c": "inside the basin", "w4t": "outside"}
    s = sudoku_curves()
    for q, pts in s.items():
        c, ls, mk, lab = style[q]
        xs, ys = zip(*pts)
        ax1.plot(xs, ys, color=c, ls=ls, marker=mk, ms=3, lw=1.3)
        text = lab + (f"\n{where[q]}" if q in where else "")
        ax1.annotate(text, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=6.5, color=c)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([1, 4, 16, 64])
    ax1.set_xticklabels(["1", "4", "16", "64"])
    ax1.set_xlabel("supervision steps (depth)")
    ax1.set_ylabel("Sudoku exact accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.set_xlim(0.8, 64 * 9)
    fs.panel_label(ax1, "a")
    m = maze_curves()
    for q, pts in m.items():
        c, ls, mk, lab = style[q]
        xs, ys = zip(*pts)
        ax2.plot(xs, ys, color=c, ls=ls, marker=mk, ms=3, lw=1.3, label=lab)
    ax2.legend(loc="lower right", fontsize=6.5, handlelength=1.6, borderpad=0.3, labelspacing=0.3)
    ax2.set_xscale("log", base=10)
    ax2.set_xticks([10, 30, 100, 300])
    ax2.set_xticklabels(["10", "30", "100", "300"])
    ax2.set_xlabel("iterations at test time")
    ax2.set_ylabel("maze 33×33 solved (%)")
    ax2.set_ylim(0, 105)
    ax2.set_xlim(8, 600)
    ax2.axvline(30, color=fs.GREY_LIGHT, lw=0.8)
    ax2.annotate("trained\nto 30", (30, 5), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=fs.GREY)
    fs.panel_label(ax2, "b")
    for ax in (ax1, ax2):
        ax.grid(True, axis="y", color=fs.GREY_LIGHT, lw=0.5)
        ax.set_axisbelow(True)
    fig.subplots_adjust(wspace=0.35)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
