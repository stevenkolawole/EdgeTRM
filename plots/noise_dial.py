#!/usr/bin/env python3
"""The noise dial with three draws: exact accuracy (top) and fidelity phi_H (bottom) against
weight-noise sigma at n=16 for five TRM/HRM models; every cell is three random draws (points),
the line joins the medians. All draws on p4d at the default H (TRM 3, HRM 2); see draws().

  python plots/noise_dial.py --out figures/noise_dial.pdf
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "results" / "breadth"
spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
TASKS = [("sudoku_mlp", "TRM Sudoku\nMLP mixer"), ("sudoku_attn", "TRM Sudoku\nattention"),
         ("maze_attn", "TRM Maze\nattention"), ("hrm_sudoku", "HRM\nSudoku"), ("hrm_maze", "HRM\nMaze")]
SIG = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]


DEFAULT_H = {"sudoku_mlp": 3, "sudoku_attn": 3, "maze_attn": 3, "hrm_sudoku": 2, "hrm_maze": 2}


def draws(task, nsup=16):
    """Three weight-noise draws per sigma, all on p4d (A100) at the model's default H, n=1000.
    Draw 0 is the original p4d sweep, except HRM Sudoku, whose original sweep ran at H=3 and was rerun
    at H=2 (seed 0); draws 1-2 are the Sep 24 p4d rerun (results/breadth/p4d/noise_rerun). The earlier
    Babel draws (L40S, HRM Sudoku at H=2 against an H=3 draw 0) are no longer used."""
    H = DEFAULT_H[task]
    R = B / "p4d" / "noise_rerun"
    first = R / f"noise_{task}_H{H}_seed0.json" if task == "hrm_sudoku" else B / f"{task}_noise.json"
    files = [first] + [R / f"noise_{task}_H{H}_seed{s}.json" for s in (1, 2)]
    out = {}
    for f in files:
        recs = [r for r in json.load(open(f))["records"] if r["nsup"] == nsup and r.get("H_cycles") == H]
        assert recs, (f, H, nsup)
        for r in recs:
            q = spec(r)
            key = "fp32" if q == "fp32" else float(q.replace("noise", ""))
            out.setdefault(key, []).append((100 * r["pexact"], r["fidelity_H"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "noise_dial.pdf")
    a = ap.parse_args()
    fs.use()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, len(TASKS), figsize=(fs.TEXT_W, 2.8), sharex=True)
    for j, (task, name) in enumerate(TASKS):
        d = draws(task)
        fp = st.median(v[0] for v in d.get("fp32", [(0, 1)]))
        top, bot = axes[0, j], axes[1, j]
        top.axhline(fp, color=fs.GREY, lw=0.7, ls=":")
        xs = [s for s in SIG if s in d]
        med_acc = [st.median(v[0] for v in d[s]) for s in xs]
        med_phi = [st.median(v[1] for v in d[s]) for s in xs]
        for s in xs:
            top.scatter([s] * len(d[s]), [v[0] for v in d[s]], s=7, color=fs.BLUE, alpha=0.55, lw=0, zorder=3)
            bot.scatter([s] * len(d[s]), [v[1] for v in d[s]], s=7, color=fs.BLUE, alpha=0.55, lw=0, zorder=3)
        top.plot(xs, med_acc, color=fs.BLUE, lw=1.2)
        bot.plot(xs, med_phi, color=fs.BLUE, lw=1.2)
        bot.axhline(0.85, color=fs.RED, lw=0.7, ls="--")
        top.set_title(name, fontsize=7.5)
        top.set_ylim(0, 100)
        bot.set_ylim(0, 1.02)
        bot.set_xscale("log")
        bot.set_xticks([0.05, 0.2, 1.0])
        bot.set_xticklabels(["0.05", "0.2", "1"])
        for ax in (top, bot):
            ax.grid(True, axis="y", color=fs.GREY_LIGHT, lw=0.5)
            ax.set_axisbelow(True)
        if j == 0:
            top.set_ylabel("exact accuracy (%)")
            bot.set_ylabel(r"fidelity $\varphi_H$")
        else:
            top.set_yticklabels([])
            bot.set_yticklabels([])
        if j == 2:
            bot.set_xlabel(r"weight noise $\sigma$ (three draws per cell; line = median)")
    top = axes[0, 0]
    top.annotate("fp32", (SIG[0], st.median(v[0] for v in draws("sudoku_mlp")["fp32"])), xytext=(2, 2),
                 textcoords="offset points", fontsize=6.5, color=fs.GREY)
    axes[1, 0].annotate("0.85", (SIG[0], 0.85), xytext=(2, 2), textcoords="offset points", fontsize=6.5, color=fs.RED)
    fig.subplots_adjust(wspace=0.12, hspace=0.18)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
