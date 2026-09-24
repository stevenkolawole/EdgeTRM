#!/usr/bin/env python3
"""Angle A mechanism figure (X2): what the recursion map does at the accuracy cliff.

  (a) exact accuracy against weight-noise sigma for the Sudoku TRM with the MLP mixer, the Sudoku TRM
      with the attention mixer (same task, same recipe) and HRM: three draws per sigma (points),
      median (line), sigma_1/2 (dashed);
  (b) the same x-axis: a stability metric of the perturbed map on its own trajectory, over the puzzles
      fp32 solves (default: median spectral radius of one recursion step's Jacobian at s_T, against 1;
      fp32 at the left edge as a diamond);
  (c) every perturbed cell of every model (weight noise, quantizers, single weight groups): fraction of
      fp32-solved puzzles still solved against the one-step map perturbation delta_T = ||F~(s_T) - F(s_T)||
      / ||s_T|| at fp32's end state.

Data: the figure JSON written by `contraction.py --analyze ... --figdata <json>`.

  python plots/contraction_fig.py --data results/breadth/x2_contraction_figdata.json \
      --out figures/contraction_mechanism.pdf [--metric rho_own_sol | kick_sol | resT_sol | sigmaJ_sol]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODELS = [("sudoku_mlp", "TRM, MLP mixer", fs.BLUE), ("sudoku_attn", "TRM, attention", fs.RED),
          ("hrm_sudoku", "HRM", fs.AMBER), ("maze_attn", "TRM Maze", fs.GREY), ("dt_maze33_T300", "DT maze 33", fs.INK),
          ("dt_maze13_T100", "DT maze 13", fs.GREY_LIGHT)]
LABEL = {"rho_own_sol": r"spectral radius $\rho(J(s_T))$",
         "kick_sol": "growth of a random state\nkick over 16 steps",
         "resT_sol": "residual at step 16",
         "sigmaJ_sol": r"$\|J(s_T)\|_2$",
         "frac_settled_sol": "fraction settled\n(residual < 0.1)",
         "Lambda_sol": r"Lyapunov number $\Lambda$"}
LOG = {"kick_sol", "resT_sol", "sigmaJ_sol"}
ONE = {"rho_own_sol", "kick_sol", "Lambda_sol"}


def _med(vs):
    vs = [v for v in vs if v is not None]
    return float(np.median(vs)) if vs else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "results" / "breadth" / "x2_contraction_figdata.json")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "contraction_mechanism.pdf")
    ap.add_argument("--metric", default="rho_own_sol")
    ap.add_argument("--curves", default="sudoku_mlp,sudoku_attn,hrm_sudoku,maze_attn", help="models in panels (a), (b)")
    a = ap.parse_args()
    D = json.load(open(a.data))
    fs.use()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(fs.TEXT_W, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], wspace=0.55)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    curves = a.curves.split(",")
    for key, name, col in MODELS:
        if key not in D or key not in curves:
            continue
        d = D[key]
        sig = np.asarray(d["sigma"], float)
        for x, accs in zip(sig, d["acc_draws"]):
            axA.scatter([x] * len(accs), [100 * v for v in accs], s=6, color=col, alpha=0.5, lw=0, zorder=3)
        axA.plot(sig, [100 * v for v in d["acc_med"]], color=col, lw=1.2, label=name)
        sh = d["sigma_half"]["value"]
        for ax in (axA, axB):
            if sh:
                ax.axvline(sh, color=col, lw=0.6, ls="--")
        vals = d["metrics"][a.metric]
        for x, vs in zip(sig, vals):
            vs = [v for v in vs if v is not None]
            axB.scatter([x] * len(vs), vs, s=6, color=col, alpha=0.5, lw=0, zorder=3)
        axB.plot(sig, [_med(vs) for vs in vals], color=col, lw=1.2)
        f0 = d["fp32"].get(a.metric)
        if f0 is not None:
            x0 = 0.013
            axB.scatter([x0], [f0], marker="D", s=11, color=col, zorder=4)
    for ax in (axA, axB):
        ax.set_xscale("log")
        ax.set_xlabel("weight noise $\\sigma$\n")
    axA.set_ylim(0, 100)
    axA.set_ylabel("exact accuracy (%)")
    h, l = axA.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(l), fontsize=6.5, bbox_to_anchor=(0.5, 1.07), frameon=False)
    axB.set_ylabel(LABEL.get(a.metric, a.metric) + "\n(puzzles fp32 solves)")
    if a.metric in ONE:
        axB.axhline(1.0, color=fs.INK, lw=0.6)
    if a.metric in LOG:
        axB.set_yscale("log")
    # (c) retained fraction against the one-step perturbation, every cell of every model
    MK = {"noise": "o", "quant": "s", "group": "^"}
    for key, name, col in MODELS:
        if key not in D:
            continue
        for c in D[key].get("cells", []):
            if c["deltaT"] is None or c["retained"] is None:
                continue
            axC.scatter([c["deltaT"]], [100 * c["retained"]], s=8, marker=MK[c["kind"]], color=col, alpha=0.7, lw=0,
                        label=None)
    for kind, mk in MK.items():
        axC.scatter([], [], marker=mk, s=8, color=fs.GREY, label={"noise": "weight noise", "quant": "quantizer",
                                                                   "group": "one weight group"}[kind])
    axC.set_xscale("log")
    axC.set_ylim(-3, 103)
    axC.set_xlabel(r"one-step $\delta_T$" + "\n", labelpad=1)
    axC.set_xticks([0.01, 0.1, 1])
    axC.set_xticklabels(["0.01", "0.1", "1"])
    axC.set_ylabel("fp32-solved puzzles\nstill solved (%)")
    axC.legend(loc="lower left", fontsize=5.5, handletextpad=0.2)
    for ax, l in zip((axA, axB, axC), "abc"):
        fs.panel_label(ax, l)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.03)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
