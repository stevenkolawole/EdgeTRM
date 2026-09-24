#!/usr/bin/env python3
"""Angle A mechanism figure (X2): what happens to the recursion map at the accuracy cliff.

  (a) exact accuracy against weight-noise sigma for the Sudoku TRM with the MLP mixer and with the
      attention mixer (same task, same recipe): three draws per sigma (points), median (line),
      sigma_1/2 (dashed);
  (b) the same x-axis: a mechanism metric of the perturbed map on its own trajectory (default: the
      median spectral radius of one recursion step's Jacobian at the final state, over the puzzles
      fp32 solves; band = median of the per-draw quartiles), against 1;
  (c) predicted against measured sigma_1/2 for every model (label-free routes from contraction.py).

Data: the figure JSON written by `contraction.py --analyze ... --figdata <json>`.

  python plots/contraction_fig.py --data results/breadth/contraction_figdata.json \
      --out figures/contraction_mechanism.pdf [--metric rho_own_sol]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAIR = [("sudoku_mlp", "MLP mixer", fs.BLUE, fs.BLUE_LIGHT), ("sudoku_attn", "attention mixer", fs.RED, fs.RED_LIGHT)]
LABEL = {"rho_own_sol": r"spectral radius $\rho(J)$ at $s_T$" + "\n(puzzles fp32 solves)",
         "rho_own": r"spectral radius $\rho(J)$ at $s_T$",
         "Lambda_sol": r"Lyapunov number $\Lambda$" + "\n(puzzles fp32 solves)",
         "frac_basin_switch": "basin switch\n(fraction of fp32-solved)",
         "resT": r"residual $\|s_T-s_{T-1}\|/\|s_T\|$",
         "gamma_nat_sol": r"gain along travel $\gamma$"}
NAMES = {"sudoku_mlp": "TRM Sudoku MLP", "sudoku_attn": "TRM Sudoku attn", "hrm_sudoku": "HRM Sudoku",
         "maze_attn": "TRM Maze attn"}


def _band(vals_per_sigma):
    med = [np.median([v for v in vs if v is not None]) if any(v is not None for v in vs) else np.nan
           for vs in vals_per_sigma]
    return np.asarray(med, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "results" / "breadth" / "contraction_figdata.json")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "contraction_mechanism.pdf")
    ap.add_argument("--metric", default="rho_own_sol")
    ap.add_argument("--pred", default="rho_sol", help="panel (c) route: rho_sol | rho_all | mu_lin | mu_meas")
    a = ap.parse_args()
    D = json.load(open(a.data))
    fs.use()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(fs.TEXT_W, 2.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.8], wspace=0.42)
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))
    q1k, q3k = a.metric + "_q1", a.metric + "_q3"
    for task, name, col, light in PAIR:
        if task not in D:
            continue
        d = D[task]
        sig = np.asarray(d["sigma"], float)
        for x, accs in zip(sig, d["acc_draws"]):
            axA.scatter([x] * len(accs), [100 * v for v in accs], s=7, color=col, alpha=0.55, lw=0, zorder=3)
        axA.plot(sig, [100 * v for v in d["acc_med"]], color=col, lw=1.2, label=name)
        axA.axhline(100 * d["fp32_acc"], color=col, lw=0.6, ls=":")
        sh = d["sigma_half"]["value"]
        for ax in (axA, axB):
            if sh:
                ax.axvline(sh, color=col, lw=0.7, ls="--")
        m = _band(d["metrics"][a.metric])
        axB.plot(sig, m, color=col, lw=1.2, marker="o", ms=2.5)
        if q1k in d["metrics"]:
            lo, hi = _band(d["metrics"][q1k]), _band(d["metrics"][q3k])
            axB.fill_between(sig, lo, hi, color=light, alpha=0.6, lw=0)
        f0 = d["fp32"].get(a.metric)
        if f0 is not None:
            axB.scatter([sig[0] * 0.6], [f0], marker="D", s=10, color=col, zorder=4)
    axA.set_xscale("log")
    axB.set_xscale("log")
    axA.set_ylim(0, 100)
    axA.set_ylabel("exact accuracy (%)")
    axA.set_xlabel(r"weight noise $\sigma$")
    axB.set_xlabel(r"weight noise $\sigma$")
    axB.set_ylabel(LABEL.get(a.metric, a.metric))
    if a.metric.startswith("rho") or a.metric.startswith("Lambda") or a.metric.startswith("gamma"):
        axB.axhline(1.0, color=fs.INK, lw=0.6)
    axA.legend(loc="lower left", fontsize=6.5)
    # (c) predicted vs measured sigma_1/2
    lo_, hi_ = 1e9, 0
    for key, d in D.items():
        sh = d["sigma_half"]["value"]
        route = {"rho_sol": d["sigma_star_rho"].get("sol", {}), "rho_all": d["sigma_star_rho"].get("all", {}),
                 "mu_lin": d["sigma_star_mu"].get("lin", {}), "mu_meas": d["sigma_star_mu"].get("meas", {})}[a.pred]
        pv = route.get("value")
        if sh is None or pv is None:
            continue
        col = fs.BLUE if key == "sudoku_mlp" else fs.RED if key == "sudoku_attn" else fs.GREY
        axC.scatter([sh], [pv], s=14, color=col, zorder=3)
        axC.annotate(NAMES.get(key, key), (sh, pv), xytext=(3, 2), textcoords="offset points", fontsize=5.5)
        lo_, hi_ = min(lo_, sh, pv), max(hi_, sh, pv)
    if hi_ > 0:
        lim = [lo_ / 2, hi_ * 2]
        axC.plot(lim, lim, color=fs.GREY, lw=0.6)
        axC.set_xlim(lim)
        axC.set_ylim(lim)
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(r"measured $\sigma_{1/2}$")
    axC.set_ylabel(r"predicted $\sigma^*$")
    for ax, l in zip((axA, axB, axC), "abc"):
        fs.panel_label(ax, l)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
