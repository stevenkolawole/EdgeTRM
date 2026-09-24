#!/usr/bin/env python3
"""Appendix twin of the mechanism figure (X2): every model, every mechanism metric against weight-noise
sigma. Rows: models in the figdata JSON; columns: accuracy, spectral radius at s_T over fp32-solved
puzzles, Lyapunov number, late residual, basin-switch fraction, measured vs linear-response displacement.
Draws are points, medians are lines; the fp32 value sits at the left edge (diamond).

  python plots/contraction_appendix.py --data results/breadth/contraction_figdata.json \
      --out figures/contraction_appendix.pdf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COLS = [("acc", "exact acc."), ("rho_own_sol", r"$\rho(J_T)$ solved"), ("Lambda_sol", r"$\Lambda$ solved"),
        ("resT", "residual"), ("frac_basin_switch", "basin switch"), ("frac_settled_sol", "settled (solved)"), ("sigmaJ_sol", r"$\|J_T\|_2$ solved")]
ONE = {"rho_own_sol", "Lambda_sol"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "results" / "breadth" / "contraction_figdata.json")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "contraction_appendix.pdf")
    a = ap.parse_args()
    D = json.load(open(a.data))
    keys = [k for k in D if not k.startswith("_") and D[k]["sigma"]]
    fs.use()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(keys), len(COLS), figsize=(fs.TEXT_W * 1.25, 1.25 * len(keys) + 0.3), squeeze=False)
    for i, k in enumerate(keys):
        d = D[k]
        sig = np.asarray(d["sigma"], float)
        x0 = sig[0] * 0.5
        for j, (m, title) in enumerate(COLS):
            ax = axes[i, j]
            if m == "acc":
                vals = [[100 * v for v in vs] for vs in d["acc_draws"]]
                f0 = 100 * d["fp32_acc"]
            else:
                vals = d["metrics"].get(m, [[] for _ in sig])
                f0 = d["fp32"].get(m)
            for x, vs in zip(sig, vals):
                vs = [v for v in vs if v is not None]
                ax.scatter([x] * len(vs), vs, s=5, color=fs.BLUE, alpha=0.6, lw=0)
            med = [np.median([v for v in vs if v is not None]) if any(v is not None for v in vs) else np.nan
                   for vs in vals]
            ax.plot(sig, med, color=fs.BLUE, lw=1.0)
            if m == "disp":
                dl = [np.median([v for v in vs if v is not None]) if any(v is not None for v in vs) else np.nan
                      for vs in d["metrics"]["Dlin"]]
                ax.plot(sig, dl, color=fs.RED, lw=0.9, ls="--")
                ax.set_yscale("log")
            if f0 is not None:
                ax.scatter([x0], [f0], marker="D", s=9, color=fs.INK, zorder=4)
            if m in ONE:
                ax.axhline(1, color=fs.GREY, lw=0.6)
            sh = d["sigma_half"]["value"]
            if sh:
                ax.axvline(sh, color=fs.GREY, lw=0.6, ls="--")
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(title, fontsize=7)
            if j == 0:
                ax.set_ylabel(k.replace("_", " "), fontsize=7)
            if i == len(keys) - 1:
                ax.set_xlabel(r"$\sigma$", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.subplots_adjust(wspace=0.45, hspace=0.45)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
