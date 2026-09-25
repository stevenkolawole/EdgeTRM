#!/usr/bin/env python3
"""X2 extended fit figure: (a) basin width sigma_1/2 against the one-step sensitivity S (sigma0 = 0.05), one point
per model, colour = family, line sigma_1/2 = delta*/S with delta* fitted on TRM+HRM only; (b) every quantizer cell:
retained accuracy against its one-step map perturbation delta_T, dashed line at the same delta*.
Data: results/breadth/x2_sens_fit.json (sens_analyze.py).

  python plots/sens_fit_fig.py --data results/breadth/x2_sens_fit.json --out figures/sens_fit.pdf
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
COL = {"TRM": fs.BLUE, "HRM": fs.AMBER, "DT": fs.INK, "DEQ": fs.RED, "LM": fs.GREY}
NAME = {"TRM": "TRM (Sudoku MLP/attn, variants, Maze)", "HRM": "HRM", "DT": "deep-thinking", "DEQ": "DEQ (MDEQ, DEQ-Transf.)",
        "LM": "looped LM"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "results" / "breadth" / "x2_sens_fit.json")
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "sens_fit.pdf")
    a = ap.parse_args()
    R = json.load(open(a.data))
    fs.use()
    import matplotlib.pyplot as plt
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.3), gridspec_kw={"wspace": 0.35})
    h = R["held_out"].get("TRM+HRM -> DT, DEQ") or {}
    dstar = h.get("delta_star") or R["quant_delta_star_trm_hrm"]
    seen = set()
    for r in R["rows"]:
        if r["sigma_half"] is None:
            continue
        f = r["family"]
        axA.scatter([r["S"]], [r["sigma_half"]], s=14 if not r["low_fp32"] else 10, color=COL[f],
                    marker="o" if not r["low_fp32"] else "x", label=NAME[f] if f not in seen else None, zorder=3)
        seen.add(f)
    xs = np.geomspace(0.15, 4, 50)
    axA.plot(xs, dstar / xs, color=fs.GREY, lw=0.7, ls="--")
    axA.text(0.97, 0.97, rf"$\sigma_{{1/2}}=\delta^*/S$, $\delta^*={dstar:.2f}$" + "\n(fit on TRM+HRM only)",
             transform=axA.transAxes, ha="right", va="top", fontsize=5.5, color=fs.GREY)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"sensitivity $S=\delta_T/\sigma$ at $\sigma$=0.05")
    axA.set_ylabel(r"measured basin width $\sigma_{1/2}$")
    axA.legend(fontsize=5.5, loc="lower left", handletextpad=0.2, borderaxespad=0.2)
    for c in R["quant_cells"]:
        axB.scatter([max(c["delta"], 1e-3)], [100 * min(c["retained"], 1.3)], s=6, color=COL[c["family"]], alpha=0.7, lw=0)
    axB.axvline(R["quant_delta_star_trm_hrm"], color=fs.GREY, lw=0.7, ls="--")
    axB.scatter([], [], s=8, color=COL["LM"], label="looped LM\n(Ouro-1.4B/2.6B, Huginn)")
    axB.legend(fontsize=5.5, loc="center left", handletextpad=0.2, borderaxespad=0.2)
    axB.axhline(50, color=fs.GREY_LIGHT, lw=0.6)
    axB.set_xscale("log")
    axB.set_xlabel(r"quantizer's one-step $\delta_T$")
    axB.set_ylabel("retained accuracy (%)")
    for ax, l in zip((axA, axB), "ab"):
        fs.panel_label(ax, l)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.03)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
