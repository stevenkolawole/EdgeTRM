#!/usr/bin/env python3
"""Exhibit 6 of the rewrite: fidelity predicts accuracy retained, and its unit is the cell.
(a) Every TRM/HRM cell at supervision step 16 (quantizers, static INT8 activations, weight noise):
    accuracy retained relative to fp32 against fidelity phi_H. (b) Deep-thinking convnets at their
    largest iteration count: exact-match retained (open) and per-cell accuracy retained (filled, from
    the dt_cell reruns) against whole-map fidelity. The 0.85 line is the band the old draft used.
Prints, for each panel, the share of cells the 0.85 line classifies correctly ("retained >= 90%").

  python plots/fidelity_scatter.py --out figures/fidelity_scatter.pdf
"""
import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as fs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "results" / "breadth"
spec = lambda r: r["quant"]["spec"] if isinstance(r.get("quant"), dict) else r.get("quant")
TRM = {"sudoku_mlp": "o", "sudoku_attn": "s", "maze_attn": "^", "hrm_sudoku": "D", "hrm_maze": "v"}
DT = {"mazes": "o", "chess": "s", "prefix_sums": "^"}


def trm_cells():
    pts = []
    for task, mk in TRM.items():
        recs = []
        for suffix in ("_all", "_a8", "_noise"):
            f = B / f"{task}{suffix}.json"
            if f.exists():
                recs += json.load(open(f))["records"]
        recs = [r for r in recs if r["nsup"] == 16 and (task.startswith("hrm") or r.get("H_cycles") == 3)]
        fp = [r for r in recs if spec(r) == "fp32"]
        if not fp:
            continue
        base = fp[0]["pexact"]
        for r in recs:
            if spec(r) == "fp32" or base == 0:
                continue
            pts.append((task, mk, r["fidelity_H"], 100 * r["pexact"] / base))
    return pts


def dt_cells():
    pts = []
    for task, mk in DT.items():
        recs = []
        for suffix in ("", "_a8", "_noise"):
            f = B / f"dt_{task}{suffix}.json"
            if f.exists():
                recs += json.load(open(f))["records"]
        # largest iteration count per size, fp32 as base
        by = {}
        for r in recs:
            by.setdefault((r.get("size"), r.get("iters")), []).append(r)
        sizes = {}
        for (size, it), rr in by.items():
            if size not in sizes or (it or 0) > (sizes[size] or 0):
                sizes[size] = it
        for (size, it), rr in by.items():
            if it != sizes[size]:
                continue
            fp = [r for r in rr if spec(r) == "fp32"]
            if not fp or fp[0]["acc"] == 0:
                continue
            for r in rr:
                if spec(r) == "fp32" or r.get("fidelity") is None:
                    continue
                pts.append((task, mk, r["fidelity"], 100 * r["acc"] / fp[0]["acc"]))
    return pts


def dt_cell_level():
    pts = []
    for f in sorted(glob.glob(str(B / "p4d" / "dt_cell_mazes_*_s*.json"))):
        recs = [r for r in json.load(open(f))["records"] if r["iters"] == 300]
        fp = [r for r in recs if spec(r) == "fp32"][0]
        for r in recs:
            if spec(r) == "fp32":
                continue
            pts.append(("mazes (per cell)", "o", r["fidelity"], 100 * r["cell_acc"] / fp["cell_acc"],
                        100 * r["acc"] / fp["acc"]))
    return pts


def agreement(pts, thr=0.85, keep=90):
    ok = sum((x >= thr) == (y >= keep) for _, _, x, y in pts)
    return ok, len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "figures" / "fidelity_scatter.pdf")
    a = ap.parse_args()
    fs.use()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fs.TEXT_W, 2.2))
    trm = trm_cells()
    for task, mk in TRM.items():
        sel = [(x, y) for t, m, x, y in trm if t == task]
        if sel:
            ax1.scatter(*zip(*sel), marker=mk, s=11, color=fs.INK, alpha=0.55, lw=0, label=task.replace("_", " "))
    ok, n = agreement(trm)
    print(f"TRM/HRM: {n} cells; the 0.85 line agrees with 'retained >= 90%' on {ok} ({100*ok/n:.0f}%)")
    dt = dt_cells()
    for task, mk in DT.items():
        sel = [(x, y) for t, m, x, y in dt if t == task]
        if sel:
            ax2.scatter(*zip(*sel), marker=mk, s=11, facecolors="none", edgecolors=fs.GREY, lw=0.7, label=f"{task.replace('_', ' ')}, exact")
    ok, n = agreement(dt)
    print(f"DT exact: {n} cells; agreement {ok} ({100*ok/n:.0f}%)")
    cell = dt_cell_level()
    ax2.scatter([x for *_, x, y, _ in cell], [y for *_, x, y, _ in cell], marker="o", s=13, color=fs.BLUE, lw=0,
                label="mazes 25/33, per cell", zorder=4)
    ok, n = agreement([(t, m, x, y) for t, m, x, y, _ in cell])
    print(f"DT per-cell (mazes 25/33, 300 it): {n} cells; agreement {ok} ({100*ok/n:.0f}%)")
    okx, nx = agreement([(t, m, x, e) for t, m, x, _, e in cell])
    print(f"  the same cells by exact match: agreement {okx} ({100*okx/nx:.0f}%)")
    for ax, title in ((ax1, "TRM and HRM, step 16"), (ax2, "deep-thinking convnets")):
        ax.axvline(0.85, color=fs.RED, lw=0.8, ls="--")
        ax.axhline(90, color=fs.GREY_LIGHT, lw=0.8)
        ax.set_xlim(0, 1.02)
        ax.set_ylim(-3, 112)
        ax.set_xlabel(r"fidelity to fp32 ($\varphi$)")
        ax.set_title(title, fontsize=8)
        ax.legend(loc="upper left", fontsize=6, handlelength=1.0, borderpad=0.3, labelspacing=0.25, markerscale=1.2)
        ax.grid(True, axis="y", color=fs.GREY_LIGHT, lw=0.5)
        ax.set_axisbelow(True)
    ax1.set_ylabel("accuracy retained (% of fp32)")
    ax1.annotate("0.85", (0.85, 2), xytext=(3, 0), textcoords="offset points", fontsize=6.5, color=fs.RED)
    fs.panel_label(ax1, "a")
    fs.panel_label(ax2, "b")
    fig.subplots_adjust(wspace=0.25)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
