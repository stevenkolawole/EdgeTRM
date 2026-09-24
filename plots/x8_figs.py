#!/usr/bin/env python3
"""X8 figures from results/breadth/x8_summary.json (written by `analyze_x8.py json`).

  python plots/x8_figs.py            # -> figures/x8_defer.png, x8_detectors.png, x8_ptrm.png

defer: per model, the halting head's own rule sigmoid(q) > 0.5 (set in training, never
recalibrated) at every quantizer / noise draw: selective risk and the share of correct answers it
keeps, against accuracy retained (the basin axis).  detectors: AUROC per detector by basin class.
ptrm: single-rollout accuracy vs best-Q@K per sigma, PTRM latent noise vs our activation noise.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
S = json.load(open(sys.argv[1] if len(sys.argv) > 1 else HERE / "results" / "breadth" / "x8_summary.json"))
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
NAMES = {"sudoku_mlp": "TRM Sudoku-MLP", "sudoku_attn": "TRM Sudoku-attn", "hrm_sudoku": "HRM Sudoku",
         "maze_attn": "TRM Maze-attn", "hrm_maze": "HRM Maze"}
CLS = {"fp32": ("k", "o"), "in": ("#1b7837", "o"), "edge": ("#e08214", "s"), "out": ("#b2182b", "x")}


def defer_fig():
    tasks = [t for t in NAMES if t in S["basin"]]
    fig, axes = plt.subplots(2, len(tasks), figsize=(3.2 * len(tasks), 5.6), squeeze=False, sharey="row")
    for j, t in enumerate(tasks):
        d = "exact" if "sudoku" in t else "vshort"
        rows = S["basin"][t][d]
        for r in rows:
            if r["rule"] is None or r["retained"] is None:
                continue
            c, mk = CLS.get(r["class"], ("grey", "."))
            x = min(r["retained"], 1.1)
            if r["rule"]["risk"] is not None:
                axes[0, j].scatter(x, 100 * r["rule"]["risk"], c=c, marker=mk, s=22)
            if r["rule"]["cov_of_correct"] is not None:
                axes[1, j].scatter(x, 100 * r["rule"]["cov_of_correct"], c=c, marker=mk, s=22)
        axes[0, j].set_title(f"{NAMES[t]} ({d})", fontsize=9)
        axes[1, j].set_xlabel("accuracy retained vs fp32")
        for i in (0, 1):
            axes[i, j].axvline(0.9, ls=":", c="grey", lw=0.8)
            axes[i, j].invert_xaxis()
    axes[0, 0].set_ylabel("selective risk of accepted answers (%)")
    axes[1, 0].set_ylabel("correct answers kept (%)")
    fig.suptitle("Halting head's own rule, set in training, applied unchanged at every quantizer and noise draw", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "x8_defer.png", dpi=160)


def detectors_fig():
    tasks = [t for t in NAMES if t in S["auroc"]]
    dets = ["head", "combo", "checker", "msp_min", "resid_H", "pi_dis", "fid_H"]
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 3.2), squeeze=False, sharey=True)
    for j, t in enumerate(tasks):
        d = "exact" if "sudoku" in t else "vshort"
        rows = S["auroc"][t][d]
        for ci, cls in enumerate(["fp32", "in", "edge", "out"]):
            vals = []
            for k in dets:
                v = [r["auroc"].get(k, {}).get("auroc") for r in rows if r["class"] == cls]
                v = [x for x in v if x is not None]
                vals.append(np.mean(v) if v else np.nan)
            axes[0, j].bar(np.arange(len(dets)) + 0.2 * ci - 0.3, vals, width=0.2, color=CLS[cls][0], label=cls)
        axes[0, j].set_xticks(range(len(dets)))
        axes[0, j].set_xticklabels(dets, rotation=60, fontsize=7)
        axes[0, j].set_ylim(0.4, 1.02)
        axes[0, j].axhline(0.5, c="grey", lw=0.6)
        axes[0, j].set_title(f"{NAMES[t]} ({d})", fontsize=9)
    axes[0, 0].set_ylabel("AUROC for a wrong answer")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "x8_detectors.png", dpi=160)


def ptrm_fig():
    keys = sorted(k for k in S["ptrm"] if k.endswith("|fp32"))
    tasks = sorted({k.split("|")[0] for k in keys})
    if not tasks:
        return
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.2), squeeze=False)
    for j, t in enumerate(tasks):
        ax = axes[0, j]
        for noise, c in (("ptrm", "#2166ac"), ("anoise", "#b2182b")):
            k = f"{t}|{noise}|fp32"
            if k not in S["ptrm"]:
                continue
            rows = S["ptrm"][k]["rows"]
            sg = [r["sigma"] for r in rows]
            ax.plot(sg, [100 * r["single_mean"] for r in rows], "-o", c=c, ms=3, label=f"{noise}: single rollout")
            ax.plot(sg, [100 * r["bestq"] for r in rows], "--s", c=c, ms=3, label=f"{noise}: best-Q@K")
            ax.plot(sg, [100 * r["pass"] for r in rows], ":", c=c, lw=0.8, label=f"{noise}: pass@K")
        ax.set_xlabel("sigma")
        ax.set_title(NAMES.get(t, t), fontsize=9)
    axes[0, 0].set_ylabel("exact accuracy (%)")
    axes[0, -1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "x8_ptrm.png", dpi=160)


if __name__ == "__main__":
    defer_fig()
    detectors_fig()
    ptrm_fig()
    print("wrote", OUT / "x8_defer.png", OUT / "x8_detectors.png", OUT / "x8_ptrm.png")
