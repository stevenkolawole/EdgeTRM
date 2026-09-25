#!/usr/bin/env python3
"""M5 figure: held-out accuracy vs mean steps for the unified controller and the baselines
(results/breadth/m5_controller.json, m5_controller_dt.json) -> figures/m5_controller_frontier.pdf"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
d = json.load(open(HERE / "results/breadth/m5_controller.json"))
dt = json.load(open(HERE / "results/breadth/m5_controller_dt.json"))
panels = [("sudoku_mlp|fp32", "TRM Sudoku-MLP fp32"), ("sudoku_mlp|w3g32", "TRM Sudoku-MLP w3g32 (edge)"),
          ("sudoku_mlp|noise0.15@s2", "TRM Sudoku-MLP noise 0.15 (edge)"), ("sudoku_attn|noise0.7@s0", "TRM Sudoku-attn noise 0.7"),
          ("hrm_sudoku|w4c", "HRM Sudoku w4c"), ("33|w4t", "Deep-thinking maze 33x33 w4t")]
style = {"controller": ("C3", "-", "controller (exit+sample+polish)"), "controller_nopolish": ("C1", "--", "controller, no polish"),
         "fixed": ("k", "-", "fixed depth"), "exit": ("C0", "-", "exit only"), "bestq": ("C2", ":", "best-Q@K"),
         "seq": ("C4", ":", "sequential sampling"), "stab": ("C3", "-", "stability exit")}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
for ax, (key, title) in zip(axes.flat, panels):
    fr = (d[key]["frontier"] if "|" in key and not key[0].isdigit() else dt[key])
    for name, lst in fr.items():
        if not lst or name not in style:
            continue
        pts = sorted({(round(c, 2), a) for b, a, c, _ in lst})
        col, ls, lab = style[name]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, ls=ls, marker="o", ms=3, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("mean steps per instance")
    ax.set_ylabel("exact accuracy (%, held out)")
    ax.grid(alpha=0.3)
handles, labels = [], []
for ax in axes.flat:
    for h, l in zip(*ax.get_legend_handles_labels()):
        if l not in labels:
            handles.append(h); labels.append(l)
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = HERE / "figures" / "m5_controller_frontier.pdf"
fig.savefig(out)
print("wrote", out)
