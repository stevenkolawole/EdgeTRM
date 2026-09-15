#!/usr/bin/env python3
"""Sanity check for a trained TRM variant: puzzle-exact accuracy on the first N *training* rows of its
own dataset through breadth_sweep's build/run path. If training reported high train accuracy and this
reads ~0, the evaluation path is mismatched; if this reads high and the test read is ~0, the model
memorised the training puzzles.

    python ckpt_train_check.py --task maze_attn --ckpt <step_65100> --data data/maze-30x30-hard-1k --n 1000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_sweep as B  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="TRM dataset dir (has train/all__inputs.npy)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--nsup", type=int, default=16)
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    B.CKPT_OVERRIDE = a.ckpt
    d = Path(a.data) / "train"
    inp = np.load(d / "all__inputs.npy", mmap_mode="r")
    lab = np.load(d / "all__labels.npy", mmap_mode="r")
    pidx = np.load(d / "all__puzzle_indices.npy")
    # one row per puzzle (the first of its augmentations), so the read is over distinct puzzles
    rows = pidx[:-1][: a.n]
    inp = np.asarray(inp[rows]).astype(np.int64)
    lab = np.asarray(lab[rows]).astype(np.int64)
    print(f"{a.task}: {len(rows)} distinct training puzzles from {a.data}", flush=True)
    m, meta = B.build(a.task, 3, a.nsup, a.batch)
    c, cell, zH, zL = B.run(m, inp, lab, a.batch, a.nsup)
    print(f"train-set read: puzzle-exact {c.mean()*100:.2f}  cell {cell*100:.2f}  (H={meta['H_cycles']}, nsup={a.nsup})")


if __name__ == "__main__":
    main()
