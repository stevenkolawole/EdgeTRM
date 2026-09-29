#!/usr/bin/env python3
"""Write the puzzle file edge_loop.py reads (inputs int32, labels int64) from the Sapient test CSVs, with the
same encoding as breadth_sweep.load_qa (sudoku: '.' -> 0, digit + 1; maze: index in "# SGo" + 1), rows in
file order, so puzzle i here is puzzle i everywhere else in EdgeTRM (X3's host_fp32.npz, E1's records).

    python make_puzzles.py --task sudoku_mlp --n 100 --out assets/sudoku_mlp/puzzles_0_100.npz
    python make_puzzles.py --task maze --csv ~/path/to/maze-30x30-hard-1k/test.csv --n 50 --out ...

The CSV is looked up in the local Hugging Face cache (sapientinc/sudoku-extreme, sapientinc/maze-30x30-hard-1k)
unless --csv is given; nothing is downloaded here. Runs on the host; stdlib + numpy (Python 3.6+).
"""
import argparse
import csv
import glob
import os

import numpy as np

MAZE_CHARSET = "# SGo"
DATASETS = {"sudoku": "datasets--sapientinc--sudoku-extreme", "maze": "datasets--sapientinc--maze-30x30-hard-1k"}
KIND = {"sudoku_mlp": "sudoku", "sudoku_attn": "sudoku", "hrm_sudoku": "sudoku", "maze": "maze"}


def find_csv(kind):
    home = os.path.expanduser("~")
    root = os.environ.get("HF_HOME", os.path.join(home, ".cache", "huggingface"))
    pat = os.path.join(root, "hub", DATASETS[kind], "snapshots", "*", "test.csv")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise SystemExit("test.csv not found under %s; pass --csv" % pat)
    return hits[0]


def load_rows(path, kind, start, n):
    q, a = [], []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)
        for k, row in enumerate(r):
            if k < start:
                continue
            qq, aa = row[1], row[2]
            if kind == "sudoku":
                q.append((np.frombuffer(qq.replace(".", "0").encode(), np.uint8) - ord("0")).astype(np.int64) + 1)
                a.append((np.frombuffer(aa.encode(), np.uint8) - ord("0")).astype(np.int64) + 1)
            else:
                q.append(np.array([MAZE_CHARSET.index(c) + 1 for c in qq], dtype=np.int64))
                a.append(np.array([MAZE_CHARSET.index(c) + 1 for c in aa], dtype=np.int64))
            if len(q) >= n:
                break
    return np.stack(q), np.stack(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(KIND))
    ap.add_argument("--csv", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    kind = KIND[a.task]
    path = a.csv or find_csv(kind)
    q, lab = load_rows(path, kind, a.start, a.n)
    np.savez_compressed(a.out, inputs=q.astype(np.int32), labels=lab.astype(np.int64),
                        index=np.arange(a.start, a.start + len(q)), source=os.path.abspath(path), task=a.task)
    print("%s: %d puzzles (rows %d..%d) of %s -> %s" % (a.task, len(q), a.start, a.start + len(q) - 1, path, a.out))


if __name__ == "__main__":
    main()
