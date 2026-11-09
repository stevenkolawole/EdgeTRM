#!/usr/bin/env python3
"""Label-free check on the AGX forced-INT8 faults. Per step, each TensorRT engine's halting value q_halt (a function of
the carry) is compared with the laptop's onnxruntime fp32 run of the same graph (results/laptop/, same puzzles, same
steps): median |q_dev - q_host| over puzzles. The gate records' own host reference (host_fp32.npz) is not used: for
Sudoku it was made with the one-cycle graph (stepc_H1), while the engines run three cycles per step (stepc_H3).
usage: agx_int8_check.py <edge/results dir>"""
import json, sys
from pathlib import Path
import numpy as np

R = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
HOST = {"sudoku_mlp": "sudoku_mlp_H3", "sudoku_attn": "sudoku_attn_H3", "maze": "maze_H3", "hrm_sudoku": "hrm_sudoku_H2"}
PASS = ["fp32", "fp16", "int8", "dla0fp16"]


def q_steps(check):
    return np.array([p["q_halt"] for p in check["per_puzzle"]])          # (puzzles, steps)


for task, host in HOST.items():
    h = json.load(open(R / "laptop" / f"laptop_{host}_ort_fp32.json"))["check"]
    qh = q_steps(h)
    dist = {}
    for prec in PASS + ["int8strict"]:
        f = R / "agx" / f"agx_{task}_trt_{prec}_gate.json"
        if f.exists():
            c = json.load(open(f))["check"]
            q = q_steps(c)
            n, s = min(len(q), len(qh)), min(q.shape[1], qh.shape[1])
            dist[prec] = (np.median(np.abs(q[:n, :s] - qh[:n, :s]), 0), c["exact_count_per_step"][-1])
    ok = [dist[p][0] for p in PASS if p in dist]
    bad, ex = dist["int8strict"]
    print(f"{task}: host exact at last step {h['exact_count_per_step'][-1]}, forced-INT8 {ex}; passing engines {len(ok)}")
    print("  step  pass_max  int8strict  ratio")
    for s in (1, 2, 3, len(bad)):
        hi = max(a[s - 1] for a in ok)
        print(f"  {s:>4}  {hi:8.4f}  {bad[s - 1]:10.4f}  {bad[s - 1] / max(hi, 1e-9):6.1f}x")
