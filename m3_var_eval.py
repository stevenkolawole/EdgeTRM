#!/usr/bin/env python3
"""Evaluate written QDQ variant graphs (as they would be compiled) with the torch QDQ executor on a split of the
exported M3 data (default: held-out puzzles 1000-1999). Resumable per variant.
    python m3_var_eval.py --task sudoku_mlp --variants m3_a_dT_40_legal,m3_greedy_40_legal --out ~/m3/var_eval.json
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qdq_torch import QDQRunner

ap = argparse.ArgumentParser()
ap.add_argument("--task", default="sudoku_mlp")
ap.add_argument("--variants", required=True)
ap.add_argument("--art", default=str(Path.home() / "aihub_artifacts" / "h1"))
ap.add_argument("--export", default=str(Path.home() / "m3" / "export"))
ap.add_argument("--split", default="evalbig")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
D = dict(np.load(Path(a.export) / f"data_{a.task}.npz"))
inp, lab = D[f"{a.split}_inp"], D[f"{a.split}_lab"]
init = (D["zH0"], D["zL0"], D[f"{a.split}_emb"])
out = Path(a.out)
blob = json.load(open(out)) if out.exists() else {}
for v in a.variants.split(","):
    if v in blob:
        continue
    t0 = time.time()
    p = Path(a.art) / a.task / "infer" / ("qdq_w8a16/model.onnx" if v == "w8a16" else
                                          "qdq_w8a8/model.onnx" if v == "w8a8" else f"qdq_w8a16/variant_{v}.onnx")
    r = QDQRunner(p, None, device=dev).rollout(inp, init, int(D["nsteps"]), batch=100)
    c = ((r["preds"][-1] == lab) | (lab == 0)).all(-1)
    blob[v] = {"exact": float(c.mean()), "n": int(len(c)), "correct": c.astype(int).tolist(), "split": a.split,
               "device": dev, "secs": time.time() - t0}
    json.dump(blob, open(out, "w"))
    print(f"{v}: {100 * c.mean():.1f}% of {len(c)} ({time.time() - t0:.0f}s)", flush=True)
