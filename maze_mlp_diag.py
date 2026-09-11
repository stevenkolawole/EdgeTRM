"""Diagnostic for the trained TRM-MLP Maze checkpoint: run the first 50 test
mazes from the dataset's own token arrays and print what the model predicts."""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from trm_mlp_maze_fidelity import build, run, DEV  # noqa: E402

ck, cfgp = sys.argv[1], sys.argv[2]
d = os.path.join(HERE, "maze_mlp", "test")
inp = np.load(os.path.join(d, "all__inputs.npy")).astype(np.int64)[:50]
lab = np.load(os.path.join(d, "all__labels.npy")).astype(np.int64)[:50]
m, cfg = build(ck, cfgp, 50)
print("label histogram", np.bincount(lab.ravel(), minlength=6))
print("input histogram", np.bincount(inp.ravel(), minlength=6))
m = m.to(DEV)
with torch.no_grad():
    x = torch.from_numpy(inp).to(DEV); y = torch.from_numpy(lab).to(DEV)
    bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(50, dtype=torch.int32, device=DEV)}
    from trm_mlp_maze_fidelity import _carry
    carry = _carry(m, bt)
    for step in range(cfg["halt_max_steps"]):
        carry, out = m(carry, bt)
        preds = out["logits"].argmax(-1)
        mask = y != 0
        corr = (preds == y) & mask
        pex = (corr.sum(-1) == mask.sum(-1)).float().mean().item()
        cell = (corr.sum() / mask.sum()).item()
        print(f"step {step+1:2d}: pexact {pex*100:5.1f}%  cell {cell*100:5.1f}%  pred histogram {np.bincount(preds.cpu().numpy().ravel(), minlength=6)}  "
              f"q_halt>0: {(out['q_halt_logits']>0).float().mean().item():.2f}", flush=True)
        if carry.halted.all():
            break
