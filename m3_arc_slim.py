#!/usr/bin/env python3
"""Slim copy of the ARC-AGI-1 TRM checkpoint for M3: the 419 canonical (un-augmented) test rows and only their
puzzle-embedding rows (ids remapped to 1..K; row 0 kept), so the model fits anywhere. Weights otherwise untouched.
    python m3_arc_slim.py --out ~/edgetrm_ckpt/arc_slim
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import quant_ab as Q

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
sd = Q._load_sd(Q.CKPT_DIR / "step_518071")
cfg = yaml.safe_load((Q.CKPT_DIR / "all_config.yaml").read_text())
ident = json.loads((Q.DATA / "identifiers.json").read_text())
d = Q.DATA / "test"
inputs = np.load(d / "all__inputs.npy", mmap_mode="r"); labels = np.load(d / "all__labels.npy", mmap_mode="r")
pidx = np.load(d / "all__puzzle_indices.npy"); pid = np.load(d / "all__puzzle_identifiers.npy")
per = np.zeros(len(inputs), dtype=np.int64)
for k in range(len(pidx) - 1):
    per[pidx[k]:pidx[k + 1]] = pid[k]
canon = [i for i in range(len(per)) if per[i] != 0 and "|||" not in ident[int(per[i])]]
old = sorted({int(per[i]) for i in canon})
remap = {o: j + 1 for j, o in enumerate(old)}
W = sd["inner.puzzle_emb.weights"]
sd["inner.puzzle_emb.weights"] = torch.cat([W[:1], W[torch.tensor(old)]]).clone()
torch.save(sd, out / "sd.pt")
(out / "all_config.yaml").write_text(yaml.safe_dump(cfg))
np.savez_compressed(out / "canon.npz", inputs=np.asarray(inputs[canon]).astype(np.int64),
                    labels=np.asarray(labels[canon]).astype(np.int64),
                    pids=np.array([remap[int(per[i])] for i in canon], dtype=np.int64), rows=np.array(canon))
print(len(canon), "canonical rows,", len(old), "puzzle ids; puzzle_emb", tuple(sd["inner.puzzle_emb.weights"].shape))
