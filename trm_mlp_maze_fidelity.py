"""Reverse ablation: TRM with MLP token mixing (mlp_t=True, no positional
encoding) TRAINED BY US on Maze-30x30-hard (H100, batch 384, 130,200 steps),
evaluated like hrm_maze_fidelity.py: FP32 / INT8 / naive INT4 / calibrated
INT4, puzzle-exact + cell accuracy + carry-trajectory fidelity against the
FP32 final carry on the same mazes. If MLP-mixing collapses under naive INT4
on Maze as it does on Sudoku, the fragility is architectural on two tasks.

    ~/venvs/kvcache/bin/python trm_mlp_maze_fidelity.py --ckpt maze_mlp/step_130200 \
        --config maze_mlp/all_config.yaml [--n 1000] [--batch 50]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TinyRecursiveModels"))
sys.path.insert(0, HERE)
from huggingface_hub import hf_hub_download  # noqa: E402
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1 as TRM  # noqa: E402
from models.recursive_reasoning.trm import (TinyRecursiveReasoningModel_ACTV1Carry as C,  # noqa: E402
                                            TinyRecursiveReasoningModel_ACTV1InnerCarry as IC)
from carry_diagnostics import fake_quant_  # noqa: E402
from sudoku_calibrated import calib_quant_  # noqa: E402

np.random.seed(0)
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAZE_CHARSET = "# SGo"


def load_qa(n):
    p = hf_hub_download("sapientinc/maze-30x30-hard-1k", "test.csv", repo_type="dataset")
    q, a = [], []
    with open(p, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            qq, aa = row[1], row[2]
            q.append(np.array([MAZE_CHARSET.index(c) + 1 for c in qq], dtype=np.int64))
            a.append(np.array([MAZE_CHARSET.index(c) + 1 for c in aa], dtype=np.int64))
            if len(q) >= n:
                break
    return np.stack(q), np.stack(a)


def build(ckpt, config, batch):
    arch = yaml.safe_load(open(config))["arch"]
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    sd = {k.replace("_orig_mod.model.", "").replace("model.", ""): v for k, v in sd.items()}
    npid = sd["inner.puzzle_emb.weights"].shape[0]
    vocab = sd["inner.embed_tokens.embedding_weight"].shape[0]
    cfg = dict(batch_size=batch, seq_len=900, vocab_size=vocab, num_puzzle_identifiers=npid,
               H_cycles=arch["H_cycles"], L_cycles=arch["L_cycles"],
               H_layers=arch["H_layers"], L_layers=arch["L_layers"],
               hidden_size=arch["hidden_size"], expansion=arch["expansion"],
               num_heads=arch["num_heads"], pos_encodings=arch["pos_encodings"],
               halt_max_steps=arch["halt_max_steps"], halt_exploration_prob=arch["halt_exploration_prob"],
               forward_dtype="float32", mlp_t=arch["mlp_t"], puzzle_emb_ndim=arch["puzzle_emb_ndim"],
               puzzle_emb_len=arch.get("puzzle_emb_len", 16), no_ACT_continue=arch.get("no_ACT_continue", True))
    m = TRM(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss[:3], unexp[:3])
    return m, cfg


def _carry(m, batch):
    c = m.initial_carry(batch)
    ic = c.inner_carry
    return C(inner_carry=IC(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)), steps=c.steps.to(DEV),
             halted=c.halted.to(DEV), current_data={k: v.to(DEV) for k, v in c.current_data.items()})


@torch.no_grad()
def run(m, inp, lab, batch, max_steps):
    m = m.to(DEV)
    n = inp.shape[0]
    pex = cell_c = cell_n = 0
    finals = []
    for s in range(0, n, batch):
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        B = x.shape[0]
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
        carry = _carry(m, bt)
        out = None
        for _ in range(max_steps):
            carry, out = m(carry, bt)
            if carry.halted.all():
                break
        preds = out["logits"].argmax(-1)
        mask = y != 0
        corr = (preds == y) & mask
        pex += (corr.sum(-1) == mask.sum(-1)).sum().item()
        cell_c += corr.sum().item()
        cell_n += mask.sum().item()
        finals.append(carry.inner_carry.z_H.detach().float().cpu())
    m.cpu()
    torch.cuda.empty_cache()
    return pex / n, cell_c / cell_n, torch.cat(finals)


def fidelity(z, ref):
    return float(torch.nn.functional.cosine_similarity(z.flatten(1), ref.flatten(1), dim=-1).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out", default=os.path.expanduser("~/trm_mlp_maze_fidelity_results.json"))
    a = ap.parse_args()
    inp, lab = load_qa(a.n)
    m, cfg = build(a.ckpt, a.config, a.batch)
    print(f"TRM-MLP (mlp_t, pos none) on Maze-Hard, {len(inp)} test mazes, "
          f"H{cfg['H_cycles']}L{cfg['L_cycles']} L_layers {cfg['L_layers']}, ckpt {a.ckpt}", flush=True)
    rows = []
    _, _, ref = run(m, inp, lab, a.batch, cfg["halt_max_steps"])
    for name, q in [("FP32", None), ("INT8", lambda mm: fake_quant_(mm, 8, True)),
                    ("INT4-naive", lambda mm: fake_quant_(mm, 4, False)),
                    ("INT4-calib", lambda mm: calib_quant_(mm, 4))]:
        mm, _ = build(a.ckpt, a.config, a.batch)
        if q:
            q(mm)
        pex, cell, z = run(mm, inp, lab, a.batch, cfg["halt_max_steps"])
        fid = fidelity(z, ref)
        rows.append({"variant": name, "pexact": pex, "cell": cell, "fidelity": fid})
        print(f"{name:12s} pexact {pex*100:6.2f}%  cell {cell*100:6.2f}%  fidelity {fid:.4f}", flush=True)
    json.dump({"task": "maze-30x30-hard", "model": "TRM-MLP-mixing, trained by us (H100, batch 384, 130200 steps)",
               "ckpt": a.ckpt, "n": int(len(inp)), "rows": rows}, open(a.out, "w"), indent=2)
    print("wrote", a.out)
