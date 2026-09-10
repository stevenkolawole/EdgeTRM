"""HRM on ARC-AGI-2 from the official checkpoint (sapientinc/HRM-checkpoint-ARC-2),
same protocol as hrm_maze_fidelity.py: FP32 / INT8 / naive INT4 / calibrated
INT4, single pass (no augmentation voting) on the un-augmented test inputs of
the 120 public-evaluation tasks, with carry-trajectory fidelity against the
FP32 carry on the same inputs.

The dataset must be rebuilt with HRM's own builder (seed 42, 1000 augs), so
that the puzzle identifiers match the checkpoint's puzzle-embedding rows:

    cd hrm_arc/HRM && python dataset/build_arc_dataset.py \
        --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000

    ~/venvs/kvcache/bin/python hrm_arc_fidelity.py [--batch 32]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "TinyRecursiveModels"))
sys.path.insert(0, HERE)
from huggingface_hub import snapshot_download  # noqa: E402
from models.recursive_reasoning.hrm import HierarchicalReasoningModel_ACTV1 as HRM  # noqa: E402
from models.recursive_reasoning.hrm import (HierarchicalReasoningModel_ACTV1Carry as C,  # noqa: E402
                                            HierarchicalReasoningModel_ACTV1InnerCarry as IC)
from carry_diagnostics import fake_quant_, signals  # noqa: E402, F401
from sudoku_calibrated import calib_quant_  # noqa: E402

np.random.seed(0)
torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = os.path.join(HERE, "hrm_arc", "HRM", "data", "arc-2-aug-1000")


def load_test():
    """Un-augmented test examples of the evaluation tasks, with their identifiers."""
    meta = json.load(open(os.path.join(DATA, "test", "dataset.json")))
    inp = np.load(os.path.join(DATA, "test", "all__inputs.npy"))
    lab = np.load(os.path.join(DATA, "test", "all__labels.npy"))
    pid = np.load(os.path.join(DATA, "test", "all__puzzle_identifiers.npy"))
    pidx = np.load(os.path.join(DATA, "test", "all__puzzle_indices.npy"))
    names = json.load(open(glob.glob(os.path.join(DATA, "identifiers.json"))[0]))
    # puzzle_indices[i] is the first example of puzzle i; puzzle i has identifier pid[i]
    keep = []
    for i in range(len(pidx) - 1):
        name = names[int(pid[i])]
        if "_" in name:  # augmented copy
            continue
        keep.extend(range(int(pidx[i]), int(pidx[i + 1])))
    keep = np.array(keep)
    task = np.array([names[int(pid[np.searchsorted(pidx, k, side="right") - 1])] for k in keep])
    return inp[keep], lab[keep], pid[np.searchsorted(pidx, keep, side="right") - 1], task, meta


def load_train_pairs_for(names_wanted):
    """Demonstration (train-split) examples of the given un-augmented eval
    puzzles: these were trained on with the puzzle's own embedding row, so
    they identify that row."""
    inp = np.load(os.path.join(DATA, "train", "all__inputs.npy"))
    lab = np.load(os.path.join(DATA, "train", "all__labels.npy"))
    pid = np.load(os.path.join(DATA, "train", "all__puzzle_identifiers.npy"))
    pidx = np.load(os.path.join(DATA, "train", "all__puzzle_indices.npy"))
    names = json.load(open(os.path.join(DATA, "identifiers.json")))
    out = {}
    for i in range(len(pidx) - 1):
        name = names[int(pid[i])]
        if name in names_wanted:
            sl = slice(int(pidx[i]), int(pidx[i + 1]))
            out[name] = (inp[sl], lab[sl], int(pid[i]))
    return out


@torch.no_grad()
def align_identifiers(m, tasks, pad_id, batch, max_shift=12):
    """The rebuilt dataset has a few more identifiers than the checkpoint's
    embedding table (augmentation dedup differs), so ids after the first
    divergence are shifted. For each eval puzzle, pick the shift in
    [-max_shift, max_shift] whose embedding row best reproduces the puzzle's
    own demonstration pairs (trained on); report the chosen shift per task."""
    pairs = load_train_pairs_for(set(tasks))
    shift = {}
    m = m.to(DEV)
    npid = m.inner.puzzle_emb.weights.shape[0] if hasattr(m.inner.puzzle_emb, "weights") else None
    for t in tasks:
        x_np, y_np, base = pairs[t]
        best = (-1.0, 0)
        for d in range(-max_shift, max_shift + 1):
            cand = base + d
            if cand <= 0 or (npid is not None and cand >= npid):
                continue
            x = torch.from_numpy(x_np.astype(np.int64)).to(DEV)
            y = torch.from_numpy(y_np.astype(np.int64)).to(DEV)
            p = torch.full((x.shape[0],), cand, dtype=torch.int32, device=DEV)
            bt = {"inputs": x, "labels": y, "puzzle_identifiers": p}
            carry = _carry(m, bt)
            for _ in range(16):
                carry, out = m(carry, bt)
                if carry.halted.all():
                    break
            mask = y != pad_id
            acc = (((out["logits"].argmax(-1) == y) & mask).sum() / mask.sum()).item()
            if acc > best[0]:
                best = (acc, d)
        shift[t] = best
    m.cpu()
    torch.cuda.empty_cache()
    return shift


def build(batch, meta):
    snap = snapshot_download("sapientinc/HRM-checkpoint-ARC-2")
    ck = glob.glob(os.path.join(snap, "checkpoint*"))[0]
    arch = yaml.safe_load(open(os.path.join(snap, "all_config.yaml")))["arch"]
    sd = torch.load(ck, map_location="cpu", weights_only=False)
    sd = {k.replace("_orig_mod.model.", "").replace("model.", ""): v for k, v in sd.items()}
    npid = sd["inner.puzzle_emb.weights"].shape[0]
    vocab = sd["inner.embed_tokens.embedding_weight"].shape[0]
    if npid != meta["num_puzzle_identifiers"]:
        print(f"[warn] checkpoint has {npid} puzzle ids, rebuilt dataset {meta['num_puzzle_identifiers']}; "
              f"identifiers will be aligned per puzzle on demonstration pairs", flush=True)
    cfg = dict(batch_size=batch, seq_len=meta["seq_len"], vocab_size=vocab, num_puzzle_identifiers=npid,
               H_cycles=arch["H_cycles"], L_cycles=arch["L_cycles"],
               H_layers=arch["H_layers"], L_layers=arch["L_layers"],
               hidden_size=arch["hidden_size"], expansion=arch["expansion"],
               num_heads=arch["num_heads"], pos_encodings=arch["pos_encodings"],
               halt_max_steps=arch["halt_max_steps"], halt_exploration_prob=arch["halt_exploration_prob"],
               forward_dtype="float32", puzzle_emb_ndim=arch["puzzle_emb_ndim"])
    m = HRM(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss[:3], unexp[:3])
    return m, cfg


def _carry(m, batch):
    c = m.initial_carry(batch)
    ic = c.inner_carry
    return C(inner_carry=IC(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)), steps=c.steps.to(DEV),
             halted=c.halted.to(DEV), current_data={k: v.to(DEV) for k, v in c.current_data.items()})


@torch.no_grad()
def run(m, inp, lab, pid, batch, pad_id):
    m = m.to(DEV)
    n = inp.shape[0]
    ex_ok = np.zeros(n, dtype=bool)
    cell_c = cell_n = 0
    finals = []
    for s in range(0, n, batch):
        x = torch.from_numpy(inp[s:s + batch].astype(np.int64)).to(DEV)
        y = torch.from_numpy(lab[s:s + batch].astype(np.int64)).to(DEV)
        p = torch.from_numpy(pid[s:s + batch].astype(np.int32)).to(DEV)
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": p}
        carry = _carry(m, bt)
        out = None
        for _ in range(16):
            carry, out = m(carry, bt)
            if carry.halted.all():
                break
        preds = out["logits"].argmax(-1)
        mask = y != pad_id
        corr = (preds == y) & mask
        ex_ok[s:s + batch] = ((corr.sum(-1) == mask.sum(-1)).cpu().numpy())
        cell_c += corr.sum().item()
        cell_n += mask.sum().item()
        finals.append(carry.inner_carry.z_H.detach().float().cpu())
    m.cpu()
    torch.cuda.empty_cache()
    return ex_ok, cell_c / cell_n, torch.cat(finals)


def fidelity(z, ref):
    return float(torch.nn.functional.cosine_similarity(z.flatten(1), ref.flatten(1), dim=-1).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=os.path.expanduser("~/hrm_arc_fidelity_results.json"))
    ap.add_argument("--align", action="store_true", help="force the per-puzzle identifier alignment")
    ap.add_argument("--max_shift", type=int, default=8)
    a = ap.parse_args()
    inp, lab, pid, task, meta = load_test()
    tasks = sorted(set(task.tolist()))
    pad_id = meta.get("pad_id", 0)
    print(f"HRM ARC-AGI-2, {len(inp)} un-augmented test examples over {len(tasks)} tasks, "
          f"seq_len {meta['seq_len']}, pad_id {pad_id}", flush=True)
    m, cfg = build(a.batch, meta)
    print(f"config H{cfg['H_cycles']}L{cfg['L_cycles']} layers {cfg['H_layers']}/{cfg['L_layers']}", flush=True)
    shifts = {}
    if cfg["num_puzzle_identifiers"] != meta["num_puzzle_identifiers"] or a.align:
        shifts = align_identifiers(m, tasks, pad_id, a.batch, max_shift=a.max_shift)
        pid = pid.copy()
        for t, (acc, d) in shifts.items():
            pid[task == t] += d
        hist = {}
        for acc, d in shifts.values():
            hist[d] = hist.get(d, 0) + 1
        print(f"identifier shifts chosen: {dict(sorted(hist.items()))}; "
              f"demonstration-pair cell acc at the chosen row: mean {np.mean([v[0] for v in shifts.values()])*100:.1f}%, "
              f"min {min(v[0] for v in shifts.values())*100:.1f}%", flush=True)
    rows = []
    _, _, ref = run(m, inp, lab, pid, a.batch, pad_id)
    for name, q in [("FP32", None), ("INT8", lambda mm: fake_quant_(mm, 8, True)),
                    ("INT4-naive", lambda mm: fake_quant_(mm, 4, False)),
                    ("INT4-calib", lambda mm: calib_quant_(mm, 4))]:
        mm, _ = build(a.batch, meta)
        if q:
            q(mm)
        ex_ok, cell, z = run(mm, inp, lab, pid, a.batch, pad_id)
        task_ok = np.mean([all(ex_ok[task == t]) for t in tasks])
        fid = fidelity(z, ref)
        rows.append({"variant": name, "example_exact": float(ex_ok.mean()), "task_exact": float(task_ok),
                     "cell": cell, "fidelity": fid})
        print(f"{name:12s} example-exact {ex_ok.mean()*100:6.2f}%  task-exact {task_ok*100:6.2f}%  "
              f"cell {cell*100:6.2f}%  fidelity {fid:.4f}", flush=True)
    json.dump({"task": "ARC-AGI-2 public evaluation, single pass", "model": "HRM official ARC-2",
               "n_examples": int(len(inp)), "n_tasks": len(tasks), "rows": rows,
               "identifier_shifts": {t: {"demo_cell_acc": v[0], "shift": v[1]} for t, v in shifts.items()}},
              open(a.out, "w"), indent=2)
    print("wrote", a.out)
