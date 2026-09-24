#!/usr/bin/env python3
"""M5 on ARC-AGI-1 (TRM attention, arc_v1_public step_518071): per-step exit records on the 419
canonical (un-augmented) test rows of the 400 evaluation puzzles, at H=1 granularity (K H=1 steps =
K/3 default-H supervision steps), token-exact scoring as arc_breadth.carry_pass (every non-pad label
token correct). Not the published aug-1000 vote protocol; that needs the full augmented set (p4d).

Bundle (built on p4d from ~/edgetrm_ckpt/arc_v1_public + ~/edgetrm_data/arc1concept-aug-1000, see
M5.md): arc_slim.pt = the checkpoint with the puzzle-embedding table cut to the 400 evaluation puzzles
(row 0 kept, ids remapped 1..400); arc_canon.npz = inputs, labels, remapped pids of the canonical rows.

    python arc_exit.py --K 48 --specs fp32,w8c,w4c,w4t --outdir ~/breadth_results/m5/arc
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "TinyRecursiveModels"))
from breadth_sweep import _ActQuant, _ActNoise, _linears, parse_quant, quantize_weights_  # noqa: E402
from detector_sweep import split_spec  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUNDLE = Path(os.environ.get("M5_ARC", Path.home() / "edgetrm_ckpt" / "arc_slim"))


def build(K, batch, H=1):
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    arch = yaml.safe_load((BUNDLE / "all_config.yaml").read_text())["arch"]
    sd = torch.load(BUNDLE / "arc_slim.pt", map_location="cpu")
    cfg = dict(batch_size=batch, seq_len=900, vocab_size=sd["inner.embed_tokens.embedding_weight"].shape[0],
               num_puzzle_identifiers=sd["inner.puzzle_emb.weights"].shape[0], H_cycles=H, L_cycles=arch["L_cycles"],
               H_layers=arch["H_layers"], L_layers=arch["L_layers"], hidden_size=sd["inner.embed_tokens.embedding_weight"].shape[1],
               expansion=arch["expansion"], num_heads=arch["num_heads"], pos_encodings=arch["pos_encodings"], halt_max_steps=K,
               halt_exploration_prob=arch["halt_exploration_prob"], forward_dtype=arch["forward_dtype"],
               mlp_t=arch.get("mlp_t", False), puzzle_emb_ndim=arch["puzzle_emb_ndim"], puzzle_emb_len=arch.get("puzzle_emb_len", 16),
               no_ACT_continue=arch.get("no_ACT_continue", True))
    m = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss[:5], unexp[:5])
    return m


@torch.no_grad()
def trajectory(m, x, y, pid, K):
    B = x.shape[0]
    bt = {"inputs": x, "labels": y, "puzzle_identifiers": pid}
    c = m.initial_carry(bt)
    ic = c.inner_carry
    carry = type(c)(inner_carry=type(ic)(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)), steps=c.steps.to(DEV),
                    halted=c.halted.to(DEV), current_data={k: v.to(DEV) for k, v in c.current_data.items()})
    mask = y != 0
    rec = {k: torch.zeros(B, K, device=DEV, dtype=d) for k, d in
           (("correct", torch.bool), ("qh", torch.float16), ("resH", torch.float16), ("cell", torch.float16), ("chg", torch.int16))}
    prevH, prevP = None, None
    for k in range(K):
        carry, out = m(carry, bt)
        carry.halted.zero_()
        zH = carry.inner_carry.z_H.float()
        pred = out["logits"].argmax(-1)
        corr = (pred == y) & mask
        rec["correct"][:, k] = corr.sum(-1) == mask.sum(-1)
        rec["cell"][:, k] = corr.sum(-1).float() / mask.sum(-1).clamp_min(1)
        rec["qh"][:, k] = out["q_halt_logits"].float()
        if prevH is not None:
            rec["resH"][:, k] = (zH - prevH).flatten(1).norm(dim=1) / zH.flatten(1).norm(dim=1).clamp_min(1e-8)
            rec["chg"][:, k] = ((pred != prevP) & mask).sum(-1).clamp_max(32767).to(torch.int16)
        else:
            rec["resH"][:, k] = 1.0
            rec["chg"][:, k] = mask.sum(-1).to(torch.int16)
        prevH, prevP = zH, pred
    return {k: v.cpu().numpy() for k, v in rec.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", default="fp32")
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--allow_cpu", action="store_true")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA"
    d = np.load(BUNDLE / "arc_canon.npz")
    inp, lab, pids = d["inputs"], d["labels"], d["pids"]
    if a.n:
        inp, lab, pids = inp[:a.n], lab[:a.n], pids[:a.n]
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    for spec in [s for s in a.specs.split(",") if s]:
        path = od / f"{spec}.npz"
        if path.exists():
            print(f"  {spec}: exists, skipped", flush=True)
            continue
        base, seed = split_spec(spec)
        q = parse_quant(base)
        t0 = time.time()
        m = build(a.K, a.batch)
        torch.manual_seed(seed if seed is not None else 0)
        if q["gran"] is not None:
            quantize_weights_(m, q["bits"], q["gran"], q.get("sigma"))
        m = m.to(DEV)
        handles, obs = [], []
        if q.get("anoise") is not None:
            for mod in _linears(m):
                handles.append(mod.register_forward_pre_hook(_ActNoise(q["anoise"])))
        if q["act"] is not None:
            for mod in _linears(m):
                qa = _ActQuant(q["act"])
                obs.append(qa)
                handles.append(mod.register_forward_pre_hook(qa))
        parts = []
        for ci, s in enumerate(range(0, len(inp), a.batch)):
            x = torch.from_numpy(inp[s:s + a.batch]).to(DEV)
            y = torch.from_numpy(lab[s:s + a.batch]).to(DEV)
            p = torch.from_numpy(pids[s:s + a.batch]).to(torch.int32).to(DEV)
            if q["act"] is not None and ci == 0:
                trajectory(m, x, y, p, a.K)
                for qa in obs:
                    qa.freeze()
            parts.append(trajectory(m, x, y, p, a.K))
        for h in handles:
            h.remove()
        r = {k: np.concatenate([pp[k] for pp in parts]) for k in parts[0]}
        info = {"task": "arc_attn", "spec": spec, "quant": q, "seed": seed, "H_run": 1, "H_default": 3, "K": a.K, "n": int(len(inp)),
                "protocol": "canonical rows, token exact", "secs": time.time() - t0, "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        tmp = str(path) + ".tmp.npz"
        np.savez_compressed(tmp, info=json.dumps(info), **r)
        os.replace(tmp, path)
        c = r["correct"]
        print(f"  arc {spec:12s} token-exact at default-H step " + " ".join(
            f"{s}:{100 * c[:, 3 * s - 1].mean():.1f}" for s in (1, 2, 4, 8, 16) if 3 * s <= a.K) + f" ({time.time() - t0:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
