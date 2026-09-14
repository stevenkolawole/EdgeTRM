#!/usr/bin/env python3
"""ARC-AGI-1 breadth grid for the TRM attention checkpoint: quantizer x
recursion depth (H_cycles, n_sup), scored with the paper's own evaluator
(nb_func.evaluate_arc_per_puzzle: canonical un-augmented sample per puzzle,
inverse augmentation, per-puzzle pass@1 over the 400 evaluation tasks, cell
accuracy), plus token puzzle-exact and final-carry fidelity on the same
canonical rows.

    python arc_breadth.py --quants fp32,w8c,w4t,w4a --H 1,3 --nsup 1,4,16 \
        --out ~/breadth/results/arc_attn_H13.json
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "TinyRecursiveModels"))
import quant_ab as Q  # noqa: E402  (CKPT_DIR, DATA, _build, _load_sd)
import nb_func  # noqa: E402
from breadth_sweep import parse_quant, quantize_weights_, _ActQuant, _linears, fidelity  # noqa: E402

DEV = "cuda"
nb_func.DATA_DIR = str(Q.DATA)
nb_func.get_inner = lambda m: m
nb_func.torch = torch          # notebook-era globals the evaluator relies on
nb_func.np = np


def load_split():
    d = Q.DATA / "test"
    inputs = np.load(d / "all__inputs.npy", mmap_mode="r")
    labels = np.load(d / "all__labels.npy", mmap_mode="r")
    pidx = np.load(d / "all__puzzle_indices.npy")
    pid = np.load(d / "all__puzzle_identifiers.npy")
    per_sample = np.zeros(len(inputs), dtype=np.int32)
    for k in range(len(pidx) - 1):
        per_sample[pidx[k]:pidx[k + 1]] = pid[k]
    return inputs, labels, per_sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quants", default="fp32,w8c,w4t,w4a")
    ap.add_argument("--H", default="3")
    ap.add_argument("--nsup", default="16")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--full", action="store_true",
                    help="official protocol: all augmented test rows, inverse-augmented and voted per puzzle "
                         "(pass@1/pass@2 with test-time augmentation, comparable to the published 44.6%%); "
                         "token metrics and fidelity still on the canonical rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q._setup()
    cfg_yaml = yaml.safe_load((Q.CKPT_DIR / "all_config.yaml").read_text())
    sd = Q._load_sd(Q.CKPT_DIR / "step_518071")
    identifiers = json.loads((Q.DATA / "identifiers.json").read_text())
    inputs, labels, pids = load_split()
    canon = np.array([i for i in range(len(pids)) if pids[i] != 0 and "|||" not in identifiers[int(pids[i])]])
    ds = SimpleNamespace(inputs=inputs[canon], labels=labels[canon], per_sample_pids=pids[canon])
    loader = SimpleNamespace(dataset=ds, batch_size=a.chunk)
    if a.full:
        loader = SimpleNamespace(dataset=SimpleNamespace(inputs=inputs, labels=labels, per_sample_pids=pids),
                                 batch_size=a.chunk)
    protocol = "full_aug_vote" if a.full else "canonical_single_pass"
    print(f"ARC-AGI-1: {len(canon)} canonical (un-augmented) rows of {len(pids)}; protocol {protocol}", flush=True)
    c_inputs = np.asarray(ds.inputs).astype(np.int64)
    c_labels = np.asarray(ds.labels).astype(np.int64)
    c_pids = ds.per_sample_pids

    from models.recursive_reasoning.trm import (TinyRecursiveReasoningModel_ACTV1Carry as Carry,
                                                TinyRecursiveReasoningModel_ACTV1InnerCarry as IC)

    def build(H, nsup):
        cy = json.loads(json.dumps(cfg_yaml))
        cy["arch"]["H_cycles"] = H
        cy["arch"]["halt_max_steps"] = nsup
        return Q._build(sd, cy, batch_size=a.chunk).to(DEV).eval()

    @torch.no_grad()
    def carry_pass(m, nsup, act=None):
        """token puzzle-exact, cell accuracy and final z_H on the canonical rows
        (also the activation-quant calibration when act is set)."""
        handles, obs = [], []
        if act is not None:
            for mod in _linears(m):
                q = _ActQuant(act)
                obs.append(q)
                handles.append(mod.register_forward_pre_hook(q))
        preds = np.zeros_like(c_labels)
        finals = []
        for ci, lo in enumerate(range(0, len(c_inputs), a.chunk)):
            hi = min(lo + a.chunk, len(c_inputs))
            bt = {"inputs": torch.from_numpy(c_inputs[lo:hi]).to(DEV),
                  "labels": torch.from_numpy(c_labels[lo:hi]).to(DEV),
                  "puzzle_identifiers": torch.from_numpy(c_pids[lo:hi]).to(torch.int32).to(DEV)}

            def fwd():
                c = m.initial_carry(bt)
                ic = c.inner_carry
                carry = Carry(inner_carry=IC(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)), steps=c.steps.to(DEV),
                              halted=c.halted.to(DEV), current_data={k: v.to(DEV) for k, v in c.current_data.items()})
                out = None
                for _ in range(nsup):
                    carry, out = m(carry, bt)
                    if carry.halted.all():
                        break
                return carry, out
            if act is not None and ci == 0:
                fwd()
                for q in obs:
                    q.freeze()
            carry, out = fwd()
            preds[lo:hi] = out["logits"].argmax(-1).cpu().numpy()
            finals.append(carry.inner_carry.z_H.detach().float().cpu())
        mask = c_labels != 0
        corr = (preds == c_labels) & mask
        return ({"pexact_token": float((corr.sum(-1) == mask.sum(-1)).mean()), "cell_token": float(corr.sum() / mask.sum())},
                torch.cat(finals), handles)

    @torch.no_grad()
    def evaluate(m, nsup, act=None):
        stats, z, handles = carry_pass(m, nsup, act)          # activation scales frozen here if act
        p1, p2, cell, ms, n = nb_func.evaluate_arc_per_puzzle(m, loader, device=DEV, n_sup_max=nsup, return_pass2=True,
                                                              fast_mode=not a.full)
        for h in handles:
            h.remove()
        stats.update({"arc_pass1": p1, "arc_pass2": p2, "cell": cell, "n_puzzles": n, "protocol": protocol})
        return stats, z

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():
        try:
            records = json.load(open(out_path))["records"]
        except (json.JSONDecodeError, ValueError):
            pass
    done = {(r["H_cycles"], r["nsup"], r["quant"]["spec"]) for r in records}
    quants = [q for q in a.quants.split(",") if q]
    for H in [int(h) for h in a.H.split(",")]:
        for nsup in [int(x) for x in a.nsup.split(",")]:
            t0 = time.time()
            m = build(H, nsup)
            ref, ref_z = evaluate(m, nsup)
            del m
            torch.cuda.empty_cache()
            if (H, nsup, "fp32") not in done:
                records.append({"task": "arc_attn", "H_cycles": H, "L_cycles": cfg_yaml["arch"]["L_cycles"], "nsup": nsup,
                                "quant": parse_quant("fp32"), **ref, "fidelity_H": 1.0, "secs": time.time() - t0})
                json.dump({"task": "arc_attn", "records": records}, open(out_path, "w"))
            print(f"  H={H} nsup={nsup} fp32       pass1 {ref['arc_pass1']*100:6.2f} pass2 {ref['arc_pass2']*100:6.2f} "
                  f"cell {ref['cell']*100:6.2f} tok {ref['pexact_token']*100:6.2f} ({time.time()-t0:.0f}s)", flush=True)
            for spec in quants:
                if spec == "fp32" or (H, nsup, spec) in done:
                    continue
                q = parse_quant(spec)
                t1 = time.time()
                mq = build(H, nsup)
                quantize_weights_(mq, q["bits"], q["gran"])
                r, z = evaluate(mq, nsup, act=q["act"])
                del mq
                torch.cuda.empty_cache()
                fid = fidelity(z, ref_z)
                records.append({"task": "arc_attn", "H_cycles": H, "L_cycles": cfg_yaml["arch"]["L_cycles"], "nsup": nsup,
                                "quant": q, **r, "fidelity_H": fid, "secs": time.time() - t1})
                json.dump({"task": "arc_attn", "records": records}, open(out_path, "w"))
                print(f"  H={H} nsup={nsup} {spec:10s} pass1 {r['arc_pass1']*100:6.2f} pass2 {r['arc_pass2']*100:6.2f} "
                      f"cell {r['cell']*100:6.2f} tok {r['pexact_token']*100:6.2f} fidH {fid:.4f} ({time.time()-t1:.0f}s)", flush=True)
    print("wrote", out_path, len(records))


if __name__ == "__main__":
    main()
