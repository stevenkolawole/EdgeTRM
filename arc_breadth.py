#!/usr/bin/env python3
"""ARC-AGI-1 breadth grid for the TRM attention checkpoint: quantizer x
recursion depth (H_cycles, n_sup) with single-pass pass@1 over the 400
evaluation tasks (one row per (task, test pair), the canonical rows of
quant_ab.py), token puzzle-exact, cell accuracy and final-carry fidelity.

    python arc_breadth.py --quants fp32,w8c,w4t,w4a --H 1,3 --nsup 1,4,16 \
        --out ~/breadth/results/arc_attn_H13.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "TinyRecursiveModels"))
import quant_ab as Q  # noqa: E402  (checkpoint/data paths, _build, _canonical_rows)
from breadth_sweep import parse_quant, quantize_weights_, _ActQuant, _linears, fidelity  # noqa: E402

DEV = "cuda"


def crop(tokens):
    g = np.asarray(tokens).reshape(30, 30)
    rows = np.where((g >= 2).any(axis=1))[0]
    cols = np.where((g >= 2).any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    return (g[: rows.max() + 1, : cols.max() + 1] - 2).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quants", default="fp32,w8c,w4t,w4a")
    ap.add_argument("--H", default="3")
    ap.add_argument("--nsup", default="16")
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Q._setup()
    cfg_yaml = yaml.safe_load((Q.CKPT_DIR / "all_config.yaml").read_text())
    sd = Q._load_sd(Q.CKPT_DIR / "step_518071")
    identifiers = json.loads((Q.DATA / "identifiers.json").read_text())
    idx, pids, names = Q._canonical_rows(identifiers)
    inputs = np.load(Q.DATA / "test" / "all__inputs.npy", mmap_mode="r")[idx].astype(np.int64)
    labels = np.load(Q.DATA / "test" / "all__labels.npy", mmap_mode="r")[idx].astype(np.int64)
    test_puzzles = json.loads((Q.DATA / "test_puzzles.json").read_text())
    pair_inputs = {name: [(np.array(p["input"], dtype=np.uint8), np.array(p["output"], dtype=np.uint8))
                          for p in puz["test"]] for name, puz in test_puzzles.items()}
    row_match, seen = [], set()
    for i, name in enumerate(names):
        if name not in pair_inputs:
            continue
        in_grid = crop(inputs[i])
        for j, (pin, _) in enumerate(pair_inputs[name]):
            if pin.shape == in_grid.shape and np.array_equal(pin, in_grid) and (name, j) not in seen:
                seen.add((name, j))
                row_match.append((i, name, j))
                break
    print(f"ARC-AGI-1: {len(inputs)} canonical rows, {len(row_match)} matched (task, pair) rows, "
          f"{len(pair_inputs)} tasks", flush=True)

    from models.recursive_reasoning.trm import (TinyRecursiveReasoningModel_ACTV1Carry as Carry,
                                                TinyRecursiveReasoningModel_ACTV1InnerCarry as IC)

    def build(H, nsup):
        cy = json.loads(json.dumps(cfg_yaml))
        cy["arch"]["H_cycles"] = H
        cy["arch"]["halt_max_steps"] = nsup
        return Q._build(sd, cy, batch_size=a.chunk).to(DEV).eval()

    @torch.no_grad()
    def run(m, nsup, act=None):
        preds = np.zeros_like(labels)
        finals = []
        handles, obs = [], []
        if act is not None:
            for mod in _linears(m):
                q = _ActQuant(act)
                obs.append(q)
                handles.append(mod.register_forward_pre_hook(q))
        for ci, lo in enumerate(range(0, len(inputs), a.chunk)):
            hi = min(lo + a.chunk, len(inputs))
            bt = {"inputs": torch.from_numpy(inputs[lo:hi]).to(DEV),
                  "labels": torch.from_numpy(labels[lo:hi]).to(DEV),
                  "puzzle_identifiers": torch.from_numpy(pids[lo:hi]).to(torch.int32).to(DEV)}

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
                fwd()                       # observe activation ranges on the first chunk
                for q in obs:
                    q.freeze()
            carry, out = fwd()
            preds[lo:hi] = out["logits"].argmax(-1).cpu().numpy()
            finals.append(carry.inner_carry.z_H.detach().float().cpu())
        for h in handles:
            h.remove()
        mask = labels != 0
        corr = (preds == labels) & mask
        per_task = {}
        for i, name, j in row_match:
            per_task.setdefault(name, {})[j] = bool(np.array_equal(crop(preds[i]), pair_inputs[name][j][1]))
        task_scores = {n: sum(per_task.get(n, {}).get(j, False) for j in range(len(p))) / len(p)
                       for n, p in pair_inputs.items()}
        return {"arc_pass1": float(np.mean(list(task_scores.values()))),
                "pexact_token": float((corr.sum(-1) == mask.sum(-1)).mean()),
                "cell": float(corr.sum() / mask.sum()),
                "task_correct": {n: s for n, s in task_scores.items()}}, torch.cat(finals)

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
            ref, ref_z = run(m, nsup)
            del m
            torch.cuda.empty_cache()
            if (H, nsup, "fp32") not in done:
                records.append({"task": "arc_attn", "H_cycles": H, "L_cycles": cfg_yaml["arch"]["L_cycles"], "nsup": nsup,
                                "quant": parse_quant("fp32"), **ref, "fidelity_H": 1.0, "secs": time.time() - t0})
                json.dump({"task": "arc_attn", "records": records}, open(out_path, "w"))
            print(f"  H={H} nsup={nsup} fp32       pass1 {ref['arc_pass1']*100:6.2f} tok {ref['pexact_token']*100:6.2f} "
                  f"cell {ref['cell']*100:6.2f} ({time.time()-t0:.0f}s)", flush=True)
            for spec in quants:
                if spec == "fp32" or (H, nsup, spec) in done:
                    continue
                q = parse_quant(spec)
                t1 = time.time()
                mq = build(H, nsup)
                quantize_weights_(mq, q["bits"], q["gran"])
                r, z = run(mq, nsup, act=q["act"])
                del mq
                torch.cuda.empty_cache()
                fid = fidelity(z, ref_z)
                records.append({"task": "arc_attn", "H_cycles": H, "L_cycles": cfg_yaml["arch"]["L_cycles"], "nsup": nsup,
                                "quant": q, **r, "fidelity_H": fid, "secs": time.time() - t1})
                json.dump({"task": "arc_attn", "records": records}, open(out_path, "w"))
                print(f"  H={H} nsup={nsup} {spec:10s} pass1 {r['arc_pass1']*100:6.2f} tok {r['pexact_token']*100:6.2f} "
                      f"cell {r['cell']*100:6.2f} fidH {fid:.4f} ({time.time()-t1:.0f}s)", flush=True)
    print("wrote", out_path, len(records))


if __name__ == "__main__":
    main()
