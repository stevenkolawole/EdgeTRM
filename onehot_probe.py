#!/usr/bin/env python3
"""X1 pre-registered probe (reports/edgetrm_exp/X1.md, "one-hot attention explains catch-up").

For a compressed TRM (weights quantized or weight-noised as in difficulty_sweep.py), follow its own
trajectory for K ACT steps on n puzzles and, at steps t in STEPS, measure from its carry z_t:
  mix_err   every token-mixer call inside step t (attention module, or the MLP mixer mlp_t) is recomputed
            with the fp32 twin module on the identical input: ||A_q(x) - A_f(x)|| / ||A_f(x)||, per call,
            averaged over calls (per-puzzle norms over tokens and channels, then mean over puzzles)
  step_err  ||F_q(z_t) - F(z_t)|| / ||F(z_t)|| on z_H after one ACT step from the same carry
  ent       attention entropy of the compressed model's own maps, normalised by log(S), mean over
            heads, queries, calls, puzzles; maxp = mean max probability (attention models only)
  acc       puzzle-exact accuracy of the compressed trajectory after step t
The fp32 model is also followed on its own trajectory (entropy / accuracy reference).

    cd ~/edgetrm_src && python onehot_probe.py --task sudoku_attn --quants fp32,w3c,w4t --n 128 --K 512 \
        --out ~/breadth_results/x1rb/onehot_sudoku_attn.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "TinyRecursiveModels"))
from breadth_sweep import DEV, build, parse_quant, quantize_weights_  # noqa: E402
from difficulty_sweep import load_meta, load_rows, split_spec, _fresh  # noqa: E402
from models.layers import apply_rotary_pos_emb  # noqa: E402

STEPS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


def mixers(m):
    """name -> module for every token mixer (self_attn or mlp_t) in the model."""
    return {n: mod for n, mod in m.named_modules() if n.endswith("self_attn") or n.endswith("mlp_t")}


class Probe:
    def __init__(self, mq, mf):
        self.on = False
        self.fp = mixers(mf)
        self.stats = {"err": [], "ent": [], "maxp": []}
        self.handles = []
        for name, mod in mixers(mq).items():
            self.handles.append(mod.register_forward_hook(self._hook(name, mod), with_kwargs=True))

    def _hook(self, name, mod):
        def fn(module, args, kwargs, output):
            if not self.on:
                return None
            self.on = False                              # the twin may be this very module (fp32 reference)
            try:
                return self._measure(name, module, args, kwargs, output)
            finally:
                self.on = True
        return fn

    def _measure(self, name, module, args, kwargs, output):
        if True:
            with torch.no_grad():
                twin = self.fp[name]
                ref = twin(*args, **kwargs)
                num = (output.float() - ref.float()).flatten(1).norm(dim=1)
                den = ref.float().flatten(1).norm(dim=1).clamp_min(1e-8)
                self.stats["err"].append((num / den).mean().item())
                if name.endswith("self_attn"):
                    x = kwargs.get("hidden_states", args[1] if len(args) > 1 else None)
                    cs = kwargs.get("cos_sin", args[0] if args else None)
                    B, S, _ = x.shape
                    qkv = module.qkv_proj(x).view(B, S, module.num_heads + 2 * module.num_key_value_heads,
                                                  module.head_dim)
                    q = qkv[:, :, :module.num_heads]
                    k = qkv[:, :, module.num_heads:module.num_heads + module.num_key_value_heads]
                    if cs is not None:
                        q, k = apply_rotary_pos_emb(q, k, cs[0], cs[1])
                    q, k = q.transpose(1, 2).float(), k.transpose(1, 2).float()
                    if k.shape[1] != q.shape[1]:
                        k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
                    p = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(module.head_dim), dim=-1)
                    ent = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(S)
                    self.stats["ent"].append(ent.mean().item())
                    self.stats["maxp"].append(p.max(-1).values.mean().item())
            return None

    def take(self):
        out = {k: (float(np.mean(v)) if v else None) for k, v in self.stats.items()}
        self.stats = {"err": [], "ent": [], "maxp": []}
        return out


@torch.no_grad()
def acc_of(out, y):
    preds = out["logits"].argmax(-1)
    mask = y != 0
    return (((preds == y) | ~mask).all(-1)).float().mean().item()


@torch.no_grad()
def attn_stats_fp32(m, bt, carry):
    """entropy of the fp32 model's own maps during one step (reference)."""
    pr = Probe(m, m)
    pr.on = True
    carry2, out = m(carry, bt)
    for h in pr.handles:
        h.remove()
    s = pr.take()
    return carry2, out, s["ent"], s["maxp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sudoku_mlp", "sudoku_attn"])
    ap.add_argument("--quants", required=True)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--K", type=int, default=512)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = load_meta()
    rows_all = meta["strat"]["rows"]
    bins = meta["strat"]["bin"]
    seen, deep = {}, []
    for r, b in zip(rows_all, bins):                     # the deep set: first 300 of each rating bin
        if seen.get(b, 0) < 300:
            deep.append(r)
            seen[b] = seen.get(b, 0) + 1
    rows = deep[::max(1, len(deep) // a.n)][:a.n]
    inp, lab, _ = load_rows(rows)
    x = torch.from_numpy(inp).to(DEV)
    y = torch.from_numpy(lab).to(DEV)
    B = x.shape[0]
    bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
    steps = [s for s in STEPS if s <= a.K]
    doc = {"task": a.task, "rows": rows, "K": a.K, "steps": steps, "records": []}
    if Path(a.out).exists():
        doc["records"] = json.load(open(a.out))["records"]
    done = {r["spec"] for r in doc["records"]}
    mf, _ = build(a.task, None, a.K, B)
    mf = mf.to(DEV).eval()
    for spec in [q.strip() for q in a.quants.split(",") if q.strip()]:
        if spec in done:
            continue
        t0 = time.time()
        rec = {"spec": spec, "n": B, "steps": steps, "acc": [], "mix_err": [], "step_err": [], "ent": [], "maxp": []}
        if spec == "fp32":
            carry = _fresh(mf, bt)
            for t in range(1, a.K + 1):
                if t in steps:
                    carry, out, ent, maxp = attn_stats_fp32(mf, bt, carry)
                    rec["acc"].append(acc_of(out, y)); rec["ent"].append(ent); rec["maxp"].append(maxp)
                    rec["mix_err"].append(0.0); rec["step_err"].append(0.0)
                else:
                    carry, out = mf(carry, bt)
        else:
            base, seed = split_spec(spec)
            q = parse_quant(base)
            mq, _ = build(a.task, None, a.K, B)
            torch.manual_seed(seed)
            np.random.seed(seed)
            quantize_weights_(mq, q["bits"], q["gran"], q.get("sigma"))
            mq = mq.to(DEV).eval()
            pr = Probe(mq, mf)
            carry = _fresh(mq, bt)
            for t in range(1, a.K + 1):
                if t in steps:
                    with torch.no_grad():
                        cf, _ = mf(carry, bt)                   # fp32 step from the compressed model's own carry
                    pr.on = True
                    with torch.no_grad():
                        carry, out = mq(carry, bt)
                    pr.on = False
                    s = pr.take()
                    zq, zf = carry.inner_carry.z_H.float(), cf.inner_carry.z_H.float()
                    se = ((zq - zf).flatten(1).norm(dim=1) / zf.flatten(1).norm(dim=1).clamp_min(1e-8)).mean().item()
                    rec["acc"].append(acc_of(out, y)); rec["mix_err"].append(s["err"]); rec["step_err"].append(se)
                    rec["ent"].append(s["ent"]); rec["maxp"].append(s["maxp"])
                else:
                    with torch.no_grad():
                        carry, out = mq(carry, bt)
            for h in pr.handles:
                h.remove()
            del mq
            torch.cuda.empty_cache()
        rec["secs"] = time.time() - t0
        doc["records"].append(rec)
        tmp = a.out + ".tmp"
        json.dump(doc, open(tmp, "w"))
        os.replace(tmp, a.out)
        f = lambda v: "--" if v is None else f"{v:.3f}"
        print(f"{spec:12s} ({rec['secs']:.0f}s) " + " ".join(
            f"t{t}:acc{100*rec['acc'][i]:.0f}/err{f(rec['mix_err'][i])}/ent{f(rec['ent'][i])}"
            for i, t in enumerate(steps) if t in (1, 16, 64, 512)), flush=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
