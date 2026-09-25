#!/usr/bin/env python3
"""M2 on looped LMs: stochastic-rounding (SR) draws of one bf16 master as the diversity source for Ouro, GSM8K.

One process = one model, one arm, K draws, GSM8K test[offset:offset+n]. Arms:
  bf16                 greedy, the bf16 model (K = 1)
  rtn:<spec>           greedy, round-to-nearest 4-bit weights (K = 1), spec as looped_lm_eval (w4c, w4g32, ...)
  sr:<spec>            K greedy decodes, draw k uses its own SR rounding of the bf16 master onto the spec's grid
                       (draw 0 = RTN), core Linears only (model.model.layers: the loop), as X4's scope 'core'
  temp:<T>             K sampled decodes of the bf16 model (top_k 0, temperature T, seed k)
  srtemp:<spec>:<T>    K sampled decodes, draw k = SR draw k and sampling seed k (both diversity sources)
Every draw's outputs go to <outdir>/<model>_s<steps>/<arm>/d<k>.jsonl via looped_decode.gen_pass (resumable at
batch granularity). Afterwards each draw's generated tokens are scored teacher-forced under the SAME draw's weights:
mean log-prob of the generated tokens (a self-confidence signal) -> d<k>_lp.jsonl. The read-out
(m2_ouro_analyze.py) applies majority vote, confidence-weighted vote and max-confidence selection at K' <= K.

    HF_HOME=... PYTHONPATH=<tf4554> python m2_ouro_sr.py --model ouro14 --arm sr:w4g32 --K 8 --n 500 \
        --outdir /opt/dlami/nvme/edgetrm/results/m2/ouro
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import looped_decode as D  # noqa: E402
import looped_lm_eval as L  # noqa: E402


def load_gsm8k(n, offset):
    import glob
    import pandas as pd
    root = os.environ.get("HF_HOME", str(Path.home() / "hf_cache"))
    files = sorted(glob.glob(root + "/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet"))
    assert files, f"GSM8K parquet not found under {root}/hub"
    df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
    rows = df.iloc[offset:offset + n]
    return [(q, a.split("####")[-1].strip().replace(",", ""), a) for q, a in zip(rows["question"], rows["answer"])]


def core_linears(model):
    return [m for m in model.model.layers.modules() if isinstance(m, torch.nn.Linear)]


@torch.no_grad()
def sr_linear_(lin, W, bits, gran, gen):
    """SR onto looped_lm_eval.quantize_linear_'s grid (same scales); q = floor(W/s + u), u ~ U[0,1)."""
    Wf = W.float()
    U = torch.rand(Wf.shape, generator=gen, device=Wf.device)
    if gran == "a":
        qmax = 2 ** bits - 1
        wmin, wmax = Wf.amin(1, keepdim=True), Wf.amax(1, keepdim=True)
        scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
        zp = torch.round(-wmin / scale)
        Wq = (torch.clamp(torch.floor(Wf / scale + U) + zp, 0, qmax) - zp) * scale
    else:
        qmax = 2 ** (bits - 1) - 1
        if gran in ("t", "c"):
            s = ((Wf.abs().max() if gran == "t" else Wf.abs().amax(1, keepdim=True)) / qmax).clamp_min(1e-8)
            Wq = torch.floor(Wf / s + U).clamp(-qmax, qmax) * s
        else:
            g = int(gran[1:])
            o, i_ = Wf.shape
            pad = (-i_) % g
            Wp = F.pad(Wf, (0, pad)).view(o, -1, g)
            Up = F.pad(U, (0, pad)).view(o, -1, g)
            s = (Wp.abs().amax(2, keepdim=True) / qmax).clamp_min(1e-8)
            Wq = (torch.floor(Wp / s + Up).clamp(-qmax, qmax) * s).view(o, -1)[:, :i_]
    lin.weight.data = Wq.to(W.dtype)


def set_weights(lins, master, how, spec, k, seed):
    for lin, W in zip(lins, master):
        lin.weight.data = W.clone()
    if how == "bf16":
        return
    q = L.parse_quant(spec)
    if how == "rtn" or k == 0:
        for lin in lins:
            L.quantize_linear_(lin, q["bits"], q["gran"])
        return
    gen = torch.Generator(device=master[0].device).manual_seed(seed * 7907 + 1000 + k)
    for lin, W in zip(lins, master):
        sr_linear_(lin, W, q["bits"], q["gran"], gen)


@torch.no_grad()
def self_logprob(fam, model, tok, data, recs, batch):
    """Mean log-prob of each record's generated ids under the current weights (teacher forced on its own output)."""
    prompts = [fam.prompt_text(tok, q) for q, _, _ in data]
    pad = fam.pad_id(tok)
    out = {}
    idxs = sorted(recs)
    for b0 in range(0, len(idxs), batch):
        ii = idxs[b0:b0 + batch]
        seqs, plen = [], []
        for i in ii:
            p = tok.encode(prompts[i], add_special_tokens=False)
            g = recs[i].get("ids") or []
            seqs.append(p + g)
            plen.append(len(p))
        T = max(len(x) for x in seqs)
        ids = torch.full((len(ii), T), pad, dtype=torch.long)
        att = torch.zeros((len(ii), T), dtype=torch.long)
        for r, x in enumerate(seqs):
            ids[r, T - len(x):] = torch.tensor(x)
            att[r, T - len(x):] = 1
        ids, att = ids.to(D.DEV), att.to(D.DEV)
        logits = model(ids, attention_mask=att, use_cache=False).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
        for r, i in enumerate(ii):
            ng = len(seqs[r]) - plen[r]
            if ng <= 0:
                out[i] = {"idx": i, "mean_lp": -99.0, "n_gen": 0}
                continue
            v = lp[r, T - 1 - ng:T - 1]
            out[i] = {"idx": i, "mean_lp": float(v.mean()), "min_lp": float(v.min()), "n_gen": ng}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--kstart", type=int, default=0, help="run draws kstart..K-1 only (split one arm across processes)")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    parts = a.arm.split(":")
    how = parts[0]
    spec = parts[1] if how in ("rtn", "sr", "srtemp") else None
    T = float(parts[-1]) if how in ("temp", "srtemp") else None
    K = 1 if how in ("bf16", "rtn") else a.K
    fam = D.family(a.model)
    model, tok = fam.build(a.steps)
    lins = core_linears(model)
    master = [l.weight.data.clone() for l in lins]
    data = load_gsm8k(a.n, a.offset)
    arm_name = a.arm.replace(":", "_") + (f"_seed{a.seed}" if a.seed else "")
    od = Path(a.outdir) / f"{a.model}_s{a.steps}_o{a.offset}_n{a.n}" / arm_name
    od.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "hub": fam.hub_id, "steps": a.steps, "arm": a.arm, "K": K, "n": a.n, "offset": a.offset,
            "max_new": a.max_new, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "n_core_linears": len(lins), "seed": a.seed}
    json.dump(meta, open(od / f"meta_k{a.kstart}.json", "w"))
    log = lambda s: print(f"[{a.model} {a.arm}] {s}", flush=True)  # noqa: E731
    for k in range(a.kstart if K > 1 else 0, K):
        set_weights(lins, master, "bf16" if how in ("bf16", "temp") else ("rtn" if how == "rtn" else "sr"), spec, k, a.seed)
        decode = "greedy" if T is None else f"topk0t{T}s{1000 * a.seed + k}"
        t0 = time.time()
        recs, secs = D.gen_pass(fam, model, tok, data, decode, a.max_new, a.batch, od / f"d{k}.jsonl", log)
        acc = sum(r["correct"] for r in recs) / max(1, len(recs))
        lpf = od / f"d{k}_lp.jsonl"
        if not lpf.exists():
            rd = {r["idx"]: r for r in recs}
            lp = self_logprob(fam, model, tok, data, rd, max(4, a.batch // 2))
            with open(lpf, "w") as f:
                for i in sorted(lp):
                    f.write(json.dumps(lp[i]) + "\n")
        log(f"draw {k}: acc {100 * acc:.1f} ({time.time() - t0:.0f}s)")
    log(f"done; peak reserved {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
