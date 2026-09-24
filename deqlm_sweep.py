#!/usr/bin/env python3
"""X6 (angle A transfer), second DEQ: the published DEQ-Transformer on WikiText-103 (locuslab/deq,
DEQ-Sequence; 23.1 test ppl reported) under the breadth-sweep perturbations, solver budget varied.

    python deqlm_sweep.py --specs noise0.02,noise0.05 --seed 0 --budgets 2,5,10,20,30,60 \
        --out ~/x6/results/deqlm_noise_s0.json

Evaluation as the repo's `--eval` path: test split, eval batch 16, tgt_len 150, mem_len 300, DEQ mode,
Anderson forward solver (m=6, stop at relative residual 1e-3, as published) with the iteration cap
("threshold") set to each budget; plus plain fixed-point iteration (picard) at the same caps. Each
(cell, solver, budget) is a full pass over the test set because the memory of segment t+1 is the
equilibrium of segment t.
Perturbation scope (deq_sweep.py formulas): every Linear/Conv1d weight outside the tied adaptive
embedding/softmax (the DEQ layer f: qkv, r, o, ff1, ff2, weight norm folded; plus the input-injection
conv). Embeddings, projections tied to them, LayerNorm and biases untouched, as in the TRM sweeps.
Read-outs per (cell, solver, budget): test perplexity; per segment the relative residual
||f(z)-z||/||f(z)|| of the returned state (recomputed, one uncounted evaluation), steps used, cosine of
the returned state to the fp32 state at the same budget (lockstep), displacement from the fp32 state at
the largest budget, and the segment NLL.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deq_sweep import parse_spec, quantize, _wn_norm

DEQ_ROOT = Path(os.environ.get("DEQ_ROOT", Path.home() / "x6" / "deq"))
DS = DEQ_ROOT / "DEQ-Sequence"
CKPT = Path(os.environ.get("X6_CKPT", "/data/hf_cache/skolawol/x6/ckpt")) / "deq_transformer_wt103.pkl"
CACHE = Path(os.environ.get("X6_CACHE", "/data/user_data/skolawol/x6"))
DEV = "cuda"


# ----------------------------------------------------------------------------- data
def wt103():
    """Vocabulary exactly as the repo's Corpus (Transformer-XL Vocab, special ['<eos>'], case kept, order
    = Counter.most_common over train.txt) and the encoded test/valid streams; built from the HF copy of
    wikitext-103-v1 (one row per line of wiki.*.tokens; empty rows are the blank lines)."""
    d = CACHE / "wt103"
    cache = d / "encoded.pt"
    if cache.exists():
        return torch.load(cache)
    import pyarrow.parquet as pq
    d.mkdir(parents=True, exist_ok=True)
    hf = os.environ.get("HF_HOME", "/data/hf_cache/skolawol")
    base = glob.glob(hf + "/hub/datasets--Salesforce--wikitext/snapshots/*/wikitext-103-v1")
    assert base, "wikitext-103-v1 not in HF cache"

    def lines(split):
        for fpath in sorted(glob.glob(base[0] + f"/{split}-*.parquet")):
            for t in pq.read_table(fpath).column("text").to_pylist():
                yield t

    counter = Counter()
    for t in lines("train"):
        counter.update(t.strip().split())
    idx2sym = ["<eos>"] + [s for s, _ in counter.most_common()]
    sym2idx = {s: i for i, s in enumerate(idx2sym)}

    def enc(split):
        out = []
        for t in lines(split):
            out.extend(sym2idx.get(s, None) for s in t.strip().split())
            out.append(0)
        assert all(x is not None for x in out), f"OOV in {split}"
        return torch.tensor(out, dtype=torch.long)

    res = {"n_token": len(idx2sym), "test": enc("test"), "valid": enc("validation")}
    tmp = cache.with_suffix(f".tmp{os.getpid()}")
    torch.save(res, tmp)
    os.replace(tmp, cache)
    return res


def batches(stream, bsz=16, bptt=150):
    n_step = stream.size(0) // bsz
    data = stream.narrow(0, 0, n_step * bsz).view(bsz, -1).t().contiguous().to(DEV)
    for i in range(0, data.size(0) - 1, bptt):
        L = min(bptt, data.size(0) - 1 - i)
        yield data[i:i + L], data[i + 1:i + 1 + L], L


# ----------------------------------------------------------------------------- model
def load_deqlm(n_token):
    for p in [str(DEQ_ROOT), str(DS)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    torch.set_default_device(DEV)
    from models.deq_transformer import DEQTransformerLM
    m = DEQTransformerLM(n_token, 2, 24, 10, 700, 70, 48000, dropout=0.05, dropatt=0.0, tie_weights=True,
                         d_embed=700, div_val=4, tie_projs=[False, True, True, True], pre_lnorm=False,
                         wnorm=True, local_size=0, pretrain_steps=0, tgt_len=150, mem_len=300,
                         cutoffs=[20000, 40000, 200000], load="", f_solver=None, b_solver=None,
                         stop_mode="rel", logging=lambda *a, **k: None)
    sd = torch.load(CKPT, map_location=DEV, weights_only=False)
    if not isinstance(sd, dict) or "state_dict" in sd:
        sd = sd["state_dict"] if isinstance(sd, dict) else sd.state_dict()
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not missing, f"missing {missing[:8]}"
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:4]}", flush=True)
    m.reset_length(150, 300)
    return m.eval()


@torch.no_grad()
def perturb_lm_(m, spec, seed):
    q = parse_spec(spec)
    if q["gran"] is None:
        return q
    gen = torch.Generator(device=DEV).manual_seed(int(seed))
    rel = []
    for name, mod in m.named_modules():
        if not isinstance(mod, (nn.Linear, nn.Conv1d)) or name.startswith("crit") or name.startswith("word_emb"):
            continue
        if hasattr(mod, "weight_v"):
            W = mod.weight_v.data * (mod.weight_g.data / _wn_norm(mod.weight_v.data))
        else:
            W = mod.weight.data
        if q["gran"] == "noise":
            Wp = W + q["sigma"] * W.std() * torch.randn(W.shape, generator=gen, device=W.device, dtype=W.dtype)
        else:
            Wp = quantize(W, q["bits"], q["gran"])
        rel.append(float((Wp - W).norm() / W.norm().clamp_min(1e-12)))
        if hasattr(mod, "weight_v"):
            g = _wn_norm(Wp)
            v = Wp.clone()
            v[g.flatten() == 0] = 1.0
            mod.weight_v.data, mod.weight_g.data = v, g
        else:
            mod.weight.data = Wp
    q["weight_rel_err_mean"] = float(np.mean(rel))
    q["n_tensors"] = len(rel)
    return q


# ----------------------------------------------------------------------------- solvers
def _rel(fz, z):
    b = z.shape[0]
    return (fz - z).reshape(b, -1).norm(dim=1) / (fz.reshape(b, -1).norm(dim=1) + 1e-9)


class Solver:
    """anderson: the repo's lib.solvers.anderson (m=6, lam=1e-4, beta=1, stop at batch rel residual
    < 1e-3, return the lowest-residual iterate) with torch.solve -> torch.linalg.solve; picard: z <- f(z),
    last iterate. Records per call: steps used and the per-sequence residual of the returned state."""

    def __init__(self, kind):
        self.kind, self.log = kind, []

    @torch.no_grad()
    def __call__(self, f, x0, threshold, stop_mode="rel", **kw):
        if self.kind == "picard":
            x = x0
            for _ in range(threshold):
                x = f(x)
            z, used = x, threshold
        else:
            z, used = self._anderson(f, x0, threshold)
        fz = f(z)
        r = _rel(fz, z)
        self.log.append({"steps": used, "res": r.cpu().numpy()})
        return {"result": z}

    def _anderson(self, f, x0, threshold, m=6, lam=1e-4, eps=1e-3):
        bsz, d, L = x0.shape
        if threshold <= 2:                        # the repo's loop needs >= 3; fall back to iteration
            x = x0
            for _ in range(threshold):
                x = f(x)
            return x, threshold
        X = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
        Fm = torch.zeros(bsz, m, d * L, dtype=x0.dtype, device=x0.device)
        X[:, 0], Fm[:, 0] = x0.reshape(bsz, -1), f(x0).reshape(bsz, -1)
        X[:, 1], Fm[:, 1] = Fm[:, 0], f(Fm[:, 0].reshape_as(x0)).reshape(bsz, -1)
        H = torch.zeros(bsz, m + 1, m + 1, dtype=x0.dtype, device=x0.device)
        H[:, 0, 1:] = H[:, 1:, 0] = 1
        y = torch.zeros(bsz, m + 1, 1, dtype=x0.dtype, device=x0.device)
        y[:, 0] = 1
        lowest, best, best_k = 1e8, X[:, 1].view_as(x0).clone(), 1
        k = 1
        for k in range(2, threshold):
            n = min(k, m)
            G = Fm[:, :n] - X[:, :n]
            H[:, 1:n + 1, 1:n + 1] = torch.bmm(G, G.transpose(1, 2)) + lam * torch.eye(n, dtype=x0.dtype, device=x0.device)[None]
            alpha = torch.linalg.solve(H[:, :n + 1, :n + 1], y[:, :n + 1])[:, 1:n + 1, 0]
            X[:, k % m] = (alpha[:, None] @ Fm[:, :n])[:, 0]
            Fm[:, k % m] = f(X[:, k % m].reshape_as(x0)).reshape(bsz, -1)
            gx = Fm[:, k % m] - X[:, k % m]
            rel = gx.norm().item() / (1e-5 + Fm[:, k % m].norm().item())
            if rel < lowest:
                lowest, best, best_k = rel, X[:, k % m].view_as(x0).clone(), k
            if rel < eps:
                break
        return best, k + 1


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def lm_step(m, data, target, mems, f_thres):
    if not mems:
        mems = m.init_mems()
    else:
        mems = [x.permute(1, 2, 0).contiguous() for x in mems]
    qlen, bsz = data.size()
    mlen = 0 if mems[0].nelement() == 0 else mems[0].size(2)
    m.pos_drop.reset_mask(1, m.d_model, mlen + qlen)
    m.func.reset(bsz, qlen, mlen + qlen)
    hidden, new_mems, _, _ = m._forward(data, mems=mems, f_thres=f_thres, b_thres=f_thres, train_step=-1,
                                        compute_jac_loss=False)
    pred = hidden[-target.size(0):]
    loss = m.crit(pred.reshape(-1, pred.size(-1)), target.contiguous().view(-1)).view(target.size(0), -1)
    return loss, hidden, new_mems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budgets", default="2,5,10,20,30,60")
    ap.add_argument("--solvers", default="anderson,picard")
    ap.add_argument("--max_segments", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    t0 = time.time()
    data = wt103()
    stream = data[a.split]
    print(f"wt103 vocab {data['n_token']}, {a.split} tokens {len(stream)} ({time.time()-t0:.0f}s)", flush=True)
    specs = ["fp32"] + [s for s in a.specs.split(",") if s and s != "fp32"]
    budgets = sorted(int(b) for b in a.budgets.split(","))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out.exists():
        try:
            records = json.load(open(out))["records"]
        except Exception:
            records = []
    done = {(r["spec"], r["solver"], r["budget"]) for r in records}
    models, qinfo = {}, {}
    for s in specs:
        models[s] = load_deqlm(data["n_token"])
        qinfo[s] = perturb_lm_(models[s], s, a.seed)
    print(f"  {len(models)} models; weight rel err " +
          ", ".join(f"{s}:{qinfo[s].get('weight_rel_err_mean', 0):.3f}" for s in specs), flush=True)
    arrays = {}
    if records and out.with_suffix(".npz").exists():                  # resume after preemption
        try:
            arrays = dict(np.load(str(out.with_suffix(".npz"))))
        except Exception:
            arrays = {}
    kmax = budgets[-1]
    for sv in a.solvers.split(","):
        for k in budgets:
            todo = [s for s in specs if (s, sv, k) not in done]
            if not todo:
                continue
            t1 = time.time()
            # lockstep: fp32 at this budget (cosine), fp32 at the largest budget (displacement), cells
            run = ["fp32"] + [s for s in todo if s != "fp32"]
            solv = {s: Solver(sv) for s in run}
            ref_solv = Solver("anderson")
            mems = {s: [] for s in run}
            ref_mems = []
            stats = {s: {"nll": 0.0, "len": 0, "cos": [], "disp": [], "seg_nll": []} for s in run}
            for si, (x, y, L) in enumerate(batches(stream)):
                if a.max_segments and si >= a.max_segments:
                    break
                models["fp32"].f_solver = ref_solv
                _, zref, ref_mems = lm_step(models["fp32"], x, y, ref_mems, kmax)
                zsame = None
                for s in run:
                    models[s].f_solver = solv[s]
                    loss, z, mems[s] = lm_step(models[s], x, y, mems[s], k)
                    if s == "fp32":
                        zsame = z
                    st = stats[s]
                    st["nll"] += L * float(loss.mean())
                    st["len"] += L
                    st["seg_nll"].append(float(loss.mean()))
                    zf, rf, sf = z.reshape(-1, z.size(-1)), zref.reshape(-1, z.size(-1)), zsame.reshape(-1, z.size(-1))
                    st["cos"].append(float(F.cosine_similarity(zf, sf, dim=1).mean()))
                    st["disp"].append(float((zf - rf).norm() / rf.norm().clamp_min(1e-12)))
            for s in run:
                if (s, sv, k) in done:
                    continue
                st, lg = stats[s], solv[s].log
                res = np.concatenate([e["res"] for e in lg])
                rec = {"model": "deq_transformer_wt103", "spec": s, "quant": qinfo[s], "seed": a.seed, "solver": sv,
                       "budget": k, "split": a.split, "segments": len(lg), "tokens": st["len"],
                       "ppl": math.exp(st["nll"] / st["len"]), "nll": st["nll"] / st["len"],
                       "res_median": float(np.median(res)), "res_mean": float(np.mean(np.clip(res, 0, 1e6))),
                       "res_p90": float(np.percentile(res, 90)),
                       "frac_res_le": {str(t): float((res <= t).mean()) for t in (1e-1, 3e-2, 1e-2, 1e-3)},
                       "steps_mean": float(np.mean([e["steps"] for e in lg])),
                       "cos_mean": float(np.mean(st["cos"])), "disp_mean": float(np.mean(st["disp"])),
                       "ref_res_median": float(np.median(np.concatenate([e["res"] for e in ref_solv.log]))),
                       "secs": time.time() - t1}
                seg_res = np.array([np.median(e["res"]) for e in lg])
                rec["spearman_segres_segnll"] = None
                if len(seg_res) > 3:
                    from scipy.stats import spearmanr
                    rec["spearman_segres_segnll"] = float(spearmanr(seg_res, st["seg_nll"]).correlation)
                records.append(rec)
                arrays[f"{s}|{sv}|{k}|seg_nll"] = np.array(st["seg_nll"], np.float32)
                arrays[f"{s}|{sv}|{k}|seg_res"] = seg_res.astype(np.float32)
                print(f"  {sv:8s} k={k:3d} {s:10s} ppl {rec['ppl']:8.2f} res {rec['res_median']:.1e} "
                      f"steps {rec['steps_mean']:5.1f} cos {rec['cos_mean']:.4f} disp {rec['disp_mean']:.3f}", flush=True)
            json.dump({"records": records}, open(out, "w"))
            np.savez_compressed(str(out.with_suffix(".npz")), **arrays)
            print(f"  ({sv} k={k}: {time.time()-t1:.0f}s)", flush=True)
    print("wrote", out, len(records), f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
