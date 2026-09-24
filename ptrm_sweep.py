#!/usr/bin/env python3
"""X8, PTRM reconciliation: single-trajectory accuracy vs best-of-K with halting-head selection,
under PTRM's latent noise and under our activation noise, at the same sigma, on our checkpoints.

Noise types (--noise):
  ptrm    PTRM (Sghaier et al. 2605.19943, Fig. 4 algorithm): before every supervision step,
          z_L += eps, eps ~ N(0, sigma^2 I) (absolute sigma; z_L is the output of a post-RMSNorm
          block, so its per-element RMS is ~1 and absolute ~ relative; measured and stored as
          zL_rms).  One kick per supervision step, to the latent only.
  anoise  ours (breadth_sweep._ActNoise): Gaussian noise at the input of every linear layer on
          every call, std = sigma x that input's std.  Many kicks per supervision step, to every
          activation including the heads' inputs.
For each sigma, K independent rollouts (seed = 7919*k + round(1000*sigma)) from the model's own
initial carry, D = nsup supervision steps; per puzzle and rollout: exact correctness, halting
logit, a hash of the predicted grid (for mode@K).  Summary: mean single-rollout accuracy (and its
spread over rollouts), pass@K, best-Q@K (rollout with the highest halting logit), mode@K, for
K' = 1, 2, 4, 8, ..., K (first K' rollouts).  sigma = 0 is the deterministic baseline.
Resumable per sigma.

    python ptrm_sweep.py --task sudoku_mlp --noise ptrm --sigmas 0.1,0.3,1.0 --K 16 --n 1000 \
        --out ~/breadth_results/x8/ptrm_sudoku_mlp_ptrm.json
"""
import argparse
import json
import os
import socket
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breadth_sweep as bs  # noqa: E402
from breadth_sweep import DEV, TRM_TASKS, HRM_TASKS, _ActNoise, _linears, build, load_qa, parse_quant, quantize_weights_  # noqa: E402
from detector_sweep import explicit_carry  # noqa: E402

HASH_W = None


@torch.no_grad()
def rollout(m, inp, lab, batch, nsup, noise, sigma, seed):
    """One stochastic rollout over all puzzles. Returns correct[n], qhalt[n], hash[n], zL_rms."""
    global HASH_W
    torch.manual_seed(seed)
    handles = []
    if noise == "anoise" and sigma > 0:
        for mod in _linears(m):
            handles.append(mod.register_forward_pre_hook(_ActNoise(sigma)))
    corr, qh, hs, rms = [], [], [], []
    for s in range(0, inp.shape[0], batch):
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        B = x.shape[0]
        if HASH_W is None or HASH_W.shape[0] != x.shape[1]:
            g = torch.Generator().manual_seed(99)
            HASH_W = torch.randint(1, 2 ** 31 - 1, (x.shape[1],), generator=g, dtype=torch.int64).to(DEV)
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
        carry = explicit_carry(m, bt)
        out = None
        for _ in range(nsup):
            if noise == "ptrm" and sigma > 0:
                ic = carry.inner_carry
                carry.inner_carry = type(ic)(z_H=ic.z_H, z_L=ic.z_L + sigma * torch.randn_like(ic.z_L))
            carry, out = m(carry, bt)
        pred = out["logits"].argmax(-1)
        corr.append((pred == y).all(1).cpu().numpy())
        qh.append(out["q_halt_logits"].float().cpu().numpy())
        hs.append((pred.long() * HASH_W).sum(1).cpu().numpy())
        zl = carry.inner_carry.z_L.float()
        rms.append(zl.pow(2).mean((1, 2)).sqrt().cpu().numpy())
    for h in handles:
        h.remove()
    return np.concatenate(corr), np.concatenate(qh), np.concatenate(hs), float(np.mean(np.concatenate(rms)))


def summarize(C, Q, Hh):
    """C, Q, Hh: [K, n]. Metrics for K' = 1, 2, 4, ..., K."""
    K, n = C.shape
    out = {}
    Ks = sorted({k for k in (1, 2, 4, 8, 16, 25, 32, 64, 100) if k <= K} | {K})
    for k in Ks:
        c, q, h = C[:k], Q[:k], Hh[:k]
        best = q.argmax(0)
        bestq = c[best, np.arange(n)]
        mode = np.zeros(n, bool)
        for i in range(n):
            cnt = Counter(h[:, i].tolist())
            top = max(cnt.values())
            j = next(j for j in range(k) if cnt[h[j, i]] == top)       # first rollout with a modal answer
            mode[i] = c[j, i]
        out[k] = {"pass": float(c.any(0).mean()), "bestq": float(bestq.mean()), "mode": float(mode.mean())}
    single = C.mean(1)                                                      # per-rollout accuracy
    out["single_mean"] = float(single.mean())
    out["single_sd"] = float(single.std(ddof=1)) if K > 1 else 0.0
    out["single_min"], out["single_max"] = float(single.min()), float(single.max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TRM_TASKS) + list(HRM_TASKS))
    ap.add_argument("--noise", required=True, choices=("ptrm", "anoise"))
    ap.add_argument("--sigmas", default="0.1,0.2,0.3,0.5,1.0")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--nsup", type=int, default=16)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--wspec", default="fp32", help="weight quantizer applied first (default fp32)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow_cpu", action="store_true", help="local logic smoke only")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA"
    t = TRM_TASKS.get(a.task) or HRM_TASKS[a.task]
    batch = a.batch or t["batch"]
    inp, lab = load_qa(t["data"], a.n)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs = []
    if out_path.exists():
        try:
            recs = json.load(open(out_path)).get("records", [])
        except (json.JSONDecodeError, ValueError):
            pass
    done = {r["sigma"] for r in recs}

    def dump():
        tmp = str(out_path) + ".tmp"
        json.dump({"task": a.task, "noise": a.noise, "wspec": a.wspec, "n": int(len(inp)), "nsup": a.nsup,
                   "records": recs}, open(tmp, "w"))
        os.replace(tmp, out_path)

    m, meta = build(a.task, None, a.nsup, batch)
    q = parse_quant(a.wspec)
    if q["gran"] is not None:
        torch.manual_seed(0)
        quantize_weights_(m, q["bits"], q["gran"], q.get("sigma"))
    m = m.to(DEV)
    sigmas = [0.0] + [float(s) for s in a.sigmas.split(",") if s.strip()]
    print(f"{a.task} noise={a.noise} wspec={a.wspec} n={len(inp)} K={a.K} D={a.nsup} sigmas={sigmas} done={sorted(done)}", flush=True)
    for sg in sigmas:
        if sg in done:
            continue
        t1 = time.time()
        K = 1 if sg == 0 else a.K
        C, Q, Hh, R = [], [], [], []
        for k in range(K):
            c, qq, hh, rms = rollout(m, inp, lab, batch, a.nsup, a.noise, sg, 7919 * k + int(round(1000 * sg)))
            C.append(c); Q.append(qq); Hh.append(hh); R.append(rms)
        C, Q, Hh = np.stack(C), np.stack(Q), np.stack(Hh)
        summ = summarize(C, Q, Hh)
        recs.append({"sigma": sg, "K": K, "summary": summ, "zL_rms": float(np.mean(R)),
                     "correct": C.astype(np.int8).tolist(), "qhalt": np.round(Q, 3).tolist(),
                     "hash": Hh.astype(str).tolist(), "secs": time.time() - t1, "host": socket.gethostname(),
                     "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__, "H_cycles": meta["H_cycles"]})
        dump()
        print(f"  sigma {sg:4.2f}: single {summ['single_mean']*100:5.1f} (sd {summ['single_sd']*100:.1f}) "
              f"pass@{K} {summ[K]['pass']*100:5.1f} bestQ@{K} {summ[K]['bestq']*100:5.1f} mode@{K} {summ[K]['mode']*100:5.1f} "
              f"zL_rms {np.mean(R):.3f} ({time.time()-t1:.0f}s)", flush=True)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
