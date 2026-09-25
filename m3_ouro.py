#!/usr/bin/env python3
"""M3 recursion-aware calibration on Ouro-1.4B (looped LM; X4's harness looped_decode.py, run from ~/breadth/lm).

One loop of Ouro is F(h) = norm(layers_1..24(h)); the model applies it total_ut_steps = 4 times. The looped
"fixed-point" quantity is delta = ||F~(h_3) - F(h_3)|| / ||h_3||, the one-loop perturbation at the input of the last
loop (h_3 from the bf16 model), averaged over calibration sequences.

Calibration data: GSM8K *train* problems (prompt + reference solution, chat template), disjoint from the GSM8K test
problems evaluated. Format: w4c (per-channel INT4 on every Linear in the looped stack, X4's 'core' scope).
  --clip none|mse|omse|dT   per-tensor clip ratio of the per-row absmax scales
  --calib gptq              GPTQ rounding with the layer Hessian from the calibration forward (all 4 loops)
Evaluation: X4's teacher-forced pass against X4's bf16 reference (NLL, top-1 agreement, KL, latent fidelity) and
greedy generation on GSM8K test[:n] (accuracy), written to <outdir>/<tag>/summary.json.

    cd ~/breadth/lm && PYTHONPATH=~/pylibs/tf4554 HF_HOME=~/hf_cache HF_HUB_OFFLINE=1 python ~/m3/m3_ouro.py \
        --clip dT --calib gptq --outdir ~/m3/ouro
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HOME = Path.home()
sys.path.insert(0, str(HOME / "breadth" / "lm"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import looped_decode as LD  # noqa: E402
import looped_lm_eval as L  # noqa: E402

CLIPS = [round(x, 3) for x in np.linspace(0.25, 1.0, 16)]


def q_c(W, bits, r):
    """per-channel symmetric INT-b with the row absmax scaled by r."""
    Wf = W.float()
    qmax = 2 ** (bits - 1) - 1
    s = (r * Wf.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-8)
    return (torch.round(Wf / s).clamp(-qmax, qmax) * s).to(W.dtype), s


@torch.no_grad()
def gptq(W, H, s, qmax, blocksize=128, percdamp=0.01):
    W = W.clone().float()
    cols = W.shape[1]
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += percdamp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    s = s.reshape(-1, 1).float()
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i:i + 1]
            q = torch.clamp(torch.round(w / s), -qmax, qmax) * s
            Q1[:, i:i + 1] = q
            e = (w - q) / Hi[i, i]
            W1[:, i:] -= e @ Hi[i:i + 1, i:]
            E1[:, i:i + 1] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro14")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--clip", default="none", choices=["none", "mse", "omse", "dT"])
    ap.add_argument("--calib", default="", choices=["", "gptq"])
    ap.add_argument("--n_cal", type=int, default=16)
    ap.add_argument("--cal_offset", type=int, default=100, help="GSM8K train rows used for calibration")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    t0 = time.time()
    tag = f"{a.model}_s{a.steps}_w{a.bits}c_{a.clip}{'_' + a.calib if a.calib else ''}"
    tdir = Path(a.outdir) / tag
    tdir.mkdir(parents=True, exist_ok=True)

    def log(s):
        print(time.strftime("%H:%M:%S ") + s, flush=True)

    fam = LD.family(a.model)
    model, tok = fam.build(a.steps)
    fam.steps = a.steps
    for p in model.parameters():
        p.requires_grad_(False)
    lins = [(n, m) for n, m in model.model.layers.named_modules() if isinstance(m, torch.nn.Linear)]
    W0 = {n: m.weight.data.clone() for n, m in lins}
    mod = dict(lins)
    log(f"{tag}: {len(lins)} looped Linear tensors")

    # ---- calibration sequences (GSM8K train, prompt + reference solution)
    import pandas as pd
    import glob
    files = sorted(glob.glob(str(HOME / "hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet")))
    df = pd.read_parquet(files[0])
    seqs = []
    for q, sol in zip(df["question"][a.cal_offset:a.cal_offset + a.n_cal], df["answer"][a.cal_offset:a.cal_offset + a.n_cal]):
        txt = fam.prompt_text(tok, q) + fam.gold_text(tok, sol)
        seqs.append(tok.encode(txt, return_tensors="pt", add_special_tokens=False).to(LD.DEV))
    inner = model.model

    @torch.no_grad()
    def loops(ids):
        out, hl, _ = inner(ids, use_cache=False)
        return hl

    @torch.no_grad()
    def F_step(h):
        pos = torch.arange(h.shape[1], device=h.device)[None]
        pe = inner.rotary_emb(h, pos)
        for layer in inner.layers[:inner.config.num_hidden_layers]:
            h = layer(h, attention_mask=None, position_ids=pos, past_key_value=None, use_cache=False,
                      cache_position=pos[0], position_embeddings=pe, current_ut=a.steps - 1)
        return inner.norm(h)

    ref = []
    for ids in seqs:
        hl = loops(ids)
        ref.append((hl[-2], hl[-1]))
    chk = F_step(ref[0][0])
    log(f"  check: ||F(h_3) - h_4|| / ||h_4|| = {float((chk - ref[0][1]).float().norm() / ref[0][1].float().norm()):.2e}")

    def set_w(alloc):
        for n, m in lins:
            m.weight.data = alloc.get(n, W0[n])

    @torch.no_grad()
    def delta(alloc):
        set_w(alloc)
        v = [float((F_step(h3) - h4).float().norm() / h3.float().norm()) for h3, h4 in ref]
        set_w({})
        return float(np.mean(v))

    # ---- inputs / Hessians of every looped Linear (bf16 forward, all loops)
    X, Hs, cnt = {}, {}, {}
    if a.clip == "omse" or a.calib == "gptq":
        parts = [lins[i::4] for i in range(4)] if a.calib == "gptq" else [lins]
        for part in parts:                              # a quarter of the Hessians on the GPU at a time
            hooks = []
            for n, m in part:
                def hk(m_, args, n=n):
                    x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
                    if a.calib == "gptq":
                        if n not in Hs:
                            Hs[n] = x.T @ x
                        else:
                            Hs[n].addmm_(x.T, x)
                        cnt[n] = cnt.get(n, 0) + x.shape[0]
                    if a.clip == "omse" and len(X.get(n, [])) < 32:
                        idx = torch.randint(0, x.shape[0], (min(128, x.shape[0]),), device=x.device)
                        X.setdefault(n, []).append(x[idx])
                hooks.append(m.register_forward_pre_hook(hk))
            for ids in seqs:
                loops(ids)
            for h in hooks:
                h.remove()
            for n, _ in part:                           # to host memory: two processes share the 40 GB card
                if n in Hs and Hs[n].is_cuda:
                    Hs[n] = Hs[n].div_(cnt[n]).cpu()
            torch.cuda.empty_cache()
        X = {n: torch.cat(v) for n, v in X.items()}
        log(f"  inputs collected ({time.time()-t0:.0f}s)")

    # ---- clip per tensor
    clips = {}
    if a.clip != "none":
        for n, _ in lins:
            vals = []
            for r in CLIPS:
                Wq, _ = q_c(W0[n], a.bits, r)
                if a.clip == "mse":
                    vals.append(float((Wq.float() - W0[n].float()).pow(2).sum()))
                elif a.clip == "omse":
                    vals.append(float((X[n] @ (Wq.float() - W0[n].float()).T).pow(2).sum()))
                else:
                    vals.append(delta({n: Wq}))
            clips[n] = CLIPS[int(np.argmin(vals))]
        log(f"  clips ({a.clip}) chosen ({time.time()-t0:.0f}s): median {np.median(list(clips.values())):.3f}")
    Q = {}
    qmax = 2 ** (a.bits - 1) - 1
    for n, _ in lins:
        Wq, s = q_c(W0[n], a.bits, clips.get(n, 1.0))
        if a.calib == "gptq":
            Wq = gptq(W0[n], Hs.pop(n).to(W0[n].device), s, qmax).to(W0[n].dtype)
            torch.cuda.empty_cache()
        Q[n] = Wq
    X, Hs = {}, {}
    torch.cuda.empty_cache()
    d_lo = delta(Q)
    log(f"  one-loop delta at h_3: {d_lo:.4f} ({time.time()-t0:.0f}s)")
    set_w(Q)

    # ---- X4's evaluation protocol
    data = L.load_gsm8k(a.n)
    ref_path = HOME / "breadth" / "lm" / "results" / "q6" / "ref" / f"{a.model}_s{a.steps}_n{a.n}_v2.pt"
    tf = None
    if ref_path.exists():
        R = torch.load(ref_path)
        recs, _ = LD.tf_pass(fam, model, tok, data, ref=R)
        tf = {k: float(np.mean([r[k] for r in recs])) for k in ("nll", "tf_acc", "tf_agree", "kl", "fidelity",
                                                                 "tf_ans_acc")}
        log(f"  tf: {tf}")
    recs, secs = LD.gen_pass(fam, model, tok, data, "greedy", a.max_new, a.batch, tdir / "greedy.jsonl", log)
    g = LD.summarize(recs)
    summary = {"tag": tag, "clip": a.clip, "calib": a.calib, "bits": a.bits, "steps": a.steps, "clips": clips,
               "delta_last_loop": d_lo, "n_cal": a.n_cal, "cal": f"gsm8k train[{a.cal_offset}:{a.cal_offset + a.n_cal}]",
               "tf": tf, "gen_greedy": g, "secs": time.time() - t0}
    (tdir / "summary.json").write_text(json.dumps(summary, indent=1))
    log(f"{tag}: greedy acc {g['acc']*100:.1f} (n={g['n']}), degenerate {g['degenerate']*100:.1f}")


if __name__ == "__main__":
    main()
