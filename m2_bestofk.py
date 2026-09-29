#!/usr/bin/env python3
"""M2: best-of-K latent sampling with the model's own halting head as verifier, for COMPRESSED
recursive reasoners (parallel test-time scaling past the basin edge).

Extends X8's ptrm_sweep.py (PTRM, Sghaier et al. 2605.19943, Fig. 4) in four ways:
  * weight specs: any breadth_sweep quantizer (w4c, w4t, w3g32, w3c, w3a, w2g32, ...) and weight
    noise with a seeded draw ("noise0.15@s1"; the same CPU RNG draw as detector_sweep / X8);
  * per-step records of every rollout (exact correctness and halting logit at every supervision
    step), so one run with depth D gives every (D' <= D, K' <= K) combination and the sequential
    arm (K = 1, sigma = 0, D up to 512) of the matched-compute comparison;
  * label-free checker outputs and answer hashes at chosen steps (Sudoku: row/col/box conflicts +
    given mismatches + non-digits, 0 <=> a valid completed grid, which on Sudoku-Extreme's unique
    solutions equals exact; maze: valid S-G simple path and valid-and-shortest (X1 maze_rescore /
    X8 detector_sweep definitions)), so the read-out can apply max-Q, majority vote, Q-weighted
    vote, halt-first, the checker oracle and random selection offline;
  * optional fp32 polish (M1 x M2): after the D compressed steps, P noise-free steps with the fp32
    weights from every rollout's end state; exact, halting logit (fp32 head) and hash per polish step.

Noise (--noise):
  ptrm    before every supervision step z_L += sigma * eps, eps ~ N(0, I)  (PTRM's absolute sigma).
          z_L is the output of a post-RMSNorm block (and the init L_init ~ N(0,1)), so its per-element
          RMS is ~1 and absolute sigma ~ sigma x RMS(z_L); the measured RMS is stored per step.
  rel     the same kick with std = sigma x the per-puzzle RMS of z_L at that step (exactly relative).
  anoise  breadth_sweep._ActNoise: noise at every linear input, std = sigma x the input's std.
  none    deterministic (sigma ignored; K forced to 1).
Rollouts are vectorized: each chunk of B puzzles is replicated K times (batch K*B), rollout k of
puzzle i sits at row k*B + i.  Noise comes from a CUDA generator seeded per (spec seed, sigma,
chunk), so a cell is reproducible given the chunk size.

Reduced-precision arithmetic (--compute fp16 | bf16; default fp32 = every earlier record):
  Every supervision step (and every polish step) runs under torch.autocast(cuda, dtype), so the linears, the attention
  and the halting head compute in the reduced dtype while the WEIGHTS stay whatever the spec says (fp32, or RTN /
  SR values dequantized to fp32) and the carry z_H / z_L stays fp32 (the residual add promotes back).  Nothing else
  changes: same rows, seeds, records.  Cell files carry the suffix _c16 / _cbf16 and the index record the field
  "compute", so fp32-compute cells are never overwritten.
  Question ("free dither"): the dither result (M2.md "Dither" / "Dither controls") is that a FRESH per-step weight
  perturbation at the stochastic-rounding variance lifts the compressed loop above its own fp32 accuracy at depth 512,
  and the lattice is incidental (fp32 weights + per-step Gaussian at the SR variance matches).  Reduced-precision
  arithmetic injects a fresh per-step rounding error of the activations at zero cost.  Is it big enough to act?
  Order of magnitude (weight scales measured on the released checkpoints, p4d Sep 28, tools: see HANDOVER_SEP26):
  the per-step rounding of an fp16 activation is a relative error of at most 2^-11 = 4.9e-4 per element (uniform,
  std 2.8e-4); bf16 is 2^-8 = 3.9e-3 (std 2.3e-3).  A relative per-element error on a linear's INPUT has the same
  effect on the output as a relative per-element error on that row of the WEIGHT, so the comparable weight-side
  number is the SR per-weight std relative to the weight RMS, sqrt(mean s^2 f (1 - f)) / rms(W), over the
  CastedLinears: Sudoku-MLP w4c 0.20 (s = row absmax / 7; rms(W) 0.022, SR std 0.0043), w3g32 0.29; Sudoku-attn w3c
  0.42, w3g32 0.28, w4t 0.43; HRM w3c 0.44.  So the fresh fp16 rounding is 700-1500x smaller in std (5e5-2e6x in
  variance) than the perturbation the dither / gdither arms apply, and bf16 90-200x (8e3-4e4x).  (RTN's FIXED error
  is 0.14-0.31 relative on the same cells.)  The p4d smoke (n = 16, D = 64) confirms the perturbation is live: the
  halting logit differs from fp32 compute at step 0 by 0.045 (fp16) / 0.25 (bf16) and the trajectories diverge with
  depth (per-step correctness differs at 34 / 58 of 64 steps) while the final accuracy is unchanged.
  Predictions, written before any cell ran:
  (a) "free dither works": at D = 512 the compressed model under fp16 / bf16 compute rises above its own fp32-compute
      RTN number, in the direction of the gdither arm (MLP w4c 82.1 -> towards 95; attn w3c 89.4 -> 94; HRM w3c 66.4
      -> 75), bf16 more than fp16, and the fp32-weight model gains as well.  Then a device gets the per-step
      regulariser from its fp16 / bf16 tensor cores alone and the re-draw is unnecessary.
  (b) "too weak" (EXPECTED, by the ~2-3 orders of magnitude above): fp16 and bf16 compute reproduce the fp32-compute
      trajectories to within a few tenths of a point at every checkpoint (16 / 64 / 512) for both fp32 and RTN weights,
      and adding fp16 compute on top of dither (K = 3) changes the dither number by no more than its seed spread.
      Then the free perturbation is far below the useful range and the device needs the explicit re-draw (or an
      explicit per-step noise source); the reduced-precision run also doubles as the "reduced precision does no harm"
      check for the fp16 device paths.
  A third outcome to watch for: bf16 HURTS (a fixed 2^-8 error on the halting logit or on the settled state moves the
      readout without any regularising benefit); then compute precision is a fault, not a free lunch.

Output: one .npz per cell (task, spec, noise, sigma, D, K) under --outdir, plus a JSON index line
(--index) with the cell's summary.  Resumable per cell (skip if the .npz exists).

    python m2_bestofk.py --task sudoku_mlp --specs fp32,w3g32,noise0.15@s0 --noise ptrm \
        --sigmas 0.3 --K 32 --D 16 --n 1000 --outdir ~/m2_results/cells --index ~/m2_results/index.jsonl
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from breadth_sweep import DEV, TRM_TASKS, HRM_TASKS, _ActNoise, _linears, build, load_qa, parse_quant, quantize_weights_  # noqa: E402
from detector_sweep import SUD_BLANK, MZ_WALL, sudoku_check, maze_check, bfs_dist, split_spec  # noqa: E402


def init_inner(m, B, S, D):
    ic = m.inner.empty_carry(B)
    return type(ic)(z_H=m.inner.H_init.to(DEV).view(1, 1, D).expand(B, S, D).clone(),
                    z_L=m.inner.L_init.to(DEV).view(1, 1, D).expand(B, S, D).clone())


def make_hash_w(L):
    g = torch.Generator().manual_seed(99)
    return torch.randint(1, 2 ** 31 - 1, (L,), generator=g, dtype=torch.int64).to(DEV)


COMPUTE = "fp32"                       # set from --compute; fp16 / bf16 = autocast every forward (arithmetic only)
_AC_DTYPE = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def fwd(model, carry, bt):
    """One supervision step under the chosen compute precision. Weights and carry stay as they are; only the arithmetic
    inside the step (linears, attention, halting head) runs in the reduced dtype."""
    dt = _AC_DTYPE[COMPUTE]
    if dt is None:
        return model.inner(carry, bt)
    with torch.autocast(device_type="cuda", dtype=dt):
        return model.inner(carry, bt)


def build_spec(task, spec, D, batch):
    """CPU build + weight quantizer / weight-noise draw exactly as detector_sweep (seed after build)."""
    base, seed = split_spec(spec)
    q = parse_quant("fp32" if base.startswith("gq") else base)          # gq<fmt>: GPTQ weights (M5, m5_gptq.py)
    m, meta = build(task, None, D, batch)
    torch.manual_seed(seed if seed is not None else 0)
    if q["gran"] is not None:
        quantize_weights_(m, q["bits"], q["gran"], q.get("sigma"))
    if base.startswith("gq"):
        import m5_gptq
        m5_gptq.load_if_gptq(m, task, base)
    assert q.get("act") is None, "activation fake-quant specs are not supported here"
    return m.to(DEV).eval(), meta, q, seed


@torch.no_grad()
def run_cell(m, mref, inp, lab, kind, bfs, K, D, noise, sigma, seed, chunk, chk_steps, polish, hash_w, polish_from="last"):
    n = inp.shape[0]
    corr = np.zeros((K, n, D), np.uint8)
    qh = np.zeros((K, n, D), np.float16)
    rms = np.zeros(D)
    hs = {k: np.zeros((K, n), np.int64) for k in chk_steps}
    chk = {k: np.zeros((K, n), np.int16) for k in chk_steps}          # sudoku: checker score; maze: n violations
    vsh = {k: np.zeros((K, n), np.uint8) for k in chk_steps} if kind == "maze" else None
    pol = None
    if polish:
        pol = {"corr": np.zeros((K, n, polish), np.uint8), "q": np.zeros((K, n, polish), np.float16),
               "hash": np.zeros((K, n, polish), np.int64), "chk": np.zeros((K, n, polish), np.int16)}
        if kind == "maze":
            pol["vshort"] = np.zeros((K, n, polish), np.uint8)
    handles = []
    if noise == "anoise" and sigma > 0:
        torch.manual_seed(seed)
        for mod in _linears(m):
            handles.append(mod.register_forward_pre_hook(_ActNoise(sigma)))
    try:
        for ci, s in enumerate(range(0, n, chunk)):
            x0 = torch.from_numpy(inp[s:s + chunk]).to(DEV)
            y0 = torch.from_numpy(lab[s:s + chunk]).to(DEV)
            B = x0.shape[0]
            x, y = x0.repeat(K, 1), y0.repeat(K, 1)
            bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(K * B, dtype=torch.int32, device=DEV)}
            S = x.shape[1] + m.inner.puzzle_emb_len
            Dh = m.inner.config.hidden_size
            carry = init_inner(m, K * B, S, Dh)
            gen = torch.Generator(device=DEV).manual_seed(int(seed) * 1000003 + ci)
            if polish and polish_from == "best":
                best_q = torch.full((K * B,), -float("inf"), device=DEV)
                best_H, best_L = carry.z_H.clone(), carry.z_L.clone()
            for t in range(D):
                if noise in ("ptrm", "rel") and sigma > 0:
                    zl = carry.z_L
                    eps = torch.randn(zl.shape, generator=gen, device=DEV, dtype=zl.dtype)
                    scale = sigma if noise == "ptrm" else sigma * zl.float().pow(2).mean((1, 2), keepdim=True).sqrt()
                    carry = type(carry)(z_H=carry.z_H, z_L=zl + scale * eps)
                carry, logits, (q_halt, _) = fwd(m, carry, bt)
                pred = logits.argmax(-1)
                corr[:, s:s + B, t] = (pred == y).all(1).view(K, B).cpu().numpy()
                qh[:, s:s + B, t] = q_halt.float().view(K, B).cpu().numpy()
                if polish and polish_from == "best":
                    upd = q_halt.float() > best_q
                    best_q = torch.where(upd, q_halt.float(), best_q)
                    best_H = torch.where(upd.view(-1, 1, 1), carry.z_H, best_H)
                    best_L = torch.where(upd.view(-1, 1, 1), carry.z_L, best_L)
                rms[t] += float(carry.z_L.float().pow(2).mean().sqrt()) * B / n
                if (t + 1) in chk_steps:
                    k = t + 1
                    hs[k][:, s:s + B] = (pred.long() * hash_w).sum(1).view(K, B).cpu().numpy()
                    if kind == "sudoku":
                        chk[k][:, s:s + B] = sudoku_check(pred, x)["chk"].view(K, B).cpu().numpy()
                    else:
                        mc = maze_check(pred.cpu().numpy(), x.cpu().numpy(), np.tile(bfs[s:s + B], K))
                        chk[k][:, s:s + B] = np.minimum(mc["chk"], 32000).reshape(K, B)
                        vsh[k][:, s:s + B] = mc["vshort"].reshape(K, B)
            if polish:
                if polish_from == "best":
                    carry = type(carry)(z_H=best_H, z_L=best_L)
                for p in range(polish):
                    carry, logits, (q_halt, _) = fwd(mref, carry, bt)
                    pred = logits.argmax(-1)
                    pol["corr"][:, s:s + B, p] = (pred == y).all(1).view(K, B).cpu().numpy()
                    pol["q"][:, s:s + B, p] = q_halt.float().view(K, B).cpu().numpy()
                    pol["hash"][:, s:s + B, p] = (pred.long() * hash_w).sum(1).view(K, B).cpu().numpy()
                    if kind == "sudoku":
                        pol["chk"][:, s:s + B, p] = sudoku_check(pred, x)["chk"].view(K, B).cpu().numpy()
                    else:
                        mc = maze_check(pred.cpu().numpy(), x.cpu().numpy(), np.tile(bfs[s:s + B], K))
                        pol["chk"][:, s:s + B, p] = np.minimum(mc["chk"], 32000).reshape(K, B)
                        pol["vshort"][:, s:s + B, p] = mc["vshort"].reshape(K, B)
    finally:
        for h in handles:
            h.remove()
    return corr, qh, rms, hs, chk, vsh, pol


@torch.no_grad()
def sr_quantize_(m, master, bits, gran, gen):
    """Stochastic rounding of the master weights onto breadth_sweep's grid for (bits, gran): same scales (and zero
    points) as round-to-nearest, but q = floor(W/s + u), u ~ U[0,1) (unbiased; E[q s] = W). One draw per call."""
    import torch.nn.functional as F
    for mod, W in zip(_linears(m), master):
        U = torch.rand(W.shape, generator=gen, device=W.device, dtype=W.dtype)
        if gran == "a":
            qmax = 2 ** bits - 1
            wmin = W.amin(dim=1, keepdim=True); wmax = W.amax(dim=1, keepdim=True)
            scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
            zp = torch.round(-wmin / scale)
            mod.weight.data = (torch.clamp(torch.floor(W / scale + U) + zp, 0, qmax) - zp) * scale
            continue
        qmax = 2 ** (bits - 1) - 1
        if gran in ("t", "c"):
            sc = (W.abs().max() if gran == "t" else W.abs().amax(dim=1, keepdim=True)) / qmax
            sc = sc.clamp_min(1e-8)
            mod.weight.data = torch.floor(W / sc + U).clamp(-qmax, qmax) * sc
        else:
            g = int(gran[1:])
            out_f, in_f = W.shape
            pad = (-in_f) % g
            Wp = F.pad(W, (0, pad)).view(out_f, -1, g)
            Up = F.pad(U, (0, pad)).view(out_f, -1, g)
            sc = (Wp.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-8)
            mod.weight.data = (torch.floor(Wp / sc + Up).clamp(-qmax, qmax) * sc).view(out_f, -1)[:, :in_f]


def sudoku_symmetry(rng):
    """Random Sudoku symmetry: band/row and stack/column permutations, optional transpose, digit relabelling.
    Returns (P, dmap, dinv): transformed flat grid t = dmap[g[P]]; inverse g[P] = dinv[t]."""
    def order():
        b = rng.permutation(3)
        w = [rng.permutation(3) for _ in range(3)]            # one within-band permutation per band
        return np.array([3 * b[i] + w[i][j] for i in range(3) for j in range(3)])
    R, C = order(), order()
    P = (R[:, None] * 9 + C[None, :]).reshape(-1)
    if rng.random() < 0.5:
        P = P.reshape(9, 9).T.reshape(-1).copy()
    dmap = np.arange(11)
    dmap[2:11] = 2 + rng.permutation(9)
    dinv = np.argsort(dmap)
    return P, torch.from_numpy(dmap).to(DEV), torch.from_numpy(dinv).to(DEV)


@torch.no_grad()
def gmatch_perturb_(m, master, bits, gran, gen):
    """Control for SR: Gaussian weight noise with the SAME per-element variance as stochastic rounding on the spec's
    grid (s^2 f (1 - f), f = frac(W / s)), zero mean, no lattice: the unstructured perturbation of equal size."""
    import torch.nn.functional as F
    for mod, W in zip(_linears(m), master):
        Z = torch.randn(W.shape, generator=gen, device=W.device, dtype=W.dtype)
        if gran == "a":
            qmax = 2 ** bits - 1
            sc = ((W.amax(dim=1, keepdim=True) - W.amin(dim=1, keepdim=True)) / qmax).clamp_min(1e-8)
        elif gran in ("t", "c"):
            qmax = 2 ** (bits - 1) - 1
            sc = ((W.abs().max() if gran == "t" else W.abs().amax(dim=1, keepdim=True)) / qmax).clamp_min(1e-8)
        else:
            g = int(gran[1:]); qmax = 2 ** (bits - 1) - 1
            out_f, in_f = W.shape
            pad = (-in_f) % g
            Wp = F.pad(W, (0, pad)).view(out_f, -1, g)
            sc = (Wp.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-8).expand(-1, -1, g).reshape(out_f, -1)[:, :in_f]
        f = W / sc - torch.floor(W / sc)
        mod.weight.data = W + sc * torch.sqrt(f * (1 - f)) * Z


@torch.no_grad()
def run_cell_multi(get_model, get_xform, inp, lab, kind, bfs, K, D, noise, sigma, seed, chunk, chk_steps, hash_w,
                   step_hook=None):
    """K deterministic (or kicked) rollouts; rollout k runs model get_model(k) on the puzzles transformed by
    get_xform(k) (None = identity). Correctness and the checker are frame-invariant; answer hashes are taken in the
    original frame so votes pool across transforms. Same output arrays as run_cell (no polish)."""
    n = inp.shape[0]
    corr = np.zeros((K, n, D), np.uint8); qh = np.zeros((K, n, D), np.float16); rms = np.zeros(D)
    hs = {k: np.zeros((K, n), np.int64) for k in chk_steps}
    chk = {k: np.zeros((K, n), np.int16) for k in chk_steps}
    vsh = {k: np.zeros((K, n), np.uint8) for k in chk_steps} if kind == "maze" else None
    for k in range(K):
        m = get_model(k)
        xf = get_xform(k)
        for ci, s in enumerate(range(0, n, chunk)):
            x = torch.from_numpy(inp[s:s + chunk]).to(DEV); y = torch.from_numpy(lab[s:s + chunk]).to(DEV)
            if xf is not None:
                P, dmap, dinv = xf
                Pt = torch.from_numpy(P).to(DEV)
                x, y = dmap[x[:, Pt]], dmap[y[:, Pt]]
            B = x.shape[0]
            bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
            carry = init_inner(m, B, x.shape[1] + m.inner.puzzle_emb_len, m.inner.config.hidden_size)
            gen = torch.Generator(device=DEV).manual_seed(int(seed) * 1000003 + 7 * ci + 100003 * k)
            for t in range(D):
                if step_hook is not None:
                    step_hook(k, t)
                if noise in ("ptrm", "rel") and sigma > 0:
                    eps = torch.randn(carry.z_L.shape, generator=gen, device=DEV, dtype=carry.z_L.dtype)
                    carry = type(carry)(z_H=carry.z_H, z_L=carry.z_L + sigma * eps)
                carry, logits, (q_halt, _) = fwd(m, carry, bt)
                pred = logits.argmax(-1)
                corr[k, s:s + B, t] = (pred == y).all(1).cpu().numpy()
                qh[k, s:s + B, t] = q_halt.float().cpu().numpy()
                rms[t] += float(carry.z_L.float().pow(2).mean().sqrt()) * B / (n * K)
                if (t + 1) in chk_steps:
                    po = pred
                    if xf is not None:
                        po = torch.empty_like(pred)
                        po[:, Pt] = dinv[pred]
                    hs[t + 1][k, s:s + B] = (po.long() * hash_w).sum(1).cpu().numpy()
                    if kind == "sudoku":
                        chk[t + 1][k, s:s + B] = sudoku_check(pred, x)["chk"].cpu().numpy()
                    else:
                        mc = maze_check(pred.cpu().numpy(), x.cpu().numpy(), bfs[s:s + B])
                        chk[t + 1][k, s:s + B] = np.minimum(mc["chk"], 32000)
                        vsh[t + 1][k, s:s + B] = mc["vshort"]
    return corr, qh, rms, hs, chk, vsh, None


def weight_setter(m, master, bits, gran, draw0_rtn, seed, gmatch=False):
    """get_model for the weight samplers: SR draws (draw 0 = RTN if draw0_rtn) or Gaussian weight-noise draws."""
    def get(k):
        if gran == "noise":
            for mod, W in zip(_linears(m), master):
                mod.weight.data = W.clone()
            torch.manual_seed(int(seed) * 7907 + 1000 + k)
            quantize_weights_(m, None, "noise", bits)          # bits carries sigma here
        elif k == 0 and draw0_rtn:
            for mod, W in zip(_linears(m), master):
                mod.weight.data = W.clone()
            quantize_weights_(m, bits, gran)
        elif gmatch:
            gmatch_perturb_(m, master, bits, gran, torch.Generator(device=DEV).manual_seed(int(seed) * 7907 + 1000 + k))
        else:
            sr_quantize_(m, master, bits, gran, torch.Generator(device=DEV).manual_seed(int(seed) * 7907 + 1000 + k))
        return m
    return get


def quick_summary(corr, qh, hs, chk, D):
    """Headline numbers at the last step (the full read-out is m2_analyze.py)."""
    c, q = corr[:, :, D - 1].astype(bool), qh[:, :, D - 1].astype(np.float32)
    K, n = c.shape
    out = {"single_mean": float(c.mean()), "single_sd": float(c.mean(1).std(ddof=1)) if K > 1 else 0.0}
    for k in sorted({1, 2, 4, 8, 16, 32, 64, K}):
        if k > K:
            continue
        cc, qq = c[:k], q[:k]
        out[f"pass@{k}"] = float(cc.any(0).mean())
        out[f"bestq@{k}"] = float(cc[qq.argmax(0), np.arange(n)].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TRM_TASKS) + list(HRM_TASKS))
    ap.add_argument("--specs", default="fp32")
    ap.add_argument("--noise", default="ptrm", choices=("ptrm", "rel", "anoise", "none"))
    ap.add_argument("--sigmas", default="0.3")
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=0, help="puzzles per chunk (batch = K*chunk)")
    ap.add_argument("--chk_steps", default="", help="steps with checker + hash (default: D, plus 16/32/64/.. <= D)")
    ap.add_argument("--polish", type=int, default=0, help="fp32 polish steps after the D compressed steps")
    ap.add_argument("--polish_from", default="last", choices=("last", "best"),
                    help="polish from each rollout's last state, or from its state at its max-q step (kept on the fly)")
    ap.add_argument("--master", default="fp32", help="weights the draws are made from: fp32 or an RTN spec (e.g. w8c)")
    ap.add_argument("--wsample", default="none", choices=("none", "sr", "sr0", "wnoise", "gmatch", "dither", "gdither"),
                    help="sr: each rollout uses its own stochastic-rounding draw of the weights on the spec's grid; "
                         "sr0: the same with draw 0 = round-to-nearest")
    ap.add_argument("--tta", action="store_true", help="rollout k runs on a random Sudoku symmetry of each puzzle (k=0 identity)")
    ap.add_argument("--ensemble", default="", help="comma list of checkpoints (released = the default one); rollout k uses model k")
    ap.add_argument("--ckpt", default="", help="trained-variant checkpoint instead of the released one (use with --tag)")
    ap.add_argument("--offset", type=int, default=0, help="first test row (held-out sets: e.g. 1000)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--compute", default="fp32", choices=("fp32", "fp16", "bf16"),
                    help="arithmetic precision of every step (autocast); weights stay as the spec says. fp16 -> _c16, bf16 -> _cbf16")
    ap.add_argument("--allow_cpu", action="store_true")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA on this node"
    global COMPUTE
    COMPUTE = a.compute
    csuf = {"fp32": "", "fp16": "_c16", "bf16": "_cbf16"}[a.compute]
    if a.ckpt:
        import breadth_sweep as bsw
        assert a.tag, "--ckpt needs a --tag so cell names differ from the released model's"
        bsw.CKPT_OVERRIDE = a.ckpt
    t = TRM_TASKS.get(a.task) or HRM_TASKS[a.task]
    kind = t["data"]
    inp, lab = load_qa(kind, a.n + a.offset)
    inp, lab = inp[a.offset:], lab[a.offset:]
    n = len(inp)
    bfs = np.array([bfs_dist(g, 30) for g in inp]) if kind == "maze" else None
    chk_steps = sorted({int(s) for s in a.chk_steps.split(",") if s.strip()} |
                       {a.D} | {s for s in (16, 32, 64, 128, 256, 512) if s <= a.D})
    if a.chunk:
        chunk = a.chunk
    else:
        per = 2048 if kind == "sudoku" else 192          # rows per forward batch
        chunk = max(1, min(n, per // (1 if a.noise == "none" else a.K)))
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hash_w = make_hash_w(inp.shape[1])
    mref = None
    if a.polish:
        mref, _ = build(a.task, None, a.D, 256)
        mref = mref.to(DEV).eval()
    sigmas = [0.0] if a.noise == "none" else [float(s) for s in a.sigmas.split(",") if s.strip()]
    specs = [s.strip() for s in a.specs.split(",") if s.strip()]
    print(f"{a.task} n={n} D={a.D} K={a.K} noise={a.noise} sigmas={sigmas} specs={specs} chunk={chunk} "
          f"chk_steps={chk_steps} polish={a.polish} compute={a.compute} dev={DEV}", flush=True)
    for spec in specs:
        todo = []
        for sg in sigmas:
            multi = a.wsample != "none" or a.tta or a.ensemble
            K = (len(a.ensemble.split(",")) if a.ensemble else a.K) if multi else (1 if (a.noise == "none" or sg == 0) else a.K)
            name = f"{a.task}__{spec}__{a.noise}{sg:g}__D{a.D}_K{K}{('_P%d' % a.polish) if a.polish else ''}{'b' if a.polish and a.polish_from == 'best' else ''}{('_' + a.wsample) if a.wsample != 'none' else ''}{('_m' + a.master) if a.master != 'fp32' else ''}{'_tta' if a.tta else ''}{('_ens%d' % len(a.ensemble.split(','))) if a.ensemble else ''}{('_o%d' % a.offset) if a.offset else ''}{csuf}{a.tag}.npz"
            if not (outdir / name).exists():
                todo.append((sg, K, name))
        if not todo:
            print(f"  {spec}: all cells done", flush=True)
            continue
        get_model = None
        step_hook = None
        if a.ensemble:
            import breadth_sweep as bsw
            base, wseed = split_spec(spec)
            q = parse_quant(base)
            models = []
            for ck in a.ensemble.split(","):
                bsw.CKPT_OVERRIDE = None if ck == "released" else ck
                mk, meta = build(a.task, None, a.D, 256)
                if q["gran"] is not None:
                    torch.manual_seed(wseed if wseed is not None else 0)
                    quantize_weights_(mk, q["bits"], q["gran"], q.get("sigma"))
                models.append(mk.to(DEV).eval())
            bsw.CKPT_OVERRIDE = None
            m = models[0]
            get_model = lambda k: models[k]
        elif a.wsample != "none":
            base, wseed = split_spec(spec)
            q = parse_quant(base)
            if a.wsample in ("wnoise",):
                assert q["gran"] == "noise", "wnoise needs a noise<sigma> spec"
            else:
                assert q["bits"] is not None and q["gran"] not in (None, "noise"), "wsample needs a quantizer spec"
            m, meta = build(a.task, None, a.D, 256)
            m = m.to(DEV).eval()
            if a.master != "fp32":
                qm = parse_quant(a.master)
                quantize_weights_(m, qm["bits"], qm["gran"])
            master = [mod.weight.data.clone() for mod in _linears(m)]
            get_model = weight_setter(m, master, q["sigma"] if a.wsample == "wnoise" else q["bits"], q["gran"],
                                      a.wsample == "sr0", 7919 * (wseed or 0) + 17, gmatch=a.wsample == "gmatch")
            if a.wsample == "dither":                # SR weights RE-DRAWN before every supervision step (dithered quantization)
                _dseed = 7919 * (wseed or 0) + 17
                get_model = lambda k, _m=m: _m
                step_hook = lambda k, t, _m=m, _ms=master, _q=q, _s=_dseed: sr_quantize_(
                    _m, _ms, _q["bits"], _q["gran"], torch.Generator(device=DEV).manual_seed(_s * 7907 + 100003 * (k + 1) + t))
            if a.wsample == "gdither":               # control: master + FRESH zero-mean Gaussian every step, per-weight variance = SR's
                _dseed = 7919 * (wseed or 0) + 17
                get_model = lambda k, _m=m: _m
                step_hook = lambda k, t, _m=m, _ms=master, _q=q, _s=_dseed: gmatch_perturb_(
                    _m, _ms, _q["bits"], _q["gran"], torch.Generator(device=DEV).manual_seed(_s * 7907 + 100003 * (k + 1) + t))
        else:
            m, meta, q, wseed = build_spec(a.task, spec, a.D, 256)
            if a.tta:
                get_model = lambda k, _m=m: _m
        for sg, K, name in todo:
            t1 = time.time()
            seed = 7919 * (wseed or 0) + int(round(1000 * sg)) + 17
            if get_model is not None:
                ch = a.chunk or (1000 if kind == "sudoku" else 128)
                if a.tta:
                    assert kind == "sudoku", "TTA implemented for Sudoku symmetries"
                    rng = np.random.default_rng(12345 + 1000 * (wseed or 0))
                    xfs = [None] + [sudoku_symmetry(rng) for _ in range(K - 1)]
                    get_xform = lambda k: xfs[k]
                else:
                    get_xform = lambda k: None
                corr, qh, rms, hs, chk, vsh, pol = run_cell_multi(get_model, get_xform, inp, lab, kind, bfs, K, a.D,
                                                                  a.noise, sg, seed, ch, chk_steps, hash_w, step_hook)
            else:
                corr, qh, rms, hs, chk, vsh, pol = run_cell(m, mref, inp, lab, kind, bfs, K, a.D, a.noise, sg, seed,
                                                            chunk, chk_steps, a.polish, hash_w, a.polish_from)
            arrs = {"corr": corr, "q": qh, "zl_rms": rms}
            for k in chk_steps:
                arrs[f"hash_{k}"] = hs[k]
                arrs[f"chk_{k}"] = chk[k]
                if vsh is not None:
                    arrs[f"vshort_{k}"] = vsh[k]
            if pol is not None:
                for kk, vv in pol.items():
                    arrs[f"pol_{kk}"] = vv
            if kind == "maze":
                arrs["bfs"] = bfs
            tmp = outdir / (name + ".tmp.npz")
            np.savez_compressed(tmp, **arrs)
            os.replace(tmp, outdir / name)
            summ = quick_summary(corr, qh, hs, chk, a.D)
            rec = {"task": a.task, "spec": spec, "wseed": wseed, "noise": a.noise, "sigma": sg, "K": K, "D": a.D,
                   "polish": a.polish, "polish_from": a.polish_from, "wsample": a.wsample, "master": a.master, "tta": a.tta, "ensemble": a.ensemble, "ckpt": a.ckpt, "offset": a.offset, "compute": a.compute, "n": n, "chunk": chunk, "file": name, "H_cycles": meta["H_cycles"],
                   "zl_rms_last": float(rms[-1]), "summary": summ, "secs": time.time() - t1,
                   "host": socket.gethostname(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "torch": torch.__version__, "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
            with open(a.index, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"  {spec:12s} {a.noise}{sg:<4g} K={K:2d} D={a.D}: single {summ['single_mean']*100:5.1f} "
                  f"pass@{K} {summ[f'pass@{K}']*100:5.1f} bestQ@{K} {summ[f'bestq@{K}']*100:5.1f} "
                  f"zL_rms {rms[-1]:.3f} ({time.time()-t1:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
