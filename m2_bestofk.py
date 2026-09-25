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


def build_spec(task, spec, D, batch):
    """CPU build + weight quantizer / weight-noise draw exactly as detector_sweep (seed after build)."""
    base, seed = split_spec(spec)
    q = parse_quant(base)
    m, meta = build(task, None, D, batch)
    torch.manual_seed(seed if seed is not None else 0)
    if q["gran"] is not None:
        quantize_weights_(m, q["bits"], q["gran"], q.get("sigma"))
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
                carry, logits, (q_halt, _) = m.inner(carry, bt)
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
                    carry, logits, (q_halt, _) = mref.inner(carry, bt)
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


@torch.no_grad()
def run_cell_wsample(m, master, bits, gran, draw0_rtn, inp, lab, kind, bfs, K, D, noise, sigma, seed, chunk, chk_steps,
                     hash_w):
    """K rollouts, rollout k uses its own stochastic-rounding draw of the weights (draw 0 = round-to-nearest if
    draw0_rtn); optional PTRM kicks on top. Same output arrays as run_cell (no polish)."""
    n = inp.shape[0]
    corr = np.zeros((K, n, D), np.uint8); qh = np.zeros((K, n, D), np.float16); rms = np.zeros(D)
    hs = {k: np.zeros((K, n), np.int64) for k in chk_steps}
    chk = {k: np.zeros((K, n), np.int16) for k in chk_steps}
    vsh = {k: np.zeros((K, n), np.uint8) for k in chk_steps} if kind == "maze" else None
    for k in range(K):
        if gran == "noise":                                   # weight-noise ensemble: draw k = its own Gaussian draw
            for mod, W in zip(_linears(m), master):
                mod.weight.data = W.clone()
            torch.manual_seed(int(seed) * 7907 + 1000 + k)
            quantize_weights_(m, None, "noise", bits)          # bits carries sigma here
        elif k == 0 and draw0_rtn:
            for mod, W in zip(_linears(m), master):
                mod.weight.data = W.clone()
            quantize_weights_(m, bits, gran)
        else:
            sr_quantize_(m, master, bits, gran, torch.Generator(device=DEV).manual_seed(int(seed) * 7907 + 1000 + k))
        for ci, s in enumerate(range(0, n, chunk)):
            x = torch.from_numpy(inp[s:s + chunk]).to(DEV); y = torch.from_numpy(lab[s:s + chunk]).to(DEV)
            B = x.shape[0]
            bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
            carry = init_inner(m, B, x.shape[1] + m.inner.puzzle_emb_len, m.inner.config.hidden_size)
            gen = torch.Generator(device=DEV).manual_seed(int(seed) * 1000003 + 7 * ci + 100003 * k)
            for t in range(D):
                if noise in ("ptrm", "rel") and sigma > 0:
                    eps = torch.randn(carry.z_L.shape, generator=gen, device=DEV, dtype=carry.z_L.dtype)
                    carry = type(carry)(z_H=carry.z_H, z_L=carry.z_L + sigma * eps)
                carry, logits, (q_halt, _) = m.inner(carry, bt)
                pred = logits.argmax(-1)
                corr[k, s:s + B, t] = (pred == y).all(1).cpu().numpy()
                qh[k, s:s + B, t] = q_halt.float().cpu().numpy()
                rms[t] += float(carry.z_L.float().pow(2).mean().sqrt()) * B / (n * K)
                if (t + 1) in chk_steps:
                    hs[t + 1][k, s:s + B] = (pred.long() * hash_w).sum(1).cpu().numpy()
                    if kind == "sudoku":
                        chk[t + 1][k, s:s + B] = sudoku_check(pred, x)["chk"].cpu().numpy()
                    else:
                        mc = maze_check(pred.cpu().numpy(), x.cpu().numpy(), bfs[s:s + B])
                        chk[t + 1][k, s:s + B] = np.minimum(mc["chk"], 32000)
                        vsh[t + 1][k, s:s + B] = mc["vshort"]
    return corr, qh, rms, hs, chk, vsh, None


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
    ap.add_argument("--wsample", default="none", choices=("none", "sr", "sr0", "wnoise"),
                    help="sr: each rollout uses its own stochastic-rounding draw of the weights on the spec's grid; "
                         "sr0: the same with draw 0 = round-to-nearest")
    ap.add_argument("--offset", type=int, default=0, help="first test row (held-out sets: e.g. 1000)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--allow_cpu", action="store_true")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA on this node"
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
          f"chk_steps={chk_steps} polish={a.polish} dev={DEV}", flush=True)
    for spec in specs:
        todo = []
        for sg in sigmas:
            K = 1 if ((a.noise == "none" or sg == 0) and a.wsample == "none") else a.K
            name = f"{a.task}__{spec}__{a.noise}{sg:g}__D{a.D}_K{K}{('_P%d' % a.polish) if a.polish else ''}{'b' if a.polish and a.polish_from == 'best' else ''}{('_' + a.wsample) if a.wsample != 'none' else ''}{('_m' + a.master) if a.master != 'fp32' else ''}{('_o%d' % a.offset) if a.offset else ''}{a.tag}.npz"
            if not (outdir / name).exists():
                todo.append((sg, K, name))
        if not todo:
            print(f"  {spec}: all cells done", flush=True)
            continue
        if a.wsample != "none":
            base, wseed = split_spec(spec)
            q = parse_quant(base)
            if a.wsample == "wnoise":
                assert q["gran"] == "noise", "wnoise needs a noise<sigma> spec"
            else:
                assert q["bits"] is not None and q["gran"] not in (None, "noise"), "wsample needs a quantizer spec"
            m, meta = build(a.task, None, a.D, 256)
            m = m.to(DEV).eval()
            if a.master != "fp32":
                qm = parse_quant(a.master)
                quantize_weights_(m, qm["bits"], qm["gran"])
            master = [mod.weight.data.clone() for mod in _linears(m)]
        else:
            m, meta, q, wseed = build_spec(a.task, spec, a.D, 256)
        for sg, K, name in todo:
            t1 = time.time()
            seed = 7919 * (wseed or 0) + int(round(1000 * sg)) + 17
            if a.wsample != "none":
                ch = a.chunk or (1000 if kind == "sudoku" else 128)
                corr, qh, rms, hs, chk, vsh, pol = run_cell_wsample(m, master, q["sigma"] if a.wsample == "wnoise" else q["bits"],
                                                                    q["gran"], a.wsample == "sr0",
                                                                    inp, lab, kind, bfs, K, a.D, a.noise, sg, seed, ch,
                                                                    chk_steps, hash_w)
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
                   "polish": a.polish, "polish_from": a.polish_from, "wsample": a.wsample, "master": a.master, "offset": a.offset, "n": n, "chunk": chunk, "file": name, "H_cycles": meta["H_cycles"],
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
