#!/usr/bin/env python3
"""Recursion-aware calibration (M3) and the layer-wise calibrations it is compared with.

Every method returns, per weight tensor, a quantized tensor in the SAME format (per-tensor or per-channel INT-b
with one scale per tensor/row); they differ only in how the scale and the rounding are chosen, from the same
unlabeled calibration inputs (the scoring puzzles; evaluation uses separate test puzzles).

Scale (clip ratio r of absmax; m3_mixed.choose_clips): absmax, pct (99.9th percentile of |W|), aciq (Laplace
  optimum, Banner et al.), mse (weight error), omse (layer-output error on trajectory inputs), dT (ours: the one-step
  map perturbation at the fixed point, delta_T), dTj (ours, joint coordinate descent).
Rounding (this file):
  gptq      GPTQ (Frantar et al.) per tensor with the layer Hessian X^T X from inputs along the fp32 trajectory
            (non-sequential: recursion shares weights across depth)
  ar_layer  AdaRound (Nagel et al.): learned rounding, layer-output reconstruction
  ar_block  BRECQ-style: learned rounding, reconstruction of each block's output (one application of a
            transformer/MLP-mixer block, or the DT recurrent block), inputs sampled along the trajectory
  ar_step   ours: learned rounding minimising the one-step map perturbation at the fixed point,
            ||F~(s_T) - F(s_T)||^2 / ||s_T||^2 over unlabeled puzzles (all tensors jointly), plus the read-out
            head's layer error (the head is outside F)
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

ZETA, GAMMA = 1.1, -0.1


def scale_of(W, fmt_p):
    """per-tensor or per-output-row scale for a parsed format {'bits','gran','clip'} (weights as [out, ...])."""
    qmax = 2 ** (fmt_p["bits"] - 1) - 1
    r = fmt_p.get("clip", 1.0)
    Wf = W.reshape(W.shape[0], -1).float()
    if fmt_p["gran"] == "t":
        s = (r * Wf.abs().max() / qmax).clamp_min(1e-12).reshape(1)
        return s.view(*([1] * W.dim())), qmax
    if fmt_p["gran"] == "c":
        s = (r * Wf.abs().amax(1) / qmax).clamp_min(1e-12)
        return s.view(-1, *([1] * (W.dim() - 1))), qmax
    if fmt_p["gran"].startswith("g") or fmt_p["gran"] == "mx":
        k = fmt_p.get("block") or int(fmt_p["gran"][1:])
        o, n = Wf.shape
        pad = (-n) % k
        am = (r * F.pad(Wf, (0, pad)).view(o, -1, k).abs().amax(2)).clamp_min(1e-12)       # [out, groups]
        s = torch.exp2(torch.ceil(torch.log2(am / qmax))) if fmt_p["gran"] == "mx" else am / qmax
        return s.repeat_interleave(k, dim=1)[:, :n].reshape(W.shape), qmax                  # elementwise scale
    raise ValueError(f"calibrated rounding supports per-tensor/per-channel/grouped formats, not {fmt_p}")


class SoftQuant(torch.nn.Module):
    """AdaRound parametrization: W_q = s * clamp(floor(W/s) + h(V), -qmax, qmax)."""

    def __init__(self, W, s, qmax):
        super().__init__()
        self.register_buffer("s", s)
        x = W.float() / s
        self.register_buffer("fl", torch.floor(x))
        frac = (x - torch.floor(x)).clamp(1e-4, 1 - 1e-4)
        p = ((frac - GAMMA) / (ZETA - GAMMA)).clamp(1e-4, 1 - 1e-4)
        self.V = torch.nn.Parameter(torch.log(p / (1 - p)))
        self.qmax = qmax
        self.hard = False

    def h(self):
        return (torch.sigmoid(self.V) * (ZETA - GAMMA) + GAMMA).clamp(0, 1)

    def forward(self, _W):
        r = (self.V >= 0).float() if self.hard else self.h()
        return self.s * torch.clamp(self.fl + r, -self.qmax, self.qmax)

    def reg(self, beta):
        return (1 - (2 * self.h() - 1).abs().pow(beta)).mean()


@torch.no_grad()
def collect_inputs(ctx, S, tensors, max_rows=4096, hessian=False, blocks=None, max_block_calls=48):
    """Along the fp32 trajectory of the scoring puzzles: per tensor, sampled input rows (and X^T X when hessian);
    per block module, sampled (args, kwargs, output) calls."""
    rows = {n: [] for n in tensors}
    H = {n: None for n in tensors} if hessian else None
    cnt = {n: 0 for n in tensors}
    calls = {b: [] for b in (blocks or {})}
    hs = []
    gen = torch.Generator(device="cpu").manual_seed(0)

    def lin_hook(n):
        def f(mod, args):
            if not hessian and len(rows[n]) >= 64:
                return
            x = args[0].detach()
            if ctx.family == "dt":
                x = F.unfold(x, mod.kernel_size, dilation=mod.dilation, padding=mod.padding, stride=mod.stride)
                x = x.transpose(1, 2)
            x = x.reshape(-1, x.shape[-1]).float()
            if hessian:
                H[n] = x.T @ x if H[n] is None else H[n] + x.T @ x
                cnt[n] += x.shape[0]
            idx = torch.randint(0, x.shape[0], (min(256, x.shape[0]),), generator=gen).to(x.device)
            rows[n].append(x[idx])
        return f

    def _slice(v, B, k):
        return v[:k].detach() if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == B else v

    def blk_hook(b):
        def f(mod, args, kwargs, out):
            if len(calls[b]) >= max_block_calls or float(torch.rand(1, generator=gen)) > 0.25:
                return
            B = out.shape[0]
            k = min(B, 8)
            calls[b].append(([_slice(a, B, k) for a in args], {kk: _slice(v, B, k) for kk, v in kwargs.items()},
                             out[:k].detach()))
        return f
    for n in tensors:
        hs.append(ctx.tmod[n].register_forward_pre_hook(lin_hook(n)))
    for b, mod in (blocks or {}).items():
        hs.append(mod.register_forward_hook(blk_hook(b), with_kwargs=True))
    ctx.set({})
    for c in S.chunks:
        Fn = ctx.rec(c["x"])
        z = Fn.z0()
        for _ in range(ctx.T):
            z = Fn(z)
        if ctx.family != "dt":
            ctx.m.inner.lm_head(z[:, 0])
    for h in hs:
        h.remove()
    X = {}
    for n in tensors:
        if rows[n]:
            Xn = torch.cat(rows[n])
            if Xn.shape[0] > max_rows:
                Xn = Xn[torch.randperm(Xn.shape[0], generator=gen)[:max_rows].to(Xn.device)]
            X[n] = Xn
    for b in calls:                                          # keep a random subset of the calls
        L = calls[b]
        if len(L) > max_block_calls:
            idx = torch.randperm(len(L), generator=gen)[:max_block_calls].tolist()
            calls[b] = [L[i] for i in idx]
    if hessian:
        H = {n: (H[n] / max(cnt[n], 1)) if H[n] is not None else None for n in tensors}
    return X, H, calls


@torch.no_grad()
def gptq_tensor(W, Hm, s, qmax, blocksize=128, percdamp=0.01):
    """GPTQ with a fixed scale grid s (per tensor or per row); W as [out, in]."""
    W = W.clone().float()
    cols = W.shape[1]
    Hm = Hm.clone().float()
    dead = torch.diag(Hm) == 0
    Hm[dead, dead] = 1
    W[:, dead] = 0
    damp = percdamp * torch.mean(torch.diag(Hm))
    Hm += damp * torch.eye(cols, device=W.device)
    L = torch.linalg.cholesky(Hm)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    full = s.numel() == W.numel()                        # grouped: one scale per element's group
    s2 = s.reshape(W.shape) if full else (s.reshape(-1, 1) if s.numel() > 1 else s.reshape(1, 1))
    Q = torch.zeros_like(W)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i:i + 1]
            d = Hinv1[i, i]
            sc = s2[:, i1 + i:i1 + i + 1] if full else s2
            q = torch.clamp(torch.round(w / sc), -qmax, qmax) * sc
            Q1[:, i:i + 1] = q
            err = (w - q) / d
            W1[:, i:] -= err @ Hinv1[i:i + 1, i:]
            Err1[:, i:i + 1] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return Q


def block_modules(ctx):
    """The step's blocks: TRM/HRM reasoning blocks (one attention/MLP-mixer layer each), DT recurrent block."""
    if ctx.family == "dt":
        return {"recur_block": ctx.m.recur_block}
    out = {}
    inner = ctx.m.inner
    for lvl in ("L_level", "H_level"):
        mod = getattr(inner, lvl, None)
        if mod is None or not hasattr(mod, "layers"):
            continue
        for i, layer in enumerate(mod.layers):
            out[f"inner.{lvl}.layers.{i}"] = layer
    return out


def calibrate(ctx, S, lo, method, parse_fmt, iters=1500, lr=1e-2, lam=1.0, batch=32, log=print):
    """lo: {group: fmt with clip}; returns {group: quantized tensor}. Groups must be whole tensors (no split)."""
    assert all(ctx.tensor_of[g] == g for g in ctx.groups), "calibrated rounding needs whole-tensor groups"
    t0 = time.time()
    tensors = list(ctx.groups)
    P = {g: parse_fmt(lo[g]) for g in tensors}
    SQ = {g: scale_of(ctx.W0[g], P[g]) for g in tensors}
    ctx.set({})
    if method == "gptq":
        X, H, _ = collect_inputs(ctx, S, tensors, hessian=True)
        out = {}
        for g in tensors:
            W = ctx.W0[g]
            s, qmax = SQ[g]
            Wf = W.reshape(W.shape[0], -1)
            if H[g] is None or H[g].shape[0] != Wf.shape[1]:
                out[g] = torch.clamp(torch.round(W / s), -qmax, qmax) * s
                continue
            sg = s.reshape(Wf.shape) if s.numel() == W.numel() else s.reshape(-1)
            out[g] = gptq_tensor(Wf, H[g], sg, qmax).reshape(W.shape).to(W.dtype)
        log(f"  gptq done ({time.time()-t0:.0f}s)")
        return out
    # AdaRound family
    blocks = block_modules(ctx) if method == "ar_block" else None
    X, _, calls = collect_inputs(ctx, S, tensors, blocks=blocks)
    in_block = set()
    if blocks:
        for b in blocks:
            in_block |= {g for g in tensors if g.startswith(b + ".")}
    head = {g for g in tensors if ("lm_head" in g or "q_head" in g)}
    sq = {}
    for g in tensors:
        s, qmax = SQ[g]
        sq[g] = SoftQuant(ctx.W0[g], s, qmax).to(ctx.W0[g].device)
        ctx.tmod[g].weight.data = ctx.W0[g].clone()
        parametrize.register_parametrization(ctx.tmod[g], "weight", sq[g], unsafe=True)
    params = [p for m_ in sq.values() for p in m_.parameters()]
    opt = torch.optim.Adam(params, lr=lr)
    gen = torch.Generator(device="cpu").manual_seed(1)
    states = [(c["x"], c["zT"], c["zT1"]) for c in S.chunks]
    fp_out = {g: (X[g] @ ctx.W0[g].reshape(ctx.W0[g].shape[0], -1).float().T) for g in X}

    def layer_loss(gs):
        L = torch.zeros((), device=params[0].device)
        for g in gs:
            if g not in X:
                continue
            Wq = ctx.tmod[g].weight.reshape(ctx.W0[g].shape[0], -1).float()
            L = L + ((X[g] @ Wq.T - fp_out[g]).pow(2).mean() / fp_out[g].pow(2).mean().clamp_min(1e-12))
        return L

    warm = int(0.2 * iters)
    for it in range(iters):
        beta = 20 - (20 - 2) * max(0, it - warm) / max(1, iters - warm)
        use_reg = it >= warm
        with torch.enable_grad():
            if method == "ar_layer":
                loss = layer_loss(tensors)
            elif method == "ar_block":
                loss = layer_loss([g for g in tensors if g not in in_block])
                for b, mod in blocks.items():
                    L = calls[b]
                    if not L:
                        continue
                    a_, kw, out = L[int(torch.randint(0, len(L), (1,), generator=gen))]
                    y = mod(*a_, **kw)
                    loss = loss + (y - out).pow(2).mean() / out.pow(2).mean().clamp_min(1e-12)
            elif method == "ar_step":
                x, zT, zT1 = states[int(torch.randint(0, len(states), (1,), generator=gen))]
                idx = torch.randperm(x.shape[0], generator=gen)[:batch].to(x.device)
                Fn = ctx.rec(x[idx])
                from contraction import bnorm, sdpa_math
                with sdpa_math():
                    zq = Fn(zT[idx])
                d = (bnorm(zq - zT1[idx]) / bnorm(zT[idx])).pow(2).mean()
                loss = d + layer_loss(head)
            else:
                raise ValueError(method)
            if it == 0:
                loss0 = float(loss.detach()) + 1e-12       # objectives are normalised by their starting value
            reg = sum(m_.reg(beta) for m_ in sq.values()) / len(sq) if use_reg else torch.zeros((), device=params[0].device)
            tot = loss / loss0 + lam * reg
            opt.zero_grad(set_to_none=True)
            tot.backward()
            opt.step()
        if it % max(1, iters // 5) == 0 or it == iters - 1:
            soft = float(np.mean([((m_.h() > 0.01) & (m_.h() < 0.99)).float().mean().item() for m_ in sq.values()]))
            log(f"  {method} it {it}: loss {float(loss):.5f} reg {float(reg):.4f} beta {beta:.1f} soft {soft:.3f} "
                f"({time.time()-t0:.0f}s)")
    out = {}
    with torch.no_grad():
        for g in tensors:
            sq[g].hard = True
            out[g] = ctx.tmod[g].weight.detach().clone()
            parametrize.remove_parametrizations(ctx.tmod[g], "weight", leave_parametrized=False)
            ctx.tmod[g].weight.data = ctx.W0[g]
    ctx.set({})
    return out
