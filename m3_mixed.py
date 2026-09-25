#!/usr/bin/env python3
"""M3: sensitivity-guided mixed precision under the formats edge hardware supports (weights).

Every weight tensor (TRM/HRM CastedLinear, DT Conv) is one group. A constrained low format (per-tensor
INT-b: TFLM's only INT4 fully-connected form, the collapse setting) is the default; a budget of groups is
kept at a high format (per-tensor INT8). Which groups are kept is chosen by a label-free sensitivity
measured on UNLABELED inputs disjoint from the evaluation set; accuracy is then measured on the
evaluation set against uniform baselines at matched average bits.

Sensitivity scores per group g (all label-free unless marked):
  dT_noise   X2's S = delta_T / sigma with Gaussian noise (sigma0 x W.std) on g only, times the group's
             quantization error in the same units (sigma_eq = rms(Q(W)-W) / std(W)): predicted delta_T of Q
  dT_q       delta_T with g at the low format: ||F~(s_T) - F(s_T)|| / ||s_T|| at fp32's end state
  dT_traj    the same one-step perturbation averaged over the fp32 trajectory t = 0..T-1
  kl         KL(p_fp32 || p_q) of the answer read-out after T steps with g at the low format
  lin        CompreSSM-style input-output gain: the one-step perturbation of g injected at every step and
             propagated through the fp32 linearisation, d_{t+1} = J(z_t) d_t + delta_g(z_t), read out
             through the head (||C d_T|| / ||C s_T||); the finite-horizon Gramian (H2-type) counterpart of
             ranking by Hankel singular values, restricted to the input direction the quantizer injects
  hawq       HAWQ-V2: Tr(H_g)/n_g x ||Q(W_g) - W_g||^2, Hutchinson trace of the Hessian of the cross-entropy
             at the fp32 model's own predictions (pseudo-labels; label-free) through the last recursion step
  hawq_lbl   the same with the true labels (classic HAWQ; NOT label-free, a reference)
  hawq_full  pseudo-label Hessian through the whole T-step unroll (small batch)
  dT_greedy  iterative: promote, one at a time, the group whose promotion lowers delta_T of the current
             mixed model the most per bit (captures interactions)
  random     random order (several draws); small_first: cheapest groups first (naive control)
Allocation: groups ranked by score / cost (cost = n_g x (b_hi - b_lo)), kept high in that order while the
average bits stay within the budget (dT_greedy: its own order).

    python m3_mixed.py --task sudoku_mlp --lo w4t --hi w8t --n 1000 --n_score 256 \
        --out ~/breadth_results/m3/mixed_sudoku_mlp_w4t.json
Resumable: scores and evaluated allocations are cached in --out.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import breadth_sweep as bs  # noqa: E402
import contraction as C  # noqa: E402

DEV = bs.DEV


# ----------------------------------------------------------------------------- formats
def parse_fmt(f):
    """'w4t' | 'w8c' | 'w4g32' | 'mx4' (MXInt4-like: block 32, shared power-of-two scale, INT4 elements) |
    'w4s<k>' (output channels split into k contiguous slices, one per-tensor scale each: k per-tensor INT4
    ops + concat, expressible on TFLM) ; optional clip suffix '@<r>' (scale = r x absmax / qmax)."""
    if "@" in f:
        base, r = f.split("@")
        return {**parse_fmt(base), "clip": float(r)}
    if f.startswith("mx"):
        return {"bits": int(f[2:]), "gran": "mx", "block": 32}
    body = f[1:]
    i = 0
    while body[i].isdigit():
        i += 1
    return {"bits": int(body[:i]), "gran": body[i:]}


@torch.no_grad()
def qtensor(W, fmt):
    """Fake-quantize one weight tensor (any shape; rows = output channels, flattened input dims)."""
    p = parse_fmt(fmt)
    b, g = p["bits"], p["gran"]
    r = p.get("clip", 1.0)
    shp = W.shape
    Wf = W.reshape(shp[0], -1).float()
    qmax = 2 ** (b - 1) - 1
    if g == "t":
        s = (r * Wf.abs().max() / qmax).clamp_min(1e-12)
        Q = torch.round(Wf / s).clamp(-qmax, qmax) * s
    elif g.startswith("s"):
        k = int(g[1:])
        Q = torch.empty_like(Wf)
        for rows in torch.tensor_split(torch.arange(Wf.shape[0], device=Wf.device), k):
            if len(rows) == 0:
                continue
            sl = Wf[rows]
            s = (r * sl.abs().max() / qmax).clamp_min(1e-12)
            Q[rows] = torch.round(sl / s).clamp(-qmax, qmax) * s
    elif g == "c":
        s = (r * Wf.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-12)
        Q = torch.round(Wf / s).clamp(-qmax, qmax) * s
    elif g == "a":
        qm = 2 ** b - 1
        lo, hi = Wf.amin(1, keepdim=True), Wf.amax(1, keepdim=True)
        s = ((hi - lo) / qm).clamp_min(1e-12)
        z = torch.round(-lo / s)
        Q = (torch.clamp(torch.round(Wf / s) + z, 0, qm) - z) * s
    else:                                            # blocks along the input dim: g<N> float scale, mx power of 2
        blk = p.get("block") or int(g[1:])
        o, i_ = Wf.shape
        pad = (-i_) % blk
        Wp = F.pad(Wf, (0, pad)).view(o, -1, blk)
        am = Wp.abs().amax(2, keepdim=True).clamp_min(1e-12)
        if g == "mx":
            s = torch.exp2(torch.ceil(torch.log2(am / qmax)))
        else:
            s = am / qmax
        Q = (torch.round(Wp / s).clamp(-qmax - (1 if g == "mx" else 0), qmax) * s).view(o, -1)[:, :i_]
    return Q.reshape(shp).to(W.dtype)


def fmt_bits(fmt, W):
    """Average stored bits per weight including scale overhead (fp16 scales; MX: 8-bit shared exponent)."""
    p = parse_fmt(fmt)
    b, g = p["bits"], p["gran"]
    rows = W.shape[0]
    cols = W.numel() // rows
    if g == "t":
        return b + 32.0 / W.numel()
    if g.startswith("s"):
        return b + 32.0 * int(g[1:]) / W.numel()
    if g == "c":
        return b + 16.0 / cols
    if g == "a":
        return b + 32.0 / cols
    if g == "mx":
        return b + 8.0 / 32
    return b + 16.0 / int(g[1:])


# ----------------------------------------------------------------------------- model context
class Ctx:
    """The model on the GPU, its weight groups, the fp32 originals, and a cache of quantized copies.
    split=k: every tensor's output channels are cut into k contiguous slices and each slice is its own group
    (its own per-tensor scale); a slice of a tensor is expressible as a separate op + concat."""

    def __init__(self, task, T, batch, split=1):
        self.task, self.T, self.batch = task, T, batch
        if task in C.DT_TASKS:
            self.family = "dt"
            self.m = C.build_dt().to(DEV).eval()
            mods = [(n, mod) for n, mod in self.m.named_modules() if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d))]
            self.pl = 0
        else:
            self.family = "trm" if task in bs.TRM_TASKS else "hrm"
            self.m, self.meta = bs.build(task, None, T, batch)
            self.m = self.m.to(DEV).eval()
            mods = [(n, mod) for n, mod in self.m.named_modules() if mod.__class__.__name__ == "CastedLinear"]
            self.pl = self.m.inner.puzzle_emb_len
        for p_ in self.m.parameters():
            p_.requires_grad_(False)
        self.tensors = [n for n, _ in mods]
        self.tmod = dict(mods)
        self.T0 = {n: mod.weight.data.clone() for n, mod in mods}
        self.groups, self.W0, self.tensor_of, self.rows = [], {}, {}, {}
        for n in self.tensors:
            R = self.T0[n].shape[0]
            k = min(split, R) if split > 1 and self.T0[n].numel() >= 4096 else 1
            parts = torch.tensor_split(torch.arange(R, device=self.T0[n].device), k)
            for i, rows in enumerate(parts):
                g = n if k == 1 else f"{n}#{i}"
                self.groups.append(g)
                self.tensor_of[g] = n
                self.rows[g] = None if k == 1 else rows
                self.W0[g] = self.T0[n] if k == 1 else self.T0[n][rows].clone()
        self.slices_of = {n: [g for g in self.groups if self.tensor_of[g] == n] for n in self.tensors}
        self.mod = {g: self.tmod[self.tensor_of[g]] for g in self.groups}
        self.numel = {g: self.W0[g].numel() for g in self.groups}
        self.N = sum(self.numel.values())
        self.cur = {g: self.W0[g] for g in self.groups}
        self._qc = {}

    def Wq(self, g, fmt):
        if fmt == "fp32":
            return self.W0[g]
        k = (g, fmt)
        if k not in self._qc:
            self._qc[k] = qtensor(self.W0[g], fmt)
        return self._qc[k]

    def _apply(self, n):
        sl = self.slices_of[n]
        if len(sl) == 1:
            self.tmod[n].weight.data = self.cur[sl[0]]
        else:
            W = torch.empty_like(self.T0[n])
            for g in sl:
                W[self.rows[g]] = self.cur[g]
            self.tmod[n].weight.data = W

    def set(self, alloc):
        """alloc: {group: fmt} (missing groups -> fp32)."""
        for g in self.groups:
            self.cur[g] = self.Wq(g, alloc.get(g, "fp32"))
        for n in self.tensors:
            self._apply(n)

    def set_one(self, g, W):
        self.cur[g] = W
        self._apply(self.tensor_of[g])

    def bits(self, alloc):
        return sum(self.numel[g] * fmt_bits(alloc.get(g, "fp32"), self.W0[g]) if alloc.get(g, "fp32") != "fp32"
                   else self.numel[g] * 32 for g in self.groups) / self.N

    def rec(self, x):
        return C.DTRecursion(self.m, x) if self.family == "dt" else C.Recursion(self.m, self.family, x)

    def logits(self, Fn, z):
        """answer logits (B, cells, classes)."""
        if self.family == "dt":
            out = self.m.head(z).float()                    # (B, 2, H, W)
            return out.flatten(2).transpose(1, 2)
        return self.m.inner.lm_head(z[:, 0])[:, self.pl:].float()

    def answer_mask(self, x, y):
        if self.family == "dt":
            return torch.ones(y.shape[0], y[0].numel(), dtype=torch.bool, device=y.device)
        return y != 0


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def rollout(ctx, inp, lab, keep_final=True):
    """exact per example, cell acc, final answer state (for fidelity)."""
    T, bsz = ctx.T, ctx.batch
    corr, cells, zs = [], [], []
    for s in range(0, len(inp), bsz):
        x = torch.from_numpy(inp[s:s + bsz]).to(DEV)
        y = torch.from_numpy(lab[s:s + bsz]).to(DEV)
        Fn = ctx.rec(x)
        z = Fn.z0()
        for _ in range(T):
            z = Fn(z)
        ro = Fn.readout(z, y)
        corr.append(ro["correct"].cpu())
        cells.append(ro["cellacc"].cpu())
        if keep_final:
            zs.append(Fn.answer_state(z).float().cpu())
    return torch.cat(corr).numpy().astype(bool), float(torch.cat(cells).mean()), (torch.cat(zs) if keep_final else None)


def fid(z, ref):
    return float(F.cosine_similarity(z.flatten(1), ref.flatten(1), dim=-1).mean())


# ----------------------------------------------------------------------------- scores
class ScoreSet:
    """fp32 trajectory on the unlabeled scoring inputs, chunked."""

    def __init__(self, ctx, inp, lab):
        self.ctx = ctx
        self.chunks = []
        bsz = ctx.batch
        with torch.no_grad():
            for s in range(0, len(inp), bsz):
                x = torch.from_numpy(inp[s:s + bsz]).to(DEV)
                y = torch.from_numpy(lab[s:s + bsz]).to(DEV)
                ctx.set({})
                Fn = ctx.rec(x)
                z = Fn.z0()
                for _ in range(ctx.T - 1):
                    z = Fn(z)
                zTm1 = z
                zT = Fn(z)
                zT1 = Fn(zT)
                lg = ctx.logits(Fn, zT)
                self.chunks.append({"x": x, "y": y, "zTm1": zTm1, "zT": zT, "zT1": zT1, "logits": lg})

    @torch.no_grad()
    def delta_T(self, alloc):
        """mean over inputs of ||F~(s_T) - F(s_T)|| / ||s_T|| (F~ = the model under `alloc`)."""
        ctx = self.ctx
        ctx.set(alloc)
        vals = []
        for c in self.chunks:
            Fn = ctx.rec(c["x"])
            zq = Fn(c["zT"])
            vals.append(C.bnorm(zq - c["zT1"]) / C.bnorm(c["zT"]))
        ctx.set({})
        return float(torch.cat(vals).mean())

    @torch.no_grad()
    def kl(self, alloc):
        ctx = self.ctx
        ctx.set(alloc)
        vals = []
        for c in self.chunks:
            Fn = ctx.rec(c["x"])
            z = Fn.z0()
            for _ in range(ctx.T):
                z = Fn(z)
            lq = ctx.logits(Fn, z)
            p = F.log_softmax(c["logits"], -1)
            q = F.log_softmax(lq, -1)
            k = (p.exp() * (p - q)).sum(-1)                 # (B, cells)
            msk = ctx.answer_mask(c["x"], c["y"]).float()
            vals.append((k * msk).sum(-1) / msk.sum(-1).clamp_min(1))
        ctx.set({})
        return float(torch.cat(vals).mean())

    def lin_gain(self, g, fmt):
        """d_{t+1} = J(z_t) d_t + (F~_g(z_t) - F(z_t)) along the fp32 trajectory; returns
        (median ||C d_T|| / ||C s_T||, mean one-step delta over t)."""
        ctx = self.ctx
        ctx.set({})
        Wq, W0 = ctx.Wq(g, fmt), ctx.W0[g]
        gains, dtraj = [], []
        for c in self.chunks:
            Fn = ctx.rec(c["x"])
            z = Fn.z0()
            d = torch.zeros_like(z)
            dsum = torch.zeros(z.shape[0], device=z.device)
            for _ in range(ctx.T):
                with torch.no_grad():
                    ctx.set_one(g, Wq)
                    zq = Fn(z)
                    ctx.set_one(g, W0)
                zn, Jd = C.fjvp(Fn, z, d)
                dl = zq - zn
                dsum += C.bnorm(dl) / C.bnorm(z).clamp_min(1e-12)
                d = Jd + dl
                z = zn
            with C.sdpa_math():
                Cz, Cd = torch.func.jvp(lambda zz: ctx.logits(Fn, zz), (z,), (d,))   # head is linear for TRM/HRM
            with torch.no_grad():
                gains.append(C.bnorm(Cd) / C.bnorm(Cz).clamp_min(1e-12))
                dtraj.append(dsum / ctx.T)
        g_ = torch.cat(gains)
        return float(g_.median()), float(g_.mean()), float(torch.cat(dtraj).mean())

    def hessian_trace(self, n_vec=16, labels=False, full=False, max_inputs=64, mb=16, seed=0):
        """Hutchinson Tr(H_g) per group of the cross-entropy at the answer read-out. Pseudo-labels = fp32
        argmax unless labels. full: through the whole T-step unroll from z_0 (else from s_{T-1} detached).
        Micro-batches of `mb` inputs (the double-backward graph of a recursion is large)."""
        ctx = self.ctx
        ctx.set({})
        params = [ctx.tmod[n].weight for n in ctx.tensors]
        for p_ in params:
            p_.requires_grad_(True)
        tr = {g: 0.0 for g in ctx.groups}
        gen = torch.Generator(device=DEV).manual_seed(seed)
        used = 0
        for c in self.chunks:
            for s0 in range(0, c["x"].shape[0], mb):
                if used >= max_inputs:
                    break
                sl = slice(s0, min(s0 + mb, c["x"].shape[0], s0 + max_inputs - used))
                x, y = c["x"][sl], c["y"][sl]
                nb = x.shape[0]
                used += nb
                with C.sdpa_math(), torch.enable_grad():
                    Fn = ctx.rec(x)
                    if full:
                        z = Fn.z0()
                        for _ in range(ctx.T):
                            z = Fn(z)
                    else:
                        z = Fn(c["zTm1"][sl].detach())
                    lg = ctx.logits(Fn, z)
                    msk = ctx.answer_mask(x, y)
                    if labels:
                        tflat = y.reshape(nb, -1) if ctx.family == "dt" else y
                    else:
                        tflat = c["logits"][sl].argmax(-1)
                    loss = F.cross_entropy(lg[msk], tflat[msk], reduction="sum") / msk.sum()
                    gl = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
                    used_i = [i for i, g_ in enumerate(gl) if g_ is not None]      # q_head is not in the answer loss
                    grads = [gl[i] for i in used_i]
                    pu = [params[i] for i in used_i]
                    for k in range(n_vec):
                        vs = [torch.randint(0, 2, p_.shape, generator=gen, device=DEV).float() * 2 - 1 for p_ in pu]
                        Hv = torch.autograd.grad(grads, pu, grad_outputs=vs, retain_graph=k < n_vec - 1, allow_unused=True)
                        for n, v, h in zip([ctx.tensors[i] for i in used_i], vs, Hv):
                            if h is None:
                                continue
                            vh = (v * h).reshape(h.shape[0], -1).sum(1)
                            for g in ctx.slices_of[n]:
                                r_ = ctx.rows[g]
                                tr[g] += float(vh.sum() if r_ is None else vh[r_].sum()) * nb / n_vec
                        del Hv
                del grads, gl, loss, lg, z
                torch.cuda.empty_cache()
        for p_ in params:
            p_.requires_grad_(False)
        torch.cuda.empty_cache()
        return {g: tr[g] / max(used, 1) for g in ctx.groups}


CLIPS = [round(x, 3) for x in np.linspace(0.25, 1.0, 16)]


@torch.no_grad()
def layer_inputs(ctx, S, max_rows=8192):
    """Inputs each tensor sees on the fp32 trajectory of the scoring puzzles (rows subsampled)."""
    got = {n: [] for n in ctx.tensors}
    hs = []
    for n in ctx.tensors:
        def hook(mod, args, n=n):
            x = args[0].detach().reshape(-1, args[0].shape[-1]) if ctx.family != "dt" else None
            if x is not None:
                idx = torch.randint(0, x.shape[0], (min(512, x.shape[0]),), device=x.device)
                got[n].append(x[idx].float())
        hs.append(ctx.tmod[n].register_forward_pre_hook(hook))
    ctx.set({})
    for c in S.chunks:
        Fn = ctx.rec(c["x"])
        z = Fn.z0()
        for _ in range(ctx.T):
            z = Fn(z)
        ctx.logits(Fn, z)
        ctx.m.inner.q_head(z[:, 0, 0]) if ctx.family != "dt" else None
    for h in hs:
        h.remove()
    out = {}
    for n, L in got.items():
        X = torch.cat(L) if L else None
        if X is not None and X.shape[0] > max_rows:
            X = X[torch.randperm(X.shape[0], device=X.device)[:max_rows]]
        out[n] = X
    return out


def choose_clips(ctx, S, lo, hi, how):
    """Per group, the clip ratio of the low format: 'mse' = min ||Q(W) - W||^2 (data-free), 'omse' = min
    ||X (Q(W) - W)^T||^2 on the layer's own inputs along the fp32 trajectory (standard data-aware PTQ), 'dT' =
    min delta_T with only this group at the low format and the rest high (label-free, unlabeled inputs)."""
    out = {}
    X = layer_inputs(ctx, S) if how == "omse" else None
    for g in ctx.groups:
        best, bv = 1.0, float("inf")
        for r in CLIPS:
            f = f"{lo}@{r}"
            if how == "mse":
                v = float((ctx.Wq(g, f) - ctx.W0[g]).float().pow(2).sum())
            elif how == "omse":
                Xg = X[ctx.tensor_of[g]]
                D = (ctx.Wq(g, f) - ctx.W0[g]).float()
                v = float((Xg @ D.reshape(D.shape[0], -1).T).pow(2).sum()) if Xg is not None else \
                    float(D.pow(2).sum())
            else:
                v = S.delta_T({**{h: hi for h in ctx.groups}, g: f})
            if v < bv:
                best, bv = r, v
        out[g] = best
    return out


def group_scores(ctx, S, lo, hi, sigma0, draws, want, log, out=None):
    """{score name: {group: value}}; larger = more damage at the low format. `out`: partial results to reuse."""
    out = out if out is not None else {}
    want = {w for w in want if w not in out or (w in ("lin", "dT_traj") and "lin" not in out)}
    base_hi = {g: hi for g in ctx.groups}
    if "dT_q" in want:
        out["dT_q"] = {g: S.delta_T({**base_hi, g: lo[g]}) for g in ctx.groups}
        log("dT_q", out["dT_q"])
    if "dT_noise" in want:
        sens, sig_eq = {}, {}
        for g in ctx.groups:
            W0 = ctx.W0[g]
            sig_eq[g] = float((ctx.Wq(g, lo[g]) - W0).float().pow(2).mean().sqrt() / W0.float().std())
            vals = []
            for d in range(draws):
                gen = torch.Generator(device=DEV).manual_seed(20_000 + 97 * d + ctx.groups.index(g))
                Wn = W0 + torch.randn(W0.shape, generator=gen, device=DEV) * (sigma0 * W0.std())
                key = (g, f"noise{sigma0}_{d}")
                ctx._qc[key] = Wn
                vals.append(S.delta_T({g: f"noise{sigma0}_{d}"}))
                del ctx._qc[key]
            sens[g] = float(np.mean(vals)) / sigma0
        out["S_group"] = sens
        out["sigma_eq"] = sig_eq
        out["dT_noise"] = {g: sens[g] * sig_eq[g] for g in ctx.groups}
        log("dT_noise", out["dT_noise"])
    if "kl" in want:
        out["kl"] = {g: S.kl({**base_hi, g: lo[g]}) for g in ctx.groups}
        log("kl", out["kl"])
    if "lin" in want or "dT_traj" in want:
        lg, lgm, dtr = {}, {}, {}
        for g in ctx.groups:
            lg[g], lgm[g], dtr[g] = S.lin_gain(g, lo[g])
        out["lin"], out["lin_mean"], out["dT_traj"] = lg, lgm, dtr
        log("lin", lg)
    err2 = {g: float((ctx.Wq(g, lo[g]) - ctx.W0[g]).float().pow(2).sum()) for g in ctx.groups}
    for name, kw in (("hawq", dict()), ("hawq_lbl", dict(labels=True)), ("hawq_full", dict(full=True))):
        if name not in want:
            continue
        try:
            kw2 = dict(kw)
            small = ctx.family != "dt" and ctx.m.inner.config.seq_len < 200
            kw2.update(max_inputs=64 if small else 16, mb=8 if small else 2)
            if name == "hawq_full":
                kw2.update(max_inputs=16 if small else 4, mb=2 if small else 1, n_vec=8)
            t = S.hessian_trace(**kw2)
            out[name + "_trace"] = t
            out[name] = {g: t[g] / ctx.numel[g] * err2[g] for g in ctx.groups}
            log(name, out[name])
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            for p_ in ctx.m.parameters():
                p_.requires_grad_(False)
            out[name + "_error"] = f"OOM: {str(e)[:200]}"
            print(f"  {name}: OOM, skipped", flush=True)
    return out


def greedy_dT(ctx, S, lo, hi, max_bits):
    """Promote one group at a time (lo -> hi): the one whose promotion lowers delta_T most per bit."""
    alloc = dict(lo)
    order, trace = [], [S.delta_T(alloc)]
    cost = {g: ctx.numel[g] * (fmt_bits(hi, ctx.W0[g]) - fmt_bits(lo[g], ctx.W0[g])) for g in ctx.groups}
    cur = trace[0]
    while len(order) < len(ctx.groups):
        best, bv = None, -1e30
        for g in ctx.groups:
            if g in order:
                continue
            d = S.delta_T({**alloc, g: hi})
            v = (cur - d) / cost[g]
            if v > bv:
                best, bv, bd = g, v, d
        order.append(best)
        alloc[best] = hi
        cur = bd
        trace.append(cur)
        if ctx.bits(alloc) > max_bits + 1e-9:
            break
    return order, trace


def order_by(score, ctx, lo, hi):
    cost = {g: ctx.numel[g] * (fmt_bits(hi, ctx.W0[g]) - fmt_bits(lo[g], ctx.W0[g])) for g in ctx.groups}
    return sorted(ctx.groups, key=lambda g: -(score[g] / max(cost[g], 1e-12)))


def alloc_for_budget(ctx, order, lo, hi, budget, prefix=False):
    """Keep groups high in `order` while the average bits stay <= budget: skip the ones that do not fit
    (knapsack fill), or, with prefix=True, stop at the first one that does not fit (tests the ranking)."""
    alloc = dict(lo)
    for g in order:
        trial = {**alloc, g: hi}
        if ctx.bits(trial) <= budget + 1e-9:
            alloc = trial
        elif prefix:
            break
    return alloc


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--lo", default="w4t")
    ap.add_argument("--hi", default="w8t")
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000, help="evaluation inputs: test indices 0..n-1")
    ap.add_argument("--n_score", type=int, default=256, help="unlabeled scoring inputs: test indices n..n+n_score-1")
    ap.add_argument("--score_offset", type=int, default=-1, help="first scoring index (default: n)")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--sigma0", type=float, default=0.05)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--scores", default="dT_q,dT_noise,kl,lin,hawq,hawq_lbl,hawq_full")
    ap.add_argument("--budgets", default="", help="average-bit targets; default lo+{0.1,0.25,0.5,1,2}")
    ap.add_argument("--uniform", default="", help="uniform baseline formats")
    ap.add_argument("--random_draws", type=int, default=5)
    ap.add_argument("--ckpt", default="", help="TRM: a trained variant's step_N file (arch from its all_config.yaml)")
    ap.add_argument("--split", type=int, default=1, help="groups = k output-channel slices per tensor")
    ap.add_argument("--singles", type=int, default=1, help="evaluate every tensor alone low / alone high")
    ap.add_argument("--xfer", default="", help="another run's output JSON: evaluate its allocation orders here (seed transfer)")
    ap.add_argument("--clips_from", default="", help="use the clip ratios chosen in another run (seed transfer)")
    ap.add_argument("--clip", default="none", choices=["none", "mse", "omse", "dT"],
                    help="clip ratio of the low format per group: none (absmax), mse (data-free), dT (label-free)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    if a.task in C.DT_TASKS:
        T = a.T or C.DT_TASKS[a.task]["T"]
        batch = a.batch or C.DT_TASKS[a.task]["batch"]
        so = a.score_offset if a.score_offset >= 0 else a.n
        inp, lab = C.load_dt(a.task, so + a.n_score)
    else:
        T = a.T or 16
        t = bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task]
        batch = a.batch or (128 if t["seq"] < 200 else 16)
        so = a.score_offset if a.score_offset >= 0 else a.n
        inp, lab = bs.load_qa(t["data"], so + a.n_score)
    ev_inp, ev_lab = inp[:a.n], lab[:a.n]
    sc_inp, sc_lab = inp[so:so + a.n_score], lab[so:so + a.n_score]
    assert len(sc_inp) == a.n_score, f"only {len(sc_inp)} scoring inputs available"
    assert so >= a.n, "scoring inputs must be disjoint from the evaluation inputs"

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.load(open(out)) if out.exists() else {}
    blob.update({"task": a.task, "lo": a.lo, "hi": a.hi, "clip": a.clip, "T": T, "n": a.n, "n_score": a.n_score, "score_offset": so,
                 "sigma0": a.sigma0, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__})
    blob.setdefault("evals", {})

    def save():
        tmp = out.with_suffix(".tmp")
        json.dump(blob, open(tmp, "w"))
        tmp.replace(out)

    t0 = time.time()
    if a.ckpt:
        bs.CKPT_OVERRIDE = a.ckpt
        blob["ckpt"] = a.ckpt
    ctx = Ctx(a.task, T, batch, split=a.split)
    blob["split"] = a.split
    blob["groups"] = [{"name": g, "numel": ctx.numel[g], "shape": list(ctx.W0[g].shape),
                       "bits_lo": fmt_bits(a.lo, ctx.W0[g]), "bits_hi": fmt_bits(a.hi, ctx.W0[g])} for g in ctx.groups]
    print(f"{a.task} ({ctx.family}): {len(ctx.groups)} groups, {ctx.N} weights, T={T}, eval n={a.n}, "
          f"score n={a.n_score} (offset {so}), lo={a.lo} hi={a.hi}", flush=True)

    # fp32 reference on the evaluation set
    ctx.set({})
    ref_c, ref_cell, ref_z = rollout(ctx, ev_inp, ev_lab)
    blob["fp32"] = {"exact": float(ref_c.mean()), "cell": ref_cell, "correct": ref_c.astype(int).tolist()}
    print(f"  fp32 exact {ref_c.mean()*100:.1f} cell {ref_cell*100:.2f} ({time.time()-t0:.0f}s)", flush=True)

    def evaluate(key, alloc, meta):
        if key in blob["evals"]:
            return blob["evals"][key]
        t1 = time.time()
        ctx.set(alloc)
        c, cell, z = rollout(ctx, ev_inp, ev_lab)
        ctx.set({})
        r = {"exact": float(c.mean()), "cell": cell, "fidelity_H": fid(z, ref_z), "bits": ctx.bits(alloc),
             "n_hi": sum(1 for g in ctx.groups if alloc.get(g) == a.hi), "hi_groups": sorted(g for g in ctx.groups if alloc.get(g) == a.hi),
             "correct": c.astype(int).tolist(), "secs": time.time() - t1, **meta}
        blob["evals"][key] = r
        save()
        print(f"  {key:40s} bits {r['bits']:.3f} n_hi {r['n_hi']:2d} exact {r['exact']*100:5.1f} cell {cell*100:5.2f} "
              f"fid {r['fidelity_H']:.4f} ({r['secs']:.0f}s)", flush=True)
        return r

    # uniform baselines
    lo_b = parse_fmt(a.lo)["bits"]
    uni = [u for u in (a.uniform.split(",") if a.uniform else
                       [f"w{lo_b}t", f"w{lo_b+1}t", f"w{lo_b+2}t", "w8t", f"w{lo_b}c", f"w{lo_b}g32", f"w{lo_b}g128",
                        f"mx{lo_b}", f"w{lo_b+1}c", f"w{lo_b}s4", f"w{lo_b}s16"]) if u]
    for u in uni:
        evaluate(f"uniform:{u}", {g: u for g in ctx.groups}, {"method": "uniform", "fmt": u})

    S = None
    want = set(a.scores.split(","))
    ts = time.time()

    def log(name, d):
        top = sorted(d, key=lambda g: -d[g])[:5]
        print(f"  score {name}: top {[(g, round(d[g], 5)) for g in top]} ({time.time()-ts:.0f}s)", flush=True)

    # low format per group (optionally with a chosen clip ratio)
    if a.clips_from:
        src = json.load(open(a.clips_from))
        blob["clips"] = src["clips"]
        blob["clips_from"] = a.clips_from
        LO = {g: f"{a.lo}@{blob['clips'][g]}" for g in ctx.groups}
        evaluate(f"uniform_lo_clip:xfer", dict(LO), {"method": "uniform", "fmt": f"{a.lo}+clip_from_{Path(a.clips_from).stem}"})
    elif a.clip == "none":
        LO = {g: a.lo for g in ctx.groups}
    else:
        if "clips" not in blob:
            S = S or ScoreSet(ctx, sc_inp, sc_lab)
            blob["clips"] = choose_clips(ctx, S, a.lo, a.hi, a.clip)
            save()
        LO = {g: f"{a.lo}@{blob['clips'][g]}" for g in ctx.groups}
        print(f"  clips ({a.clip}): {blob['clips']}", flush=True)
        evaluate(f"uniform_lo_clip:{a.clip}", dict(LO), {"method": "uniform", "fmt": f"{a.lo}+clip_{a.clip}"})
    blob["LO"] = LO

    # scores (on the unlabeled scoring inputs)
    if "scores" not in blob or set(blob.get("score_list", [])) != want:
        S = S or ScoreSet(ctx, sc_inp, sc_lab)
        part = blob.setdefault("scores_partial", {})

        def log2(name, d):
            log(name, d)
            save()
        sc = group_scores(ctx, S, LO, a.hi, a.sigma0, a.draws, want, log2, out=part)
        max_b = fmt_bits(a.hi, next(iter(ctx.W0.values())))
        gorder, gtrace = greedy_dT(ctx, S, LO, a.hi, max_b)
        sc["dT_greedy_order"], sc["dT_greedy_trace"] = gorder, gtrace
        print(f"  dT_greedy order {gorder[:6]} delta_T trace {[round(v, 4) for v in gtrace[:8]]}", flush=True)
        sc["delta_T_uniform_lo"] = S.delta_T(dict(LO))
        sc["delta_T_uniform_hi"] = S.delta_T({g: a.hi for g in ctx.groups})
        sc["secs"] = time.time() - ts
        blob["scores"], blob["score_list"] = sc, sorted(want)
        save()
    del S
    torch.cuda.empty_cache()
    sc = blob["scores"]

    # allocations at matched budgets
    budgets = [float(b) for b in a.budgets.split(",")] if a.budgets else [lo_b + d for d in (0.1, 0.25, 0.5, 1.0, 2.0)]
    orders = {}
    for name in ("dT_q", "dT_noise", "kl", "lin", "dT_traj", "hawq", "hawq_lbl", "hawq_full"):
        if name in sc:
            orders[name] = order_by(sc[name], ctx, LO, a.hi)
    orders["dT_greedy"] = sc["dT_greedy_order"] + [g for g in ctx.groups if g not in sc["dT_greedy_order"]]
    orders["small_first"] = sorted(ctx.groups, key=lambda g: ctx.numel[g])
    rng = np.random.default_rng(1234)
    for d in range(a.random_draws):
        orders[f"random{d}"] = [ctx.groups[i] for i in rng.permutation(len(ctx.groups))]
    blob["orders"] = orders
    # ground truth for the score comparison: accuracy with one tensor alone low (rest high) / alone high
    if a.singles and len(ctx.groups) <= 40:
        for g in ctx.groups:
            evaluate(f"single_lo:{g}", {**{h: a.hi for h in ctx.groups}, g: LO[g]}, {"method": "single_lo", "group": g})
            evaluate(f"single_hi:{g}", {**LO, g: a.hi}, {"method": "single_hi", "group": g})
    if a.xfer:
        src = json.load(open(a.xfer))
        assert [g["name"] for g in src["groups"]] == ctx.groups, "transfer needs the same tensor names"
        for name in ("dT_q", "dT_noise", "kl", "lin", "hawq", "dT_greedy"):
            if name in src.get("orders", {}):
                orders[f"xfer_{name}"] = src["orders"][name]
        blob["xfer_from"] = a.xfer
    for B in budgets:
        for name, order in orders.items():
            alloc = alloc_for_budget(ctx, order, LO, a.hi, B)
            key = "alloc:" + ",".join(sorted(g for g in ctx.groups if alloc[g] == a.hi))
            evaluate(key, alloc, {"method": "mixed"})
            blob.setdefault("curve", {}).setdefault(name, {})[f"{B:.3f}"] = key
            alloc = alloc_for_budget(ctx, order, LO, a.hi, B, prefix=True)
            key = "alloc:" + ",".join(sorted(g for g in ctx.groups if alloc[g] == a.hi))
            evaluate(key, alloc, {"method": "mixed"})
            blob.setdefault("curve_prefix", {}).setdefault(name, {})[f"{B:.3f}"] = key
        save()
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
