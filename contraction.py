#!/usr/bin/env python3
"""Angle A mechanism (agent X2): is the trained recursion map locally contractive, and does the
perturbation that collapses accuracy push its contraction factor to >= 1?

The recursion map is one supervision step F(z; x) of TRM/HRM (the paper's unit of depth, n_sup),
acting on the stacked latent state z = (z_H, z_L); the answer after T steps is lm_head(z_H(T)).
For fp32 and for every compressed copy of the model, on its OWN trajectory z_0 .. z_{T+1}, per input:

  d[t]      ||z_t - z_{t-1}||, t = 1..T+1        -> empirical contraction q_t = d[t+1] / d[t]
  growth[t] ||J(z_t) v_t|| with v_{t+1} = J(z_t) v_t / ||.|| (tangent propagation of one random
            direction along the trajectory; its late geometric mean is the finite-time Lyapunov
            number Lambda: < 1 perturbations die out along the trajectory, >= 1 they do not)
  rho[p]    spectral radius of J(z_p), power iteration with JVPs (geometric mean of the last
            `tail` growth factors), p in --points (default 4 and T)
  sigma[p]  spectral norm of J(z_p), power iteration on J^T J (JVP + VJP), p in --sigma_points
and, against the fp32 run on the same inputs:
  disp      ||z_T - z_T^fp32|| / ||z_T^fp32||          (displacement of the final state)
  phi_H     cos(z_H(T), z_H^fp32(T))                     (the paper's fidelity, per input)
  delta[t]  ||F~(z_t^fp32) - F(z_t^fp32)|| / ||z_t^fp32|| (the one-step perturbation of the map,
            measured on the fp32 trajectory; t = 0..T)
The Banach picture predicts disp ~ delta[T] / (1 - c) inside the basin (c the contraction) and a
break (c >= 1, disp ~ O(1)) outside it.  J is the Jacobian of F with respect to z (weights fixed,
input embedding fixed); jvp/vjp by torch.func, attention on the math SDPA kernel.

    cd ~/edgetrm_src && python contraction.py --task sudoku_mlp --n 256 \
        --cells fp32,w8c,w4c,w3g32,w4t,noise0.05:3,noise0.1:3,noise0.15:3,noise0.2:3 \
        --out ~/breadth_results/contraction_sudoku_mlp.json
    python contraction.py --analyze results/breadth/babel/contraction_*.json --summary out.json

Resumable: records are keyed by (spec, draw) and skipped when already in --out.
"""
import argparse
import glob
import json
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ----------------------------------------------------------------------------- helpers
def _torch():
    import torch
    return torch


@contextmanager
def sdpa_math():
    """Math SDPA kernel: composite ops, so forward-mode AD and double use of the graph work."""
    torch = _torch()
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        with sdpa_kernel(SDPBackend.MATH):
            yield
    except ImportError:
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            yield


def bnorm(t):
    return t.flatten(1).norm(dim=1)


def unit(t):
    return t / bnorm(t).clamp_min(1e-30).view(-1, *([1] * (t.dim() - 1)))


class Recursion:
    """One supervision step of TRM / HRM as a pure function of the stacked state z = (z_H, z_L).
    Mirrors TinyRecursiveReasoningModel_ACTV1_Inner.forward / HRM's without the no_grad split."""

    def __init__(self, m, family, x_tok):
        torch = _torch()
        self.inner = m.inner
        cfg = m.inner.config
        self.H, self.L, self.family = cfg.H_cycles, cfg.L_cycles, family
        self.pl = self.inner.puzzle_emb_len
        pid = torch.zeros(x_tok.shape[0], dtype=torch.int32, device=x_tok.device)
        with torch.no_grad():
            self.x = self.inner._input_embeddings(x_tok, pid).float()
        self.cs = self.inner.rotary_emb() if hasattr(self.inner, "rotary_emb") else None
        self.hl = self.inner.L_level if family == "trm" else self.inner.H_level

    def z0(self):
        torch = _torch()
        B, S, D = self.x.shape
        return torch.stack([self.inner.H_init.float().expand(B, S, D),
                            self.inner.L_init.float().expand(B, S, D)], 1).contiguous()

    def __call__(self, z):
        torch = _torch()
        zH, zL = z[:, 0], z[:, 1]
        for _ in range(self.H):
            for _ in range(self.L):
                zL = self.inner.L_level(zL, zH + self.x, cos_sin=self.cs)
            zH = self.hl(zH, zL, cos_sin=self.cs)
        return torch.stack([zH, zL], 1)

    def answer_state(self, z):
        return z[:, 0]                               # z_H: what the read-out head sees

    def flip_ratios(self, zref, deltas):
        """Decoder-margin test of a state displacement: per puzzle, min over answer cells of
        mu_i / |(w_top1 - w_top2) . Delta_H,i| with mu_i the fp32 logit margin at zref (the read-out is
        linear in z_H, so this is exact to first order in the head). < 1: the displacement flips a cell.
        Returns {name: (min ratio per puzzle, fraction of cells flipped)}."""
        torch = _torch()
        with torch.no_grad():
            W = self.inner.lm_head.weight.float()
            lg = self.inner.lm_head(zref[:, 0])[:, self.pl:].float()
            top = lg.topk(2, dim=-1)
            mu = top.values[..., 0] - top.values[..., 1]
            wd = W[top.indices[..., 0]] - W[top.indices[..., 1]]            # (B, S, D)
            res = {}
            for name, D in deltas.items():
                ch = (wd * D[:, 0][:, self.pl:].float()).sum(-1).abs()        # (B, S)
                ratio = mu / ch.clamp_min(1e-12)
                res[name] = (ratio.amin(-1), (ratio < 1).float().mean(-1))
        return res

    def readout(self, z, y):
        """puzzle-exact correct, cell accuracy, halting-head logit, predicted tokens, and the per-cell
        logit margin (top1 - top2) of the read-out from z (min and mean over answer cells)."""
        torch = _torch()
        with torch.no_grad():
            lg = self.inner.lm_head(z[:, 0])[:, self.pl:].float()
            pred = lg.argmax(-1)
            mask = y != 0
            cc = (pred == y) & mask
            correct = cc.sum(-1) == mask.sum(-1)
            cell = cc.sum(-1).float() / mask.sum(-1).clamp_min(1).float()
            qh = self.inner.q_head(z[:, 0, 0]).float()[..., 0]
            top2 = lg.topk(2, dim=-1).values
            mg = (top2[..., 0] - top2[..., 1])
            mmin = torch.where(mask, mg, torch.full_like(mg, float("inf"))).amin(-1)
            mmean = (mg * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        return {"correct": correct, "cellacc": cell, "qhalt": qh, "pred": pred, "margin_min": mmin, "margin_mean": mmean}


class DTRecursion:
    """Deep-thinking recall net (Bansal et al. 2022): z_0 = projection(x), z_{t+1} = recur_block([z_t, x]),
    answer = head(z_T); mirrors DTNet.forward. Correct = every cell of the masked prediction matches."""

    def __init__(self, net, x, period=1):
        torch = _torch()
        self.net, self.x, self.period = net, x, period
        with torch.no_grad():
            self._z0 = net.projection(x)

    def z0(self):
        return self._z0.clone()

    def __call__(self, z):
        torch = _torch()
        for _ in range(self.period):                     # period 2: analyse G o G (DT nets end in 2-cycles)
            z = self.net.recur_block(torch.cat([z, self.x], 1))
        return z

    def answer_state(self, z):
        return z

    def readout(self, z, y):
        torch = _torch()
        with torch.no_grad():
            out = self.net.head(z).float()
            B = z.shape[0]
            pred = out.argmax(1).view(B, -1) * self.x.max(1)[0].view(B, -1)
            tg = y.view(B, -1)
            eq = pred == tg
            mg = (out[:, 1] - out[:, 0]).abs().view(B, -1)
        return {"correct": eq.all(1), "cellacc": eq.float().mean(1), "qhalt": torch.zeros(B, device=z.device),
                "pred": pred, "margin_min": mg.amin(1), "margin_mean": mg.mean(1)}


def fjvp(Fn, z, v):
    torch = _torch()
    with sdpa_math():
        out, u = torch.func.jvp(Fn, (z,), (v,))
    return out.detach(), u.detach()


def power_rho(Fn, z, v, K, tail):
    """Spectral radius of J(z) by power iteration (JVPs); geometric mean of the last `tail`
    growth factors (robust to a complex dominant pair). Returns (rho, std of the tail log-growth,
    the final direction, the whole log-growth trace (K, B))."""
    torch = _torch()
    logs = []
    for _ in range(K):
        _, u = fjvp(Fn, z, v)
        n = bnorm(u)
        logs.append(n.clamp_min(1e-30).log())
        v = unit(u)
    L = torch.stack(logs[-tail:])
    return L.mean(0).exp(), L.std(0), v, torch.stack(logs)


def nl_growth(Fn, z, v, eps, ks):
    """Nonlinear check of the linearisation: kick z by eps*||z|| along v and iterate the true map;
    returns {k: ||F^k(z + kick) - F^k(z)|| / ||kick||}. Growth > 1 at large k = the kick is not absorbed."""
    torch = _torch()
    scale = eps * bnorm(z)
    a, b = z, z + v * scale.view(-1, *([1] * (z.dim() - 1)))
    res = {}
    with torch.no_grad(), sdpa_math():
        for k in range(1, max(ks) + 1):
            a, b = Fn(a), Fn(b)
            if k in ks:
                res[k] = bnorm(b - a) / scale
    return res


def power_sigma(Fn, z, v, K):
    """Spectral norm of J(z) by power iteration on J^T J (JVP forward, VJP back)."""
    torch = _torch()
    with sdpa_math():
        _, vjp_fn = torch.func.vjp(Fn, z)
    s = None
    for _ in range(K):
        _, u = fjvp(Fn, z, v)
        s = bnorm(u)
        (w,) = vjp_fn(u)
        v = unit(w.detach())
    del vjp_fn
    return s


# ----------------------------------------------------------------------------- one chunk
NL_EPS, NL_KS = 1e-4, (1, 4, 8, 16)
def kept_steps(T):
    """Steps t (0..T) at which per-step quantities are stored: all of them up to 40 steps, else a
    coarse grid (1..10, then 20, 30, 40, 60, 80, 100, 150, 200, ...) plus T-1 and T."""
    if T + 1 <= 41:
        return list(range(T + 1))
    ks = set(range(0, 11)) | {20, 30, 40, 60, 80} | set(range(100, T + 1, 50)) | {T - 1, T}
    return sorted(k for k in ks if k <= T)


SUBSAMPLED = ("disp_t", "phiH_t", "delta")          # already on kept_steps(T); not re-indexed by to_lists


def _settle(r, tol):
    """First step t (1-based) from which r_t stays below tol; len(r)+1 if it never does. r: (steps, B)."""
    torch = _torch()
    idx = torch.arange(r.shape[0], device=r.device).view(-1, 1).expand_as(r)
    last = torch.where(r >= tol, idx, torch.full_like(idx, -1)).amax(0)
    return (last + 2).float()


def eval_chunk(m, family, x, y, T, points, sigma_points, K_rho, K_sig, tail, seed, ref=None, keep_states=False,
               m_ref=None, K_lin=60, period=1, xs_steps=16):
    """Trajectory, tangent growth, Jacobian spectra, read-out; displacement vs `ref` (fp32 states at the
    kept steps and one step after each, on CPU). Only the states at the Jacobian points stay on the GPU.
    With `m_ref` (the fp32 model) also: linear-response displacement through the fp32 resolvent,
    cross-seeding (basin identity) and the decoder-margin flip ratios."""
    torch = _torch()
    mk = (lambda net: DTRecursion(net, x, period)) if family == "dt" else (lambda net: Recursion(net, family, x))
    Fn = mk(m)
    dev = x.device
    ks = kept_steps(T)
    need = set(ks) | {k + 1 for k in ks}                 # states the reference must keep (for delta_t)
    jpts = set(points) | set(sigma_points) | {T, T + 1}
    z = Fn.z0()
    g = torch.Generator(device=dev).manual_seed(seed)
    v = unit(torch.randn(z.shape, generator=g, device=dev))
    keep = {0: z.cpu()} if keep_states else None
    onpts, tang = {}, {}
    if 0 in jpts:
        onpts[0], tang[0] = z, v
    growth, dl, d2l, disp_t, phi_t = [], [], [], [], []
    ref_on = (lambda t: ref["states"][t].to(dev)) if ref is not None else None
    prev, correct1 = None, None
    for t in range(T + 1):                               # z_1 .. z_{T+1}
        zn, u = fjvp(Fn, z, v)
        growth.append(bnorm(u))
        dl.append(bnorm(zn - z))
        if prev is not None:
            d2l.append(bnorm(zn - prev))                 # ||z_{t+1} - z_{t-1}||  (2-cycle test)
        v = unit(u)
        prev, z = z, zn
        if t == 0:
            correct1 = Fn.readout(z, y)["correct"]       # T1 substitution check: decode z_1
        if (t + 1) in jpts:
            onpts[t + 1], tang[t + 1] = z, v
        if keep_states and (t + 1) in need:
            keep[t + 1] = z.cpu()
        if ref is not None and t in ks:                  # displacement from fp32 at depth t+1
            rz = ref_on(t + 1)
            disp_t.append(bnorm(z - rz) / bnorm(rz))
            phi_t.append(torch.nn.functional.cosine_similarity(Fn.answer_state(z).flatten(1),
                                                               Fn.answer_state(rz).flatten(1), dim=-1))
    zT, zT1 = onpts[T], onpts[T + 1]
    ro = Fn.readout(zT, y)
    d = torch.stack(dl)                                  # (T+1, B): d[t-1] = ||z_t - z_{t-1}||
    d2 = torch.stack(d2l)                                # (T, B):   d2[t-1] = ||z_{t+1} - z_{t-1}||
    G = torch.stack(growth)                              # (T+1, B)
    zn_T = bnorm(zT)
    r = d / zn_T
    out = {"correct": ro["correct"].int(), "correct1": correct1.int(), "cellacc": ro["cellacc"], "qhalt": ro["qhalt"],
           "znorm": zn_T, "margin_min": ro["margin_min"], "margin_mean": ro["margin_mean"],
           "d": d, "growth": G,
           "lam_late": torch.exp(torch.log(G[T // 2:].clamp_min(1e-30)).mean(0)),     # finite-time Lyapunov number
           "qT": d[T] / d[T - 1].clamp_min(1e-30),                                     # ||z_{T+1}-z_T|| / ||z_T-z_{T-1}||
           "resT": d[T - 1] / zn_T,                                                    # ||z_T-z_{T-1}|| / ||z_T||
           "res2T": d2[T - 1] / zn_T,                                                  # ||z_{T+1}-z_{T-1}|| / ||z_T||
           "settle01": _settle(r[:T], 0.01), "settle05": _settle(r[:T], 0.05)}        # settle time (critical slowing)
    dout = unit(zT1 - zT)
    _, u = fjvp(Fn, zT, dout)
    out["gamma_nat"] = bnorm(u)                          # gain along the direction of travel
    with torch.no_grad(), sdpa_math():                   # path independence (Anil et al. 2022): random initial carry
        z0 = Fn.z0()
        w = torch.randn(z0.shape, generator=g, device=dev)
        if family == "dt":                               # DT carry is a ReLU map: non-negative, same norm as z_0
            w = w.clamp_min(0)
            w = w * (bnorm(z0) / bnorm(w).clamp_min(1e-30)).view(-1, *([1] * (w.dim() - 1)))
        for _ in range(T):
            w = Fn(w)
        rw = Fn.readout(w, y)
        out["pi_dist"] = bnorm(w - zT) / zn_T
        out["pi_agree"] = (rw["pred"] == ro["pred"]).float().mean(-1)
        out["pi_correct"] = rw["correct"].int()
    for p in points:
        rr, cv, vdom, trace = power_rho(Fn, onpts[p], tang[p], K_rho, tail)
        out[f"rho{p}"], out[f"rhocv{p}"] = rr, cv
        if p == T:
            out["rho_trace"] = trace.exp()                                             # (K, B) growth per iteration
            for tag, vv in (("dom", vdom), ("rand", unit(torch.randn(vdom.shape, generator=g, device=dev))),
                            ("nat", dout)):
                for k, val in nl_growth(Fn, zT, vv, NL_EPS, NL_KS).items():
                    out[f"nl_{tag}_{k}"] = val
    for p in sigma_points:
        out[f"sigma{p}"] = power_sigma(Fn, onpts[p], tang[p], K_sig)
    if ref is not None:
        with torch.no_grad():
            rT, rT1 = ref_on(T), ref_on(T + 1)
            out["agree"] = (ro["pred"] == ref["pred"].to(dev)).float().mean(-1)        # cells matching fp32's answer
            out["disp_t"] = torch.stack(disp_t)                                        # (len(ks), B), t = k + 1
            out["phiH_t"] = torch.stack(phi_t)
            out["disp"] = bnorm(zT - rT) / bnorm(rT)
            aT, raT = Fn.answer_state(zT), Fn.answer_state(rT)
            out["dispH"] = bnorm(aT - raT) / bnorm(raT)
            out["phi_H"] = torch.nn.functional.cosine_similarity(aT.flatten(1), raT.flatten(1), dim=-1)
            out["phi_full"] = torch.nn.functional.cosine_similarity(zT.flatten(1), rT.flatten(1), dim=-1)
            dl2 = []
            with sdpa_math():
                for k in ks:                             # one-step map perturbation on the fp32 trajectory, t = k
                    a0, a1 = ref_on(k), ref_on(k + 1)
                    dl2.append(bnorm(Fn(a0) - a1) / bnorm(a0))
                dv = Fn(rT) - rT1                        # the vector delta_T
            out["delta"] = torch.stack(dl2)
            out["deltaT"] = out["delta"][ks.index(T)]
        if m_ref is not None:
            Fr = mk(m_ref)
            # linear response: Delta = J_fp32(s_T) Delta + delta_T, by Neumann iteration (converges iff rho < 1)
            Dl, conv = dv.clone(), None
            for _ in range(K_lin):
                _, JD = fjvp(Fr, rT, Dl)
                Dn = JD + dv
                conv = bnorm(Dn - Dl) / bnorm(Dn).clamp_min(1e-30)
                Dl = Dn
            out["Dlin"] = bnorm(Dl) / bnorm(rT)
            out["lin_conv"] = conv
            with torch.no_grad(), sdpa_math():
                # cross-seeding: fp32 map from the perturbed end state vs from its own; perturbed map from fp32's
                a, b, c, e = zT.clone(), rT.clone(), rT.clone(), zT.clone()
                for _ in range(xs_steps):
                    a, b, c, e = Fr(a), Fr(b), Fn(c), Fn(e)
                out["xs_ref_dist"] = bnorm(a - b) / bnorm(b)            # fp32 map: from s~_T vs from s_T
                rox, rob = Fr.readout(a, y), Fr.readout(b, y)
                out["xs_ref_correct"] = rox["correct"].int()             # fp32 map decodes right from s~_T
                out["xs_ref_agree"] = (rox["pred"] == rob["pred"]).float().mean(-1)
                out["xs_own_dist"] = bnorm(c - e) / bnorm(e)            # perturbed map: from s_T vs from s~_T
                out["xs_own_correct"] = Fn.readout(c, y)["correct"].int()
            if family != "dt":
                fr = Fr.flip_ratios(rT, {"lin": Dl, "meas": zT - rT})
                out["flip_lin"], out["flipfrac_lin"] = fr["lin"]
                out["flip_meas"], out["flipfrac_meas"] = fr["meas"]
    kept = {"states": keep, "pred": ro["pred"].cpu()} if keep_states else None
    return out, kept


def arnoldi_top(m, family, x, T, k=8, ncv=40, period=1):
    """Top-k eigenvalues of J(s_T) per input (batch of 1 each) by ARPACK on a JVP LinearOperator:
    checks the power-iteration rho for complex pairs and eigenvalues near -1."""
    torch = _torch()
    import scipy.sparse.linalg as sla
    res = []
    for i in range(x.shape[0]):
        xi = x[i:i + 1]
        Fn = DTRecursion(m, xi, period) if family == "dt" else Recursion(m, family, xi)
        z = Fn.z0()
        with torch.no_grad(), sdpa_math():
            for _ in range(T):
                z = Fn(z)
        shape, N = z.shape, z.numel()
        cnt = [0]

        def mv(vec):
            cnt[0] += 1
            vv = torch.from_numpy(np.asarray(vec, dtype=np.float32).reshape(shape)).to(z.device)
            _, u = fjvp(Fn, z, vv)
            return u.reshape(-1).double().cpu().numpy()
        op = sla.LinearOperator((N, N), matvec=mv, dtype=np.float64)
        try:
            vals = sla.eigs(op, k=k, which="LM", ncv=ncv, tol=1e-3, maxiter=20, return_eigenvectors=False)
            ok = True
        except sla.ArpackNoConvergence as err:
            vals, ok = err.eigenvalues, False
        vals = sorted(vals, key=lambda c: -abs(c))
        res.append({"i": i, "converged": ok, "matvecs": cnt[0], "eig": [[float(c.real), float(c.imag)] for c in vals]})
    return res


def to_lists(o, T):
    res, ks = {}, kept_steps(T)
    for k, v in o.items():
        a = v.detach().float().cpu().numpy()
        if a.ndim == 2 and k == "rho_trace":        # (K, B) power-iteration growth factors
            a = a.T
        elif a.ndim == 2 and k in SUBSAMPLED:
            a = a.T
        elif a.ndim == 2:                            # (steps, B) -> per input list of kept steps
            a = a[[k for k in ks if k < a.shape[0]]].T
        res[k] = np.round(a, 6).tolist()
    return res


def cat_chunks(chunks):
    torch = _torch()
    keys = chunks[0].keys()
    res = {}
    for k in keys:
        v = chunks[0][k]
        res[k] = torch.cat([c[k] for c in chunks], dim=-1 if v.dim() == 2 else 0)
    return res


# ----------------------------------------------------------------------------- cells
def parse_cells(s):
    """'fp32,w4t,noise0.1:3,noise0.5#2' -> [('fp32',0), ('w4t',0), ('noise0.1',0), ('noise0.1',1), ('noise0.1',2),
    ('noise0.5',2)]  (spec:k = draws 0..k-1; spec#k = draw k only)"""
    cells = []
    for tok in [t.strip() for t in s.split(",") if t.strip()]:
        if "#" in tok:
            spec, k = tok.split("#")
            cells.append((spec, int(k)))
        elif ":" in tok:
            spec, k = tok.split(":")
            cells += [(spec, i) for i in range(int(k))]
        else:
            cells.append((tok, 0))
    return cells


DT_TASKS = {"dt_maze9": dict(size=9, T=100, batch=64), "dt_maze13": dict(size=13, T=100, batch=64),
            "dt_maze33": dict(size=33, T=300, batch=32)}
DT_DIR = Path(__import__("os").environ.get("X2_DT_DIR", Path.home() / "edgetrm_ckpt" / "dt_mazes"))  # 256-instance subsets + ckpt


def load_dt(task, n):
    s = DT_TASKS[task]["size"]
    x = np.load(DT_DIR / f"maze{s}_inputs.npy")[:n].astype(np.float32)
    y = np.load(DT_DIR / f"maze{s}_solutions.npy")[:n].astype(np.int64)
    return x, y


def build_dt():
    torch = _torch()
    from dt_models.dt_net_2d import dt_net_recall_2d      # copy of deep-thinking's deepthinking/models
    net = dt_net_recall_2d(128, in_channels=3)             # training-rusty-Tayla: dt_net_recall_2d, width 128
    sd = torch.load(DT_DIR / "model_best.pth", map_location="cpu")
    sd = sd.get("net", sd)
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    return net.eval()


def dt_quantize_(net, bits, gran, sigma=None):
    """Identical to dt_breadth.quantize_ (every Conv weight; t / c / a / gN / noise)."""
    torch = _torch()
    F = torch.nn.functional
    with torch.no_grad():
        for mod in net.modules():
            if not isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d)):
                continue
            W = mod.weight.data
            if gran == "noise":
                mod.weight.data = W + torch.randn_like(W) * (sigma * W.std())
                continue
            o = W.shape[0]
            Wf = W.reshape(o, -1)
            if gran == "a":
                qmax = 2 ** bits - 1
                wmin, wmax = Wf.amin(1, keepdim=True), Wf.amax(1, keepdim=True)
                scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
                zp = torch.round(-wmin / scale)
                Wq = (torch.clamp(torch.round(Wf / scale) + zp, 0, qmax) - zp) * scale
            else:
                qmax = 2 ** (bits - 1) - 1
                if gran == "t":
                    sc = (Wf.abs().max() / qmax).clamp_min(1e-8)
                    Wq = torch.round(Wf / sc).clamp(-qmax, qmax) * sc
                elif gran == "c":
                    sc = (Wf.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-8)
                    Wq = torch.round(Wf / sc).clamp(-qmax, qmax) * sc
                else:
                    g = int(gran[1:])
                    i_ = Wf.shape[1]
                    pad = (-i_) % g
                    Wp = F.pad(Wf, (0, pad)).view(o, -1, g)
                    sc = (Wp.abs().amax(2, keepdim=True) / qmax).clamp_min(1e-8)
                    Wq = (torch.round(Wp / sc).clamp(-qmax, qmax) * sc).view(o, -1)[:, :i_]
            mod.weight.data = Wq.reshape(W.shape)


GROUPS = {                                               # weight groups for the per-group sensitivity
    "mixer": lambda n: ".self_attn." in n or ".mlp_t." in n,     # token mixer: attention or sequence-MLP
    "mlp": lambda n: ".mlp." in n,                               # channel MLP (SwiGLU)
    "head": lambda n: "lm_head" in n or "q_head" in n,           # read-out and halting heads
    "core": lambda n: ".self_attn." in n or ".mlp_t." in n or ".mlp." in n,
}


def make_model(task, cell, T, batch):
    """spec[@group]: the perturbation applied to every CastedLinear, or only to one weight group."""
    torch = _torch()
    import breadth_sweep as bs
    spec, draw = cell
    group = None
    if "@" in spec:
        spec, group = spec.split("@")
    q = dict(bs.parse_quant(spec), group=group)
    if task in DT_TASKS:
        m, meta = build_dt(), {"H_cycles": None, "L_cycles": None}
        quant = dt_quantize_
    else:
        m, meta = bs.build(task, None, T, batch)
        quant = bs.quantize_weights_
    if q["gran"] is not None:
        target = m
        if group is not None:
            assert task not in DT_TASKS, "weight groups are defined for TRM/HRM"
            mods = [mod for name, mod in m.named_modules()
                    if mod.__class__.__name__ == "CastedLinear" and GROUPS[group](name)]
            assert mods, f"no CastedLinear in group {group}"
            q["n_linears"] = len(mods)
            target = torch.nn.ModuleList(mods)
        if q["gran"] == "noise":
            torch.manual_seed(10_000 + 1000 * draw + int(round(q["sigma"] * 1000)))
        quant(target, q["bits"], q["gran"], q.get("sigma"))
    assert q.get("act") is None and q.get("anoise") is None, "activation perturbations make F stochastic / flat"
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(bs.DEV).eval(), meta, q


def main():
    if "--analyze" in sys.argv:                      # read-out mode: numpy only (no torch needed)
        ap = argparse.ArgumentParser()
        ap.add_argument("--analyze", nargs="*")
        ap.add_argument("--summary", default="")
        ap.add_argument("--figdata", default="", help="write the figure-ready per-task curves (task_curves) here")
        a = ap.parse_args()
        analyze(a.analyze, a.summary)
        if a.figdata:
            json.dump(task_curves(a.analyze), open(a.figdata, "w"), indent=1)
            print("wrote", a.figdata)
        return
    torch = _torch()
    import breadth_sweep as bs
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(bs.TRM_TASKS) + list(bs.HRM_TASKS) + list(DT_TASKS))
    ap.add_argument("--cells", default="fp32,w8c,w4c,w4t")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--offset", type=int, default=0, help="first test index (disjoint input sets)")
    ap.add_argument("--T", type=int, default=0, help="recursion steps (TRM/HRM: n_sup, default 16; DT: iterations)")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--points", default="4,T", help="steps at which rho(J) is estimated")
    ap.add_argument("--sigma_points", default="T", help="steps at which ||J||_2 is estimated")
    ap.add_argument("--K_rho", type=int, default=60)
    ap.add_argument("--K_sig", type=int, default=20)
    ap.add_argument("--tail", type=int, default=20)
    ap.add_argument("--K_lin", type=int, default=60, help="Neumann iterations of the fp32 resolvent (linear response)")
    ap.add_argument("--period", type=int, default=1, help="DT: analyse G^period (2 for the nets that end in 2-cycles)")
    ap.add_argument("--arnoldi", type=int, default=0, help="fp32: ARPACK top eigenvalues of J(s_T) on this many inputs")
    ap.add_argument("--check", action="store_true", help="compare the fp32 trajectory with the model's own ACT loop")
    ap.add_argument("--cpu", action="store_true", help="smoke test without a GPU (CUDA_VISIBLE_DEVICES='')")
    ap.add_argument("--out")
    ap.add_argument("--analyze", nargs="*")
    ap.add_argument("--summary", default="")
    a = ap.parse_args()
    if a.analyze is not None:
        return analyze(a.analyze, a.summary)
    assert torch.cuda.is_available() or a.cpu, "NO CUDA"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    T = a.T or (DT_TASKS[a.task]["T"] if a.task in DT_TASKS else 16)
    pts = sorted({T if p == "T" else int(p) for p in a.points.split(",") if p})
    spts = sorted({T if p == "T" else int(p) for p in a.sigma_points.split(",") if p})
    if a.task in DT_TASKS:
        family = "dt"
        batch = a.batch or DT_TASKS[a.task]["batch"]
        inp, lab = load_dt(a.task, a.offset + a.n)
    else:
        family = "trm" if a.task in bs.TRM_TASKS else "hrm"
        t = bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task]
        batch = a.batch or (64 if t["seq"] < 200 else 8)
        inp, lab = bs.load_qa(t["data"], a.offset + a.n)
    inp, lab = inp[a.offset:], lab[a.offset:]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"task": a.task, "family": family, "n": int(len(inp)), "offset": a.offset, "T": T, "points": pts,
            "sigma_points": spts, "K_rho": a.K_rho, "K_sig": a.K_sig, "tail": a.tail, "batch": batch,
            "K_lin": a.K_lin, "period": a.period, "nl_eps": NL_EPS, "nl_ks": list(NL_KS),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__, "records": []}
    if out_path.exists():
        try:
            old = json.load(open(out_path))
            if (old.get("n") == blob["n"] and old.get("T") == T and old.get("offset", 0) == a.offset
                    and old.get("period", 1) == a.period):
                blob["records"] = old["records"]
        except (json.JSONDecodeError, ValueError):
            print(f"{out_path} unreadable; starting fresh", flush=True)
    done = {(r["spec"], r["draw"]) for r in blob["records"]}
    cells = parse_cells(a.cells)
    if ("fp32", 0) not in cells:
        cells = [("fp32", 0)] + cells
    print(f"{a.task} ({family}): n={len(inp)} batch={batch} T={T} rho@{pts} sigma@{spts}; "
          f"{len(done)} cells already; cells={cells}", flush=True)
    chunks = [(s, min(s + batch, len(inp))) for s in range(0, len(inp), batch)]
    ref_states, m_ref = {}, None
    for cell in [("fp32", 0)] + [c for c in cells if c != ("fp32", 0)]:
        is_ref = cell == ("fp32", 0)
        if cell in done and not is_ref:
            continue
        t0 = time.time()
        m, meta, q = make_model(a.task, cell, T, batch)
        outs = []
        for ci, (s, e) in enumerate(chunks):
            x = torch.from_numpy(inp[s:e]).to(bs.DEV)
            y = torch.from_numpy(lab[s:e]).to(bs.DEV)
            o, st = eval_chunk(m, family, x, y, T, pts, spts, a.K_rho, a.K_sig, a.tail, seed=ci,
                               ref=None if is_ref else ref_states[ci], keep_states=is_ref,
                               m_ref=None if is_ref else m_ref, K_lin=a.K_lin, period=a.period)
            if is_ref:
                ref_states[ci] = st
            outs.append(o)
        o = cat_chunks(outs)
        extra = {}
        if is_ref:
            if a.check and family != "dt":
                check_against_model(m, inp, lab, batch, T, ref_states, bs)
            if a.arnoldi and cell not in done:
                ta = time.time()
                xa = torch.from_numpy(inp[:a.arnoldi]).to(bs.DEV)
                extra["arnoldi"] = arnoldi_top(m, family, xa, T, period=a.period)
                print(f"  arnoldi ({time.time()-ta:.0f}s): " + "; ".join(
                    f"#{r_['i']} |l|max {abs(complex(*r_['eig'][0])):.3f} ({'ok' if r_['converged'] else 'partial'}, {r_['matvecs']} mv)"
                    for r_ in extra["arnoldi"] if r_["eig"]), flush=True)
            m_ref = m                                     # the fp32 model stays on the GPU (resolvent, cross-seeding)
        else:
            del m
        torch.cuda.empty_cache()
        rec = {"spec": cell[0], "draw": cell[1], "quant": q, "H_cycles": meta["H_cycles"], "L_cycles": meta["L_cycles"],
               "acc": float(o["correct"].float().mean()), "cellacc": float(o["cellacc"].mean()),
               "secs": time.time() - t0, "steps_kept": kept_steps(T), **extra, "per_input": to_lists(o, T)}
        med = lambda k: float(o[k].float().median()) if k in o else float("nan")
        lam = o["lam_late"]
        print(f"  {cell[0]:>10s}#{cell[1]} acc {rec['acc']*100:5.1f}  rho_T med {med(f'rho{T}'):.3f}  "
              f"sigma_T med {med(f'sigma{T}') if spts else float('nan'):.3f}  Lambda_late med {float(lam.median()):.3f}  "
              f"q_T med {float((o['d'][T] / o['d'][T-1]).median()):.3f}  "
              + (f"phi_H {float(o['phi_H'].mean()):.3f} disp med {med('disp'):.4f} delta_T med {float(o['deltaT'].median()):.4f} "
                 f"Dlin med {med('Dlin'):.4f} xs_ref_correct {float(o['xs_ref_correct'].float().mean()):.3f}"
                 if not is_ref else "") + f"  ({rec['secs']:.0f}s)", flush=True)
        if cell in done:                              # fp32 recomputed for the reference only
            continue
        blob["records"].append(rec)
        done.add(cell)
        tmp = out_path.with_suffix(".tmp")
        json.dump(blob, open(tmp, "w"))
        tmp.replace(out_path)
    print("wrote", out_path, len(blob["records"]), "records", flush=True)


def check_against_model(m, inp, lab, batch, T, ref_states, bs):
    """The reimplemented step must reproduce the model's own ACT loop (same z_H after T steps)."""
    torch = _torch()
    n = min(batch, len(inp))
    with sdpa_math():
        c, cell, zH, zL = bs.run(m, inp[:n], lab[:n], batch, T)
    m.to(bs.DEV)
    mine = ref_states[0]["states"][T][:, 0]
    diff = float((mine - zH).abs().max())
    print(f"  check: model ACT loop vs reimplementation, max|dz_H| = {diff:.2e} (n={n}, acc {c.mean()*100:.1f})", flush=True)
    assert diff < 1e-3, "reimplemented recursion does not match the model"


# ----------------------------------------------------------------------------- analysis
def _q(a, qs=(25, 50, 75)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return [None] * len(qs)
    return [float(np.percentile(a, q)) for q in qs]


def _scalars(P, T):
    """Per-input Lambda_late, q_T, res_T, delta_T (stored directly by newer runs; derived otherwise)."""
    if "lam_late" in P:
        return (np.asarray(P["lam_late"], float), np.asarray(P["qT"], float), np.asarray(P["resT"], float),
                np.asarray(P["deltaT"], float) if "deltaT" in P else None)
    d, gr, zn = np.asarray(P["d"], float), np.asarray(P["growth"], float), np.asarray(P["znorm"], float)
    lam = np.exp(np.log(np.clip(gr[:, T // 2:], 1e-30, None)).mean(1))
    qT = d[:, T] / np.clip(d[:, T - 1], 1e-30, None)
    resT = d[:, T - 1] / zn
    dT = np.asarray(P["delta"], float)[:, T] if "delta" in P else None
    return lam, qT, resT, dT


def cell_summary(r, T, ref=None, conv_tol=0.01):
    P = r["per_input"]
    corr = np.asarray(P["correct"], bool)
    lam, qT, resT, dT = _scalars(P, T)
    s = {"spec": r["spec"], "draw": r["draw"], "acc": r["acc"], "cellacc": r["cellacc"], "n": int(corr.size),
         "Lambda_late_q": _q(lam), "frac_Lambda_lt1": float(np.mean(lam < 1)),
         "qT_q": _q(qT), "resT_q": _q(resT), "frac_converged": float(np.mean(resT < conv_tol)),
         "Lambda_q_correct": _q(lam[corr]) if corr.any() else None,
         "Lambda_q_wrong": _q(lam[~corr]) if (~corr).any() else None,
         "resT_q_correct": _q(resT[corr]) if corr.any() else None,
         "resT_q_wrong": _q(resT[~corr]) if (~corr).any() else None,
         "auroc_Lambda_wrong": auroc(lam, ~corr), "auroc_resT_wrong": auroc(resT, ~corr)}
    rk = f"rho{T}"
    if rk in P:
        rho = np.asarray(P[rk], float)
        s.update({"rhoT_q": _q(rho), "frac_rhoT_lt1": float(np.mean(rho < 1)), "rhoT_mean": float(np.mean(rho)),
                  "rhoT_q_correct": _q(rho[corr]) if corr.any() else None,
                  "rhoT_q_wrong": _q(rho[~corr]) if (~corr).any() else None,
                  "auroc_rhoT_wrong": auroc(rho, ~corr)})
    for k in [k for k in P if k.startswith("nl_")]:
        a = np.asarray(P[k], float)
        s[f"{k}_q"] = _q(a)
        s[f"{k}_q_correct"] = _q(a[corr]) if corr.any() else None
        s[f"{k}_q_wrong"] = _q(a[~corr]) if (~corr).any() else None
        s[f"frac_{k}_lt1"] = float(np.mean(a < 1))
    for k in ("margin_min", "margin_mean"):
        if k in P:
            a = np.asarray(P[k], float)
            s[f"{k}_q"] = _q(a)
            s[f"{k}_q_correct"] = _q(a[corr]) if corr.any() else None
    for k in P:
        if k in ("rho_trace",):
            continue
        if (k.startswith("rho") and not k.startswith("rhocv") and k != rk) or k.startswith("sigma"):
            s[f"{k}_q"] = _q(P[k])
            if k.startswith("sigma"):
                sg = np.asarray(P[k], float)
                s[f"frac_{k}_lt1"] = float(np.mean(sg < 1))
                s[f"{k}_q_correct"] = _q(sg[corr]) if corr.any() else None
    def qsplit(key, a=None):
        a = np.asarray(P[key], float) if a is None else a
        s[f"{key}_q"] = _q(a)
        s[f"{key}_q_correct"] = _q(a[corr]) if corr.any() else None
        s[f"{key}_q_wrong"] = _q(a[~corr]) if (~corr).any() else None
    for key in ("gamma_nat", "settle01", "settle05", "res2T", "pi_dist"):
        if key in P:
            qsplit(key)
    if "settle01" in P:
        s["frac_never_settle01"] = float(np.mean(np.asarray(P["settle01"]) > T))
        s["frac_never_settle05"] = float(np.mean(np.asarray(P["settle05"]) > T))
    if "res2T" in P:                                  # fixed point / 2-cycle / other, per input (tol 0.01)
        r1, r2 = resT, np.asarray(P["res2T"], float)
        s["frac_fixed"] = float(np.mean(r1 < conv_tol))
        s["frac_2cycle"] = float(np.mean((r1 >= conv_tol) & (r2 < conv_tol)))
    for key in ("pi_agree", "pi_correct", "correct1"):
        if key in P:
            s[f"{key}_mean"] = float(np.mean(P[key]))
    if "disp" in P:
        disp = np.asarray(P["disp"], float)
        s.update({"phi_H": float(np.mean(P["phi_H"])), "phi_H_q": _q(P["phi_H"]), "disp_q": _q(disp),
                  "deltaT_q": _q(dT), "amp_q": _q(disp / np.clip(dT, 1e-12, None)),
                  "amp_q_correct": _q((disp / np.clip(dT, 1e-12, None))[corr]) if corr.any() else None})
        if "agree" in P:
            s["agree_mean"] = float(np.mean(P["agree"]))
        if "disp_t" in P:                            # displacement from fp32 at each depth t (kept steps)
            ks = r.get("steps_kept") or list(range(T + 1))
            dt = np.asarray(P["disp_t"], float)       # column j <-> t = ks[j] + 1
            s["disp_t_med"] = {int(ks[j] + 1): float(np.median(dt[:, j])) for j in range(dt.shape[1])}
        if "Dlin" in P:
            Dlin = np.asarray(P["Dlin"], float)
            lc = np.asarray(P["lin_conv"], float)
            ok = lc < 1e-2                             # Neumann series converged (rho_fp32 < 1 along delta)
            s["frac_lin_converged"] = float(ok.mean())
            s["Dlin_q"] = _q(Dlin)
            s["R_disp_over_Dlin_q"] = _q(disp[ok] / np.clip(Dlin[ok], 1e-12, None)) if ok.sum() >= 3 else None
            s["spearman_disp_Dlin"] = spearman(disp[ok], Dlin[ok]) if ok.sum() >= 5 else None
        if "xs_ref_correct" in P:
            s["xs_ref_correct_mean"] = float(np.mean(P["xs_ref_correct"]))
            s["xs_own_correct_mean"] = float(np.mean(P["xs_own_correct"]))
            s["xs_ref_agree_mean"] = float(np.mean(P["xs_ref_agree"]))
            s["xs_ref_dist_q"] = _q(P["xs_ref_dist"])
            s["xs_own_dist_q"] = _q(P["xs_own_dist"])
        if ref is not None:
            RP = ref["per_input"]
            rlam, rqT, rres, _ = _scalars(RP, T)
            rrho = np.asarray(RP.get(rk, [np.nan] * len(disp)), float)
            rcorr = np.asarray(RP["correct"], bool)
            conv = rres < conv_tol                      # fp32 trajectory converged at T
            s["frac_ref_converged"] = float(conv.mean())
            own = np.asarray(P.get(rk, rrho), float)
            for name, c in (("rho_fp32", rrho), ("Lambda_fp32", rlam), ("rho_own", own), ("qT_fp32", rqT)):
                pred = dT / np.clip(1 - c, 1e-3, None)
                ok = (c < 1) & conv
                s[f"pred_{name}_ok_n"] = int(ok.sum())
                if ok.sum() >= 5:
                    s[f"ratio_disp_pred_{name}_q"] = _q(disp[ok] / pred[ok])
                    s[f"spearman_disp_pred_{name}"] = spearman(disp[ok], pred[ok])
                    s[f"disp_q_ok_{name}"] = _q(disp[ok])
                    s[f"pred_q_ok_{name}"] = _q(pred[ok])
            s["spearman_disp_deltaT_conv"] = spearman(disp[conv], dT[conv]) if conv.sum() >= 5 else None
            s["disp_q_conv"] = _q(disp[conv]) if conv.any() else None
            s["disp_q_ref_correct"] = _q(disp[rcorr]) if rcorr.any() else None
            s["acc_on_ref_correct"] = float(corr[rcorr].mean()) if rcorr.any() else None
            if rcorr.any():                             # the puzzles whose answer can be lost: fp32 solves them
                if rk in P:
                    rho_ = np.asarray(P[rk], float)[rcorr]
                    s["rhoT_q_refcorrect"] = _q(rho_)
                    s["frac_rhoT_lt1_refcorrect"] = float(np.mean(rho_ < 1))
                s["resT_q_refcorrect"] = _q(resT[rcorr])
                s["frac_settled_refcorrect"] = float(np.mean(resT[rcorr] < 0.1))
                if "nl_rand_16" in P:
                    s["kick_q_refcorrect"] = _q(np.asarray(P["nl_rand_16"], float)[rcorr])
                if "sigma16" in P:
                    s["sigmaJ_q_refcorrect"] = _q(np.asarray(P["sigma16"], float)[rcorr])
                s["Lambda_q_refcorrect"] = _q(lam[rcorr])
                s["frac_Lambda_lt1_refcorrect"] = float(np.mean(lam[rcorr] < 1))
                if "gamma_nat" in P:
                    s["gamma_nat_q_refcorrect"] = _q(np.asarray(P["gamma_nat"], float)[rcorr])
            if "xs_ref_correct" in P and rcorr.any():
                xr = np.asarray(P["xs_ref_correct"], bool)
                # basin switch: fp32 solves the puzzle, but the fp32 map started from the perturbed end state does not
                s["frac_basin_switch"] = float(np.mean(~xr[rcorr]))
                s["frac_wrong_but_fp32_recovers"] = float(np.mean(xr[rcorr] & ~corr[rcorr]))
            for key in ("flip_lin", "flip_meas"):
                if key in P and rcorr.any():
                    fl = np.asarray(P[key], float)[rcorr]
                    s[f"{key}_q_refcorrect"] = _q(fl)
                    s[f"frac_{key}_lt1_refcorrect"] = float(np.mean(fl < 1))
                    s[f"auroc_{key}_wrong_refcorrect"] = auroc(-fl, ~corr[rcorr])
    return s


def _rank(a):
    """Average ranks (ties averaged), 1-based, numpy only."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and a[order[j + 1]] == a[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auroc(score, positive):
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    ok = np.isfinite(score)
    score, positive = score[ok], positive[ok]
    npos, nneg = positive.sum(), (~positive).sum()
    if npos == 0 or nneg == 0:
        return None
    rk = _rank(score)
    return float((rk[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5:
        return None
    ra, rb = _rank(a[ok]), _rank(b[ok])
    return float(np.corrcoef(ra, rb)[0, 1])


def analyze(files, summary_path):
    """Per-cell table for every contraction_*.json; each file's own fp32 record is the reference for
    its cells (split jobs of one task each recompute the identical deterministic fp32 run)."""
    files = sorted(set(f for p in files for f in glob.glob(p)))
    allsum = {}
    fmt = lambda x, p=3: "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{p}f}"
    med = lambda q: None if not q else q[1]
    for f in files:
        j = json.load(open(f))
        T = j["T"]
        recs = j["records"]
        ref = next((r for r in recs if r["spec"] == "fp32"), None)
        rows = [cell_summary(r, T, ref) for r in recs]
        key = Path(f).stem.replace("contraction_", "")
        allsum[key] = {"task": j["task"], "file": Path(f).name, "n": j["n"], "T": T, "gpu": j.get("gpu"),
                       "torch": j.get("torch"), "cells": rows}
        print(f"\n### {key}  (task {j['task']}, n={j['n']}, T={T}, {j.get('gpu')})")
        print("| cell | acc | phi_H | agree | Lambda med (corr / wrong) | rho_T med (corr / wrong) | ||J_T|| med | "
              "kick@16 rand med | frac Lambda<1 | res_T med | disp med | delta_T med | basin switch | acc on fp32-solved |")
        print("|---|" + "---|" * 12)
        for s in rows:
            lc, lw = med(s.get("Lambda_q_correct")), med(s.get("Lambda_q_wrong"))
            rc, rw = med(s.get("rhoT_q_correct")), med(s.get("rhoT_q_wrong"))
            print(f"| {s['spec']}#{s['draw']} | {s['acc']*100:.1f} | {fmt(s.get('phi_H'))} | {fmt(s.get('agree_mean'))} | "
                  f"{fmt(med(s['Lambda_late_q']))} ({fmt(lc)} / {fmt(lw)}) | {fmt(med(s.get('rhoT_q')))} ({fmt(rc)} / {fmt(rw)}) | "
                  f"{fmt(med(s.get(f'sigma{T}_q')), 1)} | {fmt(med(s.get('nl_rand_16_q')))} | {fmt(s.get('frac_Lambda_lt1'), 2)} | "
                  f"{fmt(med(s['resT_q']))} | {fmt(med(s.get('disp_q')))} | {fmt(med(s.get('deltaT_q')))} | "
                  f"{fmt(s.get('frac_basin_switch'), 2)} | {fmt(s.get('acc_on_ref_correct'), 2)} |")
    if summary_path:
        json.dump(allsum, open(summary_path, "w"), indent=1)
        print("wrote", summary_path)


METRICS = {   # per-cell scalar (from cell_summary) -> name in the figure data
    "rho_own": lambda c: (c.get("rhoT_q") or [None] * 3)[1],
    "rho_own_q1": lambda c: (c.get("rhoT_q") or [None] * 3)[0],
    "rho_own_q3": lambda c: (c.get("rhoT_q") or [None] * 3)[2],
    "frac_rho_lt1": lambda c: c.get("frac_rhoT_lt1"),
    # over the puzzles fp32 solves (for fp32 itself: its correct puzzles)
    "rho_own_sol": lambda c: (c.get("rhoT_q_refcorrect") or c.get("rhoT_q_correct") or [None] * 3)[1],
    "rho_own_sol_q1": lambda c: (c.get("rhoT_q_refcorrect") or c.get("rhoT_q_correct") or [None] * 3)[0],
    "rho_own_sol_q3": lambda c: (c.get("rhoT_q_refcorrect") or c.get("rhoT_q_correct") or [None] * 3)[2],
    "frac_rho_lt1_sol": lambda c: c.get("frac_rhoT_lt1_refcorrect"),
    "Lambda_sol": lambda c: (c.get("Lambda_q_refcorrect") or c.get("Lambda_q_correct") or [None] * 3)[1],
    "gamma_nat_sol": lambda c: (c.get("gamma_nat_q_refcorrect") or c.get("gamma_nat_q_correct") or [None] * 3)[1],
    "Lambda": lambda c: (c.get("Lambda_late_q") or [None] * 3)[1],
    "frac_Lambda_lt1": lambda c: c.get("frac_Lambda_lt1"),
    "gamma_nat": lambda c: (c.get("gamma_nat_q") or [None] * 3)[1],
    "resT": lambda c: (c.get("resT_q") or [None] * 3)[1],
    "settle05": lambda c: (c.get("settle05_q") or [None] * 3)[1],
    "frac_never_settle05": lambda c: c.get("frac_never_settle05"),
    "frac_fixed": lambda c: c.get("frac_fixed"),
    "phi_H": lambda c: c.get("phi_H"),
    "disp": lambda c: (c.get("disp_q") or [None] * 3)[1],
    "deltaT": lambda c: (c.get("deltaT_q") or [None] * 3)[1],
    "Dlin": lambda c: (c.get("Dlin_q") or [None] * 3)[1],
    "frac_lin_converged": lambda c: c.get("frac_lin_converged"),
    "R": lambda c: (c.get("R_disp_over_Dlin_q") or [None] * 3)[1],
    "frac_basin_switch": lambda c: c.get("frac_basin_switch"),
    "xs_ref_agree": lambda c: c.get("xs_ref_agree_mean"),
    "pi_agree": lambda c: c.get("pi_agree_mean"),
    "agree": lambda c: c.get("agree_mean"),
    "kick_rand16": lambda c: (c.get("nl_rand_16_q") or [None] * 3)[1],
    "kick_dom16": lambda c: (c.get("nl_dom_16_q") or [None] * 3)[1],
    "frac_kick_rand16_lt1": lambda c: c.get("frac_nl_rand_16_lt1"),
    "acc_on_ref_correct": lambda c: c.get("acc_on_ref_correct"),
    "resT_sol": lambda c: (c.get("resT_q_refcorrect") or c.get("resT_q_correct") or [None] * 3)[1],
    "resT_sol_q1": lambda c: (c.get("resT_q_refcorrect") or c.get("resT_q_correct") or [None] * 3)[0],
    "resT_sol_q3": lambda c: (c.get("resT_q_refcorrect") or c.get("resT_q_correct") or [None] * 3)[2],
    "frac_settled_sol": lambda c: c.get("frac_settled_refcorrect"),
    "kick_sol": lambda c: (c.get("kick_q_refcorrect") or c.get("nl_rand_16_q_correct") or [None] * 3)[1],
    "sigmaJ_sol": lambda c: (c.get("sigmaJ_q_refcorrect") or c.get("sigma16_q_correct") or [None] * 3)[1],
    "flip_meas_frac": lambda c: c.get("frac_flip_meas_lt1_refcorrect"),
    "sigma_max": lambda c: next((v[1] for k, v in c.items() if k.startswith("sigma") and k.endswith("_q") and v), None),
}


def _sigma_half(sig, acc, a0):
    """First sigma where the median accuracy falls below a0/2, log-linear interpolation."""
    prev = None
    for x, y in zip(sig, acc):
        if y < a0 / 2:
            if prev is None:
                return {"value": None, "bound": f"< {x}"}
            x0, y0 = prev
            f = (y0 - a0 / 2) / max(y0 - y, 1e-12)
            return {"value": float(np.exp(np.log(x0) + f * (np.log(x) - np.log(x0)))), "bound": None}
        prev = (x, y)
    return {"value": None, "bound": f"> {sig[-1]}" if sig else None}


def task_curves(files):
    """Figure data per task: accuracy and mechanism metrics against weight-noise sigma (all draws), the
    quantizer / weight-group cells, measured sigma_1/2 and the two label-free predictions of it."""
    from collections import defaultdict
    groups = defaultdict(list)
    for f in sorted(set(f for p in files for f in glob.glob(p))):
        j = json.load(open(f))
        groups[(j["task"], j.get("period", 1), j["T"])].append((f, j))
    out = {}
    for (task, period, T), blobs in groups.items():
        key = task + (f"_p{period}" if period != 1 else "") + (f"_T{T}" if task.startswith("dt_") else "")
        fp, noise, others = None, defaultdict(list), []
        for f, j in blobs:
            ref = next((r for r in j["records"] if r["spec"] == "fp32"), None)
            for r in j["records"]:
                if r["spec"] == "fp32":
                    if fp is None:
                        fp = cell_summary(r, T)
                    continue
                c = cell_summary(r, T, ref)
                q = r["quant"]
                if q.get("gran") == "noise" and not q.get("group"):
                    noise[float(q["sigma"])].append(c)
                else:
                    others.append(c)
        if fp is None:
            continue
        sig = sorted(noise)
        a0 = fp["acc"]
        acc_med = [float(np.median([c["acc"] for c in noise[x]])) for x in sig]
        d = {"task": task, "period": period, "T": T, "n": fp["n"], "fp32": {k: fn(fp) for k, fn in METRICS.items()},
             "fp32_acc": a0, "sigma": sig, "acc_draws": [[c["acc"] for c in noise[x]] for x in sig], "acc_med": acc_med,
             "metrics": {k: [[fn(c) for c in noise[x]] for x in sig] for k, fn in METRICS.items()},
             "others": [{"spec": c["spec"], "draw": c["draw"], "acc": c["acc"], **{k: fn(c) for k, fn in METRICS.items()}}
                        for c in others]}
        d["fp32"].update({"rho_q_correct": fp.get("rhoT_q_correct"), "rho_q_wrong": fp.get("rhoT_q_wrong"),
                          "Lambda_q_correct": fp.get("Lambda_q_correct"), "Lambda_q_wrong": fp.get("Lambda_q_wrong"),
                          "resT_q_correct": fp.get("resT_q_correct"), "resT_q_wrong": fp.get("resT_q_wrong"),
                          "gamma_q_correct": fp.get("gamma_nat_q_correct"), "gamma_q_wrong": fp.get("gamma_nat_q_wrong"),
                          "auroc_rho_wrong": fp.get("auroc_rhoT_wrong"), "auroc_Lambda_wrong": fp.get("auroc_Lambda_wrong"),
                          "auroc_resT_wrong": fp.get("auroc_resT_wrong"), "correct1": fp.get("correct1_mean"),
                          "pi_agree": fp.get("pi_agree_mean"), "frac_2cycle": fp.get("frac_2cycle")})
        sh = _sigma_half(sig, acc_med, a0) if sig else {"value": None, "bound": None}
        d["sigma_half"] = sh
        # sigma*_rho: linear fit of median rho_own on sigma <= sigma_1/2 / 2 (with fp32 at sigma 0), extrapolated to 1;
        # over all inputs ("all") and over the puzzles fp32 solves ("sol")
        lim = (sh["value"] or (float(sh["bound"][2:]) if sh.get("bound") and sh["bound"].startswith(">") else None))
        d["sigma_star_rho"] = {}
        for tag, mk_ in (("all", "rho_own"), ("sol", "rho_own_sol")):
            rho0 = d["fp32"][mk_]
            pred = {"value": None, "note": None, "rho_fp32": rho0}
            if rho0 is None:
                pred["note"] = "no rho"
            elif rho0 >= 1:
                pred["note"] = f"fp32 median rho_T {rho0:.2f} >= 1: bifurcation route not applicable"
            else:
                xs, ys = [0.0], [rho0]
                for i, x in enumerate(sig):
                    vals = [m for m in d["metrics"][mk_][i] if m is not None]
                    if lim is not None and x <= lim / 2 and vals:
                        xs.append(x)
                        ys.append(float(np.median(vals)))
                if len(xs) >= 2:
                    b, a_ = np.polyfit(xs, ys, 1)
                    pred.update({"value": float((1 - a_) / b) if b > 0 else None, "fit": [float(a_), float(b)],
                                 "points": len(xs),
                                 "note": None if b > 0 else "rho_own does not rise with sigma below sigma_1/2 / 2"})
                else:
                    pred["note"] = "no sigma below sigma_1/2 / 2"
            d["sigma_star_rho"][tag] = pred
        # sigma*_kick: label-free, reference-free cliff detector: the sigma at which a random state kick stops
        # dying out, i.e. the median (over draws) of the median 16-step growth crosses 1 (log-log interpolation)
        kk = [float(np.median([v for v in vs if v is not None])) if any(v is not None for v in vs) else None
              for vs in d["metrics"]["kick_rand16"]]
        k0 = d["fp32"]["kick_rand16"]
        sk = {"value": None, "note": None, "kick_fp32": k0, "kick_med": kk}
        if k0 is None:
            sk["note"] = "no kick"
        elif k0 >= 1:
            sk["note"] = "fp32 kick growth >= 1"
        else:
            px, py = 0.0, k0
            for x, y in zip(sig, kk):
                if y is None:
                    continue
                if y >= 1:
                    if px == 0.0:
                        sk["value"], sk["note"] = None, f"< {x}"
                    else:
                        f = (np.log(1) - np.log(py)) / (np.log(y) - np.log(py))
                        sk["value"] = float(np.exp(np.log(px) + f * (np.log(x) - np.log(px))))
                    break
                px, py = x, y
            else:
                sk["note"] = f"> {sig[-1]}" if sig else None
        d["sigma_star_kick"] = sk
        # sensitivity S = delta_T / sigma (one-step map perturbation at fp32's end state per unit noise), and
        # the one-step perturbation at the measured cliff (log-log interpolation of median delta_T to sigma_1/2)
        dm = [float(np.median([v for v in vs if v is not None])) if any(v is not None for v in vs) else None
              for vs in d["metrics"]["deltaT"]]
        d["deltaT_med"] = dm
        d["sensitivity"] = [None if y is None else y / x for x, y in zip(sig, dm)]
        dh = None
        if sh["value"] is not None:
            xs_ = [x for x, y in zip(sig, dm) if y is not None]
            ys_ = [y for y in dm if y is not None]
            if len(xs_) >= 2:
                dh = float(np.exp(np.interp(np.log(sh["value"]), np.log(xs_), np.log(ys_))))
        d["deltaT_at_sigma_half"] = dh
        # sigma*_mu: decoder-margin route from the smallest sigma: per fp32-solved puzzle sigma_flip = sigma0 * min-cell
        # margin ratio; the median over puzzles is where half of them flip
        d["sigma_star_mu"] = {}
        if sig:
            s0 = sig[0]
            for kind in ("lin", "meas"):
                vals = [c.get(f"flip_{kind}_q_refcorrect") for c in noise[s0]]
                vals = [v[1] for v in vals if v]
                d["sigma_star_mu"][kind] = {"value": float(s0 * np.median(vals)) if vals else None, "sigma0": s0,
                                            "per_draw": [float(s0 * v) for v in vals]}
            d["sigma_star_mu"]["lin_converged_frac_at_sigma0"] = [c.get("frac_lin_converged") for c in noise[s0]]
        out[key] = d
    return out


if __name__ == "__main__":
    main()
