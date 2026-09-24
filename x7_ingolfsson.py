#!/usr/bin/env python3
"""X7: head-to-head with Ingolfsson et al. (arXiv 2607.16237) on our TRM/HRM checkpoints.

Three modes, one evaluation engine (fp32 reference and every cell run in lockstep, one ACT step at a
time, so every metric is read out at every step 1..nsup against fp32 at the same step):

  ladder     their W4A4 granularity ladder (per-tensor static/dynamic, per-token, group-32/16,
             per-channel weights, MXInt4 block-32 power-of-two), weight-only and activation-only
             rows, W8A8; default depth.
  depth      test-time dose-response on one fixed checkpoint and one fixed quantizer: nsup 1..32 at
             the default H/L, and H_cycles / L_cycles sweeps at nsup 1..16 (static activation scales
             are calibrated once at the default depth and reused at every depth).
  biasnoise  zero-mean noise vs fixed-direction bias at matched Frobenius norm, weights and
             activations, sigma sweep x draws.

    python x7_ingolfsson.py --task sudoku_mlp --mode ladder --out ~/breadth_results/x7/ladder_sudoku_mlp.json
    python x7_ingolfsson.py --analyze 'results/breadth/babel/x7/*.json' --summary results/breadth/x7_summary.json

Cell grammar (a '+'-joined weight part and activation part; '@k' = calibration seed or noise draw):
  fp32
  w<bits><g>        weights of every CastedLinear; g in t (per-tensor absmax), c (per-output-channel),
                    g32/g16 (groups along the input dim, float absmax scale), mx (block 32 along the
                    input dim, shared power-of-two scale 2^(floor(log2 absmax)-(bits-2)), symmetric
                    elements in [-qmax, qmax]: the OCP MXINT convention)
  a<bits><g>        input activation of every CastedLinear; g in ts (per-tensor static absmax scale,
                    observed over 200 calibration puzzles through the weight-quantized model at the
                    default depth, then frozen; @k re-draws the calibration set), td (per-tensor dynamic
                    absmax), tok (per row of the input, dynamic), g32/g16 (groups along the last dim,
                    dynamic), mx (as for weights, along the last dim, dynamic)
  an<s>  x + s*std(x)*G_t, G_t fresh at every call              (zero-mean in time and in elements)
  af<s>  x + s*std(x)*G,   G fixed per layer over the last two dims (step-coherent, element-mean 0)
  ab<s>  x + s*std(x)*b,   b = sign*|g|/rms(g) over the last dim  (step-coherent, sign-consistent bias)
  wn<s>  W + s*std(W)*G/rms(G)                                   (fixed, zero-mean)
  wb<s>  W + s*std(W)*sign*|G|/rms(G), same G as wn               (fixed, sign-consistent)
All five noise arms have per-call (per-tensor) Frobenius norm exactly s*std*sqrt(numel).
Embeddings stay in fp32 everywhere (as in 2607.16237).

Per record: per-step exact / cell accuracy / fidelity phi_H (mean cosine of z_H vs fp32 at the same
step) / Maze valid-path and valid-shortest-path rates; per-example correctness at the final step and
at steps 1,2,4,8,16,32; sigma_eq of the quantizer (activation RMS(Q(x)-x)/std(x) averaged over all
calls; weight RMS(Wq-W)/std(W) averaged over tensors); cross-step coherence of the activation
perturbation ||mean_t eps_t|| / RMS_t ||eps_t|| per (layer, call index within a step) on the first
examples of the first batch (1 = the same perturbation every step, ~1/sqrt(T) = independent).
"""
import argparse
import copy
import glob
import json
import math
import os
import sys
import time
import zlib
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_sweep as bs  # noqa: E402  (adds TinyRecursiveModels to sys.path)

DEV = bs.DEV

TASKS = {
    # n and the calibration pool follow the breadth sweeps (Sudoku first 1000 test rows, Maze first 500)
    "sudoku_mlp": dict(n=1000, pool=(1000, 3000), H=3, L=6, batch=256, coh=4,
                       Hs=[1, 2, 3, 4, 6], Ls=[1, 2, 4, 6, 8]),
    "sudoku_attn": dict(n=1000, pool=(1000, 3000), H=3, L=6, batch=256, coh=4,
                        Hs=[1, 2, 3, 4, 6], Ls=[1, 2, 4, 6, 8]),
    "maze_attn": dict(n=500, pool=(500, 1000), H=3, L=4, batch=50, coh=1,
                      Hs=[1, 2, 3, 4, 6], Ls=[1, 2, 4, 8]),
    "hrm_sudoku": dict(n=1000, pool=(1000, 3000), H=2, L=2, batch=256, coh=4,
                       Hs=[1, 2, 3, 4], Ls=[1, 2, 4]),
}
CALIB_N = 200
KEEP_STEPS = (1, 2, 4, 8, 16, 32)


# ----------------------------------------------------------------------------- cells
def parse_cell(name):
    """-> dict(name, w=(kind, bits|sigma), a=(kind, bits|sigma), seed)."""
    base, seed = (name.split("@") + ["0"])[:2]
    seed = int(seed)
    cell = {"name": name, "w": None, "a": None, "seed": seed}
    if base == "fp32":
        return cell
    for part in base.split("+"):
        arm = part[:2]
        if arm in ("an", "af", "ab", "wn", "wb"):
            sig = float(part[2:])
            if arm[0] == "w":
                cell["w"] = (arm, sig)
            else:
                cell["a"] = (arm, sig)
        else:
            kind = part[0]
            i = 1
            while i < len(part) and part[i].isdigit():
                i += 1
            bits, g = int(part[1:i]), part[i:]
            if kind == "w":
                assert g in ("t", "c", "mx") or (g.startswith("g") and g[1:].isdigit()), part
                cell["w"] = (g, bits)
            elif kind == "a":
                assert g in ("ts", "td", "tok", "mx") or (g.startswith("g") and g[1:].isdigit()), part
                cell["a"] = (g, bits)
            else:
                raise ValueError(part)
    return cell


def _stable(*xs):
    return zlib.crc32(("|".join(str(x) for x in xs)).encode()) & 0x7FFFFFFF


# ----------------------------------------------------------------------------- quantizers
def q_group(x, bits, g):
    """symmetric absmax quantization with one float scale per group of g along the last dim."""
    qmax = 2 ** (bits - 1) - 1
    last = x.shape[-1]
    pad = (-last) % g
    xp = F.pad(x, (0, pad)).reshape(*x.shape[:-1], -1, g)
    s = (xp.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    q = torch.round(xp / s).clamp(-qmax, qmax) * s
    return q.reshape(*x.shape[:-1], -1)[..., :last]


def q_rows(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = (x.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
    return torch.round(x / s).clamp(-qmax, qmax) * s


def q_tensor(x, bits, absmax=None):
    qmax = 2 ** (bits - 1) - 1
    s = max(float(x.abs().max()) if absmax is None else absmax, 1e-12) / qmax
    return torch.round(x / s).clamp(-qmax, qmax) * s


def q_mx(x, bits, block=32):
    """OCP-style MXINT: per block of 32 along the last dim, shared scale 2^(floor(log2 absmax)-(bits-2)),
    symmetric integer elements in [-qmax, qmax] (block max lands in [2^(bits-2), 2^(bits-1)) before rounding)."""
    qmax = 2 ** (bits - 1) - 1
    last = x.shape[-1]
    pad = (-last) % block
    xp = F.pad(x, (0, pad)).reshape(*x.shape[:-1], -1, block)
    amax = xp.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp_min(1e-30))) - (bits - 2)
    s = torch.pow(2.0, e)
    q = torch.where(amax > 0, torch.round(xp / s).clamp(-qmax, qmax) * s, torch.zeros_like(xp))
    return q.reshape(*x.shape[:-1], -1)[..., :last]


def linears(m):
    return [mod for mod in m.modules() if mod.__class__.__name__ == "CastedLinear"]


@torch.no_grad()
def transform_weights_(m, spec, seed):
    """apply the weight part of a cell in place; returns per-tensor sigma_eq = RMS(dW)/std(W)."""
    if spec is None:
        return []
    kind, p = spec
    out = []
    for li, mod in enumerate(linears(m)):
        W = mod.weight.data
        if kind in ("wn", "wb"):
            gen = torch.Generator().manual_seed(_stable("w", seed, li))
            G = torch.randn(W.shape, generator=gen, dtype=W.dtype)
            if kind == "wb":
                sign = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
                G = sign * G.abs()
            G = G / G.pow(2).mean().sqrt()
            Wq = W + p * W.std() * G
        elif kind == "t":
            Wq = q_tensor(W, p)
        elif kind == "c":
            Wq = q_rows(W, p)
        elif kind == "mx":
            Wq = q_mx(W, p)
        else:
            Wq = q_group(W, p, int(kind[1:]))
        out.append(float((Wq - W).pow(2).mean().sqrt() / W.std().clamp_min(1e-12)))
        mod.weight.data = Wq
    return out


class ActHook:
    """forward pre-hook on a CastedLinear: activation quantizer or perturbation of its input."""

    def __init__(self, kind, p, seed, li, coh_rows=0):
        self.kind, self.p, self.seed, self.li = kind, p, seed, li
        self.absmax = None            # 'ts': static scale (set by calibration)
        self.observe = False          # calibration pass: record absmax, do not quantize
        self.obs_max = 0.0
        self.fixed = None             # 'af'/'ab' fixed direction (lazily created)
        self.gen = None               # 'an' generator
        self.seq_sum, self.seq_n = 0.0, 0     # sigma_eq accumulators
        self.coh_rows = coh_rows      # coherence on the first rows of the first batch
        self.coh_on = False
        self.call = 0                 # call index within the current ACT step
        self.coh = {}                 # call index -> [sum eps, sum ||eps||^2, count]

    def new_step(self):
        self.call = 0

    def _perturb(self, x):
        k, p = self.kind, self.p
        if k == "ts":
            return q_tensor(x, p, self.absmax)
        if k == "td":
            return q_tensor(x, p)
        if k == "tok":
            return q_rows(x, p)
        if k == "mx":
            return q_mx(x, p)
        if k.startswith("g"):
            return q_group(x, p, int(k[1:]))
        sd = x.std()
        if k == "an":
            if self.gen is None:
                self.gen = torch.Generator(device=x.device).manual_seed(_stable("an", self.seed, self.li))
            G = torch.randn(x.shape, generator=self.gen, device=x.device, dtype=x.dtype)
            G = G / G.pow(2).mean().sqrt()
            return x + p * sd * G
        shape = tuple(x.shape[-2:]) if (k == "af" and x.dim() >= 3) else (x.shape[-1],)
        if self.fixed is None or tuple(self.fixed.shape) != shape:
            gen = torch.Generator().manual_seed(_stable(k, self.seed, self.li, shape))
            G = torch.randn(shape, generator=gen)
            if k == "ab":
                sign = 1.0 if torch.rand(1, generator=gen).item() < 0.5 else -1.0
                G = sign * G.abs()
            self.fixed = (G / G.pow(2).mean().sqrt()).to(x.device, x.dtype)
        return x + p * sd * self.fixed

    def __call__(self, mod, args):
        x = args[0]
        if self.observe:
            self.obs_max = max(self.obs_max, float(x.detach().abs().max()))
            return None
        xq = self._perturb(x)
        eps = xq - x
        self.seq_sum += float(eps.pow(2).mean().sqrt() / x.std().clamp_min(1e-12))
        self.seq_n += 1
        if self.coh_on and self.coh_rows:
            e = eps[: self.coh_rows].detach().float()
            acc = self.coh.get(self.call)
            if acc is None:
                self.coh[self.call] = [e.clone(), float(e.pow(2).sum()), 1]
            elif acc[0].shape == e.shape:
                acc[0] += e
                acc[1] += float(e.pow(2).sum())
                acc[2] += 1
        self.call += 1
        return (xq,) + tuple(args[1:])

    def coherence(self):
        vals = [float(s.norm() / c) / math.sqrt(q / c) for s, q, c in self.coh.values() if c > 1 and q > 0]
        return vals


class OutProbe:
    """per-application output perturbation of a linear along the cell's own trajectory:
    rho = ||y - W0 x|| / ||W0 x||, x = the unperturbed input, W0 = the fp32 weight (first batch, first rows).
    This is the layer-level epsilon of Prop. 1 in 2607.16237, on one axis for quantizers and noise arms."""

    def __init__(self, W0, b0, rows):
        self.W0, self.b0, self.rows = W0, b0, rows
        self.on = False
        self.step = 0
        self.x = None
        self.acc = {}                  # step -> list of rho

    def pre(self, mod, args):
        if self.on:
            self.x = args[0][: self.rows].detach()
        return None

    def post(self, mod, args, out):
        if self.on and self.x is not None:
            yref = F.linear(self.x, self.W0, self.b0)
            d = out[: self.rows] - yref
            self.acc.setdefault(self.step, []).append(float(d.norm() / yref.norm().clamp_min(1e-12)))
            self.x = None
        return None


# ----------------------------------------------------------------------------- models and data
def build_model(task, H, L, nsup, batch):
    if task in bs.TRM_TASKS:
        saved = bs.TRM_TASKS[task]
        bs.TRM_TASKS[task] = dict(saved, L=L)
        try:
            m, meta = bs.build_trm(task, H, nsup, batch)
        finally:
            bs.TRM_TASKS[task] = saved
    else:
        m, meta = bs.build_hrm(task, H, nsup, batch, L=L)
    return m.eval(), meta


def init_carry(m, bt):
    c = m.initial_carry(bt)
    ic = c.inner_carry
    return type(c)(inner_carry=type(ic)(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)), steps=c.steps.to(DEV),
                   halted=c.halted.to(DEV), current_data={k: v.to(DEV) for k, v in c.current_data.items()})


def make_batch(inp, lab, s, e):
    x = torch.from_numpy(inp[s:e]).to(DEV)
    y = torch.from_numpy(lab[s:e]).to(DEV)
    return {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(x.shape[0], dtype=torch.int32, device=DEV)}


# Maze scoring: tokens are MAZE_CHARSET "# SGo" + 1 -> wall 1, open 2, S 3, G 4, path 5
def bfs_dist(inp_row, side=30):
    grid = inp_row.reshape(side, side)
    s = tuple(np.argwhere(grid == 3)[0])
    g = tuple(np.argwhere(grid == 4)[0])
    dist = {s: 0}
    dq = deque([s])
    while dq:
        r, c = dq.popleft()
        if (r, c) == g:
            return dist[(r, c)]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < side and 0 <= nc < side and grid[nr, nc] != 1 and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                dq.append((nr, nc))
    return -1


def maze_valid(pred, inp, dist, side=30):
    """valid: every predicted path cell is open, the path cells plus S and G form ONE 4-connected component
    containing S and G. valid-shortest: valid and exactly dist-1 path cells."""
    from scipy import ndimage
    sg = (inp == 3) | (inp == 4)
    path = (pred == 5) & ~sg
    if (path & (inp == 1)).any():
        return False, False
    comp = (path | sg).reshape(side, side)
    labl, _ = ndimage.label(comp)
    labl = labl.reshape(-1)
    ls, lg = labl[np.argmax(inp == 3)], labl[np.argmax(inp == 4)]
    ok = bool(ls != 0 and ls == lg and (labl[path] == ls).all())
    return ok, bool(ok and int(path.sum()) == dist - 1)


# ----------------------------------------------------------------------------- calibration
@torch.no_grad()
def calibrate(task, cell, cinp, clab, batch):
    """static per-tensor activation absmax per linear, observed through the weight-quantized model at the
    default depth (nsup 16) over the calibration puzzles."""
    t = TASKS[task]
    m, _ = build_model(task, t["H"], t["L"], 16, batch)
    transform_weights_(m, cell["w"], cell["seed"])
    m = m.to(DEV)
    hooks = []
    for li, mod in enumerate(linears(m)):
        h = ActHook("ts", cell["a"][1], cell["seed"], li)
        h.observe = True
        hooks.append((h, mod.register_forward_pre_hook(h)))
    for s in range(0, len(cinp), batch):
        bt = make_batch(cinp, clab, s, s + batch)
        carry = init_carry(m, bt)
        for _ in range(16):
            carry, _ = m(carry, bt)
    out = [h.obs_max for h, _ in hooks]
    for _, hd in hooks:
        hd.remove()
    del m
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- evaluation engine
@torch.no_grad()
def evaluate(task, H, L, nsup, cells, inp, lab, batch, calib_cache, cinp_all, clab_all, pool_lo, maze_d):
    """fp32 + cells in lockstep; returns one record per cell (fp32 first)."""
    t = TASKS[task]
    base, meta = build_model(task, H, L, nsup, batch)
    entries = [{"cell": parse_cell("fp32"), "model": base, "hooks": [], "sig_w": []}]
    for name in cells:
        cell = parse_cell(name)
        m = copy.deepcopy(base)
        W0 = [(mod.weight.data.clone().to(DEV), None if mod.bias is None else mod.bias.data.clone().to(DEV))
              for mod in linears(m)]
        sig_w = transform_weights_(m, cell["w"], cell["seed"])
        m = m.to(DEV)
        probes = []
        for li, mod in enumerate(linears(m)):
            pr = OutProbe(W0[li][0], W0[li][1], t["coh"])
            probes.append((pr, mod.register_forward_pre_hook(pr.pre), mod.register_forward_hook(pr.post)))
        hooks = []
        if cell["a"] is not None:
            kind, p = cell["a"]
            absmax = None
            if kind == "ts":
                key = (json.dumps(cell["w"]), p, cell["seed"])
                if key not in calib_cache:
                    rng = np.random.RandomState(_stable("calib", cell["seed"]))
                    idx = rng.choice(len(cinp_all), CALIB_N, replace=False)
                    calib_cache[key] = calibrate(task, cell, cinp_all[idx], clab_all[idx], batch)
                absmax = calib_cache[key]
            for li, mod in enumerate(linears(m)):
                h = ActHook(kind, p, cell["seed"], li, coh_rows=t["coh"])
                if absmax is not None:
                    h.absmax = absmax[li]
                hooks.append((h, mod.register_forward_pre_hook(h)))
        entries.append({"cell": cell, "model": m, "hooks": hooks, "sig_w": sig_w, "probes": probes})
    entries[0]["model"] = base.to(DEV)
    entries[0]["probes"] = []

    n = len(inp)
    is_maze = maze_d is not None
    K = len(entries)
    acc = [{"exact": np.zeros(nsup), "cellc": np.zeros(nsup), "celln": np.zeros(nsup), "fid": np.zeros(nsup),
            "valid": np.zeros(nsup), "vshort": np.zeros(nsup),
            "keep": {s: [] for s in KEEP_STEPS if s <= nsup}, "keep_valid": [], "keep_vshort": []} for _ in range(K)]
    t0 = time.time()
    for bi, s in enumerate(range(0, n, batch)):
        bt = make_batch(inp, lab, s, s + batch)
        y = bt["labels"]
        mask = y != 0
        carries = [init_carry(e["model"], bt) for e in entries]
        for e in entries:
            for h, _ in e["hooks"]:
                h.coh_on = bi == 0
            for pr, _, _ in e["probes"]:
                pr.on = bi == 0
        inp_np = inp[s:s + batch]
        for step in range(nsup):
            zref = None
            for k, e in enumerate(entries):
                for h, _ in e["hooks"]:
                    h.new_step()
                for pr, _, _ in e["probes"]:
                    pr.step = step + 1
                carries[k], out = e["model"](carries[k], bt)
                zH = carries[k].inner_carry.z_H
                if k == 0:
                    zref = zH
                    fid = torch.ones(zH.shape[0], device=DEV)
                else:
                    fid = F.cosine_similarity(zH.flatten(1).float(), zref.flatten(1).float(), dim=-1)
                preds = out["logits"].argmax(-1)
                corr = (preds == y) & mask
                ex = corr.sum(-1) == mask.sum(-1)
                a = acc[k]
                a["exact"][step] += int(ex.sum())
                a["cellc"][step] += int(corr.sum())
                a["celln"][step] += int(mask.sum())
                a["fid"][step] += float(fid.sum())
                if (step + 1) in a["keep"]:
                    a["keep"][step + 1].extend(ex.int().cpu().tolist())
                if is_maze:
                    pn = preds.cpu().numpy()
                    vv = [maze_valid(pn[i], inp_np[i], maze_d[s + i]) for i in range(len(pn))]
                    a["valid"][step] += sum(v for v, _ in vv)
                    a["vshort"][step] += sum(w for _, w in vv)
                    if step + 1 == nsup:
                        a["keep_valid"].extend(int(v) for v, _ in vv)
                        a["keep_vshort"].extend(int(w) for _, w in vv)
                if step + 1 == nsup:
                    a.setdefault("final", []).extend(ex.int().cpu().tolist())
        print(f"    batch {bi}: {min(s + batch, n)}/{n} ({time.time() - t0:.0f}s)", flush=True)

    recs = []
    for k, e in enumerate(entries):
        a = acc[k]
        cell = e["cell"]
        hooks = [h for h, _ in e["hooks"]]
        coh = [c for h in hooks for c in h.coherence()]
        rho = {}
        for pr, _, _ in e["probes"]:
            for st, v in pr.acc.items():
                rho.setdefault(st, []).extend(v)
        rec = {"task": task, "cell": cell["name"], "w": cell["w"], "a": cell["a"], "seed": cell["seed"],
               "H_cycles": meta["H_cycles"], "L_cycles": meta["L_cycles"], "nsup": nsup, "n": n,
               "exact": (a["exact"] / n).tolist(), "cell_acc": (a["cellc"] / np.maximum(a["celln"], 1)).tolist(),
               "fidelity_H": (a["fid"] / n).tolist(),
               "correct": a["final"], "correct_at": {str(k2): v for k2, v in a["keep"].items()},
               "sigma_eq_w": float(np.mean(e["sig_w"])) if e["sig_w"] else None,
               "sigma_eq_w_max": float(np.max(e["sig_w"])) if e["sig_w"] else None,
               "sigma_eq_a": (float(sum(h.seq_sum for h in hooks) / max(sum(h.seq_n for h in hooks), 1))
                              if hooks else None),
               "coherence_a": float(np.mean(coh)) if coh else None,
               "coherence_a_median": float(np.median(coh)) if coh else None,
               "coherence_steps": nsup,
               "rho_out_by_step": ({str(st): float(np.mean(v)) for st, v in sorted(rho.items())} if rho else None),
               "rho_out": float(np.mean([x for v in rho.values() for x in v])) if rho else None,
               "calib_absmax": (calib_cache.get((json.dumps(cell["w"]), cell["a"][1], cell["seed"]))
                                if cell["a"] and cell["a"][0] == "ts" else None)}
        if is_maze:
            rec["valid"] = (a["valid"] / n).tolist()
            rec["valid_shortest"] = (a["vshort"] / n).tolist()
            rec["correct_valid"] = a["keep_valid"]
            rec["correct_valid_shortest"] = a["keep_vshort"]
        recs.append(rec)
        for _, hd in e["hooks"]:
            hd.remove()
        for _, h1, h2 in e["probes"]:
            h1.remove()
            h2.remove()
    del entries
    torch.cuda.empty_cache()
    return recs


# ----------------------------------------------------------------------------- plans
LADDER = ["w8t+a8ts@0", "w8t+a8ts@1", "w8t+a8ts@2",
          "w4t", "w4c", "w4g32", "w4mx",
          "a4ts@0", "a4ts@1", "a4ts@2", "a4td", "a4tok", "a4g32", "a4mx",
          "w4t+a4ts@0", "w4t+a4ts@1", "w4t+a4ts@2", "w4t+a4td", "w4t+a4tok", "w4t+a4g32", "w4t+a4g16",
          "w4c+a4tok", "w4g32+a4g32", "w4mx+a4mx"]
DEPTH_CELLS = ["w4c", "w4t", "w4mx+a4mx", "w4t+a4ts@0"]
SIGMAS = {"sudoku_mlp": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0],
          "sudoku_attn": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
          "hrm_sudoku": [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5],
          "maze_attn": [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]}
# the sign-consistent weight bias has ~100x the per-application output perturbation (rho_out) of wn at the
# same Frobenius norm (its mean is a rank-1 component with a large operator norm), so it gets its own grid;
# arms are compared on the rho_out axis as well as on sigma.
SIGMAS_WB = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
ARMS = ["an", "af", "ab", "wn", "wb"]


def plan(task, mode, a):
    """-> list of (H, L, nsup, [cells])"""
    t = TASKS[task]
    if a.cells:
        cells = [c for c in a.cells.split(",") if c]
    elif mode == "ladder":
        cells = LADDER
    elif mode == "depth":
        cells = DEPTH_CELLS
    else:
        sig = [float(x) for x in a.sigmas.split(",")] if a.sigmas else SIGMAS[task]
        arms = a.arms.split(",") if a.arms else ARMS
        draws = [int(x) for x in a.draws.split(",")]
        cells = [f"{arm}{s:g}@{d}" for d in draws for arm in arms
                 for s in (SIGMAS_WB if (arm == "wb" and not a.sigmas) else sig)]
    if mode == "depth":
        confs = [(t["H"], t["L"], a.nsup or 32)]
        confs += [(h, t["L"], 16) for h in t["Hs"] if h != t["H"]]
        confs += [(t["H"], l, 16) for l in t["Ls"] if l != t["L"]]
    else:
        confs = [(t["H"], t["L"], a.nsup or 16)]
    return [(H, L, ns, cells) for H, L, ns in confs]


def _f(x):
    return "-" if x is None else f"{x:.3f}"


def run(a):
    task = a.task
    t = TASKS[task]
    n = a.n or t["n"]
    batch = a.batch or t["batch"]
    kind = (bs.TRM_TASKS.get(task) or bs.HRM_TASKS[task])["data"]
    lo, hi = t["pool"]
    allq, alla = bs.load_qa(kind, max(hi, n))
    inp, lab = allq[:n], alla[:n]
    cinp_all, clab_all = allq[lo:hi], alla[lo:hi]
    assert n <= lo, "evaluation rows overlap the calibration pool"
    maze_d = None
    if kind == "maze":
        maze_d = [bfs_dist(r) for r in inp]
        ref_len = [int((lab[i] == 5).sum()) for i in range(len(lab))]
        agree = float(np.mean([maze_d[i] - 1 == ref_len[i] for i in range(len(lab))]))
        print(f"maze: reference path length == BFS shortest - 1 for {agree*100:.1f}% of {len(lab)}", flush=True)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"task": task, "mode": a.mode, "n": n, "records": [], "calib_n": CALIB_N, "calib_pool": [lo, hi]}
    if out_path.exists():
        try:
            data = json.load(open(out_path))
        except (json.JSONDecodeError, ValueError):
            print(f"  {out_path} unreadable; starting fresh", flush=True)
    done = {(r["cell"], r["H_cycles"], r["L_cycles"], r["nsup"]) for r in data["records"]}
    have_fp = {(r["H_cycles"], r["L_cycles"], r["nsup"]) for r in data["records"] if r["cell"] == "fp32"}
    calib_cache = {}
    for H, L, ns, cells in plan(task, a.mode, a):
        todo = [c for c in cells if (c, H, L, ns) not in done]
        print(f"{task} {a.mode} H={H} L={L} nsup={ns}: {len(todo)}/{len(cells)} cells to run", flush=True)
        if not todo and (H, L, ns) in have_fp:
            continue
        groups = [todo[i:i + a.group] for i in range(0, len(todo), a.group)] or [[]]
        for grp in groups:
            t1 = time.time()
            recs = evaluate(task, H, L, ns, grp, inp, lab, batch, calib_cache, cinp_all, clab_all, lo, maze_d)
            for r in recs:
                if r["cell"] == "fp32":
                    if (H, L, ns) in have_fp:
                        continue
                    have_fp.add((H, L, ns))
                data["records"].append(r)
                e = r["exact"]
                extra = f" valid {r['valid'][-1]*100:5.1f} vshort {r['valid_shortest'][-1]*100:5.1f}" if "valid" in r else ""
                print(f"  H={H} L={L} nsup={ns} {r['cell']:14s} exact@1 {e[0]*100:5.1f} @final {e[-1]*100:5.1f} "
                      f"cell {r['cell_acc'][-1]*100:5.1f} fidH {r['fidelity_H'][-1]:.3f}{extra} "
                      f"sig_w {_f(r['sigma_eq_w'])} sig_a {_f(r['sigma_eq_a'])} coh {_f(r['coherence_a'])} rho {_f(r['rho_out'])}", flush=True)
            tmp = str(out_path) + ".tmp"
            json.dump(data, open(tmp, "w"))
            os.replace(tmp, out_path)
            print(f"  group of {len(grp)} done in {time.time() - t1:.0f}s; wrote {out_path}", flush=True)
    print("ALL DONE", out_path, len(data["records"]), "records", flush=True)


# ----------------------------------------------------------------------------- analysis
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def _agg(recs, key, idx=-1):
    v = [r[key][idx] for r in recs if key in r]
    return {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v)), "k": len(v)} if v else None


def analyze(pattern, summary_path):
    files = sorted(glob.glob(pattern))
    recs = []
    for f in files:
        d = json.load(open(f))
        if not isinstance(d, dict) or "records" not in d or "mode" not in d:
            continue
        for r in d["records"]:
            r["_mode"] = d["mode"]
            r["_file"] = os.path.basename(f)
            recs.append(r)
    S = {"files": [os.path.basename(f) for f in files], "ladder": {}, "depth": {}, "biasnoise": {}}

    def base_name(c):
        return c.split("@")[0]

    # one depth per model: ladder and biasnoise read only the default (H, L) at nsup 16 (TRM H=3, HRM H=2),
    # and every quantized cell is paired with the fp32 record of the same (task, H, L, nsup)
    def at_default(r):
        t = TASKS[r["task"]]
        return r["H_cycles"] == t["H"] and r["L_cycles"] == t["L"] and r["nsup"] == 16

    def fp32_of(r):
        return next((q for q in recs if q["task"] == r["task"] and q["cell"] == "fp32" and q["H_cycles"] == r["H_cycles"]
                     and q["L_cycles"] == r["L_cycles"] and q["nsup"] == r["nsup"] and q["n"] == r["n"]), None)

    # ---- ladder
    for task in TASKS:
        rr = [r for r in recs if r["task"] == task and r["_mode"] == "ladder" and at_default(r)]
        if not rr:
            continue
        fp = [fp32_of(rr[0])] if fp32_of(rr[0]) else []
        out = {}
        for bname in dict.fromkeys(base_name(r["cell"]) for r in rr):
            g = [r for r in rr if base_name(r["cell"]) == bname]
            n = g[0]["n"]
            ex = _agg(g, "exact")
            row = {"n": n, "draws": len(g), "exact": ex,
                   "exact_ci95_first": wilson(round(g[0]["exact"][-1] * n), n),
                   "cell": _agg(g, "cell_acc"), "fidelity_H": _agg(g, "fidelity_H"),
                   "exact_step1": _agg(g, "exact", 0),
                   "sigma_eq_w": g[0]["sigma_eq_w"], "sigma_eq_a": float(np.mean([r["sigma_eq_a"] for r in g])) if g[0]["sigma_eq_a"] is not None else None,
                   "coherence_a": float(np.mean([r["coherence_a"] for r in g])) if g[0]["coherence_a"] is not None else None,
                   "rho_out": float(np.mean([r["rho_out"] for r in g])) if g[0].get("rho_out") is not None else None,
                   "rho_out_step1": (float(np.mean([r["rho_out_by_step"]["1"] for r in g]))
                                     if g[0].get("rho_out_by_step") else None),
                   "exact_curve_mean": np.mean([r["exact"] for r in g], 0).tolist()}
            if "valid" in g[0]:
                row["valid"] = _agg(g, "valid")
                row["valid_shortest"] = _agg(g, "valid_shortest")
            if fp:
                row["exact_minus_fp32"] = ex["mean"] - fp[0]["exact"][-1]
                row["fp32_config"] = [fp[0]["H_cycles"], fp[0]["L_cycles"], fp[0]["nsup"]]
            out[bname] = row
        S["ladder"][task] = {"H_cycles": TASKS[task]["H"], "L_cycles": TASKS[task]["L"], "nsup": 16, "cells": out}

    # ---- depth
    for task in TASKS:
        rr = [r for r in recs if r["task"] == task and r["_mode"] == "depth"]
        if not rr:
            continue
        t = TASKS[task]
        out = {"nsup_curve": {}, "H_sweep": {}, "L_sweep": {}}
        for r in rr:
            key = r["cell"]
            if r["H_cycles"] == t["H"] and r["L_cycles"] == t["L"] and r["nsup"] >= 32:
                f0 = fp32_of(r)
                out["nsup_curve"][key] = {"exact": r["exact"], "cell": r["cell_acc"], "fidelity_H": r["fidelity_H"],
                                          "gap_to_fp32": (np.array(r["exact"]) - np.array(f0["exact"])).tolist() if f0 else None,
                                          **({"valid": r["valid"], "valid_shortest": r["valid_shortest"]} if "valid" in r else {})}
        for sweep, fld, dflt in (("H_sweep", "H_cycles", t["H"]), ("L_sweep", "L_cycles", t["L"])):
            other = "L_cycles" if fld == "H_cycles" else "H_cycles"
            odef = t["L"] if fld == "H_cycles" else t["H"]
            for r in rr:
                if r[other] != odef:
                    continue
                v = r[fld]
                i16 = min(16, r["nsup"]) - 1
                d = out[sweep].setdefault(r["cell"], {})
                f0 = fp32_of(r)
                d[str(v)] = {"exact16": r["exact"][i16], "exact1": r["exact"][0], "cell16": r["cell_acc"][i16],
                             "gap16": (r["exact"][i16] - f0["exact"][i16]) if f0 else None,
                             "gap1": (r["exact"][0] - f0["exact"][0]) if f0 else None,
                             "fid16": r["fidelity_H"][i16],
                             **({"valid16": r["valid"][i16], "vshort16": r["valid_shortest"][i16]} if "valid" in r else {})}
        S["depth"][task] = out

    # ---- bias vs noise
    for task in TASKS:
        rr = [r for r in recs if r["task"] == task and r["_mode"] == "biasnoise" and at_default(r)]
        if not rr:
            continue
        f0 = fp32_of(rr[0])
        fp_ex = f0["exact"][-1] if f0 else None
        out = {"fp32_exact": fp_ex, "n": rr[0]["n"], "H_cycles": TASKS[task]["H"], "L_cycles": TASKS[task]["L"],
               "nsup": 16, "arms": {}}
        for arm in ARMS:
            ar = [r for r in rr if r["cell"].startswith(arm) and r["cell"] != "fp32"]
            if not ar:
                continue
            sig = sorted({float(base_name(r["cell"])[2:]) for r in ar})
            draws = sorted({r["seed"] for r in ar})
            rows = {}
            for s in sig:
                g = [r for r in ar if float(base_name(r["cell"])[2:]) == s]
                rows[f"{s:g}"] = {"exact": _agg(g, "exact"), "cell": _agg(g, "cell_acc"), "fidelity_H": _agg(g, "fidelity_H"),
                                  "exact_step1": _agg(g, "exact", 0), "fidelity_step1": _agg(g, "fidelity_H", 0),
                                  "coherence_a": (float(np.mean([r["coherence_a"] for r in g]))
                                                  if g[0]["coherence_a"] is not None else None),
                                  "rho_out": float(np.mean([r["rho_out"] for r in g])) if g[0].get("rho_out") is not None else None,
                                  **({"valid": _agg(g, "valid"), "valid_shortest": _agg(g, "valid_shortest")} if "valid" in g[0] else {})}
            half, rhalf = {}, {}
            for d in draws:
                pts = sorted(((float(base_name(r["cell"])[2:]), r) for r in ar if r["seed"] == d), key=lambda z: z[0])
                curve = [(sg, r["exact"][-1]) for sg, r in pts]
                half[str(d)] = sigma_half(curve, fp_ex) if fp_ex else None
                if fp_ex and all(r.get("rho_out") is not None for _, r in pts):
                    rhalf[str(d)] = sigma_half([(r["rho_out"], r["exact"][-1]) for _, r in pts], fp_ex)
            out["arms"][arm] = {"sigmas": rows, "sigma_half_by_draw": half, "rho_half_by_draw": rhalf}
        S["biasnoise"][task] = out
    # flat point lists for figures: accuracy against the common perturbation axis (rho_out), every cell
    pts = []
    for r in recs:
        if r["cell"] == "fp32" or r["_mode"] not in ("ladder", "biasnoise") or not at_default(r):
            continue
        f0 = fp32_of(r)
        b = base_name(r["cell"])
        arm = b[:2] if r["_mode"] == "biasnoise" else "quant"
        pts.append({"task": r["task"], "mode": r["_mode"], "cell": r["cell"], "arm": arm, "seed": r["seed"],
                    "sigma": float(b[2:]) if arm != "quant" else None, "rho_out": r.get("rho_out"),
                    "rho_out_step1": (r.get("rho_out_by_step") or {}).get("1"),
                    "sigma_eq_a": r.get("sigma_eq_a"), "sigma_eq_w": r.get("sigma_eq_w"),
                    "coherence_a": r.get("coherence_a"), "exact": r["exact"][-1], "exact_step1": r["exact"][0],
                    "fp32_exact": f0["exact"][-1] if f0 else None, "H_cycles": r["H_cycles"], "L_cycles": r["L_cycles"],
                    "cell_acc": r["cell_acc"][-1], "fidelity_H": r["fidelity_H"][-1],
                    "valid": r["valid"][-1] if "valid" in r else None,
                    "valid_shortest": r["valid_shortest"][-1] if "valid_shortest" in r else None})
    S["fig_points"] = pts
    S["fp32"] = {task: next((r["exact"][-1] for r in recs if r["task"] == task and r["cell"] == "fp32"
                             and at_default(r)), None)
                 for task in TASKS}
    json.dump(S, open(summary_path, "w"), indent=1)
    print_summary(S)
    print("wrote", summary_path)


def sigma_half(curve, fp_ex):
    """first sigma where exact < fp32/2, linearly interpolated from the grid point before it."""
    prev = None
    for s, e in curve:
        if e < fp_ex / 2:
            if prev is None:
                return {"bracket": [None, s], "interp": None, "below_grid": True}
            s0, e0 = prev
            frac = (e0 - fp_ex / 2) / max(e0 - e, 1e-9)
            return {"bracket": [s0, s], "interp": s0 + frac * (s - s0)}
        prev = (s, e)
    return {"bracket": [curve[-1][0] if curve else None, None], "interp": None}


def print_summary(S):
    for task, L in S["ladder"].items():
        rows = L["cells"]
        print(f"\n== ladder {task} (H={L['H_cycles']} L={L['L_cycles']} nsup={L['nsup']}, n={next(iter(rows.values()))['n']})"
              f" exact% [min-max over draws] cell% fidH sig_w sig_a coh")
        for name, r in rows.items():
            ex = r["exact"]
            rng = f"[{ex['min']*100:.1f}-{ex['max']*100:.1f}]" if r["draws"] > 1 else ""
            v = f" valid {r['valid']['mean']*100:.1f} vshort {r['valid_shortest']['mean']*100:.1f}" if "valid" in r else ""
            f = lambda x: "-" if x is None else f"{x:.3f}"
            print(f"  {name:14s} {ex['mean']*100:5.1f} {rng:13s} cell {r['cell']['mean']*100:5.1f} fid {r['fidelity_H']['mean']:.3f}"
                  f" sw {f(r['sigma_eq_w'])} sa {f(r['sigma_eq_a'])} coh {f(r['coherence_a'])} rho {f(r['rho_out'])}{v}")
    for task, d in S["depth"].items():
        print(f"\n== depth {task}: exact% at nsup 1,2,4,8,16,32 (default H/L)")
        for c, v in d["nsup_curve"].items():
            e = v["exact"]
            print(f"  {c:14s} " + " ".join(f"{e[i-1]*100:5.1f}" for i in KEEP_STEPS if i <= len(e)))
        for sw in ("H_sweep", "L_sweep"):
            print(f"  {sw} exact% at nsup 16:")
            for c, v in d[sw].items():
                print(f"    {c:14s} " + " ".join(f"{k}:{v[k]['exact16']*100:.1f}"
                                                 + ("" if v[k]["gap16"] is None or c == "fp32" else f"({v[k]['gap16']*100:+.1f})")
                                                 for k in sorted(v, key=int)))
    for task, d in S["biasnoise"].items():
        print(f"\n== bias vs noise {task} (n={d['n']}, fp32 {d['fp32_exact']*100:.1f}): exact% mean [min-max] per sigma; sigma_half by draw")
        for arm, v in d["arms"].items():
            line = " ".join(f"{s}:{x['exact']['mean']*100:.1f}[{x['exact']['min']*100:.0f}-{x['exact']['max']*100:.0f}]"
                            f"(rho {x['rho_out']:.2f})" if x.get("rho_out") is not None else
                            f"{s}:{x['exact']['mean']*100:.1f}[{x['exact']['min']*100:.0f}-{x['exact']['max']*100:.0f}]"
                            for s, x in v["sigmas"].items())
            hh = {k: (None if h is None else h["interp"] if h["interp"] is not None else f"<{h['bracket'][1]}"
                      if h.get("below_grid") else f">{h['bracket'][0]}") for k, h in v["sigma_half_by_draw"].items()}
            rh = {k: (None if h is None else round(h["interp"], 3) if h["interp"] is not None else f"<{h['bracket'][1]:.3f}"
                      if h.get("below_grid") else f">{h['bracket'][0]:.3f}") for k, h in v.get("rho_half_by_draw", {}).items()}
            print(f"  {arm}: {line}\n      sigma_half {hh}  rho_half {rh}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(TASKS))
    ap.add_argument("--mode", choices=["ladder", "depth", "biasnoise"])
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--nsup", type=int, default=0)
    ap.add_argument("--cells", default="", help="comma list overriding the mode's cell plan")
    ap.add_argument("--sigmas", default="")
    ap.add_argument("--arms", default="")
    ap.add_argument("--draws", default="0,1,2")
    ap.add_argument("--group", type=int, default=8, help="cells run in lockstep with one fp32 pass")
    ap.add_argument("--chain", default="", help="file with one argument line per run, executed in order "
                                                "(fewer SLURM jobs; each run resumes on its own)")
    ap.add_argument("--analyze", default="")
    ap.add_argument("--summary", default="x7_summary.json")
    a = ap.parse_args()
    if a.analyze:
        analyze(a.analyze, a.summary)
        return
    print(f"device {DEV} {torch.cuda.get_device_name(0) if DEV == 'cuda' else ''}", flush=True)
    if a.chain:
        import shlex
        lines = [ln.strip() for ln in open(os.path.expanduser(a.chain)) if ln.strip() and not ln.startswith("#")]
        for i, ln in enumerate(lines):
            sub = ap.parse_args(shlex.split(ln))
            print(f"=== chain {i + 1}/{len(lines)}: {ln}", flush=True)
            torch.manual_seed(0)
            np.random.seed(0)
            run(sub)
        print("CHAIN DONE", flush=True)
        return
    torch.manual_seed(0)
    np.random.seed(0)
    run(a)


if __name__ == "__main__":
    main()
