#!/usr/bin/env python3
"""FI: hardware-fault injection on the compressed recursive reasoners (MLSys 2027), fresh vs fixed faults.

Hypothesis (ICLR "A Tilted Bowl Is Not a Slippery Slope"): a looped model tolerates FRESH errors (drawn anew every
loop; the loop settles back) and is moved by FIXED errors (the same every loop; the settled state shifts).  Hardware
faults come in both kinds: transient bit flips and rounding = fresh; stuck-at bits, a wrong quantization scale, a
broken kernel = fixed.  This script injects both kinds into the weights of the same block and records the per-step
trajectory, so the read-out at 16 / 64 / 512 steps is the test.

Arms (--arms, on top of a base --base fp32 | w8c; w8c = breadth_sweep RTN, per-channel symmetric int8, so the codes
are real 8-bit two's-complement integers and the bit flips hit them; fp32 flips hit the 23 mantissa bits):
  none          the base model, no fault (reference; K forced to 1)
  bitflip<p>    FRESH: before EVERY supervision step every bit of every linear's weight tensor is flipped independently
                with probability p (int8 codes: 8 bits incl. the sign; fp32: the 23 mantissa bits), a fresh draw each
                step; the weights are rebuilt from the intact master for the next step, so a flip lives one step.
  stuck<p>      FIXED: one draw of the same flips (same p, same generator family) applied once and kept for all steps.
  scale<f>      FIXED: one randomly chosen linear layer's weights multiplied by (1 + f) for all steps (a wrong
                quantization scale / a mis-calibrated kernel); the layer is drawn per seed k, so K = 3 covers 3 layers.
  scalefresh<f> FRESH: a different randomly chosen layer is scaled by (1 + f) at every step (the others intact).
Rates: p in {1e-6, 1e-5, 1e-4, 1e-3}; f in {0.05, 0.1, 0.2}.  K = 3 seeds per stochastic arm (seed k, step t ->
generator seed; fixed arms use the t = 0 generator of the fresh arm, so stuck<p> seed k IS the first draw of
bitflip<p> seed k).  Depth D = 512 (Sudoku, held-out rows 1000-1999, n = 1000) or 128 (Maze, rows 500-999, n = 500);
per-step exact correctness and halting logit for every seed and puzzle, checker + hash at 16 / 32 / 64 / ... / D.

Feed-forward control (c): the model run for its default 16 steps only IS the step-16 column of the same record (the
fault process on steps 1..16 does not depend on whether the run continues), so no separate cell is needed; the
read-out compares accuracy at step 16 against step 512 (Sudoku) / 128 (Maze).

Predictions, written BEFORE any cell ran:
  (a) FRESH transient flips (bitflip<p>): accuracy costs little until p is large (the p = 1e-6 .. 1e-4 arms sit
      within a few points of the fault-free base at depth), and accuracy at depth 512 (128 for Maze) is ABOVE
      accuracy at depth 16 for every p where the model has not collapsed: the loop settles back.  The same holds for
      scalefresh<f>.  By the dither result a small fresh perturbation may even lift the depth number above the base.
  (b) FIXED faults of the same magnitude (stuck<p>, scale<f>): a fixed cost that does NOT change with depth (the
      gap to the base at step 16 is about the gap at step 512; the fixed arm ends BELOW the fresh arm of the same p or
      f at depth), or collapse (accuracy near 0 at every checkpoint) once the push exceeds the room: a stuck sign or
      high mantissa bit in a few hundred weights, or a whole layer scaled by 1.2, can move the fixed point out of the
      basin.  In particular stuck<p> at depth 512 <= bitflip<p> at depth 512 at every p, and the ordering widens with p.
  (c) The feed-forward control (step 16 of the same record) cannot recover from (a): at step 16 the fresh-fault
      accuracy is below (or at best equal to) the base at step 16, and the recovery only shows in the depth column.
Read-out: fault_analyze.py (to write) / m2_analyze.py-style: for each (task, base, arm) the mean over seeds of the
exact rate at 16 / 64 / D, the seed spread, and the difference to the base at the same step.

    python fault_inject.py --task sudoku_mlp --base fp32,w8c --arms none,bitflip1e-6,bitflip1e-5 --D 512 --n 1000 \
        --offset 1000 --K 3 --outdir ~/m2_results/cells --index ~/m2_results/index_fault_sudoku_mlp.jsonl
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
from m2_bestofk import DEV, TRM_TASKS, HRM_TASKS, _linears, build, load_qa, bfs_dist, make_hash_w, run_cell_multi, quick_summary  # noqa: E402

FAMILIES = ("none", "bitflip", "stuck", "scale", "scalefresh")
FRESH = {"bitflip", "scalefresh"}


def parse_arm(arm):
    """'bitflip1e-5' -> ('bitflip', 1e-5); 'scalefresh0.1' -> ('scalefresh', 0.1); 'none' -> ('none', 0.0)."""
    if arm == "none":
        return "none", 0.0
    for fam in sorted(FAMILIES, key=len, reverse=True):          # scalefresh before scale
        if arm.startswith(fam):
            return fam, float(arm[len(fam):])
    raise ValueError(arm)


class FaultModel:
    """Holds the intact master (fp32 weights, or int8 codes + per-channel scales for w8c) of every CastedLinear and
    rewrites the live weights with a fault applied. Every apply starts from the intact master, so a fresh fault lives
    exactly one step and a fixed fault is whatever the last apply wrote."""

    def __init__(self, m, base):
        self.m = m
        self.lins = _linears(m)
        self.base = base
        self.master = [mod.weight.data.detach().clone().contiguous() for mod in self.lins]     # fp32
        if base == "w8c":
            qmax = 127
            self.scale = [(W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8) for W in self.master]
            self.code = [torch.round(W / s).clamp(-qmax, qmax).to(torch.int32) for W, s in zip(self.master, self.scale)]
            self.nbits = 8
            self.eff = [(c.float() * s) for c, s in zip(self.code, self.scale)]         # == breadth_sweep RTN w8c
        elif base == "fp32":
            self.nbits = 23
            self.eff = self.master
        else:
            raise ValueError(base)
        self.total_bits = sum(W.numel() for W in self.master) * self.nbits
        self.reset()
        self.last_nflips = 0
        self.last_layer = -1

    @torch.no_grad()
    def reset(self):
        for mod, W in zip(self.lins, self.eff):
            mod.weight.data = W.clone()

    @torch.no_grad()
    def flip(self, p, gen, rng):
        """Every bit of every weight tensor flipped independently with prob p (one draw); weights rebuilt from the
        master with the flips applied. The count per tensor is Binomial(N * nbits, p) (numpy rng); the positions are
        distinct uniform draws (torch generator), i.e. sampling without replacement, which is the Bernoulli field up
        to the negligible chance of a repeated position."""
        nfl = 0
        for i, mod in enumerate(self.lins):
            W = self.eff[i]
            N = W.numel()
            cnt = int(rng.binomial(N * self.nbits, p))
            if cnt == 0:
                mod.weight.data = W.clone()
                continue
            pos = torch.randint(0, N * self.nbits, (cnt,), generator=gen, device=DEV).unique()
            idx, bit = pos // self.nbits, (pos % self.nbits).to(torch.int32)
            mask = torch.zeros(N, dtype=torch.int32, device=DEV)
            mask.index_put_((idx,), torch.ones_like(bit) << bit, accumulate=True)   # distinct (idx, bit): add == or
            nfl += int(pos.numel())
            if self.base == "w8c":
                code = (self.code[i].reshape(-1) & 0xFF) ^ mask                  # 8-bit two's complement, incl. sign
                code = torch.where(code >= 128, code - 256, code)
                mod.weight.data = (code.float().view_as(W) * self.scale[i])
            else:
                bits = W.reshape(-1).view(torch.int32) ^ mask                        # mantissa bits 0..22 only
                mod.weight.data = bits.view(torch.float32).view_as(W).clone()
        self.last_nflips = nfl

    @torch.no_grad()
    def scale_layer(self, f, gen):
        """One randomly chosen linear scaled by (1 + f); every other linear intact."""
        j = int(torch.randint(0, len(self.lins), (1,), generator=gen, device=DEV).item())
        if self.last_layer >= 0:
            self.lins[self.last_layer].weight.data = self.eff[self.last_layer].clone()
        self.lins[j].weight.data = self.eff[j] * (1.0 + f)
        self.last_layer = j


def make_hook(fm, fam, val, seed):
    """step_hook(k, t) for run_cell_multi: fresh arms re-draw before every step; fixed arms draw once at t = 0 with
    the SAME generator as the fresh arm's step 0 (so stuck<p> seed k is bitflip<p> seed k's first draw) and keep it."""
    def gens(k, t):
        s = int(seed) * 7907 + 100003 * (k + 1) + t
        return torch.Generator(device=DEV).manual_seed(s), np.random.default_rng(s)

    def hook(k, t):
        if fam == "none":
            if t == 0:
                fm.reset()
            return
        if fam in FRESH or t == 0:
            if t == 0:
                fm.reset()
                fm.last_layer = -1
            g, rng = gens(k, t)
            if fam in ("bitflip", "stuck"):
                fm.flip(val, g, rng)
            else:
                fm.scale_layer(val, g)
            if t == 0:
                hook.draw0[k] = {"nflips": fm.last_nflips, "layer": fm.last_layer}
    hook.draw0 = {}
    return hook


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TRM_TASKS) + list(HRM_TASKS))
    ap.add_argument("--base", default="fp32,w8c", help="comma list of fp32 | w8c")
    ap.add_argument("--arms", default="none,bitflip1e-6,bitflip1e-5,bitflip1e-4,bitflip1e-3")
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--D", type=int, default=512)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--chk_steps", default="")
    ap.add_argument("--seed", type=int, default=0)
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
    chunk = a.chunk or (1000 if kind == "sudoku" else 128)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hash_w = make_hash_w(inp.shape[1])
    bases = [b.strip() for b in a.base.split(",") if b.strip()]
    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    print(f"{a.task} n={n} offset={a.offset} D={a.D} K={a.K} bases={bases} arms={arms} chunk={chunk} chk_steps={chk_steps} dev={DEV}", flush=True)
    for base in bases:
        todo = []
        for arm in arms:
            fam, val = parse_arm(arm)
            K = 1 if fam == "none" else a.K
            name = f"{a.task}__{base}__fault_{arm}__D{a.D}_K{K}_o{a.offset}{a.tag}.npz"
            if not (outdir / name).exists():
                todo.append((arm, fam, val, K, name))
        if not todo:
            print(f"  {base}: all cells done", flush=True)
            continue
        m, meta = build(a.task, None, a.D, 256)
        m = m.to(DEV).eval()
        fm = FaultModel(m, base)
        print(f"  {base}: {len(fm.lins)} linears, {fm.total_bits / fm.nbits:.0f} weights x {fm.nbits} bits = {fm.total_bits:.3g} bit positions", flush=True)
        get_model = lambda k, _m=m: _m
        get_xform = lambda k: None
        for arm, fam, val, K, name in todo:
            t1 = time.time()
            hook = make_hook(fm, fam, val, a.seed)
            corr, qh, rms, hs, chk, vsh, _ = run_cell_multi(get_model, get_xform, inp, lab, kind, bfs, K, a.D,
                                                            "none", 0.0, a.seed, chunk, chk_steps, hash_w, hook)
            arrs = {"corr": corr, "q": qh, "zl_rms": rms}
            for k in chk_steps:
                arrs[f"hash_{k}"] = hs[k]
                arrs[f"chk_{k}"] = chk[k]
                if vsh is not None:
                    arrs[f"vshort_{k}"] = vsh[k]
            if kind == "maze":
                arrs["bfs"] = bfs
            tmp = outdir / (name + ".tmp.npz")
            np.savez_compressed(tmp, **arrs)
            os.replace(tmp, outdir / name)
            summ = quick_summary(corr, qh, hs, chk, a.D)
            at = {str(s): [float(corr[:, :, s - 1].mean(1)[k]) for k in range(K)] for s in chk_steps}
            rec = {"task": a.task, "base": base, "arm": arm, "family": fam, "rate": val, "fresh": fam in FRESH,
                   "K": K, "D": a.D, "offset": a.offset, "n": n, "chunk": chunk, "seed": a.seed, "nbits": fm.nbits,
                   "n_linears": len(fm.lins), "bit_positions": int(fm.total_bits), "draw0": hook.draw0,
                   "file": name, "H_cycles": meta["H_cycles"], "zl_rms_last": float(rms[-1]),
                   "acc_at": at, "summary": summ, "secs": time.time() - t1, "host": socket.gethostname(),
                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                   "torch": torch.__version__, "utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
            with open(a.index, "a") as f:
                f.write(json.dumps(rec) + "\n")
            line = " ".join(f"@{s} {100 * np.mean(at[str(s)]):5.1f}" for s in chk_steps)
            print(f"  {base:5s} {arm:16s} K={K}: {line}  draw0={hook.draw0.get(0)} zL_rms {rms[-1]:.3f} ({time.time() - t1:.0f}s)", flush=True)
        del m, fm
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
