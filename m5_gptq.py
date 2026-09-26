#!/usr/bin/env python3
"""M5 x post-training compression: GPTQ weights for TRM / HRM on the same grids as the RTN specs (breadth_sweep formats).

Spec "gq<fmt>" (gqw3c, gqw3g32, gqw2g32, gqw4t) = GPTQ (m3_calib.gptq_tensor: Frantar et al., blocksize 128, 1% damping)
on the grid of <fmt>. Every CastedLinear is quantized, as quantize_weights_ does for the RTN spec.
Grid: symmetric, per-tensor (t) / per-channel (c) / per-group (gN) scales. The clip ratio is chosen per tensor by weight MSE
(r in 0.50..1.00 x absmax, data-free), i.e. MSE + GPTQ, one of M3's layer-wise baselines.
Hessian: X^T X of every linear's inputs along the fp32 trajectory (model's own initial carry, 16 default-H supervision
steps) of calibration puzzles rows 2048-2303 of the test set, disjoint from the dev (0-999) and held-out (1000-1999) rows.
Non-sequential, as M3: recursion shares weights across depth, so every Hessian is taken from the fp32 model.
Weights are saved to $HOME/m5/gptq/<task>/<spec>.pt (<task>__<tag> for a trained variant). Consumers
(exit_sweep, exit_polish, m2_bestofk) call load_if_gptq() after build, and wait for the file (one builder line per task).

    python m5_gptq.py --task sudoku_mlp --specs gqw3c,gqw3g32,gqw2g32,gqw4t
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breadth_sweep as bs  # noqa: E402

GDIR = Path(os.environ.get("M5_GPTQ_DIR", Path.home() / "m5" / "gptq"))


def is_gptq(spec):
    return spec.startswith("gq")


def path_of(task, spec):
    return GDIR / task / f"{spec}.pt"


def load_if_gptq(m, task, spec, wait_s=3600):
    """replace m's weights by the saved GPTQ weights (m built as fp32); wait for the builder if needed."""
    if not is_gptq(spec):
        return m
    p = path_of(task, spec)
    t0 = time.time()
    while not p.exists():
        if time.time() - t0 > wait_s:
            raise FileNotFoundError(p)
        time.sleep(20)
    sd = torch.load(p, map_location="cpu")
    for mod_name, W in sd.items():
        mod = dict(m.named_modules())[mod_name]
        mod.weight.data = W.to(mod.weight.data.device, mod.weight.data.dtype)
    return m


def grid(W, bits, gran, r):
    qmax = 2 ** (bits - 1) - 1
    if gran == "t":
        s = (r * W.abs().max() / qmax).clamp_min(1e-8).reshape(1, 1).expand_as(W)
    elif gran == "c":
        s = (r * W.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-8).expand_as(W)
    else:
        g = int(gran[1:])
        out_f, in_f = W.shape
        pad = (-in_f) % g
        Wp = torch.nn.functional.pad(W, (0, pad)).view(out_f, -1, g)
        s = (r * Wp.abs().amax(2, keepdim=True) / qmax).clamp_min(1e-8).expand_as(Wp).reshape(out_f, -1)[:, :in_f]
    return s.contiguous(), qmax


@torch.no_grad()
def build_one(task, spec, n_cal=256, cal_off=2048, nsup=16):
    import m3_calib as M3
    from detector_sweep import explicit_carry
    q = bs.parse_quant(spec[2:])
    t = bs.TRM_TASKS.get(task) or bs.HRM_TASKS[task]
    m, _ = bs.build(task, None, 10 ** 6, 64)
    m = m.to(bs.DEV).eval()
    lins = {n: mod for n, mod in m.named_modules() if mod.__class__.__name__ == "CastedLinear"}
    H = {n: None for n in lins}
    hooks = []
    for n, mod in lins.items():
        def f(mod_, args, n=n):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            H[n] = x.T @ x if H[n] is None else H[n] + x.T @ x
        hooks.append(mod.register_forward_pre_hook(f))
    inp, lab = bs.load_qa(t["data"], cal_off + n_cal)
    inp, lab = inp[cal_off:], lab[cal_off:]
    for s in range(0, n_cal, 64):
        x = torch.from_numpy(inp[s:s + 64]).to(bs.DEV)
        y = torch.from_numpy(lab[s:s + 64]).to(bs.DEV)
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(x.shape[0], dtype=torch.int32, device=bs.DEV)}
        carry = explicit_carry(m, bt)
        for _ in range(nsup):
            carry, _o = m(carry, bt)
            carry.halted.zero_()
    for h in hooks:
        h.remove()
    out = {}
    for n, mod in lins.items():
        W = mod.weight.data.float()
        best = None
        for r in [1.0 - 0.025 * i for i in range(21)]:
            s, qmax = grid(W, q["bits"], q["gran"], r)
            err = ((torch.clamp(torch.round(W / s), -qmax, qmax) * s - W) ** 2).sum().item()
            if best is None or err < best[0]:
                best = (err, r)
        s, qmax = grid(W, q["bits"], q["gran"], best[1])
        out[n] = M3.gptq_tensor(W, H[n], s, qmax).to(mod.weight.data.dtype).cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--specs", required=True)
    ap.add_argument("--ckpt", default="")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    if a.ckpt:
        bs.CKPT_OVERRIDE = a.ckpt
    for spec in a.specs.split(","):
        p = path_of(a.task, spec)
        if p.exists():
            print(f"{spec}: exists", flush=True)
            continue
        t0 = time.time()
        sd = build_one(a.task, spec)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(sd, str(p) + ".tmp")
        os.replace(str(p) + ".tmp", p)
        print(f"{a.task} {spec}: GPTQ weights saved ({len(sd)} linears, {time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
