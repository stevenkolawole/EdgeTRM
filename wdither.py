#!/usr/bin/env python3
"""Per-iteration weight re-draw ("dither") shared by the DeepThinking (dt_exit.py) and MDEQ (deq_sweep.py)
harnesses. E6 (MLSys 2027): the perturb-every-step method of m2_bestofk.py (TRM / HRM) carried to the third and
fourth looped families.

Formulas are m2_bestofk.sr_quantize_ and m2_bestofk.gmatch_perturb_ verbatim, applied to a weight viewed as
(out, fan_in) so that Conv weights (o, i, kh, kw) use the same grid as dt_breadth.quantize_ / deq_sweep.quantize
(round-to-nearest on the flattened fan-in), i.e. the dither draws live on exactly the RTN grid of the same spec:

  sr_round   q = floor(W / s + u), u ~ U[0, 1): unbiased stochastic rounding onto the spec's grid, one draw per call
  gmatch     W + s * sqrt(f (1 - f)) * z, z ~ N(0, 1), f = frac(W / s): the Gaussian control with the SAME per-weight
             variance as stochastic rounding on that grid, no lattice (m2 arm 5, "gdither")

Seeds follow m2_bestofk: draw_seed(dseed, k, t) = dseed * 7907 + 100003 * (k + 1) + t with dseed = 7919 * wseed + 17
(wseed = 0 for the released checkpoints), k = the re-draw sequence (0..K-1, the "seed" of the arm) and t = the
iteration index, so sequence k of a cell is one fixed weight trajectory shared by every puzzle / image.
"""
import re

import torch
import torch.nn.functional as F


def draw_seed(dseed, k, t):
    return int(dseed) * 7907 + 100003 * (int(k) + 1) + int(t)


def base_dseed(wseed=0):
    return 7919 * int(wseed) + 17


ARM_RE = re.compile(r"^(?P<base>[^@]+)(?:@(?P<arm>s|dither|gdither)(?P<k>\d+))?$")


def parse_arm(spec):
    """'w4c' -> ('w4c', 'none', 0); 'noise0.2@s1' -> ('noise0.2', 's', 1); 'w4c@dither2' -> ('w4c', 'dither', 2)."""
    m = ARM_RE.match(spec)
    assert m, f"bad spec {spec}"
    return m["base"], (m["arm"] or "none"), int(m["k"] or 0)


def _grid(W, bits, gran):
    """(scale broadcastable over W (out, fan_in), zero point or None, qmax); the RTN scales of the spec."""
    if gran == "a":
        qmax = 2 ** bits - 1
        wmin, wmax = W.amin(dim=1, keepdim=True), W.amax(dim=1, keepdim=True)
        sc = ((wmax - wmin) / qmax).clamp_min(1e-8)
        return sc, torch.round(-wmin / sc), qmax
    qmax = 2 ** (bits - 1) - 1
    if gran == "t":
        return (W.abs().max() / qmax).clamp_min(1e-8), None, qmax
    if gran == "c":
        return (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8), None, qmax
    g = int(gran[1:])
    out_f, in_f = W.shape
    Wp = F.pad(W, (0, (-in_f) % g)).view(out_f, -1, g)
    sc = (Wp.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-8)
    return sc.expand(-1, -1, g).reshape(out_f, -1)[:, :in_f], None, qmax


@torch.no_grad()
def sr_round(W, bits, gran, gen):
    """One stochastic-rounding draw of W (any shape; flattened to (out, fan_in)) onto the (bits, gran) grid."""
    shape = W.shape
    W2 = W.reshape(shape[0], -1)
    U = torch.rand(W2.shape, generator=gen, device=W2.device, dtype=W2.dtype)
    sc, zp, qmax = _grid(W2, bits, gran)
    if zp is not None:
        Wq = (torch.clamp(torch.floor(W2 / sc + U) + zp, 0, qmax) - zp) * sc
    else:
        Wq = torch.floor(W2 / sc + U).clamp(-qmax, qmax) * sc
    return Wq.reshape(shape)


@torch.no_grad()
def gmatch(W, bits, gran, gen):
    """Zero-mean Gaussian perturbation of W with the per-element variance of stochastic rounding on the grid."""
    shape = W.shape
    W2 = W.reshape(shape[0], -1)
    Z = torch.randn(W2.shape, generator=gen, device=W2.device, dtype=W2.dtype)
    sc, _, _ = _grid(W2, bits, gran)
    f = W2 / sc - torch.floor(W2 / sc)
    return (W2 + sc * torch.sqrt(f * (1 - f)) * Z).reshape(shape)


def redraw_(slots, arm, bits, gran, seed, device):
    """Re-draw every (put, W_master) slot from one generator seeded with `seed`, walked over the slots in order
    (m2_bestofk walks _linears(m) with one generator per step). arm: 'dither' (SR) or 'gdither' (Gaussian)."""
    gen = torch.Generator(device=device).manual_seed(int(seed))
    fn = sr_round if arm == "dither" else gmatch
    for put, W in slots:
        put(fn(W, bits, gran, gen))


def make_hook(slots, arm, bits, gran, k, device, wseed=0):
    """step_hook(t): weights re-drawn from the master before iteration t of re-draw sequence k."""
    ds = base_dseed(wseed)

    def hook(t):
        redraw_(slots, arm, bits, gran, draw_seed(ds, k, t), device)
    return hook
