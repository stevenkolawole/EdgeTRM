#!/usr/bin/env python3
"""M1: iterate cheap, finish exact (precision-switching inference for weight-tied reasoners).

X2 found that the fp32 map, run from the end state of a broken compressed TRM, recovers 95-99% of the
puzzles fp32 solves: the compressed map fails but its state stays in fp32's basin.  This harness runs
SCHEDULES of per-step precision: most recursion steps with the compressed weights ('c'), a few with a
more precise "polish" copy ('P'), and reads the answer out with the head of the last step's model.

Schedule notation (T = 16 supervision steps for TRM/HRM; DT: iterations): tokens joined by '.',
  c12.P4       12 compressed steps, then 4 polish steps (replace: same step count as the baselines)
  c16.P16      16 compressed steps, then up to 16 polish steps appended (read out after every step)
  c15.c|P@7    15 compressed steps, then one step whose LAST 7 block applications use the polish weights
               (TRM Sudoku: one step = H x (L+1) = 21 block applications; HRM: H x (L+1) = 6)
  P2.c12.P2    exact prefix + exact suffix;  (c3.P1)x4  one polish step every 4 steps
  f16, c16     the fp32 / fully compressed references;  c16/hf  compressed steps, fp32 read-out head
  'f' is always the fp32 model; 'P' is replaced by every polish model in --polish.
Compressed cells (--cells, contraction.py syntax, draws with :k): fp32 | w<b><t|c|a|gN> | mx<b> (MXINT,
  block 32, power-of-two scale) | n<B><t|a><b> (nested / Matryoshka: the top b bits of the B-bit grid) |
  noise<s> (Gaussian weight noise, the same draws as contraction.py / X2) | bf16.
Polish models (--polish): fp32 | bf16 | w<b><gran> | mx<b> | lr<r> (compressed W plus the rank-r SVD of
  W - Q(W), factors in fp16; ZeroQuant-V2 LoRC) | lq<r> (the same, activation-scaled SVD: LQER, scale =
  mean |x| per input channel on calibration inputs disjoint from the evaluation rows).

    python polish.py --task sudoku_mlp --n 1000 --cells w4t,noise0.2:3 --polish fp32,w8c,lr8 \
        --out ~/breadth_results/m1/polish_sudoku_mlp.json
Resumable: records keyed by (cell, draw) are skipped when present in --out.
"""
import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _torch():
    import torch
    return torch


# ----------------------------------------------------------------------------- quantizers
def q_mx(x, bits, block=32):
    """OCP-style MXINT (as x7_ingolfsson.q_mx): per block of 32 along the input dim, scale
    2^(floor(log2 absmax) - (bits-2)), symmetric elements in [-qmax, qmax]."""
    torch = _torch()
    F = torch.nn.functional
    qmax = 2 ** (bits - 1) - 1
    last = x.shape[-1]
    pad = (-last) % block
    xp = F.pad(x, (0, pad)).reshape(*x.shape[:-1], -1, block)
    amax = xp.abs().amax(-1, keepdim=True)
    s = torch.pow(2.0, torch.floor(torch.log2(amax.clamp_min(1e-30))) - (bits - 2))
    q = torch.where(amax > 0, torch.round(xp / s).clamp(-qmax, qmax) * s, torch.zeros_like(xp))
    return q.reshape(*x.shape[:-1], -1)[..., :last]


def q_uniform(W, bits, gran):
    """Same arithmetic as breadth_sweep.quantize_weights_ on a 2-D (out, in) matrix."""
    torch = _torch()
    F = torch.nn.functional
    if gran == "a":
        qmax = 2 ** bits - 1
        wmin, wmax = W.amin(1, keepdim=True), W.amax(1, keepdim=True)
        sc = ((wmax - wmin) / qmax).clamp_min(1e-8)
        zp = torch.round(-wmin / sc)
        return (torch.clamp(torch.round(W / sc) + zp, 0, qmax) - zp) * sc
    qmax = 2 ** (bits - 1) - 1
    if gran == "t":
        s = (W.abs().max() / qmax).clamp_min(1e-8)
        return torch.round(W / s).clamp(-qmax, qmax) * s
    if gran == "c":
        s = (W.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-8)
        return torch.round(W / s).clamp(-qmax, qmax) * s
    g = int(gran[1:])
    o, i_ = W.shape
    pad = (-i_) % g
    Wp = F.pad(W, (0, pad)).view(o, -1, g)
    s = (Wp.abs().amax(2, keepdim=True) / qmax).clamp_min(1e-8)
    return (torch.round(Wp / s).clamp(-qmax, qmax) * s).view(o, -1)[:, :i_]


def q_nested(W, B, gran, b):
    """Matryoshka-style slicing (post-training): quantize to the B-bit grid, keep the top b bits.
    gran 't': signed per-tensor symmetric, codes in [-(2^(B-1)-1), 2^(B-1)-1], sliced code
    clamp(round(q / 2^(B-b)), -2^(b-1), 2^(b-1)-1) * 2^(B-b).  gran 'a': unsigned per-channel min/max
    with zero point (MatQuant's setting), sliced code clamp(round(q / 2^(B-b)), 0, 2^b - 1) * 2^(B-b)."""
    torch = _torch()
    sh = 2 ** (B - b)
    if gran == "a":
        qmax = 2 ** B - 1
        wmin, wmax = W.amin(1, keepdim=True), W.amax(1, keepdim=True)
        sc = ((wmax - wmin) / qmax).clamp_min(1e-8)
        zp = torch.round(-wmin / sc)
        q = torch.clamp(torch.round(W / sc) + zp, 0, qmax)
        qs = torch.clamp(torch.round(q / sh), 0, 2 ** b - 1) * sh
        return (qs - zp) * sc
    qmax = 2 ** (B - 1) - 1
    if gran == "t":
        s = (W.abs().max() / qmax).clamp_min(1e-8)
    else:
        s = (W.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-8)
    q = torch.round(W / s).clamp(-qmax, qmax)
    qs = torch.clamp(torch.round(q / sh), -(2 ** (b - 1)), 2 ** (b - 1) - 1) * sh
    return qs * s


def quant_2d(W, spec):
    """Deterministic weight transform for a spec (no noise). W: (out, in) float32."""
    torch = _torch()
    if spec == "fp32":
        return W.clone()
    if spec == "bf16":
        return W.to(torch.bfloat16).float()
    m = re.fullmatch(r"mx(\d+)", spec)
    if m:
        return q_mx(W, int(m.group(1)))
    m = re.fullmatch(r"n(\d+)([tac])(\d+)", spec)
    if m:
        return q_nested(W, int(m.group(1)), m.group(2), int(m.group(3)))
    m = re.fullmatch(r"w(\d+)(t|c|a|g\d+)", spec)
    if m:
        return q_uniform(W, int(m.group(1)), m.group(2))
    raise ValueError(spec)


def storage_bits(spec, shape):
    """Bits to store one weight matrix under a spec (codes + scales; fp16 scales, int zero points)."""
    o, i_ = shape
    N = o * i_
    if spec in ("fp32",) or spec.startswith("noise"):
        return 32 * N
    if spec == "bf16":
        return 16 * N
    m = re.fullmatch(r"mx(\d+)", spec)
    if m:
        return int(m.group(1)) * N + 8 * o * (-(-i_ // 32))
    m = re.fullmatch(r"n(\d+)([tac])(\d+)", spec)
    if m:                                          # the stored object is the B-bit model
        return storage_bits(f"w{m.group(1)}{m.group(2)}", shape)
    m = re.fullmatch(r"w(\d+)(t|c|a|g\d+)", spec)
    b, g = int(m.group(1)), m.group(2)
    if g == "t":
        return b * N + 32
    if g == "c":
        return b * N + 16 * o
    if g == "a":
        return b * N + 16 * o + b * o
    return b * N + 16 * o * (-(-i_ // int(g[1:])))


# ----------------------------------------------------------------------------- models
def linear_items(m, family):
    torch = _torch()
    if family == "dt":
        return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d))]
    return [(n, mod) for n, mod in m.named_modules() if mod.__class__.__name__ == "CastedLinear"]


def build_fp32(task, T, batch):
    import contraction as C
    import breadth_sweep as bs
    if task in C.DT_TASKS:
        return C.build_dt(), "dt", {"H_cycles": None, "L_cycles": None}
    m, meta = bs.build(task, None, T, batch)
    return m, ("trm" if task in bs.TRM_TASKS else "hrm"), meta


_FACTORY = {}


def clone_fp32(m_fp32_cpu):
    """A fresh fp32 copy (copy.deepcopy fails on non-leaf cached tensors): rebuild, then load the state."""
    task, T, batch = _FACTORY["args"]
    m, _, _ = build_fp32(task, T, batch)
    m.load_state_dict(m_fp32_cpu.state_dict())
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def make_cell(m_fp32_cpu, family, spec, draw):
    """Compressed copy on the CPU.  Noise draws reproduce contraction.make_model exactly (same seed rule,
    same module order, same CPU RNG), so X2's cells are the same models."""
    torch = _torch()
    import contraction as C
    import breadth_sweep as bs
    m = clone_fp32(m_fp32_cpu)
    if spec.startswith("noise"):
        sigma = float(spec[5:])
        torch.manual_seed(10_000 + 1000 * draw + int(round(sigma * 1000)))
        if family == "dt":
            C.dt_quantize_(m, None, "noise", sigma)
        else:
            bs.quantize_weights_(m, None, "noise", sigma)
        return m
    with torch.no_grad():
        for _, mod in linear_items(m, family):
            W = mod.weight.data
            mod.weight.data = quant_2d(W.reshape(W.shape[0], -1).float(), spec).reshape(W.shape).to(W.dtype)
    return m


def make_polish(m_fp32_cpu, m_cell_cpu, family, spec, act_scale=None):
    """Polish copy on the CPU: fp32/bf16/w*/mx* from the fp32 weights; lr<r>/lq<r> = compressed weights plus
    the rank-r (activation-scaled for lq) SVD of the error W - Q(W), factors rounded to fp16."""
    torch = _torch()
    if spec == "fp32":
        return m_fp32_cpu
    m_lr = re.fullmatch(r"(lr|lq)(\d+)", spec)
    m = clone_fp32(m_cell_cpu if m_lr else m_fp32_cpu)
    with torch.no_grad():
        ref = dict(linear_items(m_fp32_cpu, family))
        for name, mod in linear_items(m, family):
            W0 = ref[name].weight.data
            shp = W0.shape
            W = W0.reshape(shp[0], -1).float()
            if m_lr:
                r = int(m_lr.group(2))
                Wq = mod.weight.data.reshape(shp[0], -1).float()
                E = W - Wq
                s = None
                if m_lr.group(1) == "lq":
                    s = act_scale[name].float().clamp_min(1e-6)
                    if s.numel() != E.shape[1]:          # conv: per input channel -> repeat over the kernel
                        s = s.repeat_interleave(E.shape[1] // s.numel())
                    E = E * s.view(1, -1)
                U, S, Vh = torch.linalg.svd(E.double(), full_matrices=False)
                r_ = min(r, S.numel())
                A = (U[:, :r_] * S[:r_]).half().float()
                Bm = Vh[:r_].half().float()
                corr = A @ Bm
                if s is not None:
                    corr = corr / s.view(1, -1)
                mod.weight.data = (Wq + corr).reshape(shp).to(W0.dtype)
            else:
                mod.weight.data = quant_2d(W, spec).reshape(shp).to(W0.dtype)
    return m


def polish_extra_bits(pspec, cspec, shape, rank=None):
    """Extra bits beyond the compressed model that the polish needs (per matrix)."""
    o, i_ = shape
    m = re.fullmatch(r"(lr|lq)(\d+)", pspec)
    if m:
        r = min(int(m.group(2)), o, i_)
        return 16 * r * (o + i_)
    mn = re.fullmatch(r"n(\d+)([tac])(\d+)", cspec or "")
    if mn and pspec == f"w{mn.group(1)}{mn.group(2)}":   # nested: the polish is the stored model itself
        return storage_bits(pspec, shape) - storage_bits(f"w{mn.group(3)}{mn.group(2)}", shape)
    return storage_bits(pspec, shape)


# ----------------------------------------------------------------------------- steppers
class TRMStepper:
    """One supervision step of TRM/HRM on the state (z_H, z_L), with a per-block-application choice of
    weights: the first N-j applications use model `key`, the last j use model `pol` (N = H x (L+1))."""

    def __init__(self, models, family, x_tok):
        torch = _torch()
        self.family = family
        self.inn = {k: m.inner for k, m in models.items()}
        any_inner = next(iter(self.inn.values()))
        cfg = any_inner.config
        self.H, self.L = cfg.H_cycles, cfg.L_cycles
        self.N = self.H * (self.L + 1)
        self.pl = any_inner.puzzle_emb_len
        pid = torch.zeros(x_tok.shape[0], dtype=torch.int32, device=x_tok.device)
        with torch.no_grad():
            xs = {k: inn._input_embeddings(x_tok, pid).float() for k, inn in self.inn.items()}
        x0 = next(iter(xs.values()))
        for k, v in xs.items():
            assert torch.equal(v, x0), f"input embeddings differ for {k} (embeddings must not be perturbed)"
        self.x = x0
        self.cs = any_inner.rotary_emb() if hasattr(any_inner, "rotary_emb") else None
        B, S, D = self.x.shape
        self._z0 = (any_inner.H_init.float().expand(B, S, D).clone(), any_inner.L_init.float().expand(B, S, D).clone())

    def z0(self):
        return self._z0

    def step(self, z, key, pol=None, j=0):
        """j > 0: the last j block applications use `pol`; j = -1: the z_H updates use `pol` (z_L updates `key`);
        j = -2: the z_L updates use `pol` (z_H updates `key`)."""
        zH, zL = z
        idx = 0

        def pick(i, is_h):
            if pol is None:
                return self.inn[key]
            if j == -1:
                return self.inn[pol if is_h else key]
            if j == -2:
                return self.inn[key if is_h else pol]
            return self.inn[pol] if i >= self.N - j else self.inn[key]
        for _ in range(self.H):
            for _ in range(self.L):
                zL = pick(idx, False).L_level(zL, zH + self.x, cos_sin=self.cs)
                idx += 1
            inn = pick(idx, True)
            zH = (inn.L_level if self.family == "trm" else inn.H_level)(zH, zL, cos_sin=self.cs)
            idx += 1
        return (zH, zL)

    def readout(self, z, key, y):
        torch = _torch()
        lg = self.inn[key].lm_head(z[0])[:, self.pl:].float()
        pred = lg.argmax(-1)
        mask = y != 0
        cc = (pred == y) & mask
        return {"correct": cc.sum(-1) == mask.sum(-1),
                "cellacc": cc.sum(-1).float() / mask.sum(-1).clamp_min(1).float(), "pred": pred}

    @staticmethod
    def flat(z):
        torch = _torch()
        return torch.cat([z[0].flatten(1), z[1].flatten(1)], 1)


class DTStepper:
    """Deep-thinking recall net: z_{t+1} = recur_block([z_t, x]), answer = head(z_T)."""

    def __init__(self, models, family, x):
        torch = _torch()
        self.nets, self.x, self.N = models, x, 1
        with torch.no_grad():
            self._z0 = next(iter(models.values())).projection(x)

    def z0(self):
        return self._z0

    def step(self, z, key, pol=None, j=0):
        torch = _torch()
        net = self.nets[pol] if (pol is not None and j >= 1) else self.nets[key]
        return net.recur_block(torch.cat([z, self.x], 1))

    def readout(self, z, key, y):
        out = self.nets[key].head(z).float()
        B = z.shape[0]
        pred = out.argmax(1).view(B, -1) * self.x.max(1)[0].view(B, -1)
        eq = pred == y.view(B, -1)
        return {"correct": eq.all(1), "cellacc": eq.float().mean(1), "pred": pred}

    @staticmethod
    def flat(z):
        return z.flatten(1)


MAZE = {"on": False}


def maze_valid(x_tok, pred, y):
    """Maze-Hard: valid simple path and valid shortest path (maze_rescore.score), per input."""
    import maze_rescore as mr
    xn, pn = x_tok.cpu().numpy(), pred.cpu().numpy()
    ref_len = (y.cpu().numpy() == mr.PATH).sum(1)
    sc = mr.score(xn, pn, ref_len)
    return sc["simple"].astype(int), sc["shortest"].astype(int)


# ----------------------------------------------------------------------------- schedules
def parse_schedule(s, pol):
    """-> (list of steps (key, polkey, j), head key, name).  'P' -> pol; 'f' -> fp32; 'c' -> compressed.
    Tokens: <k><n> (n steps of model k), c|P@j (one hybrid step), (<tokens>)x<m> (repeat)."""
    head = None
    if "/" in s:
        s, head = s.split("/")
        head = {"hf": "f", "hc": "c", "hP": pol}[head]
    steps = []
    m = re.fullmatch(r"\((.*)\)x(\d+)", s)
    if m:
        inner, _ = parse_schedule(m.group(1), pol)
        return inner * int(m.group(2)), head
    for tok in s.split("."):
        h = re.fullmatch(r"([cfP])\|([cfP])@(\d+)", tok)
        if h:
            k1, k2 = [pol if k == "P" else k for k in (h.group(1), h.group(2))]
            steps.append((k1, k2, int(h.group(3))))
            continue
        h = re.fullmatch(r"([cfP])\|([cfP])%([HL])(\d*)", tok)      # z_H (or z_L) updates on the second model
        if h:
            k1, k2 = [pol if k == "P" else k for k in (h.group(1), h.group(2))]
            steps += [(k1, k2, -1 if h.group(3) == "H" else -2)] * int(h.group(4) or 1)
            continue
        mm = re.fullmatch(r"([cfP])(\d+)", tok)
        assert mm, (s, tok)
        k = pol if mm.group(1) == "P" else mm.group(1)
        steps += [(k, None, 0)] * int(mm.group(2))
    return steps, head


def default_schedules(T, ks, inner_js, diag):
    """Per polish model.  'diag' adds the fp32-only diagnostics (basin exit, prefix / periodic)."""
    S = [f"c{T - k}.P{k}" for k in ks if k < T] + [f"P{T}", f"c{T}.P{max(ks)}"] + [f"c{T - 1}.c|P@{j}" for j in inner_js]
    S += [f"c{T}/hP", f"c{T}"]
    return S


def diag_schedules(T, ks):
    S = []
    for t in sorted({2, 4, 8, T // 2, (3 * T) // 4}):
        if 0 < t < T:
            S.append(f"c{t}.f{T}")                  # basin exit: fp32 from the compressed state at step t, full budget
    for k in (2, 4):
        if 2 * k < T:
            S.append(f"f{k}.c{T - 2 * k}.f{k}")      # exact prefix + exact suffix (same fp32 budget 2k)
            S.append(f"f{2 * k}.c{T - 2 * k}")       # exact prefix only
    for m in (4, 8):                                # one fp32 step every m steps (budget T/m)
        if T % m == 0 and m <= T:
            S.append(f"(c{m - 1}.f1)x{T // m}")
    return S


def prefix_of(steps, T):
    """(pre, a): the schedule starts with a pure-c (or pure-f) run of length a (capped at T)."""
    a = 0
    while a < len(steps) and steps[a] == ("c", None, 0):
        a += 1
    pre = "c"
    if a == 0:
        while a < len(steps) and steps[a] == ("f", None, 0):
            a += 1
        pre = "f"
    return pre, min(a, T)


def run_schedules(stp, sched_list, y, T, ref_end, ref_pred, c_cache, marks, x_tok=None):
    """Run every schedule from its cached pure-c (or pure-f) prefix state.  Returns {name: per-input final
    correct/cellacc/disp/agree, the per-step correct count after the prefix, per-input read-outs at marks}."""
    torch = _torch()
    res = {}
    for name, (steps, head) in sched_list.items():
        pre, a = prefix_of(steps, T)
        z = c_cache[pre][a]
        curve, at = [], {}
        for i in range(a, len(steps)):
            key, pol, j = steps[i]
            z = stp.step(z, key, pol, j)
            hk = pol if (pol is not None and j != 0 and j != -2) else key
            ro = stp.readout(z, hk, y)
            curve.append(int(ro["correct"].sum()))
            if (i + 1) in marks.get(name, ()):
                at[i + 1] = ro["correct"].int().cpu()
        hk = head or (steps[-1][1] if steps[-1][1] is not None and steps[-1][2] not in (0, -2) else steps[-1][0])
        ro = stp.readout(z, hk, y)
        zf = stp.flat(z)
        res[name] = {"correct": ro["correct"].int().cpu(), "cellacc": ro["cellacc"].cpu(),
                     "disp": (torch.linalg.vector_norm(zf - ref_end, dim=1) / torch.linalg.vector_norm(ref_end, dim=1)).cpu(),
                     "agree": (ro["pred"] == ref_pred).float().mean(-1).cpu(), "start": a, "curve": curve, "at": at}
        if MAZE["on"]:
            v, sh = maze_valid(x_tok, ro["pred"], y)
            res[name]["valid"], res[name]["shortest"] = torch.from_numpy(v), torch.from_numpy(sh)
    return res


# ----------------------------------------------------------------------------- main
def act_scales(m_ref, family, stp_cls, inp, lab, T, batch, dev):
    """LQER calibration: mean |x| per input channel of every linear (conv: per input channel), fp32 model,
    all T steps, on calibration inputs."""
    torch = _torch()
    items = linear_items(m_ref, family)
    acc = {n: None for n, _ in items}
    cnt = {n: 0 for n, _ in items}
    hs = []
    for n, mod in items:
        def hook(mod_, args, n=n):
            x = args[0].detach().abs().float()
            if x.dim() == 4:                               # conv input (B, C, H, W)
                v, c = x.sum((0, 2, 3)), x.shape[0] * x.shape[2] * x.shape[3]
            else:
                v, c = x.reshape(-1, x.shape[-1]).sum(0), x.reshape(-1, x.shape[-1]).shape[0]
            acc[n] = v if acc[n] is None else acc[n] + v
            cnt[n] += c
        hs.append(mod.register_forward_pre_hook(hook))
    with torch.no_grad():
        for s in range(0, len(inp), batch):
            x = torch.from_numpy(inp[s:s + batch]).to(dev)
            y = torch.from_numpy(lab[s:s + batch]).to(dev)
            stp = stp_cls({"f": m_ref}, family, x)
            z = stp.z0()
            for _ in range(T):
                z = stp.step(z, "f")
            stp.readout(z, "f", y)
    for h in hs:
        h.remove()
    torch_ = _torch()
    shp = {n: mod.weight.shape[1] for n, mod in items}      # never-called linears (q_head): unit scale
    return {n: (acc[n] / max(cnt[n], 1)).cpu() if acc[n] is not None else torch_.ones(shp[n]) for n in acc}


def load_calib(task, n, T):
    """Unlabeled calibration inputs disjoint from the evaluation rows."""
    import contraction as C
    import breadth_sweep as bs
    if task in C.DT_TASKS:                                 # other maze size (convnet: any size works)
        other = {"dt_maze33": "dt_maze13", "dt_maze13": "dt_maze9", "dt_maze9": "dt_maze13"}[task]
        return C.load_dt(other, n)
    t = bs.TRM_TASKS.get(task) or bs.HRM_TASKS[task]
    if t["data"] == "sudoku":                              # test rows 100000.. (evaluation uses rows < 5000)
        inp, lab = bs.load_qa("sudoku", 100_000 + n)
        return inp[100_000:], lab[100_000:]
    from huggingface_hub import hf_hub_download            # maze: the 1000 test rows are all evaluated -> train split
    import csv
    p = hf_hub_download("sapientinc/maze-30x30-hard-1k", "train.csv", repo_type="dataset")
    q, a = [], []
    with open(p, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            q.append(np.array([bs.MAZE_CHARSET.index(c) + 1 for c in row[1]], dtype=np.int64))
            a.append(np.array([bs.MAZE_CHARSET.index(c) + 1 for c in row[2]], dtype=np.int64))
            if len(q) >= n:
                break
    return np.stack(q), np.stack(a)


def main():
    torch = _torch()
    import contraction as C
    import breadth_sweep as bs
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--cells", default="w4t")
    ap.add_argument("--polish", default="fp32")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--inner", default="auto", help="block-level polish of the last step: j values ('auto': 1, L+1, 2(L+1))")
    ap.add_argument("--diag", default="fp32", help="polish models that also get the diagnostic schedules")
    ap.add_argument("--extra_sched", default="", help="additional schedules (';'-separated), run for every polish model")
    ap.add_argument("--only_extra", action="store_true", help="run only --extra_sched (plus the append read-out)")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--n_calib", type=int, default=128)
    ap.add_argument("--fp32_ext", type=int, default=0, help="fp32 depth curve length (default 2T)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available() or a.cpu, "NO CUDA"
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    is_dt = a.task in C.DT_TASKS
    T = a.T or (C.DT_TASKS[a.task]["T"] if is_dt else 16)
    ks = [int(k) for k in a.ks.split(",")]
    if is_dt:
        batch = a.batch or C.DT_TASKS[a.task]["batch"]
        inp, lab = C.load_dt(a.task, a.offset + a.n)
    else:
        t = bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task]
        batch = a.batch or t["batch"]
        inp, lab = bs.load_qa(t["data"], a.offset + a.n)
    inp, lab = inp[a.offset:], lab[a.offset:]
    _FACTORY["args"] = (a.task, T, batch)
    MAZE["on"] = (not is_dt) and (bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task])["data"] == "maze"
    m_cpu, family, meta = build_fp32(a.task, T, batch)
    m_cpu.eval()
    for p in m_cpu.parameters():
        p.requires_grad_(False)
    stp_cls = DTStepper if family == "dt" else TRMStepper
    cells = C.parse_cells(a.cells) if a.cells not in ("", "none", '""') else []
    pols = [p for p in a.polish.split(",") if p]
    diag = set(p for p in a.diag.split(",") if p)
    if family == "dt":
        inner_js = []
    else:
        cfg = m_cpu.inner.config
        L1 = cfg.L_cycles + 1
        inner_js = sorted({1, L1, 2 * L1} - {cfg.H_cycles * L1}) if a.inner == "auto" else \
            [int(j) for j in a.inner.split(",") if j]
    shapes = {n: list(mod.weight.shape[:1]) + [int(np.prod(mod.weight.shape[1:]))] for n, mod in linear_items(m_cpu, family)}

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"task": a.task, "family": family, "n": int(len(inp)), "offset": a.offset, "T": T, "ks": ks,
            "inner_js": inner_js, "meta": meta, "shapes": shapes,
            "gpu": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__,
            "records": []}
    if out_path.exists():
        try:
            old = json.load(open(out_path))
            if old.get("n") == blob["n"] and old.get("T") == T and old.get("offset") == a.offset:
                blob["records"] = old["records"]
        except (json.JSONDecodeError, ValueError):
            print(f"{out_path} unreadable; starting fresh", flush=True)
    done = {(r["cell"], r["draw"], r["polish"]) for r in blob["records"]}

    need_lq = any(p.startswith("lq") for p in pols)
    scales = None
    if need_lq:
        t0 = time.time()
        ci, cl = load_calib(a.task, a.n_calib, T)
        m_gpu = m_cpu.to(dev)
        scales = act_scales(m_gpu, family, stp_cls, ci, cl, T, batch, dev)
        m_cpu = m_gpu.cpu()
        print(f"calibration (LQER scales) on {len(ci)} inputs: {time.time()-t0:.0f}s", flush=True)

    m_ref = clone_fp32(m_cpu).to(dev)
    chunks = [(s, min(s + batch, len(inp))) for s in range(0, len(inp), batch)]
    print(f"{a.task} ({family}) n={len(inp)} batch={batch} T={T} ks={ks} inner={inner_js} cells={cells} "
          f"polish={pols}; {len(done)} records present", flush=True)

    # fp32 reference trajectory per chunk: end state, prediction, depth curve (kept on the CPU)
    t0 = time.time()
    ref = []
    Text = a.fp32_ext or 2 * T                           # fp32 depth curve beyond T: control for the append schedules
    f_curve = np.zeros(Text, int)
    f_at = {k: [] for k in sorted(set(ks + [T] + [T + k for k in ks] + [Text])) if k <= Text}
    with torch.no_grad():
        for (s, e) in chunks:
            x = torch.from_numpy(inp[s:e]).to(dev)
            y = torch.from_numpy(lab[s:e]).to(dev)
            stp = stp_cls({"f": m_ref}, family, x)
            z = stp.z0()
            for t_ in range(Text):
                z = stp.step(z, "f")
                ro = stp.readout(z, "f", y)
                f_curve[t_] += int(ro["correct"].sum())
                if (t_ + 1) in f_at:
                    f_at[t_ + 1].append(ro["correct"].int().cpu())
                if t_ + 1 == T:
                    ref.append({"end": stp.flat(z).cpu(), "pred": ro["pred"].cpu(), "correct": ro["correct"].int().cpu()})
                    if MAZE["on"]:
                        ref[-1]["valid"], ref[-1]["shortest"] = maze_valid(x, ro["pred"], y)
    f_correct = torch.cat([r["correct"] for r in ref]).numpy()
    blob["fp32"] = {"acc": float(f_correct.mean()), "curve": (f_curve / len(inp)).tolist(), "T_ext": Text,
                    "correct": "".join(map(str, f_correct.tolist())),
                    "at": {str(k): "".join(map(str, torch.cat(v).tolist())) for k, v in f_at.items()},
                    "secs": time.time() - t0}
    if MAZE["on"]:
        for kk in ("valid", "shortest"):
            v = np.concatenate([r[kk] for r in ref])
            blob["fp32"][kk + "_acc"], blob["fp32"][kk] = float(v.mean()), "".join(map(str, v.tolist()))
    print(f"  fp32 acc {100*f_correct.mean():.1f}  curve@{ks}: "
          f"{[round(100*f_curve[k-1]/len(inp),1) for k in ks if k <= T]} ({time.time()-t0:.0f}s)", flush=True)

    tmp = out_path.with_suffix(".tmp")
    json.dump(blob, open(tmp, "w"))
    tmp.replace(out_path)

    def enc(t):
        return "".join(map(str, t.tolist()))

    for cell in cells:
        todo = [p for p in pols if (cell[0], cell[1], p) not in done]
        if not todo:
            continue
        t0 = time.time()
        mc_cpu = make_cell(m_cpu, family, cell[0], cell[1])
        mc = mc_cpu.to(dev)
        for pname in todo:
            t1 = time.time()
            mp = make_polish(m_cpu, mc_cpu, family, pname, scales)
            mp = mp.to(dev) if pname != "fp32" else m_ref
            models = {"c": mc, "f": m_ref, "P": mp}
            scheds = [] if a.only_extra else default_schedules(T, ks, inner_js, pname in diag)
            if pname in diag and not a.only_extra:
                scheds += diag_schedules(T, ks)
            scheds += [s for s in a.extra_sched.split(";") if s]
            if a.only_extra:
                scheds += [f"c{T}.P{max(ks)}"]                  # the append read-out the log line needs
            parsed = {s: parse_schedule(s, "P") for s in dict.fromkeys(scheds)}
            app = f"c{T}.P{max(ks)}"
            marks = {app: {T + k for k in ks}}
            per = {s: [] for s in parsed}
            need = {"c": set(), "f": set()}
            for st_, _h in parsed.values():
                pre_, a_ = prefix_of(st_, T)
                need[pre_].add(a_)
            c_curve = np.zeros(T, int)
            chead = []
            with torch.no_grad():
                for ci, (s, e) in enumerate(chunks):
                    x = torch.from_numpy(inp[s:e]).to(dev)
                    y = torch.from_numpy(lab[s:e]).to(dev)
                    stp = stp_cls(models, family, x)
                    cache = {"c": {0: stp.z0()}, "f": {0: stp.z0()}}
                    z = stp.z0()
                    for t_ in range(T):
                        z = stp.step(z, "c")
                        if (t_ + 1) in need["c"]:
                            cache["c"][t_ + 1] = z
                        c_curve[t_] += int(stp.readout(z, "c", y)["correct"].sum())
                    if need["f"] - {0}:
                        z = stp.z0()
                        for t_ in range(max(need["f"])):
                            z = stp.step(z, "f")
                            if (t_ + 1) in need["f"]:
                                cache["f"][t_ + 1] = z
                    rr = run_schedules(stp, parsed, y, T, ref[ci]["end"].to(dev), ref[ci]["pred"].to(dev), cache, marks,
                                       x_tok=x)
                    for sname, o in rr.items():
                        per[sname].append(o)
                    del cache
            recs = {}
            for sname, lst in per.items():
                corr = torch.cat([o["correct"] for o in lst]).numpy()
                disp = torch.cat([o["disp"] for o in lst]).numpy()
                agree = torch.cat([o["agree"] for o in lst]).numpy()
                cellacc = torch.cat([o["cellacc"] for o in lst]).numpy()
                curve = np.sum([o["curve"] for o in lst], 0) / len(inp) if lst[0]["curve"] else []
                r = {"acc": float(corr.mean()), "cellacc": float(cellacc.mean()), "correct": enc(corr),
                     "start": lst[0]["start"], "curve": np.round(curve, 4).tolist() if len(curve) else [],
                     "disp_med": float(np.median(disp)), "agree_mean": float(agree.mean()),
                     "recovered": float(corr[f_correct == 1].mean()) if f_correct.any() else None}
                if pname == "fp32":
                    r["disp"] = np.round(disp, 4).tolist()
                    r["agree"] = np.round(agree, 4).tolist()
                if "valid" in lst[0]:
                    for kk in ("valid", "shortest"):
                        v = torch.cat([o[kk] for o in lst]).numpy()
                        r[kk + "_acc"], r[kk] = float(v.mean()), enc(v)
                if lst[0]["at"]:
                    r["at"] = {str(k): enc(torch.cat([o["at"][k] for o in lst])) for k in lst[0]["at"]}
                recs[sname] = r
            extra_bits = {n: polish_extra_bits(pname, cell[0], shp) for n, shp in shapes.items()}
            rec = {"cell": cell[0], "draw": cell[1], "polish": pname, "c_curve": (c_curve / len(inp)).tolist(),
                   "schedules": recs, "polish_extra_bits": extra_bits,
                   "cell_bits": {n: storage_bits(cell[0], shp) for n, shp in shapes.items()},
                   "secs": time.time() - t1}
            blob["records"].append(rec)
            done.add((cell[0], cell[1], pname))
            tmp = out_path.with_suffix(".tmp")
            json.dump(blob, open(tmp, "w"))
            tmp.replace(out_path)
            show = [f"c{T}/hP", f"c{T - 1}.P1", f"c{T - 2}.P2", f"c{T - 4}.P4", f"c{T - 8}.P8", f"P{T}"] + \
                [s_ for s_ in a.extra_sched.split(";") if s_]
            print(f"  {cell[0]}#{cell[1]} polish={pname}: c{T} {100*c_curve[-1]/len(inp):.1f} | " +
                  " ".join(f"{s}={100*recs[s]['acc']:.1f}" for s in show if s in recs) +
                  f" | append {app}@{[T+k for k in ks]}: " +
                  str([round(100 * np.mean([int(ch) for ch in recs[app]['at'][str(T + k)]]), 1) for k in ks]) +
                  f" ({time.time()-t1:.0f}s)", flush=True)
            if pname != "fp32":
                del mp
            torch.cuda.empty_cache()
        del mc
        torch.cuda.empty_cache()
        print(f"  cell {cell} done ({time.time()-t0:.0f}s)", flush=True)
    print("wrote", out_path, len(blob["records"]), "records", flush=True)


if __name__ == "__main__":
    main()
