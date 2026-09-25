#!/usr/bin/env python3
"""M3 recursion-aware calibration on MDEQ-Small (ImageNet), X6's harness (deq_sweep.py).

The map is the MDEQ fullstage f(z; x_inj); its fixed point z* is found by Broyden (26 steps, X6's default). A weight
format (per-tensor or per-channel INT-b on every conv/linear, weight norm folded, as X6) is calibrated on UNLABELED
calibration images (1 of the 5 val images per class) and evaluated on the other 4 (4,000 images, top-1).

Scale choice per tensor (clip ratio r of absmax): absmax, pct, aciq, mse, omse (layer-output error on the inputs the
tensor sees at the fp32 equilibrium pass), dT (ours: delta_T = ||f~(z*) - f(z*)|| / ||z*|| with only this tensor
low; tensors outside f (classifier head) fall back to omse). Rounding: nearest or GPTQ (layer Hessian from the same
inputs).

    python m3_deq.py --lo w4t --clip dT --calib gptq --out ~/breadth_results/m3/v3/mdeq_small_w4t_dT_gptq.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deq_sweep as D  # noqa: E402
from m3_mixed import qtensor, fmt_bits, parse_fmt, CLIPS  # noqa: E402
from m3_calib import gptq_tensor  # noqa: E402

DEV = "cuda"


def norm_x(xu8):
    return ((xu8.float() / 255.0) - D.MEAN.to(xu8.device)) / D.STD.to(xu8.device)


class DEQCtx:
    def __init__(self, which="small", T=26):
        self.m = D.load_mdeq(which)
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.slots = D.weight_slots(self.m)
        self.names = [n for n, _, _ in self.slots]
        self.get = {n: g for n, g, _ in self.slots}
        self.put = {n: p for n, _, p in self.slots}
        self.W0 = {n: self.get[n]().clone() for n in self.names}
        self.mods = dict(self.m.named_modules())
        self.T = T
        self._qc = {}

    def Wq(self, n, fmt):
        if fmt == "fp32":
            return self.W0[n]
        k = (n, fmt)
        if k not in self._qc:
            self._qc[k] = qtensor(self.W0[n], fmt)
        return self._qc[k]

    @torch.no_grad()
    def set(self, alloc):
        for n in self.names:
            self.put[n](self.Wq(n, alloc.get(n, "fp32")).clone())

    @torch.no_grad()
    def equilibrium(self, x):
        f, z0, cut = D.prepare(self.m, x)
        out, _ = D.solve(f, z0, self.T, {self.T}, "broyden")
        return out[self.T][0], cut

    @torch.no_grad()
    def f_at(self, x, z, cut):
        f, _, _ = D.prepare(self.m, x)
        return f(z)

    @torch.no_grad()
    def predict(self, x):
        z, cut = self.equilibrium(x)
        return D.head(self.m, z, cut).argmax(-1)


class Calib:
    """fp32 equilibria of the calibration images and f(z*) there."""

    def __init__(self, ctx, xc, bs=64):
        self.ctx = ctx
        ctx.set({})
        self.chunks = []
        for s in range(0, len(xc), bs):
            x = norm_x(xc[s:s + bs].to(DEV))
            z, cut = ctx.equilibrium(x)
            fz = ctx.f_at(x, z, cut)
            self.chunks.append((x, z, cut, fz))

    @torch.no_grad()
    def delta(self, alloc):
        self.ctx.set(alloc)
        v = []
        for x, z, cut, fz in self.chunks:
            fq = self.ctx.f_at(x, z, cut)
            v.append((fq - fz).reshape(len(z), -1).norm(dim=1) / z.reshape(len(z), -1).norm(dim=1))
        self.ctx.set({})
        return float(torch.cat(v).mean())

    @torch.no_grad()
    def inputs(self, max_calls=8, rows=256, hessian=False):
        """per slot: sampled unfolded input rows (and X^T X) on the fp32 equilibrium pass incl. the head."""
        ctx = self.ctx
        X = {n: [] for n in ctx.names}
        H = {n: None for n in ctx.names}
        cnt = {n: 0 for n in ctx.names}
        hs = []
        for n in ctx.names:
            mod = ctx.mods[n]

            def hook(mod_, args, n=n):
                if not hessian and len(X[n]) >= max_calls:
                    return
                a = args[0]
                if isinstance(mod_, nn.Conv2d):
                    if mod_.groups != 1:
                        return
                    u = F.unfold(a, mod_.kernel_size, dilation=mod_.dilation, padding=mod_.padding, stride=mod_.stride)
                    x = u.transpose(1, 2).reshape(-1, u.shape[1]).float()
                else:
                    x = a.reshape(-1, a.shape[-1]).float()
                if hessian:
                    H[n] = x.T @ x if H[n] is None else H[n] + x.T @ x
                    cnt[n] += x.shape[0]
                if len(X[n]) < max_calls:
                    X[n].append(x[torch.randint(0, x.shape[0], (min(rows, x.shape[0]),), device=x.device)])
            hs.append(mod.register_forward_pre_hook(hook))
        ctx.set({})
        for x, z, cut, fz in self.chunks[:2]:
            ctx.f_at(x, z, cut)
            D.head(ctx.m, z, cut)
        for h in hs:
            h.remove()
        Xc = {n: torch.cat(v) for n, v in X.items() if v}
        if hessian:
            H = {n: (H[n] / cnt[n]) if H[n] is not None else None for n in ctx.names}
        return Xc, H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--lo", default="w4t")
    ap.add_argument("--clip", default="none", choices=["none", "pct", "aciq", "mse", "omse", "dT"])
    ap.add_argument("--calib", default="", choices=["", "gptq"])
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--n_cal", type=int, default=256)
    ap.add_argument("--uniform", default="")
    ap.add_argument("--cal_block", type=int, default=0, help="which block of n_cal calibration images (disjoint sets)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.load(open(out)) if out.exists() else {}
    blob.setdefault("evals", {})
    blob.update({"model": f"mdeq_{a.model}", "lo": a.lo, "clip": a.clip, "calib": a.calib, "T": 26,
                 "gpu": torch.cuda.get_device_name(0)})

    def save():
        tmp = out.with_suffix(".tmp")
        json.dump(blob, open(tmp, "w"))
        tmp.replace(out)

    t0 = time.time()
    xu, y = D.load_imagenet_val(a.n)
    per = a.n // 1000
    idx = torch.arange(len(y))
    # rank of each image within its class (loader order): the last of each class calibrates, the rest evaluate
    seen = {}
    pos = torch.zeros(len(y), dtype=torch.long)
    for i, c in enumerate(y.tolist()):
        pos[i] = seen.get(c, 0)
        seen[c] = pos[i].item() + 1
    cal_pool = idx[pos == per - 1]
    cal_idx, ev_idx = cal_pool[a.cal_block * a.n_cal:(a.cal_block + 1) * a.n_cal], idx[pos != per - 1]
    blob["cal_block"] = a.cal_block
    ctx = DEQCtx(a.model)
    print(f"mdeq_{a.model}: {len(ctx.names)} weight tensors, eval {len(ev_idx)} images, calibration {len(cal_idx)} "
          f"unlabeled images ({time.time()-t0:.0f}s)", flush=True)
    xe, ye = xu[ev_idx], y[ev_idx]
    N = sum(ctx.W0[n].numel() for n in ctx.names)

    def evaluate(key, alloc):
        if key in blob["evals"]:
            return
        t1 = time.time()
        ctx.set(alloc)
        preds = []
        for s in range(0, len(xe), 64):
            preds.append(ctx.predict(norm_x(xe[s:s + 64].to(DEV))).cpu())
        ctx.set({})
        c = (torch.cat(preds) == ye).numpy()
        bits = sum(ctx.W0[n].numel() * (fmt_bits(alloc[n], ctx.W0[n]) if alloc.get(n, "fp32") != "fp32" else 32)
                   for n in ctx.names) / N
        blob["evals"][key] = {"top1": float(c.mean()), "n": int(len(c)), "bits": bits, "correct": c.astype(int).tolist(),
                              "secs": time.time() - t1}
        save()
        print(f"  {key:40s} bits {bits:.3f} top1 {100*c.mean():5.1f} ({time.time()-t1:.0f}s)", flush=True)

    evaluate("fp32", {})
    for u in [u for u in a.uniform.split(",") if u]:
        evaluate(f"uniform:{u}", {n: u for n in ctx.names})

    cal = Calib(ctx, xu[cal_idx])
    print(f"  calibration equilibria ready ({time.time()-t0:.0f}s)", flush=True)
    X = H = None
    if a.clip == "omse" or a.clip == "dT" or a.calib == "gptq":
        X, H = cal.inputs(hessian=a.calib == "gptq")
    clips = {}
    if a.clip != "none":
        for n in ctx.names:
            W = ctx.W0[n].float()
            if a.clip in ("pct", "aciq"):
                am = float(W.abs().max())
                if a.clip == "pct":
                    v = W.abs().flatten()
                    v = v[torch.randperm(v.numel(), device=v.device)[:1_000_000]] if v.numel() > 1_000_000 else v
                    r = float(torch.quantile(v, 0.999)) / am
                else:
                    r = {2: 2.83, 3: 3.89, 4: 5.03}.get(parse_fmt(a.lo)["bits"], 5.03) * float((W - W.mean()).abs().mean()) / am
                clips[n] = round(min(1.0, max(0.05, r)), 4)
                continue

            def om(f):
                Dq = (ctx.Wq(n, f) - ctx.W0[n]).float()
                Xn = X.get(n) if X else None
                if Xn is None or Xn.shape[1] != Dq.reshape(Dq.shape[0], -1).shape[1]:
                    return float(Dq.pow(2).sum())
                return float((Xn @ Dq.reshape(Dq.shape[0], -1).T).pow(2).sum())
            if a.clip == "mse":
                vals = [float((ctx.Wq(n, f"{a.lo}@{r}") - ctx.W0[n]).float().pow(2).sum()) for r in CLIPS]
            elif a.clip == "omse":
                vals = [om(f"{a.lo}@{r}") for r in CLIPS]
            else:
                vals = [cal.delta({n: f"{a.lo}@{r}"}) for r in CLIPS]
                if max(vals) - min(vals) <= 1e-6 * max(abs(max(vals)), 1e-12):   # outside f: the classifier head
                    vals = [om(f"{a.lo}@{r}") for r in CLIPS]
            clips[n] = CLIPS[int(np.argmin(vals))]
        blob["clips"] = clips
        print(f"  clips ({a.clip}) chosen ({time.time()-t0:.0f}s)", flush=True)
    LO = {n: (f"{a.lo}@{clips[n]}" if n in clips else a.lo) for n in ctx.names}
    blob["delta_T_lo"] = cal.delta(LO)
    evaluate(f"lo:{a.clip}", LO)
    if a.calib == "gptq":
        from m3_calib import scale_of
        for n in ctx.names:
            W = ctx.W0[n]
            Wf = W.reshape(W.shape[0], -1)
            s, qmax = scale_of(W, parse_fmt(LO[n]))
            if H.get(n) is None or H[n].shape[0] != Wf.shape[1]:
                q = ctx.Wq(n, LO[n])
            else:
                q = gptq_tensor(Wf, H[n], s.reshape(-1), qmax).reshape(W.shape)
            ctx._qc[(n, LO[n] + "#gptq")] = q
        LOg = {n: LO[n] + "#gptq" for n in ctx.names}
        blob["delta_T_lo_gptq"] = cal.delta(LOg)
        evaluate(f"lo:{a.clip}+gptq", LOg)
    save()
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
