#!/usr/bin/env python3
"""M5 part 5: early exit for deep-thinking maze nets (no halting head).

The recall DT net (dt_net_recall_2d, width 128, checkpoint training-rusty-Tayla) is iterated T times;
after every iteration we record, per maze:
  correct   exact solution (every cell of the masked argmax equals the target), as deep-thinking scores it
  res       ||h_t - h_{t-1}|| / ||h_t||  (convergence residual of the recurrent state)
  chg       number of cells whose masked argmax changed since the previous iteration
  conf      min over open cells of the max softmax probability of the 2-way head
from which analyze_m5.py scores label-free exit rules (residual below a threshold tuned on fp32,
answer unchanged for p iterations, confidence above a tuned threshold) against a fixed budget.

Data: first 1000 test mazes per size (copied from p4d ~/breadth/dt_data, uint8; the first 256 equal
the X2 subsets in ~/edgetrm_ckpt/dt_mazes). Quantizers as contraction.dt_quantize_ (every Conv).

    python dt_exit.py --size 33 --T 500 --specs fp32,w4c,w4t,noise0.2@s0 --outdir ~/breadth_results/m5/dt_maze33

E6 (MLSys, perturb every step): a spec `<grid>@dither<k>` re-draws every Conv weight by stochastic rounding from
the master (fp32, or an RTN --master spec such as w8c) onto <grid> BEFORE EVERY ITERATION (draw 0 before the
projection, draw t before recur_block t); `<grid>@gdither<k>` is the control (master + fresh zero-mean Gaussian at
the SR variance, no lattice). k = the re-draw sequence, seeded as m2_bestofk (wdither.draw_seed(17, k, t)), the same
weight trajectory for every maze. Record layout unchanged (per-iteration correct / res / chg / conf); file
`<spec>[_m<master>].npz`, info carries wsample / dseed / master.

    python dt_exit.py --size 33 --T 1000 --specs w4t,w4t@dither0,w4t@dither1,w4t@dither2,w4t@gdither0 \
        --outdir ~/breadth_results/e6/dt_maze33
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
from contraction import build_dt, dt_quantize_  # noqa: E402
from breadth_sweep import parse_quant  # noqa: E402
from wdither import make_hook, parse_arm  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = Path(os.environ.get("M5_DT_DATA", Path.home() / "edgetrm_ckpt" / "dt_mazes" / "full1000"))


class _ActQuant:
    """static per-tensor symmetric activation fake-quant at a conv input (dt_breadth convention)."""

    def __init__(self, bits):
        self.qmax, self.absmax, self.scale = 2 ** (bits - 1) - 1, 0.0, None

    def __call__(self, mod, args):
        x = args[0]
        if self.scale is None:
            self.absmax = max(self.absmax, float(x.detach().abs().max()))
            return None
        return (torch.round(x / self.scale).clamp(-self.qmax, self.qmax) * self.scale,) + tuple(args[1:])

    def freeze(self):
        self.scale = max(self.absmax, 1e-8) / self.qmax


@torch.no_grad()
def run(net, x, y, T, batch, tsteps=(), href=None, step_hook=None):
    """tsteps: iterations (0-based) at which the recurrent state h is captured (href None: returned as "hcap", fp16 CPU)
    or compared with the fp32 reference (cosine per maze, returned as "phi" [n, len(tsteps)]).
    step_hook(t): called before the projection (t = 0) and before recur_block t for t >= 1 (E6 per-iteration re-draw);
    the same t gives the same weights in every batch."""
    n = x.shape[0]
    out = {k: [] for k in ("correct", "res", "chg", "conf")}
    hcap, phis = {t: [] for t in tsteps}, []
    for s in range(0, n, batch):
        xb = torch.from_numpy(x[s:s + batch]).float().to(DEV)
        yb = torch.from_numpy(y[s:s + batch]).long().to(DEV).view(xb.shape[0], -1)
        B = xb.shape[0]
        mask = xb.max(1)[0].view(B, -1)                      # open cells (walls = 0)
        if step_hook is not None:
            step_hook(0)
        h = net.projection(xb)
        prevP, prevH = None, h
        phi = []
        rec = {"correct": torch.zeros(B, T, dtype=torch.bool, device=DEV), "res": torch.zeros(B, T, dtype=torch.float16, device=DEV),
               "chg": torch.zeros(B, T, dtype=torch.int16, device=DEV), "conf": torch.zeros(B, T, dtype=torch.float16, device=DEV)}
        for t in range(T):
            if step_hook is not None and t > 0:
                step_hook(t)
            h = net.recur_block(torch.cat([h, xb], 1))
            o = net.head(h)
            pred = o.argmax(1).view(B, -1) * mask.long()
            rec["correct"][:, t] = (pred == yb).all(1)
            rec["res"][:, t] = (h - prevH).flatten(1).norm(dim=1) / h.flatten(1).norm(dim=1).clamp_min(1e-8)
            rec["chg"][:, t] = ((pred != prevP).sum(1) if prevP is not None else mask.sum(1)).clamp_max(32767).to(torch.int16)
            p = torch.softmax(o.float(), 1).amax(1).view(B, -1)
            rec["conf"][:, t] = torch.where(mask > 0, p, torch.full_like(p, 2.0)).amin(1)
            if t in hcap:
                if href is None:
                    hcap[t].append(h.detach().half().cpu())
                else:
                    phi.append(torch.nn.functional.cosine_similarity(h.flatten(1), href[t][s:s + B].to(DEV).float().flatten(1), dim=-1))
            prevP, prevH = pred, h
        for k in out:
            out[k].append(rec[k].cpu().numpy())
        if phi:
            phis.append(torch.stack(phi, 1).float().cpu().numpy())
    res = {k: np.concatenate(v) for k, v in out.items()}
    if tsteps and href is None:
        res["hcap"] = {t: torch.cat(v) for t, v in hcap.items()}
    if phis:
        res["phi"] = np.concatenate(phis)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--T", type=int, default=500)
    ap.add_argument("--specs", default="fp32")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--allow_cpu", action="store_true")
    ap.add_argument("--master", default="fp32", help="E6: weights the dither / gdither draws are made from (fp32 or an RTN spec, e.g. w8c)")
    ap.add_argument("--phi", default="", help="iterations (1-based, e.g. 30,100,500) at which to store phi = cos(h, h of fp32 at the same "
                    "iteration, same maze), from an fp32 reference run in the same process")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA"
    x = np.load(DATA / f"maze{a.size}_inputs.npy")[:a.n]
    y = np.load(DATA / f"maze{a.size}_solutions.npy")[:a.n]
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    phi_steps = [int(v) for v in a.phi.split(",") if v]
    tsteps, href = tuple(v - 1 for v in phi_steps), None
    assert all(0 <= t < a.T for t in tsteps)
    if phi_steps and any(not (od / f"{s}.npz").exists() for s in a.specs.split(",") if s):
        t0 = time.time()
        href = run(build_dt().to(DEV), x, y, a.T, a.batch, tsteps).pop("hcap")
        print(f"  fp32 reference for phi at iterations {phi_steps} ({time.time() - t0:.0f}s)", flush=True)
    msuf = "" if a.master == "fp32" else f"_m{a.master}"
    for spec in [s for s in a.specs.split(",") if s]:
        base, arm, seed = parse_arm(spec)
        path = od / (f"{spec}{msuf}.npz" if arm in ("dither", "gdither") else f"{spec}.npz")
        if path.exists():
            print(f"  {spec}: exists, skipped", flush=True)
            continue
        q = parse_quant(base)
        t0 = time.time()
        net = build_dt().to(DEV)
        torch.manual_seed(seed)
        step_hook = None
        if arm in ("dither", "gdither"):                  # E6: re-draw from the master before every iteration
            assert q["bits"] is not None and q["gran"] not in (None, "noise"), f"{spec}: dither needs a quantizer grid"
            if a.master != "fp32":
                qm = parse_quant(a.master)
                dt_quantize_(net, qm["bits"], qm["gran"])
            slots = []
            for mod in net.modules():
                if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d)):
                    slots.append(((lambda W, _m=mod: setattr(_m.weight, "data", W)), mod.weight.data.clone()))
            step_hook = make_hook(slots, arm, q["bits"], q["gran"], seed, DEV)
        elif q["gran"] is not None:
            dt_quantize_(net, q["bits"], q["gran"], q.get("sigma"))
        handles, obs = [], []
        if q["act"] is not None:
            for mod in net.modules():
                if isinstance(mod, torch.nn.Conv2d):
                    qa = _ActQuant(q["act"])
                    obs.append(qa)
                    handles.append(mod.register_forward_pre_hook(qa))
            run(net, x[:a.batch], y[:a.batch], a.T, a.batch)
            for qa in obs:
                qa.freeze()
        r = run(net, x, y, a.T, a.batch, tsteps, href, step_hook)
        for hh in handles:
            hh.remove()
        info = {"task": f"dt_maze{a.size}", "spec": spec, "quant": q, "seed": seed, "T": a.T, "n": int(len(x)), "phi_steps": phi_steps,
                "base": base, "wsample": arm if arm in ("dither", "gdither") else "none", "dseed": seed, "master": a.master,
                "secs": time.time() - t0, "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__}
        tmp = f"{path}.tmp{os.getpid()}.npz"                # pid-unique: several jobs may share an outdir (fp32 reference)
        np.savez_compressed(tmp, info=json.dumps(info), **r)
        os.replace(tmp, path)
        c = r["correct"]
        print(f"  size {a.size} {spec:12s} " + " ".join(f"it{t}:{100 * c[:, t - 1].mean():.1f}" for t in (10, 30, 100, 200, 300, 500) if t <= a.T)
              + f" ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
