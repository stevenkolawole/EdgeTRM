#!/usr/bin/env python3
"""X2 extended fit: the one-step map sensitivity S of many recursive models, for predicting the basin
width sigma_1/2 across families (held out by family).

For every model the recursion map F is the one the model iterates at inference (TRM/HRM: one supervision
step on (z_H, z_L); DT: one recurrent-block application (or G o G with --period 2); MDEQ: the multiscale
fullstage f(z; x); DEQ-Transformer: the decoder layer f(z; x, mems)). With s* the fp32 end state
(TRM/HRM/DT: the state after T steps of the fp32 recursion; DEQs: the fp32 Broyden / Anderson equilibrium),
for each perturbation (weight noise x draws, quantizers) the one-step map perturbation per input is

    delta = || F~(s*) - F(s*) || / || s* ||

(F~ = the perturbed model's full map at the fp32 state, its own input injection included), and
S(sigma) = median_draws( median_inputs(delta) ) / sigma. Pre-registered (Sep 25, before any new number):
the primary sensitivity is S at sigma0 = 0.05; 0.02 and 0.1 are robustness checks.
Perturbation formulas and noise seeding are the ones each family's accuracy sweep used (breadth_sweep /
contraction.make_model for TRM/HRM/DT; deq_sweep.perturb_ / deqlm_sweep.perturb_lm_ for the DEQs, where
seed k is the same weight-space direction as accuracy-sweep seed k).

    python sens_fit.py --family trm --task sudoku_mlp [--ckpt .../step_65100 --variant NAME] --n 256 \
        --specs noise0.02:3,noise0.05:3,noise0.1:3,w8c,w4c,w4t,w3g32 --out ~/breadth_results/x2/sens_<name>.json
    python sens_fit.py --family mdeq --model small --n 1000 --specs noise0.02:3,... --out ...
    python sens_fit.py --family deqlm --segments 20 --specs noise0.02:3,... --out ...
    python sens_fit.py --analyze 'results/breadth/babel/x2/sens_*.json'      # fit + held-out predictions
Resumable: cells already in --out are skipped.
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def parse_specs(s):
    out = []
    for tok in [t for t in s.split(",") if t]:
        if ":" in tok:
            sp, k = tok.split(":")
            out += [(sp, i) for i in range(int(k))]
        else:
            out.append((tok, 0))
    return out


def _sigma(spec):
    base = spec.split("@")[0]
    return float(base[5:]) if base.startswith("noise") else None


def save(path, blob):
    tmp = Path(str(path) + ".tmp")
    json.dump(blob, open(tmp, "w"))
    tmp.replace(path)


def load_blob(path, meta):
    if Path(path).exists():
        try:
            b = json.load(open(path))
            if all(b.get(k) == v for k, v in meta.items()):
                return b
        except Exception:
            pass
    return dict(meta, records=[])


# ----------------------------------------------------------------------------- TRM / HRM / DT
def run_recursion(a):
    import torch
    import breadth_sweep as bs
    import contraction as C
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.ckpt:
        bs.CKPT_OVERRIDE = a.ckpt
    if a.task in C.DT_TASKS:
        family, T = "dt", a.T or C.DT_TASKS[a.task]["T"]
        batch = a.batch or C.DT_TASKS[a.task]["batch"]
        inp, lab = C.load_dt(a.task, a.n)
    else:
        family = "trm" if a.task in bs.TRM_TASKS else "hrm"
        T = a.T or 16
        t = bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task]
        batch = a.batch or (64 if t["seq"] < 200 else 8)
        inp, lab = bs.load_qa(t["data"], a.n)
    meta = {"family": family, "task": a.task, "variant": a.variant or a.task, "ckpt": a.ckpt, "n": int(len(inp)),
            "T": T, "period": a.period}
    blob = load_blob(a.out, meta)
    done = {(r["spec"], r["draw"]) for r in blob["records"]}
    mk = (lambda net, x: C.DTRecursion(net, x, a.period)) if family == "dt" else (lambda net, x: C.Recursion(net, family, x))
    chunks = [(s, min(s + batch, len(inp))) for s in range(0, len(inp), batch)]
    t0 = time.time()
    m0, meta_m, _ = C.make_model(a.task, ("fp32", 0), T, batch)
    ref = []
    correct = []
    with torch.no_grad(), C.sdpa_math():
        for s, e in chunks:
            x = torch.from_numpy(inp[s:e]).to(bs.DEV)
            y = torch.from_numpy(lab[s:e]).to(bs.DEV)
            Fn = mk(m0, x)
            z = Fn.z0()
            for _ in range(T):
                z = Fn(z)
            ref.append((z, Fn(z)))
            correct.append(Fn.readout(z, y)["correct"].cpu())
    acc = float(torch.cat(correct).float().mean())
    blob["fp32_acc"] = acc
    print(f"{meta['variant']}: fp32 acc {acc*100:.1f} (n={len(inp)}, T={T}; {time.time()-t0:.0f}s)", flush=True)
    del m0
    for spec, draw in parse_specs(a.specs):
        if (spec, draw) in done:
            continue
        t1 = time.time()
        m, _, q = C.make_model(a.task, (spec, draw), T, batch)
        ds = []
        with torch.no_grad(), C.sdpa_math():
            for ci, (s, e) in enumerate(chunks):
                x = torch.from_numpy(inp[s:e]).to(bs.DEV)
                z, fz = ref[ci]
                ds.append((C.bnorm(mk(m, x)(z) - fz) / C.bnorm(z)).cpu())
        d = torch.cat(ds).numpy()
        blob["records"].append({"spec": spec, "draw": draw, "sigma": _sigma(spec), "quant": q,
                                "delta_med": float(np.median(d)), "delta_mean": float(np.mean(d)),
                                "delta": np.round(d, 6).tolist(), "secs": time.time() - t1})
        print(f"  {spec}#{draw}: delta med {np.median(d):.4f} ({time.time()-t1:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()
        save(a.out, blob)
    save(a.out, blob)


# ----------------------------------------------------------------------------- MDEQ
def run_mdeq(a):
    import torch
    import deq_sweep as D
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = {"family": "mdeq", "task": f"mdeq_{a.model}", "variant": f"mdeq_{a.model}", "n": a.n, "T": a.T}
    blob = load_blob(a.out, meta)
    done = {(r["spec"], r["draw"]) for r in blob["records"]}
    X, Y = D.load_imagenet_val(a.n, "resized")
    mean, std = D.MEAN.to(D.DEV), D.STD.to(D.DEV)
    batch = a.batch or 50
    m0 = D.load_mdeq(a.model)
    ref, corr, t0 = [], [], time.time()
    with torch.no_grad():
        for i0 in range(0, len(Y), batch):
            x = ((X[i0:i0 + batch].to(D.DEV).float() / 255.0) - mean) / std
            f, z0, cut = D.prepare(m0, x)
            snaps, _ = D.solve(f, z0, a.T, {a.T}, "broyden")
            z = snaps[a.T][0]
            ref.append((x, z, f(z)))
            corr.append((D.head(m0, z, cut).argmax(1) == Y[i0:i0 + batch].to(D.DEV)).cpu())
    blob["fp32_acc"] = float(torch.cat(corr).float().mean())
    print(f"mdeq_{a.model}: fp32 top-1 {blob['fp32_acc']*100:.1f} (n={len(Y)}; {time.time()-t0:.0f}s)", flush=True)
    del m0
    for spec, draw in parse_specs(a.specs):
        if (spec, draw) in done:
            continue
        t1 = time.time()
        m = D.load_mdeq(a.model)
        q = D.perturb_(m, spec, draw)
        ds = []
        with torch.no_grad():
            for x, z, fz in ref:
                f2, _, _ = D.prepare(m, x)
                b = z.shape[0]
                ds.append(((f2(z) - fz).reshape(b, -1).norm(dim=1) / z.reshape(b, -1).norm(dim=1)).cpu())
        d = torch.cat(ds).numpy()
        blob["records"].append({"spec": spec, "draw": draw, "sigma": _sigma(spec), "quant": q,
                                "delta_med": float(np.median(d)), "delta_mean": float(np.mean(d)),
                                "delta": np.round(d, 6).tolist(), "secs": time.time() - t1})
        print(f"  {spec}#{draw}: delta med {np.median(d):.4f} (rel w err {q.get('weight_rel_err_mean', 0):.3f}; "
              f"{time.time()-t1:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()
        save(a.out, blob)
    save(a.out, blob)


# ----------------------------------------------------------------------------- DEQ-Transformer
def run_deqlm(a):
    import torch
    import deqlm_sweep as L
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    meta = {"family": "deqlm", "task": "deq_transformer", "variant": "deq_transformer", "n": a.segments,
            "T": a.T}
    blob = load_blob(a.out, meta)
    done = {(r["spec"], r["draw"]) for r in blob["records"]}
    data = L.wt103()
    stream = data["test"]

    class Capture:                                   # fp32: Anderson to the cap, keep z* and f(z*)
        def __init__(self):
            self.inner = L.Solver("anderson")

        def __call__(self, f, x0, threshold, **kw):
            z, _ = self.inner._anderson(f, x0, threshold)
            self.z, self.fz = z, f(z)
            return {"result": z}

    class OneStep:                                   # perturbed map evaluated once at fp32's z*
        def __init__(self, z, fz):
            self.z, self.fz, self.d = z, fz, None

        def __call__(self, f, x0, threshold, **kw):
            b = self.z.shape[0]
            self.d = ((f(self.z) - self.fz).reshape(b, -1).norm(dim=1) / self.z.reshape(b, -1).norm(dim=1))
            return {"result": self.z}

    m0 = L.load_deqlm(data["n_token"])
    segs, mems, t0 = [], [], time.time()
    nll, ntok = 0.0, 0
    with torch.no_grad():
        for si, (x, y, Ln) in enumerate(L.batches(stream)):
            if si >= a.segments:
                break
            cap = Capture()
            m0.f_solver = cap
            prev = [t.clone() for t in mems] if mems else []
            loss, _, mems = L.lm_step(m0, x, y, mems, a.T)
            nll += float(loss.mean()) * Ln
            ntok += Ln
            segs.append((x, y, prev, cap.z, cap.fz))
    blob["fp32_ppl"] = float(np.exp(nll / ntok))
    print(f"deq_transformer: fp32 ppl on {len(segs)} segments {blob['fp32_ppl']:.2f} ({time.time()-t0:.0f}s)", flush=True)
    del m0
    for spec, draw in parse_specs(a.specs):
        if (spec, draw) in done:
            continue
        t1 = time.time()
        m = L.load_deqlm(data["n_token"])
        q = L.perturb_lm_(m, spec, draw)
        ds = []
        with torch.no_grad():
            for x, y, prev, z, fz in segs:
                one = OneStep(z, fz)
                m.f_solver = one
                L.lm_step(m, x, y, prev, a.T)
                ds.append(one.d.cpu())
        d = torch.cat(ds).numpy()
        blob["records"].append({"spec": spec, "draw": draw, "sigma": _sigma(spec), "quant": q,
                                "delta_med": float(np.median(d)), "delta_mean": float(np.mean(d)),
                                "delta": np.round(d, 6).tolist(), "secs": time.time() - t1})
        print(f"  {spec}#{draw}: delta med {np.median(d):.4f} (rel w err {q.get('weight_rel_err_mean', 0):.3f}; "
              f"{time.time()-t1:.0f}s)", flush=True)
        del m
        torch.cuda.empty_cache()
        save(a.out, blob)
    save(a.out, blob)


def main():
    if "--analyze" in sys.argv:
        import sens_analyze
        return sens_analyze.main(sys.argv[sys.argv.index("--analyze") + 1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["trm", "mdeq", "deqlm"],
                    help="trm covers TRM, HRM and DT tasks (contraction.py builders)")
    ap.add_argument("--task", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--model", default="small")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--segments", type=int, default=20)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--period", type=int, default=1)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--specs", default="noise0.02:3,noise0.05:3,noise0.1:3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import os
    import torch
    assert torch.cuda.is_available() or os.environ.get("X2_CPU"), "NO CUDA"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.family == "trm":
        run_recursion(a)
    elif a.family == "mdeq":
        a.T = a.T or 100
        run_mdeq(a)
    else:
        a.T = a.T or 60
        run_deqlm(a)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
