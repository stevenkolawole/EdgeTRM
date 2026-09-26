#!/usr/bin/env python3
"""X2 RETURN TEST of the displaced-attractor mechanism (predictions P1-P4 in results/breadth/x2_return_predictions.md,
written before this script ran).

Per example: z* = fp32 end state after T steps (MDEQ: fp32 Broyden equilibrium), z*_q = the compressed model's own end
state (same T; MDEQ: its own Broyden equilibrium). For t in GRID, the EXACT (fp32) map is run k steps (MDEQ: an fp32
Broyden solve with budget k) from z(t) = z* + t (z*_q - z*). "Returns" = the decoded answer equals the fp32 answer from
z* after the same k steps; a latent criterion (relative distance to that fp32 state < 1e-2) is recorded too.
Return indicator = the t = 1 test; t_c = largest grid t up to which every grid point returns (0 if t = GRID[0] fails);
rho = 1 / t_c.

Compression specs (weights exactly as in each family's accuracy sweeps):
  fp32 | w<bits><gran> | noise<s> (draw k via --draws) | <spec>@mixer|mlp  (TRM/HRM/DT: contraction.make_model;
  MDEQ: deq_sweep.perturb_ with seed = draw)
  +a<b>      static per-tensor INT<b> activation fake-quant at every Linear input, scales calibrated on the chunk's
             own full compressed run (TRM/HRM; fixed e)
  anoise<s>  Gaussian noise at every Linear input, s x the input's std, redrawn every call (TRM/HRM; FRESH e)
  (combinations like w4c+a8, w4c+anoise0.2)

    python x2_return.py --family trm --task sudoku_mlp [--ckpt ...] --n 256 --cells w4t,noise0.2,+a4,anoise0.3 \
        --out ~/breadth_results/x2/ret/ret_sudoku_mlp.json
    python x2_return.py --family mdeq --model small --n 1000 --cells w4c,w4t,noise0.2 --out ...
Resumable per cell.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
LAT_EPS = 1e-2


def parse_cell(c):
    """-> (weight spec, act bits or None, anoise sigma or None, draw)"""
    draw = 0
    if "#" in c:
        c, d = c.split("#")
        draw = int(d)
    act, an = None, None
    if "+anoise" in c or c.startswith("anoise"):
        w, s = (c.split("+anoise") if "+anoise" in c else ("fp32", c[6:]))
        an, c = float(s), (w or "fp32")
    if "+a" in c:
        w, b = c.split("+a")
        act, c = int(b), (w or "fp32")
    return c or "fp32", act, an, draw


def t_c_of(ret):
    """ret: (len(GRID), B) bool -> largest grid t with all grid points up to it returning (0 if the first fails)."""
    import torch
    ok = torch.cumprod(ret.int(), 0)
    last = (ok.sum(0) - 1)
    g = torch.tensor(GRID, device=ret.device)
    return torch.where(last >= 0, g[last.clamp_min(0)], torch.zeros_like(g[:1]).expand_as(last).clone())


def summarize_cell(rec_pi):
    return rec_pi


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
        k = a.k or 100
    else:
        family = "trm" if a.task in bs.TRM_TASKS else "hrm"
        T = a.T or 16
        t_ = bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task]
        batch = a.batch or (64 if t_["seq"] < 200 else 8)
        inp, lab = bs.load_qa(t_["data"], a.n)
        k = a.k or 16
    mk = (lambda net, x: C.DTRecursion(net, x)) if family == "dt" else (lambda net, x: C.Recursion(net, family, x))
    meta = {"family": family, "task": a.task, "model": a.name or a.task, "ckpt": a.ckpt, "n": int(len(inp)), "T": T,
            "k": k, "grid": GRID, "lat_eps": LAT_EPS}
    blob = load(a.out, meta)
    done = {r["cell"] for r in blob["records"]}
    chunks = [(s, min(s + batch, len(inp))) for s in range(0, len(inp), batch)]
    m0, _, _ = C.make_model(a.task, ("fp32", 0), T, batch)
    ref = []
    t0 = time.time()
    with torch.no_grad(), C.sdpa_math():
        for s, e in chunks:
            x = torch.from_numpy(inp[s:e]).to(bs.DEV)
            y = torch.from_numpy(lab[s:e]).to(bs.DEV)
            Fn = mk(m0, x)
            z = Fn.z0()
            for _ in range(T):
                z = Fn(z)
            ro = Fn.readout(z, y)
            w = z
            for _ in range(k):
                w = Fn(w)
            ref.append({"z": z, "corr": ro["correct"], "wk": w, "pk": Fn.readout(w, y)["pred"]})
    blob["fp32_acc"] = float(torch.cat([r["corr"] for r in ref]).float().mean())
    print(f"{meta['model']}: fp32 acc {blob['fp32_acc']*100:.1f} (n={len(inp)}, T={T}, k={k}; {time.time()-t0:.0f}s)", flush=True)
    for cell in a.cells.split(","):
        if not cell or cell in done:
            continue
        t1 = time.time()
        wspec, act, an, draw = parse_cell(cell)
        mq, _, q = C.make_model(a.task, (wspec, draw), T, batch)
        per = {k_: [] for k_ in ("corr_f", "corr_q", "ret", "ret_lat", "t_c", "phi", "disp")}
        with torch.no_grad(), C.sdpa_math():
            for ci, (s, e) in enumerate(chunks):
                x = torch.from_numpy(inp[s:e]).to(bs.DEV)
                y = torch.from_numpy(lab[s:e]).to(bs.DEV)
                hooks, obs = [], []
                if an is not None:
                    torch.manual_seed(1000 + ci + 100 * draw)
                    for mod in bs._linears(mq):
                        hooks.append(mod.register_forward_pre_hook(bs._ActNoise(an)))
                if act is not None:
                    for mod in bs._linears(mq):
                        o = bs._ActQuant(act)
                        obs.append(o)
                        hooks.append(mod.register_forward_pre_hook(o))
                    Fq = mk(mq, x)
                    z = Fq.z0()
                    for _ in range(T):                   # calibration pass on this chunk's own compressed run
                        z = Fq(z)
                    for o in obs:
                        o.freeze()
                Fq = mk(mq, x)
                z = Fq.z0()
                for _ in range(T):
                    z = Fq(z)
                roq = Fq.readout(z, y)
                for h in hooks:
                    h.remove()
                R = ref[ci]
                Ff = mk(m0, x)
                d = z - R["z"]
                rets, lats = [], []
                for t in GRID:
                    w = R["z"] + t * d
                    for _ in range(k):
                        w = Ff(w)
                    rets.append((Ff.readout(w, y)["pred"] == R["pk"]).all(-1))
                    lats.append(C.bnorm(w - R["wk"]) / C.bnorm(R["wk"]) < LAT_EPS)
                ret, lat = torch.stack(rets), torch.stack(lats)
                aq, af = Fq.answer_state(z), Ff.answer_state(R["z"])
                per["corr_f"].append(R["corr"].int().cpu())
                per["corr_q"].append(roq["correct"].int().cpu())
                per["ret"].append(ret.cpu())
                per["ret_lat"].append(lat.cpu())
                per["t_c"].append(t_c_of(ret).cpu())
                per["phi"].append(torch.nn.functional.cosine_similarity(aq.flatten(1), af.flatten(1), dim=-1).cpu())
                per["disp"].append((C.bnorm(d) / C.bnorm(R["z"])).cpu())
        rec = finish(cell, q, per, t1)
        blob["records"].append(rec)
        save(a.out, blob)
        del mq
        torch.cuda.empty_cache()
    save(a.out, blob)


def finish(cell, q, per, t1):
    import torch
    cf = torch.cat(per["corr_f"]).numpy()
    cq = torch.cat(per["corr_q"]).numpy()
    ret = torch.cat(per["ret"], 1).numpy()
    lat = torch.cat(per["ret_lat"], 1).numpy()
    tc = torch.cat(per["t_c"]).numpy()
    phi = torch.cat(per["phi"]).numpy()
    disp = torch.cat(per["disp"]).numpy()
    i1 = GRID.index(1.0)
    rho = np.where(tc > 0, 1.0 / np.maximum(tc, 1e-9), np.inf)
    rec = {"cell": cell, "quant": q, "n": int(len(cf)), "acc_f": float(cf.mean()), "acc_q": float(cq.mean()),
           "retained": float(cq.mean() / max(cf.mean(), 1e-9)), "return_rate": float(ret[i1].mean()),
           "return_rate_latent": float(lat[i1].mean()), "rho_median": float(np.median(rho)),
           "frac_rho_ge1": float(np.mean(rho >= 1)), "phi_mean": float(phi.mean()), "disp_median": float(np.median(disp)),
           "return_rate_by_t": ret.mean(1).round(4).tolist(), "latent_rate_by_t": lat.mean(1).round(4).tolist(),
           "per": {"corr_f": cf.tolist(), "corr_q": cq.tolist(), "ret1": ret[i1].astype(int).tolist(),
                   "ret1_lat": lat[i1].astype(int).tolist(), "t_c": tc.round(3).tolist(), "phi": phi.round(4).tolist(),
                   "disp": disp.round(4).tolist()}, "secs": time.time() - t1}
    print(f"  {cell}: acc {rec['acc_q']*100:.1f} (fp32 {rec['acc_f']*100:.1f}) return {rec['return_rate']:.3f} "
          f"latent {rec['return_rate_latent']:.3f} rho med {rec['rho_median']:.3f} phi {rec['phi_mean']:.3f} "
          f"({rec['secs']:.0f}s)", flush=True)
    return rec


# ----------------------------------------------------------------------------- MDEQ
def run_mdeq(a):
    import torch
    import deq_sweep as D
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    T, k = a.T or 100, a.k or 26
    meta = {"family": "mdeq", "task": f"mdeq_{a.model}", "model": f"mdeq_{a.model}", "n": a.n, "T": T, "k": k,
            "grid": GRID, "lat_eps": LAT_EPS}
    blob = load(a.out, meta)
    done = {r["cell"] for r in blob["records"]}
    X, Y = D.load_imagenet_val(a.n, "resized")
    mean, std = D.MEAN.to(D.DEV), D.STD.to(D.DEV)
    batch = a.batch or 50
    m0 = D.load_mdeq(a.model)
    ref, t0 = [], time.time()
    with torch.no_grad():
        for i0 in range(0, len(Y), batch):
            x = ((X[i0:i0 + batch].to(D.DEV).float() / 255.0) - mean) / std
            y = Y[i0:i0 + batch].to(D.DEV)
            f, z0, cut = D.prepare(m0, x)
            z = D.solve(f, z0, T, {T}, "broyden")[0][T][0]
            f, _, cut = D.prepare(m0, x)
            wk = D.solve(f, z, k, {k}, "broyden")[0][k][0]
            ref.append({"x": x, "y": y, "z": z, "cut": cut, "corr": D.head(m0, z, cut).argmax(1) == y,
                        "wk": wk, "pk": D.head(m0, wk, cut).argmax(1)})
    blob["fp32_acc"] = float(torch.cat([r["corr"] for r in ref]).float().mean())
    print(f"mdeq_{a.model}: fp32 top-1 {blob['fp32_acc']*100:.1f} (n={len(Y)}; {time.time()-t0:.0f}s)", flush=True)
    for cell in a.cells.split(","):
        if not cell or cell in done:
            continue
        t1 = time.time()
        wspec, act, an, draw = parse_cell(cell)
        assert act is None and an is None, "MDEQ: weight perturbations only"
        mq = D.load_mdeq(a.model)
        q = D.perturb_(mq, wspec, draw)
        per = {k_: [] for k_ in ("corr_f", "corr_q", "ret", "ret_lat", "t_c", "phi", "disp")}
        with torch.no_grad():
            for R in ref:
                fq, z0, cut = D.prepare(mq, R["x"])
                zq = D.solve(fq, z0, T, {T}, "broyden")[0][T][0]
                cq = D.head(mq, zq, cut).argmax(1) == R["y"]
                d = zq - R["z"]
                rets, lats = [], []
                for t in GRID:
                    f, _, _ = D.prepare(m0, R["x"])
                    w = D.solve(f, R["z"] + t * d, k, {k}, "broyden")[0][k][0]
                    rets.append(D.head(m0, w, R["cut"]).argmax(1) == R["pk"])
                    b = w.shape[0]
                    lats.append((w - R["wk"]).reshape(b, -1).norm(dim=1) / R["wk"].reshape(b, -1).norm(dim=1) < LAT_EPS)
                ret, lat = torch.stack(rets), torch.stack(lats)
                b = zq.shape[0]
                per["corr_f"].append(R["corr"].int().cpu())
                per["corr_q"].append(cq.int().cpu())
                per["ret"].append(ret.cpu())
                per["ret_lat"].append(lat.cpu())
                per["t_c"].append(t_c_of(ret).cpu())
                per["phi"].append(torch.nn.functional.cosine_similarity(zq.reshape(b, -1), R["z"].reshape(b, -1), dim=1).cpu())
                per["disp"].append((d.reshape(b, -1).norm(dim=1) / R["z"].reshape(b, -1).norm(dim=1)).cpu())
        rec = finish(cell, q, per, t1)
        blob["records"].append(rec)
        save(a.out, blob)
        del mq
        torch.cuda.empty_cache()
    save(a.out, blob)


def load(path, meta):
    if Path(path).exists():
        try:
            b = json.load(open(path))
            if all(b.get(k_) == v for k_, v in meta.items()):
                return b
        except Exception:
            pass
    return dict(meta, records=[])


def save(path, blob):
    tmp = Path(str(path) + ".tmp")
    json.dump(blob, open(tmp, "w"))
    tmp.replace(path)


def main():
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["trm", "mdeq"])
    ap.add_argument("--task", default="")
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--model", default="small")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--T", type=int, default=0)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    assert torch.cuda.is_available() or os.environ.get("X2_CPU"), "NO CUDA"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    (run_recursion if a.family == "trm" else run_mdeq)(a)
    print("wrote", a.out, flush=True)


if __name__ == "__main__":
    main()
