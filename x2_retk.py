#!/usr/bin/env python3
"""X2 recovery depth and precision of the returning map (predictions K1-K2 in results/breadth/x2_retk_predictions.md,
written before this script ran).

Per example: z* = fp32 end state after T steps (MDEQ: fp32 Broyden equilibrium, budget T); z*_q = the compressed
model's own end state (same T). From z*_q the RETURNING map (fp32 | w8c | w4c) is run k steps, k in --ks (MDEQ: a
Broyden solve with budget k). Recovered = decoded answer equals the fp32 answer from z* after the same k fp32 steps;
correctness against the label is recorded too.
POST HOC fields at z* (labelled post hoc in X2.md): r_q = ||F_q(z*) - F(z*)|| / ||z*||, res_q = ||F_q(z*) - z*|| / ||z*||,
res_f = ||F(z*) - z*|| / ||z*||, the exact map's gain along u = F_q(z*) - F(z*): g = ||F(z* + u) - F(z*)|| / ||u||.

Cells as in x2_return.py (weights, +a<b>, anoise<s>, combinations; MDEQ weights only).
    python x2_retk.py --family trm --task sudoku_mlp --n 256 --cells w4t,noise0.2 --out retk_sudoku_mlp.json
Resumable per cell.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from x2_return import parse_cell, load, save  # noqa: E402

MAPS_DEFAULT = "fp32,w8c,w4c"


def _norm(v):
    return v.reshape(v.shape[0], -1).norm(dim=1)


def run_recursion(a):
    import torch
    import breadth_sweep as bs
    import contraction as C
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if a.ckpt:
        bs.CKPT_OVERRIDE = a.ckpt
    ks = sorted(int(k) for k in a.ks.split(","))
    maps = a.maps.split(",")
    if a.task in C.DT_TASKS:
        family, T = "dt", a.T or C.DT_TASKS[a.task]["T"]
        batch = a.batch or C.DT_TASKS[a.task]["batch"]
        inp, lab = C.load_dt(a.task, a.n)
    else:
        family = "trm" if a.task in bs.TRM_TASKS else "hrm"
        T = a.T or 16
        t_ = bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task]
        batch = a.batch or (64 if t_["seq"] < 200 else 8)
        inp, lab = bs.load_qa(t_["data"], a.n)
    mk = (lambda net, x: C.DTRecursion(net, x)) if family == "dt" else (lambda net, x: C.Recursion(net, family, x))
    meta = {"family": family, "task": a.task, "model": a.name or a.task, "ckpt": a.ckpt, "n": int(len(inp)), "T": T,
            "ks": ks, "maps": maps}
    blob = load(a.out, meta)
    done = {r["cell"] for r in blob["records"]}
    chunks = [(s, min(s + batch, len(inp))) for s in range(0, len(inp), batch)]
    rm = {mp: C.make_model(a.task, (mp, 0), T, batch)[0] for mp in maps}
    m0 = rm["fp32"] if "fp32" in rm else C.make_model(a.task, ("fp32", 0), T, batch)[0]
    ref, t0 = [], time.time()
    with torch.no_grad(), C.sdpa_math():
        for s, e in chunks:
            x = torch.from_numpy(inp[s:e]).to(bs.DEV)
            y = torch.from_numpy(lab[s:e]).to(bs.DEV)
            Fn = mk(m0, x)
            z = Fn.z0()
            for _ in range(T):
                z = Fn(z)
            w, pk = z, {}
            for j in range(1, ks[-1] + 1):
                w = Fn(w)
                if j in ks:
                    pk[j] = Fn.readout(w, y)["pred"]
            ref.append({"z": z, "Fz": Fn(z), "corr": Fn.readout(z, y)["correct"], "pk": pk})
    blob["fp32_acc"] = float(torch.cat([r["corr"] for r in ref]).float().mean())
    print(f"{meta['model']}: fp32 acc {blob['fp32_acc']*100:.1f} (n={len(inp)}, T={T}, ks={ks}, maps={maps}; "
          f"{time.time()-t0:.0f}s)", flush=True)
    for cell in a.cells.split(","):
        if not cell or cell in done:
            continue
        t1 = time.time()
        wspec, act, an, draw = parse_cell(cell)
        mq, _, q = C.make_model(a.task, (wspec, draw), T, batch)
        per = {k_: [] for k_ in ("corr_f", "corr_q", "r_q", "res_q", "res_f", "gain")}
        rec_k = {mp: {"ret": {k: [] for k in ks}, "corr": {k: [] for k in ks}} for mp in maps}
        with torch.no_grad(), C.sdpa_math():
            for ci, (s, e) in enumerate(chunks):
                x = torch.from_numpy(inp[s:e]).to(bs.DEV)
                y = torch.from_numpy(lab[s:e]).to(bs.DEV)
                R = ref[ci]
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
                    for _ in range(T):
                        z = Fq(z)
                    for o in obs:
                        o.freeze()
                Fq = mk(mq, x)
                z = Fq.z0()
                for _ in range(T):
                    z = Fq(z)
                roq = Fq.readout(z, y)
                Fqz = Fq(R["z"])                                  # POST HOC: compressed map at z*
                for h in hooks:
                    h.remove()
                Ff = mk(m0, x)
                nz = _norm(R["z"])
                u = Fqz - R["Fz"]
                per["r_q"].append((_norm(u) / nz).cpu())
                per["res_q"].append((_norm(Fqz - R["z"]) / nz).cpu())
                per["res_f"].append((_norm(R["Fz"] - R["z"]) / nz).cpu())
                per["gain"].append((_norm(Ff(R["z"] + u) - R["Fz"]) / _norm(u).clamp_min(1e-30)).cpu())
                per["corr_f"].append(R["corr"].int().cpu())
                per["corr_q"].append(roq["correct"].int().cpu())
                for mp in maps:
                    Fm = mk(rm[mp], x)
                    w = z
                    for j in range(1, ks[-1] + 1):
                        w = Fm(w)
                        if j in ks:
                            ro = Fm.readout(w, y)
                            rec_k[mp]["ret"][j].append((ro["pred"] == R["pk"][j]).all(-1).int().cpu())
                            rec_k[mp]["corr"][j].append(ro["correct"].int().cpu())
        blob["records"].append(finish(cell, q, per, rec_k, ks, maps, t1))
        save(a.out, blob)
        del mq
        torch.cuda.empty_cache()
    save(a.out, blob)


def finish(cell, q, per, rec_k, ks, maps, t1):
    import torch
    cat = {k_: torch.cat(v).numpy() for k_, v in per.items()}
    cf, cq = cat["corr_f"].astype(bool), cat["corr_q"].astype(bool)
    rec = {"cell": cell, "quant": q, "n": int(len(cf)), "acc_f": float(cf.mean()), "acc_q": float(cq.mean()),
           "retained": float(cq.mean() / max(cf.mean(), 1e-9)), "maps": {},
           "r_q_median": float(np.median(cat["r_q"])), "gain_median": float(np.median(cat["gain"])),
           "per": {k_: (v.astype(int).tolist() if k_.startswith("corr") else np.round(v, 5).tolist()) for k_, v in cat.items()},
           "secs": 0.0}
    line = []
    for mp in maps:
        d = {"per_ret": {}, "per_corr": {}, "rec_fc": {}, "rec_all": {}, "acc": {}}
        for k in ks:
            r = torch.cat(rec_k[mp]["ret"][k]).numpy().astype(bool)
            c = torch.cat(rec_k[mp]["corr"][k]).numpy().astype(bool)
            d["per_ret"][str(k)] = r.astype(int).tolist()
            d["per_corr"][str(k)] = c.astype(int).tolist()
            d["rec_fc"][str(k)] = float(r[cf].mean()) if cf.any() else None
            d["rec_all"][str(k)] = float(r.mean())
            d["acc"][str(k)] = float(c.mean())
        rec["maps"][mp] = d
        line.append(f"{mp} rec_fc " + " ".join(f"k{k}:{d['rec_fc'][str(k)]:.2f}" for k in ks if d['rec_fc'][str(k)] is not None))
    rec["secs"] = time.time() - t1
    print(f"  {cell}: acc {rec['acc_q']*100:.1f} (fp32 {rec['acc_f']*100:.1f}) r_q {rec['r_q_median']:.4f} "
          f"g {rec['gain_median']:.3f} | " + " | ".join(line) + f" ({rec['secs']:.0f}s)", flush=True)
    return rec


def run_mdeq(a):
    import torch
    import deq_sweep as D
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    T = a.T or 100
    ks = sorted(int(k) for k in a.ks.split(","))
    maps = a.maps.split(",")
    meta = {"family": "mdeq", "task": f"mdeq_{a.model}", "model": f"mdeq_{a.model}", "n": a.n, "T": T, "ks": ks, "maps": maps}
    blob = load(a.out, meta)
    done = {r["cell"] for r in blob["records"]}
    X, Y = D.load_imagenet_val(a.n, "resized")
    mean, std = D.MEAN.to(D.DEV), D.STD.to(D.DEV)
    batch = a.batch or 50
    m0 = D.load_mdeq(a.model)
    rm = {}
    for mp in maps:
        if mp == "fp32":
            rm[mp] = m0
        else:
            rm[mp] = D.load_mdeq(a.model)
            D.perturb_(rm[mp], mp, 0)
    ref, t0 = [], time.time()
    with torch.no_grad():
        for i0 in range(0, len(Y), batch):
            x = ((X[i0:i0 + batch].to(D.DEV).float() / 255.0) - mean) / std
            y = Y[i0:i0 + batch].to(D.DEV)
            f, z0, cut = D.prepare(m0, x)
            z = D.solve(f, z0, T, {T}, "broyden")[0][T][0]
            f, _, cut = D.prepare(m0, x)
            snaps = D.solve(f, z, ks[-1], set(ks), "broyden")[0]
            f, _, _ = D.prepare(m0, x)
            ref.append({"x": x, "y": y, "z": z, "Fz": f(z), "cut": cut, "corr": D.head(m0, z, cut).argmax(1) == y,
                        "pk": {k: D.head(m0, snaps[k][0], cut).argmax(1) for k in ks}})
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
        per = {k_: [] for k_ in ("corr_f", "corr_q", "r_q", "res_q", "res_f", "gain")}
        rec_k = {mp: {"ret": {k: [] for k in ks}, "corr": {k: [] for k in ks}} for mp in maps}
        with torch.no_grad():
            for R in ref:
                fq, z0, cut = D.prepare(mq, R["x"])
                zq = D.solve(fq, z0, T, {T}, "broyden")[0][T][0]
                cq = D.head(mq, zq, cut).argmax(1) == R["y"]
                fq, _, _ = D.prepare(mq, R["x"])
                Fqz = fq(R["z"])
                f0, _, _ = D.prepare(m0, R["x"])
                nz = _norm(R["z"])
                u = Fqz - R["Fz"]
                per["r_q"].append((_norm(u) / nz).cpu())
                per["res_q"].append((_norm(Fqz - R["z"]) / nz).cpu())
                per["res_f"].append((_norm(R["Fz"] - R["z"]) / nz).cpu())
                per["gain"].append((_norm(f0(R["z"] + u) - R["Fz"]) / _norm(u).clamp_min(1e-30)).cpu())
                per["corr_f"].append(R["corr"].int().cpu())
                per["corr_q"].append(cq.int().cpu())
                for mp in maps:
                    fm, _, _ = D.prepare(rm[mp], R["x"])
                    snaps = D.solve(fm, zq, ks[-1], set(ks), "broyden")[0]
                    for k in ks:
                        pr = D.head(rm[mp], snaps[k][0], cut).argmax(1)
                        rec_k[mp]["ret"][k].append((pr == R["pk"][k]).int().cpu())
                        rec_k[mp]["corr"][k].append((pr == R["y"]).int().cpu())
        blob["records"].append(finish(cell, q, per, rec_k, ks, maps, t1))
        save(a.out, blob)
        del mq
        torch.cuda.empty_cache()
    save(a.out, blob)


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
    ap.add_argument("--k", type=int, default=0, help="unused (kept so x2_return.py command lines carry over)")
    ap.add_argument("--ks", default="1,2,4,8,16")
    ap.add_argument("--maps", default=MAPS_DEFAULT)
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
