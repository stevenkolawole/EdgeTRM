#!/usr/bin/env python3
"""X2 mechanism test (PLAN.md section 8, zoom-out 3): does compression (a) displace the fixed point (a
permanent gap) or (b) slow convergence / delay basin entry (which depth repairs)?  Label-free, per cell,
on X1's deep-sweep puzzles (difficulty_sweep strat set, 300 per rating bin -> 1800; every 7th -> n=256),
same model build, H default, noise draws seeded exactly as difficulty_sweep (@s<seed>).

Per cell (and fp32), from the default carry, T steps of the supervision-step map (contraction.Recursion,
= the ACT loop), z_H snapshotted on a grid:
  disp_t   ||z_H^q(t) - z_H^fp32(t)|| / ||z_H^fp32(t)||, t in {16, 64, 512, 2048}   (displacement; fidelity's state)
  settle   first grid step after which ||z_H(t) - z_H(T)|| / ||z_H(T)|| stays below eps (1e-3, 1e-2) (slowdown; = T when it never settles before the end:
           steps to reach the model's OWN end state), T+ if never
  correct  exact-match at 16 / 64 / 512 / 2048
Path independence of the compressed map (first pi_n puzzles): the default carry plus pi_K random initial carries
(standard normal, generator 123 as polish.py --attractors), pi_T steps: distinct answers, fraction of random carries
reaching the default carry's answer, mean relative distance of their end z_H to the default end z_H.

    python x2_decomp.py --task sudoku_attn --cells w3a,w3g32 --ref_cache /data/.../ref_attn.pt --out ~/.../decomp_attn_a.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_sweep as bs  # noqa: E402
import contraction as C  # noqa: E402
import difficulty_sweep as DS  # noqa: E402

GRID = list(range(1, 17)) + list(range(24, 129, 8)) + list(range(160, 2049, 32))
MARKS = (16, 64, 512, 2048)
EPS = (1e-3, 1e-2)


def puzzles(n):
    meta = DS.load_meta()
    rows, bins, seen, keep = meta["strat"]["rows"], meta["strat"]["bin"], {}, []
    for r, b in zip(rows, bins):
        if seen.get(b, 0) < 300:
            keep.append(r)
            seen[b] = seen.get(b, 0) + 1
    sub = keep[::7][:n]
    inp, lab, rat = DS.load_rows(sub)
    return sub, keep, inp, lab, rat


def build_cell(task, spec, K, batch):
    base, seed = DS.split_spec(spec)
    q = bs.parse_quant(base)
    m, meta = bs.build(task, None, K, batch)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if q["gran"] is not None:
        bs.quantize_weights_(m, q["bits"], q["gran"], q.get("sigma"))
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(bs.DEV).eval(), meta


@torch.no_grad()
def trajectory(m, family, x, y, T):
    Fn = C.Recursion(m, family, x)
    z = Fn.z0()
    snaps, corr, marks = {}, {}, {}
    grid = [t for t in GRID if t <= T]
    for t in range(1, T + 1):
        z = Fn(z)
        if t in grid:
            snaps[t] = z[:, 0].clone()
        if t in MARKS:
            corr[t] = Fn.readout(z, y)["correct"].cpu()
            marks[t] = z[:, 0].clone()
    zT = snaps[grid[-1]]
    nT = C.bnorm(zT)
    d = torch.stack([C.bnorm(snaps[t] - zT) / nT for t in grid])        # (len(grid), B)
    settle = {}
    for eps in EPS:
        above = d >= eps
        idx = torch.arange(len(grid), device=d.device).view(-1, 1).expand_as(d)
        last = torch.where(above, idx, torch.full_like(idx, -1)).amax(0)   # last grid index still above eps
        st = torch.tensor([grid[i + 1] if i + 1 < len(grid) else T + 1 for i in last.tolist()])
        st[last == -1] = grid[0]
        settle[str(eps)] = st
    pred512 = None
    return {"corr": corr, "marks": marks, "settle": settle, "d_last_grid": d[-2].cpu() if len(grid) > 1 else None}


@torch.no_grad()
def path_independence(m, family, x, y, T, K, default_pred):
    Fn = C.Recursion(m, family, x)
    g = torch.Generator(device=x.device).manual_seed(123)
    z = Fn.z0()
    for _ in range(T):
        z = Fn(z)
    ro = Fn.readout(z, y)
    dflt_end, dflt_pred = z[:, 0].clone(), ro["pred"]
    preds, same, dist, oks = [dflt_pred], [], [], []
    for _ in range(K):
        z = torch.randn(Fn.z0().shape, generator=g, device=x.device)
        for _t in range(T):
            z = Fn(z)
        r2 = Fn.readout(z, y)
        preds.append(r2["pred"])
        same.append((r2["pred"] == dflt_pred).all(-1).float())
        dist.append(C.bnorm(z[:, 0] - dflt_end) / C.bnorm(dflt_end))
        oks.append(r2["correct"].float())
    P = torch.stack(preds, 1)
    nd = [len({tuple(P[b, k].tolist()) for k in range(P.shape[1])}) for b in range(P.shape[0])]
    return {"n_distinct": nd, "frac_same_as_default": torch.stack(same, 1).mean(1).cpu().tolist(),
            "rel_dist_to_default": torch.stack(dist, 1).mean(1).cpu().tolist(),
            "frac_random_correct": torch.stack(oks, 1).mean(1).cpu().tolist(),
            "default_correct": ro["correct"].int().cpu().tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sudoku_mlp", "sudoku_attn", "hrm_sudoku"])
    ap.add_argument("--cells", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--pi_n", type=int, default=64)
    ap.add_argument("--pi_T", type=int, default=512)
    ap.add_argument("--pi_K", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--ref_cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    family = "trm" if a.task in bs.TRM_TASKS else "hrm"
    sub, rows1800, inp, lab, rat = puzzles(a.n)
    x = torch.from_numpy(inp).to(bs.DEV)
    y = torch.from_numpy(lab).to(bs.DEV)
    out = Path(a.out)
    blob = {"task": a.task, "n": len(sub), "rows": sub, "rating": rat, "T": a.T, "grid": [t for t in GRID if t <= a.T],
            "pi": {"n": a.pi_n, "T": a.pi_T, "K": a.pi_K}, "gpu": torch.cuda.get_device_name(0), "records": []}
    if out.exists():
        old = json.load(open(out))
        if old["rows"] == sub and old["T"] == a.T:
            blob["records"] = old["records"]
    done = {r["spec"] for r in blob["records"]}
    ref_path = Path(a.ref_cache)
    ref = None
    if ref_path.exists():
        ref = torch.load(ref_path, weights_only=False)
        if ref.get("rows") != sub or ref.get("T") != a.T:
            ref = None
    cells = [c for c in a.cells.split(",") if c]
    if ref is None or ("fp32" in cells and "fp32" not in done):
        t0 = time.time()
        m, meta = build_cell(a.task, "fp32", a.T, a.batch)
        tr = trajectory(m, family, x, y, a.T)
        pi = path_independence(m, family, x[:a.pi_n], y[:a.pi_n], a.pi_T, a.pi_K, None)
        ref = {"rows": sub, "T": a.T, "marks": {t: v.cpu() for t, v in tr["marks"].items()},
               "corr": {t: v for t, v in tr["corr"].items()}, "settle": tr["settle"], "pi": pi,
               "H": meta["H_cycles"], "secs": time.time() - t0}
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(ref_path) + f".tmp{os.getpid()}"
        torch.save(ref, tmp)
        os.replace(tmp, ref_path)
        del m
        print(f"{a.task} fp32 (H={meta['H_cycles']}): acc " + " ".join(f"{t}:{100*float(v.float().mean()):.1f}" for t, v in tr["corr"].items())
              + f"; settle med (1e-3) {float(tr['settle']['0.001'].float().median()):.0f}; {time.time()-t0:.0f}s", flush=True)
    if "fp32" not in done:
        blob["records"].append({"spec": "fp32", "H": ref["H"], "correct": {str(t): v.int().tolist() for t, v in ref["corr"].items()},
                                "settle": {k: v.tolist() for k, v in ref["settle"].items()}, "pi": ref["pi"], "secs": ref["secs"]})
        json.dump(blob, open(out, "w"))
    for spec in cells:
        if spec == "fp32" or spec in done:
            continue
        t1 = time.time()
        m, meta = build_cell(a.task, spec, a.T, a.batch)
        tr = trajectory(m, family, x, y, a.T)
        disp = {str(t): (C.bnorm(tr["marks"][t] - ref["marks"][t].to(bs.DEV)) / C.bnorm(ref["marks"][t].to(bs.DEV))).cpu().tolist()
                for t in MARKS if t <= a.T}
        phi = {str(t): torch.nn.functional.cosine_similarity(tr["marks"][t].flatten(1), ref["marks"][t].to(bs.DEV).flatten(1),
                                                             dim=-1).cpu().tolist() for t in MARKS if t <= a.T}
        pi = path_independence(m, family, x[:a.pi_n], y[:a.pi_n], a.pi_T, a.pi_K, None)
        del m
        torch.cuda.empty_cache()
        rec = {"spec": spec, "H": meta["H_cycles"], "correct": {str(t): v.int().tolist() for t, v in tr["corr"].items()},
               "settle": {k: v.tolist() for k, v in tr["settle"].items()}, "disp": disp, "phi_H": phi, "pi": pi,
               "secs": time.time() - t1}
        blob["records"].append(rec)
        json.dump(blob, open(out, "w"))
        acc = {t: np.mean(v) for t, v in rec["correct"].items()}
        print(f"  {spec}: acc " + " ".join(f"{t}:{100*v:.1f}" for t, v in acc.items()) + "; disp med "
              + " ".join(f"{t}:{np.median(v):.3f}" for t, v in disp.items())
              + f"; settle med {np.median(rec['settle']['0.001']):.0f}; PI same {np.mean(pi['frac_same_as_default']):.2f} "
              f"({time.time()-t1:.0f}s)", flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
