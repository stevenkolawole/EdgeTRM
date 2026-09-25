#!/usr/bin/env python3
"""M5 overthinking under compression: every number of the claim, reproducible from one set of records.

Claim tested (AUDIT_SEP25 item 70): within each model, visit-then-lose V rises as fidelity phi falls, and the model's own
halting head recovers part of it. Not claimed: that fp32 never overthinks (it does, on some models; the fp32 rows are
reported per model and per trained seed).

Records: p5 H200, one self-contained exhibit (fp32 references included), exit_sweep.py --phi / dt_exit.py --phi
(p5 autofill lines tools/m5_p5_phi_lines.txt, tags m5phi_*), p5 results/m5phi/ pulled to results/m5/phi/p5/. phi and V come from the SAME trajectory: phi = cos(z_H at step k, z_H of an fp32 run of the same
puzzle at step k, in the same process), per puzzle. Cells: every released model (TRM MLP / attention Sudoku, HRM Sudoku,
TRM / HRM Maze) x weight quantizer, weight-noise and activation-noise draws (3 seeds each, every draw its own cell), the six
trained seeds (attn512 / mlp512 s0-2, each with its own fp32 reference), and the deep-thinking mazes (13 / 25 / 33).
MDEQ rows (X6, Babel: cosine to the fp32 equilibrium from the same Broyden run) are listed for completeness only; they are
another platform and enter no pooled statistic.

Per cell, over the n puzzles, K = 16 default-H supervision steps (DT: T iterations):
  acc       exact accuracy at step K;  oracle = right at some step <= K
  V_abs     share of puzzles right at some step and wrong at step K
  V_rel     V_abs / (1 - acc): share of the final errors that were right earlier (Kaya ICML'19, Lotfi 2606.00206)
  phi       mean over puzzles of phi at step K
  exit      the halting-head rule chosen on this model's fp32 cell (fewest steps within 0.5 pt of fixed K; frozen),
            capped at K; d_exit = exit - fixed K (paired); recovery = d_exit / (oracle - acc) where oracle - acc >= 2 pt
  msp, stab max-softmax exit tuned on fp32 (two-fold cross-fit, frozen) and answer-stability (2 steps): recovery likewise
CIs: 95% percentile bootstrap over puzzles (2000 draws, the same draws for every quantity of a cell).
Per model: Spearman(phi, V_rel) over cells (average ranks for ties), OLS slope of V_rel on (1 - phi) (points of V_rel per
0.1 of 1 - phi), both with a two-level bootstrap (cells resampled within the model, and each cell's V_rel and phi taken
from one of its puzzle-bootstrap draws). Pooled within-model slope: the same regression after removing each model's mean
(model fixed effects), bootstrap within models. Cells with fewer than 20 final errors have no V_rel and are left out.

    python m5_overthink.py            # table + results/breadth/m5_overthink.json + figures/m5_overthink.pdf
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

import analyze_m5 as A

RB = A.RB
PHI = Path(os.environ.get("M5_PHI_DIR", A.HERE / "results" / "m5" / "phi" / "p5"))
NB = 2000
MIN_ERR = 20
RELEASED = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze_attn", "hrm_maze"]
VAR = [(f"{a}_s{s}", t) for a, t in (("attn512", "sudoku_attn"), ("mlp512", "sudoku_mlp")) for s in range(3)]


def rank(x):
    """average ranks (ties share their mean rank)."""
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x), dtype=float)
    xs = x[o]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return r


def spearman(x, y):
    if len(x) < 3:
        return np.nan
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    d = (xc ** 2).sum()
    return float((xc * (y - y.mean())).sum() / d) if d > 0 else np.nan


def ci(b):
    b = np.asarray(b, float)
    b = b[np.isfinite(b)]
    return (float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))) if len(b) else (np.nan, np.nan)


# ----------------------------------------------------------------------------- per-cell metrics with bootstrap draws
def cell_stats(c, phi, rng, t_exit=None, t_other=None):
    """c: [n, K] bool; phi: [n] or None; t_exit: exit step per puzzle (head rule); t_other: {name: exit steps}."""
    n, K = c.shape
    fin = c[:, -1]
    anyc = c.any(1)
    vl = anyc & ~fin
    idx = rng.integers(0, n, (NB, n))

    def stat(v):
        return 100 * v.mean(), 100 * v[idx].mean(1)
    out = {"n": n, "K": K}
    for nm, v in (("acc", fin), ("oracle", anyc), ("V_abs", vl)):
        p, b = stat(v)
        out[nm] = (p, *ci(b))
        out["_" + nm] = b
    nerr = (~fin).sum()
    out["n_err"] = int(nerr)
    bv = 100 * vl[idx].sum(1) / np.maximum((~fin)[idx].sum(1), 1)
    out["V_rel"] = (100 * vl.sum() / nerr if nerr else np.nan, *ci(bv)) if nerr >= MIN_ERR else (np.nan, np.nan, np.nan)
    out["_V_rel"] = bv if nerr >= MIN_ERR else None
    if phi is not None:
        out["phi"] = (float(phi.mean()), *ci(phi[idx].mean(1)))
        out["_phi"] = phi[idx].mean(1)
    gap = anyc.astype(float) - fin
    for nm, t in ([("head", t_exit)] if t_exit is not None else []) + list((t_other or {}).items()):
        ok = c[np.arange(n), t]
        d = ok.astype(float) - fin
        bd = 100 * d[idx].mean(1)
        bg = 100 * gap[idx].mean(1)
        r = {"acc": 100 * ok.mean(), "steps": float((t + 1).mean()), "d": (100 * d.mean(), *ci(bd)),
             "rescued": 100 * (ok & ~fin).mean(), "premature": 100 * (~ok & fin).mean()}
        if 100 * gap.mean() >= 2:
            r["recovery"] = (d.mean() / gap.mean(), *ci(np.where(bg > 0, bd / np.maximum(bg, 1e-9), np.nan)))
        out[nm] = r
    return out


def trm_family(fam_dir, task, rng):
    """released model (fam_dir = task) or trained seed (var/<arch>_s<k>): cells from one --phi run directory."""
    if not (PHI / fam_dir).exists():
        return []
    old = A.M5_DIR
    A.M5_DIR = PHI
    try:
        cells = {s: c for s, c in A.load_m5(task, sub=fam_dir).items() if not s.endswith("|h32")}
    finally:
        A.M5_DIR = old
    if "fp32" not in cells:
        return []
    K = 16
    cells = {s: c.trunc(K) for s, c in cells.items() if c.K >= K}
    ref = cells["fp32"]
    sel = A.select_rule(ref)
    rows = []
    for spec in sorted(cells, key=A.spec_order):
        c = cells[spec]
        d = np.load(PHI / fam_dir / f"{spec}.npz")
        info = json.loads(str(d["info"]))
        steps = info.get("phi_steps") or []
        phi = d["phi"][:, steps.index(K)].astype(float) if (K in steps and "phi" in d.files) else None
        other = {}
        tt, _ = A.crossfit(ref, c, "msp", 0.5) if (c.msp is not None and ref.msp is not None) else (None, None)
        if tt is not None:
            other["msp"] = tt
        ts = A.exit_steps(c, "stab", 2)
        if ts is not None:
            other["stab2"] = ts
        m = cell_stats(c.c, phi, rng, A.exit_steps(c, *sel), other)
        m.update({"family": fam_dir, "task": task, "spec": spec, "rule": A.rule_name(*sel), "host_gpu": info.get("gpu"),
                  "phi_steps": {str(k): float(d["phi"][:, j].mean()) for j, k in enumerate(steps)} if "phi" in d.files else {}})
        rows.append(m)
    return rows


def dt_family(size, rng):
    rows = []
    for f in sorted(glob.glob(str(PHI / f"dt_maze{size}" / "*.npz")), key=lambda p: A.spec_order(Path(p).stem)):
        if ".tmp" in f:
            continue
        d = np.load(f)
        info = json.loads(str(d["info"]))
        c = d["correct"].astype(bool)
        cell = A.Cell(f"dt_maze{size}", info["spec"], "m5phi", c, np.zeros(c.shape, np.float32), chg=d["chg"].astype(np.int32))
        steps = info.get("phi_steps") or []
        phi = d["phi"][:, -1].astype(float) if "phi" in d.files else None
        m = cell_stats(c, phi, rng, None, {"stab10": A.exit_steps(cell, "stab", 10)})
        m.update({"family": f"dt_maze{size}", "task": f"dt_maze{size}", "spec": info["spec"], "rule": "stab10 (no head)",
                  "host_gpu": info.get("gpu"), "phi_steps": {str(k): float(d["phi"][:, j].mean()) for j, k in enumerate(steps)} if "phi" in d.files else {}})
        rows.append(m)
    return rows


def mdeq_rows(rng):
    rows = []
    R = RB.parent / "transfer" / "babel"
    for prefix, name in (("deq_small", "mdeq_small"), ("deq_xl", "mdeq_xl")):
        seen = set()
        for f in sorted(glob.glob(str(R / f"{prefix}_*.npz"))):
            d = np.load(f)
            B = list(d["budgets"])
            j26 = B.index(26)
            seed = Path(f).stem.split("_")[-1]
            for k in d.files:
                p = k.split("|")
                if len(p) != 3 or p[2] != "c1" or p[1] != "broyden":
                    continue
                spec = p[0] if (p[0] == "fp32" or "noise" not in p[0]) else f"{p[0]}@{seed}"
                if spec in seen:
                    continue
                seen.add(spec)
                c = d[k][:j26 + 1].T.astype(bool)
                ck = f"{p[0]}|broyden|cos"
                m = cell_stats(c, None, rng)
                if ck in d.files:
                    m["phi"] = (float(d[ck][j26].mean()), np.nan, np.nan)      # per-budget mean only in the record
                m.update({"family": name, "task": name, "spec": spec, "rule": "-"})
                rows.append(m)
    return rows


# ----------------------------------------------------------------------------- per-model trend
def trend(rows, rng):
    """Spearman(phi, V_rel) and OLS slope of V_rel on (1 - phi) over cells, two-level bootstrap."""
    rs = [r for r in rows if r.get("_V_rel") is not None and r.get("_phi") is not None]
    if len(rs) < 4:
        return None
    x = np.array([r["phi"][0] for r in rs])
    y = np.array([r["V_rel"][0] for r in rs])
    xa = np.array([r["V_abs"][0] for r in rs])
    bx = np.stack([r["_phi"] for r in rs], 1)            # [NB, cells]
    by = np.stack([r["_V_rel"] for r in rs], 1)
    bxa = np.stack([r["_V_abs"] for r in rs], 1)
    m = len(rs)
    rho_b, sl_b, sla_b = [], [], []
    for b in range(NB):
        j = rng.integers(0, m, m)
        rho_b.append(spearman(bx[b, j], by[b, j]))
        sl_b.append(0.1 * slope(1 - bx[b, j], by[b, j]))
        sla_b.append(0.1 * slope(1 - bx[b, j], bxa[b, j]))
    return {"cells": m, "rho": (spearman(x, y), *ci(rho_b)), "slope_Vrel_per0.1": (0.1 * slope(1 - x, y), *ci(sl_b)),
            "slope_Vabs_per0.1": (0.1 * slope(1 - x, xa), *ci(sla_b)), "phi_range": (float(x.min()), float(x.max()))}


def pooled_within(groups, rng):
    """model fixed effects: slope of V_rel on (1 - phi) after demeaning within each model; bootstrap cells within models."""
    gs = []
    for rows in groups:
        rs = [r for r in rows if r.get("_V_rel") is not None and r.get("_phi") is not None]
        if len(rs) >= 3:
            gs.append((np.array([r["phi"][0] for r in rs]), np.array([r["V_rel"][0] for r in rs]),
                       np.stack([r["_phi"] for r in rs], 1), np.stack([r["_V_rel"] for r in rs], 1)))
    if not gs:
        return None

    def fe(pairs):
        num = den = 0.0
        for x, y in pairs:
            xc = (1 - x) - (1 - x).mean()
            num += (xc * (y - y.mean())).sum()
            den += (xc ** 2).sum()
        return 0.1 * num / den if den > 0 else np.nan
    est = fe([(x, y) for x, y, _, _ in gs])
    bs = []
    for b in range(NB):
        pairs = []
        for _, _, bx, by in gs:
            j = rng.integers(0, bx.shape[1], bx.shape[1])
            pairs.append((bx[b, j], by[b, j]))
        bs.append(fe(pairs))
    return {"models": len(gs), "cells": sum(len(g[0]) for g in gs), "slope_Vrel_per0.1": (est, *ci(bs))}


# ----------------------------------------------------------------------------- report
def f3(t, w=5, p=1):
    if t is None or not np.isfinite(t[0]):
        return "-".rjust(w)
    return f"{t[0]:{w}.{p}f} [{t[1]:.{p}f},{t[2]:.{p}f}]"


def main():
    rng = np.random.default_rng(0)
    fams = [(m, m) for m in RELEASED] + [(f"var/{v}", t) for v, t in VAR]
    groups = {}
    for fd, task in fams:
        groups[fd] = trm_family(fd, task, rng)
    for size in (13, 25, 33):
        groups[f"dt_maze{size}"] = dt_family(size, rng)
    for r in mdeq_rows(rng):
        groups.setdefault(r["family"], []).append(r)
    print("per cell: acc@K | V_rel [CI] (final errors right earlier) | V_abs | oracle | phi [CI] | head exit - fixed K [CI], "
          "recovery [CI] | msp recovery | stability recovery")
    for fam, rows in groups.items():
        if not rows:
            print(f"\n=== {fam}: no records")
            continue
        print(f"\n=== {fam} (exit rule {rows[0]['rule']}, n={rows[0]['n']}, K={rows[0]['K']}, {rows[0].get('host_gpu')})")
        for r in rows:
            h = r.get("head") or r.get("stab10") or {}
            ms = r.get("msp") or {}
            st = r.get("stab2") or {}
            rec = lambda d: f3(d.get("recovery"), 5, 2) if d.get("recovery") else "    -"  # noqa: E731
            print(f"  {r['spec']:14s} {r['acc'][0]:5.1f} | {f3(r['V_rel'])} ({r['n_err']:4d} err) | {r['V_abs'][0]:5.1f} | "
                  f"{r['oracle'][0]:5.1f} | {f3(r.get('phi'), 6, 3)} | {f3(h.get('d'))} {rec(h)} | {rec(ms)} | {rec(st)}")
    print("\n=== fp32 visit-then-lose per model (the fp32 row of each model; overthinking at full precision exists where V_rel > 0)")
    for fam, rows in groups.items():
        for r in rows:
            if r["spec"] == "fp32":
                print(f"  {r['family']:16s} acc {r['acc'][0]:5.1f}  V_rel {f3(r['V_rel'])}  V_abs {f3(r['V_abs'])}  ({r['n_err']} final errors)")
    print("\n=== within each model: V rises as phi falls? Spearman(phi, V_rel) [CI] | slope of V_rel per 0.1 of (1 - phi) [CI] | "
          "slope of V_abs | cells | phi range")
    trends = {}
    for fam, rows in groups.items():
        t = trend(rows, rng)
        if t is None:
            continue
        trends[fam] = t
        print(f"  {fam:16s} rho {f3(t['rho'], 6, 2)} | V_rel {f3(t['slope_Vrel_per0.1'])} | V_abs {f3(t['slope_Vabs_per0.1'])} | "
              f"{t['cells']:3d} | {t['phi_range'][0]:.3f}-{t['phi_range'][1]:.3f}")
    pools = {"released TRM/HRM": [groups[m] for m in RELEASED], "trained seeds": [groups[f"var/{v}"] for v, _ in VAR],
             "all TRM/HRM": [groups[f] for f, _ in fams], "DT mazes": [groups[f"dt_maze{s}"] for s in (13, 25, 33)]}
    pooled = {}
    for nm, gl in pools.items():
        p = pooled_within(gl, rng)
        if p:
            pooled[nm] = p
            print(f"  pooled within-model ({nm}): slope of V_rel per 0.1 of (1 - phi) {f3(p['slope_Vrel_per0.1'])} "
                  f"({p['models']} models, {p['cells']} cells)")
    print("\n=== head exit (capped at K) vs fixed K, every cell: non-inferior = CI low > -1 pt; better = CI low > 0; worse = CI high < 0")
    for fam, rows in groups.items():
        ds = [r["head"]["d"] for r in rows if "head" in r]
        if ds:
            print(f"  {fam:16s} cells {len(ds):3d}: non-inferior {sum(d[1] > -1 for d in ds):3d}, better {sum(d[1] > 0 for d in ds):3d}, "
                  f"worse {sum(d[2] < 0 for d in ds):3d}; mean steps saved "
                  f"{np.mean([r['K'] - r['head']['steps'] for r in rows if 'head' in r]):.1f} of {rows[0]['K']}")
    out = {"cells": [{k: v for k, v in r.items() if not k.startswith("_")} for rows in groups.values() for r in rows],
           "trend": trends, "pooled_within": pooled}
    json.dump(out, open(RB / "m5_overthink.json", "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("\nwrote", RB / "m5_overthink.json")
    plot(groups)


def plot(groups):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = [f for f, rows in groups.items() if rows and not f.startswith("mdeq")]
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, f in enumerate(fams):
        rs = [r for r in groups[f] if r.get("phi") is not None]
        col = cmap(i % 20)
        for r in rs:
            if np.isfinite(r["V_rel"][0]):
                ax[0].errorbar(r["phi"][0], r["V_rel"][0], yerr=[[r["V_rel"][0] - r["V_rel"][1]], [r["V_rel"][2] - r["V_rel"][0]]],
                               fmt="o", ms=3, color=col, alpha=0.8, lw=0.6)
            h = r.get("head")
            if h and h.get("recovery"):
                ax[1].scatter(r["phi"][0], h["recovery"][0], s=12, color=col)
        ax[0].plot([], [], "o", color=col, label=f)
    ax[0].set_ylabel("V_rel: final errors right earlier (%)")
    ax[1].set_ylabel("share of the oracle gap the halting head recovers")
    for a in ax:
        a.set_xlabel("phi = cos(z_H, fp32 z_H) at step 16, same trajectory")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out = RB.parent.parent / "figures" / "m5_overthink.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
