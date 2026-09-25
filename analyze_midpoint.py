#!/usr/bin/env python3
"""Is there ONE fidelity midpoint for looped LMs (and for recursive reasoners)?  EdgeTRM X4.

Model (per cell c in group g):  correct_c ~ Binomial(n_c, p0_c * sigmoid(k_g * (phi_c - m_g)))
  p0_c  = accuracy of the cell's own bf16/fp32 reference (same directory, platform, n, depth), plugged in;
  phi_c = latent fidelity; m_g = the group's midpoint (phi at which half the reference accuracy is kept);
  k_g   = the group's slope.  Groups = the 4 looped LMs (Ouro-1.4B, Ouro-2.6B, OLMo-rec, Huginn) and, with
  --with-trm, the 5 TRM/HRM tasks.
Tests (likelihood ratio, slopes free per group in both models):
  H0: m_g equal across the looped LMs            (df = #LMs - 1)
  H0: m_g equal across LMs and TRM/HRM together   (df = #groups - 1)
Cells are not independent binomial draws of one curve (quantizer families differ), so the output also gives
the Pearson dispersion of the full model and the dispersion-scaled LR (quasi-likelihood F-style test).
Midpoint CIs: profile likelihood (chi2_1 = 3.84, and dispersion-scaled).  All fits are 1-D profiles over m
(grid 0.30-1.00 step 0.0005) with an inner search over log k (k capped at 2000: a perfectly separated group
gives a flat profile over its gap, i.e. a CI that is the bracket itself).

    python analyze_midpoint.py                    # looped LMs, pool roots of analyze_looped.py
    python analyze_midpoint.py --with-trm         # + TRM/HRM (analyze_x8.fid_cells, default depth, one platform)
    python analyze_midpoint.py --roots results/looped/p5/cross   # a single-platform exhibit only
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

np.seterr(over="ignore")

HERE = Path(__file__).resolve().parent
M_GRID = np.arange(0.30, 1.0001, 0.0005)
LOGK = np.linspace(math.log(0.5), math.log(2000.0), 120)
CHI2 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592, 7: 14.067, 8: 15.507}


def chi2_sf(x, df):
    """Survival function of chi2 (series for the regularized upper incomplete gamma; no scipy)."""
    if x <= 0:
        return 1.0
    a, xx = df / 2.0, x / 2.0
    if xx < a + 1:        # series for P
        s, t, n = 1.0 / a, 1.0 / a, 0
        while abs(t) > 1e-14 * abs(s) and n < 10000:
            n += 1
            t *= xx / (a + n)
            s += t
        p = s * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        return max(0.0, 1.0 - p)
    # continued fraction for Q
    b, c, d = xx + 1 - a, 1e300, 1.0 / (xx + 1 - a)
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-14:
            break
    return math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h


def loglik(cells, k, m):
    phi, n, y, p0 = cells
    p = p0 / (1.0 + np.exp(-k * (phi - m)))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.sum(y * np.log(p) + (n - y) * np.log(1 - p)))


def profile(cells):
    """P(m) = max_k LL(k, m) on M_GRID; also the argmax k."""
    phi, n, y, p0 = cells
    K = np.exp(LOGK)[:, None, None]
    out, arg = np.empty(len(M_GRID)), np.empty(len(M_GRID))
    for j0 in range(0, len(M_GRID), 200):
        mm = M_GRID[j0:j0 + 200][None, :, None]
        p = p0[None, None, :] / (1.0 + np.exp(-K * (phi[None, None, :] - mm)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll = (y * np.log(p) + (n - y) * np.log(1 - p)).sum(-1)       # (K, M)
        out[j0:j0 + 200] = ll.max(0)
        arg[j0:j0 + 200] = np.exp(LOGK)[ll.argmax(0)]
    return out, arg


def pearson(cells, k, m):
    phi, n, y, p0 = cells
    p = np.clip(p0 / (1.0 + np.exp(-k * (phi - m))), 1e-9, 1 - 1e-9)
    return float(np.sum((y - n * p) ** 2 / (n * p * (1 - p))))


def ci_from_profile(P, drop):
    ok = np.where(P >= P.max() - drop)[0]
    return float(M_GRID[ok.min()]), float(M_GRID[ok.max()])


def lm_cells(roots, cut):
    import analyze_looped as A
    rows, _ = A.pool_rows(roots, cut)
    g = defaultdict(list)
    for r in rows:
        n = min(r["n_gen"], r["n_tf"])
        g["LM:" + r["model"]].append((r["fid"], n, round(r["acc"] / 100 * n), r["acc_bf16"] / 100, r))
    return g


def trm_cells(platform_filter):
    import analyze_x8 as X
    g = defaultdict(list)
    cache = {}
    for task in X.TASKS:
        for c in X.fid_cells(task):
            if platform_filter == "root" and "/" in c["file"]:
                continue
            f = X.B / c["file"]
            if f not in cache:
                d = json.load(open(f))
                cache[f] = (d, {(r["H_cycles"], r["nsup"], r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]): r
                                for r in d["records"]})
            d, idx = cache[f]
            r = idx.get((c["H"], c["nsup"], c["spec"]))
            n = (r or {}).get("n") or d.get("n")
            if not n:
                continue
            f0 = idx.get((c["H"], c["nsup"], "fp32"))
            if not f0 or not f0.get("pexact"):
                continue
            p0 = f0["pexact"]
            g["TRM:" + task].append((c["phi"], n, round(c["exact"] * n), p0, c))
    return g


def arrays(cs):
    a = np.array([[c[0], c[1], c[2], c[3]] for c in cs], dtype=float)
    return a[:, 0], a[:, 1], a[:, 2], np.clip(a[:, 3], 1e-6, 1.0)


def run(groups, title):
    names = sorted(groups)
    prof = {g: profile(arrays(groups[g])) for g in names}
    ll_sep, fits = 0.0, {}
    for g in names:
        P, K = prof[g]
        j = int(P.argmax())
        fits[g] = (float(M_GRID[j]), float(K[j]), float(P[j]))
        ll_sep += P[j]
    tot = sum(prof[g][0] for g in names)
    jc = int(tot.argmax())
    m_common, ll_common = float(M_GRID[jc]), float(tot[jc])
    npar = 2 * len(names)
    N = sum(len(groups[g]) for g in names)
    X2 = sum(pearson(arrays(groups[g]), fits[g][1], fits[g][0]) for g in names)
    disp = max(1.0, X2 / max(1, N - npar))
    LR = 2 * (ll_sep - ll_common)
    df = len(names) - 1
    print(f"\n## {title}\n")
    print(f"{N} cells in {len(names)} groups.  Dispersion (Pearson X2 / df of the separate-midpoint fit): {disp:.1f}\n")
    print("| group | cells | held (ret >= 50%) / collapsed | midpoint m | slope k | 95% profile CI | 95% CI, dispersion-scaled |")
    print("|---|---|---|---|---|---|---|")
    for g in names:
        P, _ = prof[g]
        m, k, _ = fits[g]
        c1 = ci_from_profile(P, CHI2[1] / 2)
        c2 = ci_from_profile(P, disp * CHI2[1] / 2)
        cs = groups[g]
        held = sum(c[2] / max(c[1], 1) >= 0.5 * c[3] for c in cs)
        print(f"| {g} | {len(cs)} | {held} / {len(cs) - held} | {m:.3f} | {k:.0f}{' (cap)' if k >= 1999 else ''} | "
              f"{c1[0]:.3f}-{c1[1]:.3f} | {c2[0]:.3f}-{c2[1]:.3f} |")
    print(f"\nCommon midpoint (slopes free): m = {m_common:.3f}.  LR = {LR:.1f} on {df} df, p = {chi2_sf(LR, df):.3g}; "
          f"dispersion-scaled LR/{disp:.1f} = {LR / disp:.2f}, p = {chi2_sf(LR / disp, df):.3g}.")
    return {"groups": {g: {"m": fits[g][0], "k": fits[g][1], "n_cells": len(groups[g])} for g in names},
            "m_common": m_common, "LR": LR, "df": df, "p": chi2_sf(LR, df), "dispersion": disp,
            "p_scaled": chi2_sf(LR / disp, df)}


def main():
    import sys
    sys.path.insert(0, str(HERE))
    import analyze_looped as A
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=None, help="looped-LM result roots (default analyze_looped.POOL_ROOTS)")
    ap.add_argument("--cut", type=float, default=50.0, help="only used to label held/collapsed in pool_rows")
    ap.add_argument("--with-trm", action="store_true")
    ap.add_argument("--trm-platform", default="root", choices=["root", "all"],
                    help="root = only the breadth/*.json grid files (one platform); all = include babel/ and p4d/ draws")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    lm = lm_cells(a.roots or A.POOL_ROOTS, a.cut)
    out = {"lm": run(lm, "Looped LMs: one midpoint?")}
    if a.with_trm:
        trm = trm_cells(a.trm_platform)
        out["trm"] = run(trm, "TRM/HRM tasks: one midpoint?")
        out["all"] = run({**lm, **trm}, "Looped LMs + TRM/HRM: one midpoint?")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
