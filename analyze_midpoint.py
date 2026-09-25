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
S_PLATFORM = "babel"
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


def lm_S(model, loops, sigma=0.05):
    """X2 one-loop sensitivity for a looped LM: median over draws of delta_med / sigma at `loops`."""
    f = HERE / "results/breadth/babel/x2/sens" / f"lm_{model}.json"
    if not f.exists():
        return None
    recs = json.load(open(f))["records"]
    v = [r["delta_med"] / sigma for r in recs if r["loops"] == loops and abs((r.get("sigma") or -1) - sigma) < 1e-9]
    if not v:
        v = [r["delta_med"] / sigma for r in recs if r["loops"] == loops and r["spec"] in (f"n{sigma}", f"noise{sigma}")]
    return float(np.median(v)) if v else None


def predictors(groups):
    """Label-free per-group predictors: (a) answer length (tokens of the bf16 greedy output for LMs; cells
    of the answer grid for TRM/HRM: Sudoku 81, Maze 900), (b) X2 sensitivity S (sigma 0.05)."""
    if S_PLATFORM == "p4d":      # single-platform S (X2 refit on p4d, same platform as the root TRM/HRM grid)
        x2 = {}
        for f in (HERE / "results/breadth/p4d/x2/sens").glob("sens_*.json"):
            recs = json.load(open(f))["records"]
            v = [r["delta_med"] / 0.05 for r in recs if abs((r.get("sigma") or -1) - 0.05) < 1e-9]
            if v:
                x2[f.stem[len("sens_"):]] = float(np.median(v))
    else:
        x2 = {r["model"]: r["S"] for r in json.load(open(HERE / "results/breadth/x2_sens_fit.json"))["rows"]}
    default_loops = {"ouro14": 4, "ouro26": 4, "huginn": 32, "olmo_rec32": 32}
    out = {}
    for g, cs in groups.items():
        fam, name = g.split(":", 1)
        if fam == "TRM":
            length = 900.0 if "maze" in name else 81.0
            S = x2.get(name)
        else:
            lens = [c[4].get("bf16_len") for c in cs if c[4].get("bf16_len")]
            length = float(np.mean(lens)) if lens else None
            S = lm_S(name, default_loops.get(name, 4))
        out[g] = {"length": length, "S": S}
    return out


def loo_fit(xs, ys):
    """OLS y = a + b x; leave-one-out predictions."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    A = np.c_[np.ones_like(xs), xs]
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = []
    for i in range(len(xs)):
        k = np.arange(len(xs)) != i
        c, *_ = np.linalg.lstsq(A[k], ys[k], rcond=None)
        pred.append(c[0] + c[1] * xs[i])
    pred = np.array(pred)
    r2 = 1 - ((ys - A @ coef) ** 2).sum() / ((ys - ys.mean()) ** 2).sum()
    return coef, pred, r2


def predict_section(groups, fits):
    P = predictors(groups)
    print("\n## Can the per-group midpoint be predicted label-free?\n")
    print("| group | midpoint m | answer length | log10 length | X2 sensitivity S |")
    print("|---|---|---|---|---|")
    for g in sorted(fits):
        print(f"| {g} | {fits[g]['m']:.3f} | {P[g]['length'] if P[g]['length'] is None else round(P[g]['length'])} | "
              f"{'' if not P[g]['length'] else f'{math.log10(P[g][chr(108)+chr(101)+chr(110)+chr(103)+chr(116)+chr(104)]):.2f}'} | "
              f"{'' if P[g]['S'] is None else f'{P[g][chr(83)]:.2f}'} |")
    ys_all = np.array([fits[g]["m"] for g in sorted(fits)])
    print(f"\nBaseline (predict the mean of the other groups): LOO RMSE "
          f"{np.sqrt(np.mean([(ys_all[i] - np.delete(ys_all, i).mean()) ** 2 for i in range(len(ys_all))])):.3f}\n")
    print("| predictor | groups | slope | R^2 | LOO RMSE | LOO max abs error | worst group |")
    print("|---|---|---|---|---|---|---|")
    res = {}
    L_ = lambda p: math.log10(p["length"]) if p["length"] else None  # noqa: E731
    S_ = lambda p: math.log(p["S"]) if p["S"] else None  # noqa: E731
    for name, fx, fam in (("log10 answer length", L_, None), ("log S", S_, None),
                          ("log10 answer length", L_, "TRM"), ("log S", S_, "TRM"),
                          ("log10 answer length", L_, "LM"), ("log S", S_, "LM")):
        gs = [g for g in sorted(fits) if fx(P[g]) is not None and (fam is None or g.startswith(fam + ":"))]
        if fam:
            name = f"{name}, {fam} groups only"
        if len(gs) < 3:
            continue
        xs = [fx(P[g]) for g in gs]
        ys = [fits[g]["m"] for g in gs]
        coef, pred, r2 = loo_fit(xs, ys)
        err = np.abs(np.array(ys) - pred)
        base = np.sqrt(np.mean([(ys[i] - np.mean(np.delete(ys, i))) ** 2 for i in range(len(ys))]))
        print(f"| {name} | {len(gs)} | {coef[1]:+.3f} | {r2:.2f} | {np.sqrt(np.mean(err ** 2)):.3f} (baseline {base:.3f}) | "
              f"{err.max():.3f} | {gs[int(err.argmax())]} |")
        res[name] = {"groups": gs, "slope": float(coef[1]), "r2": float(r2), "loo_rmse": float(np.sqrt(np.mean(err ** 2))),
                     "baseline_rmse": float(base), "loo_max": float(err.max())}
    print("\n(c) answer-logit margin: LMs only, see lm_runs/lm_margin.py and results/looped/lm_margins_sep25.json (R2 <= 0.02 over the 4 LMs); not available for TRM/HRM.")
    return res


def variant_cells(nsup=16):
    """The 13 trained Sudoku variants (p4d, results/breadth/p4d/variants_noise: weight noise 0.05-0.7 x 3 draws
    + w8c/w4c/w3g32/w4t, n=1000, H=3), each cell against the same file's fp32 at the same nsup."""
    g = defaultdict(list)
    for f in sorted((HERE / "results/breadth/p4d/variants_noise").glob("trm_sudoku_*.json")):
        name = f.stem.replace("_seed1", "").replace("_seed2", "")
        recs = json.load(open(f))["records"]
        sp = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]  # noqa: E731
        fp = {r["nsup"]: r for r in recs if sp(r) == "fp32"}
        for r in recs:
            if sp(r) == "fp32" or r["nsup"] != nsup or nsup not in fp or not fp[nsup]["pexact"]:
                continue
            if "seed" in f.stem and not sp(r).startswith("noise"):
                continue                      # seed files repeat the deterministic quantizers
            g["VAR:" + name].append((r["fidelity_H"], r["n"], round(r["pexact"] * r["n"]), fp[nsup]["pexact"],
                                     {"spec": sp(r), "file": f.name}))
    return g


def variant_section():
    groups = variant_cells()
    fits = {}
    for gname in sorted(groups):
        P, K = profile(arrays(groups[gname]))
        j = int(P.argmax())
        fits[gname] = {"m": float(M_GRID[j]), "k": float(K[j]), "ci": ci_from_profile(P, CHI2[1] / 2),
                       "p0": groups[gname][0][3]}
    Sd = {}
    for f in (HERE / "results/breadth/p4d/x2/sens").glob("sens_var_*.json"):
        recs = json.load(open(f))["records"]
        v = [r["delta_med"] / 0.05 for r in recs if abs((r.get("sigma") or -1) - 0.05) < 1e-9]
        Sd["VAR:trm_sudoku_" + f.stem[len("sens_var_"):]] = float(np.median(v)) if v else None
    print("\n## Within one task: the 13 trained Sudoku variants (p4d fidelity cells, p4d S)\n")
    print("| variant | fp32 exact | cells | midpoint m | 95% profile CI | S (p4d, sigma 0.05) |")
    print("|---|---|---|---|---|---|")
    for gname, v in fits.items():
        print(f"| {gname[4:]} | {100 * v['p0']:.1f} | {len(groups[gname])} | {v['m']:.3f} | {v['ci'][0]:.3f}-{v['ci'][1]:.3f} | "
              f"{Sd.get(gname) if Sd.get(gname) is None else round(Sd[gname], 2)} |")
    res = {}
    for label, keep in (("all 13", lambda g: True),
                        ("fp32 >= 30% (drops the undertrained attention w256)", lambda g: fits[g]["p0"] >= 0.30),
                        ("MLP mixers only", lambda g: "mlp" in g), ("attention mixers, fp32 >= 30%", lambda g: "attn" in g and fits[g]["p0"] >= 0.30)):
        gs = [g for g in fits if keep(g) and Sd.get(g)]
        if len(gs) < 3:
            continue
        xs = [math.log(Sd[g]) for g in gs]
        ys = [fits[g]["m"] for g in gs]
        coef, pred, r2 = loo_fit(xs, ys)
        err = np.abs(np.array(ys) - pred)
        base = np.sqrt(np.mean([(ys[i] - np.mean(np.delete(ys, i))) ** 2 for i in range(len(ys))]))
        rho = spearman_np(xs, ys)
        print(f"\n{label}: {len(gs)} variants, m vs log S slope {coef[1]:+.3f}, R^2 {r2:.2f}, Spearman {rho:+.2f}, "
              f"LOO RMSE {np.sqrt(np.mean(err ** 2)):.3f} (baseline {base:.3f}), worst {err.max():.3f}")
        res[label] = {"n": len(gs), "slope": float(coef[1]), "r2": float(r2), "spearman": rho,
                      "loo_rmse": float(np.sqrt(np.mean(err ** 2))), "baseline": float(base)}
    return {"fits": fits, "S": Sd, "tests": res}


def spearman_np(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


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
    ap.add_argument("--variants", action="store_true", help="within-Sudoku test on the 13 trained variants (p4d)")
    ap.add_argument("--s-platform", default="babel", choices=["babel", "p4d"],
                    help="TRM/HRM sensitivity S source: X2's Babel fit or its p4d refit (same platform as the grid)")
    ap.add_argument("--predict", action="store_true", help="fit the per-group midpoint on label-free predictors (needs --with-trm)")
    a = ap.parse_args()
    global S_PLATFORM
    S_PLATFORM = a.s_platform
    if a.variants:
        out = {"variants": variant_section()}
        if a.json:
            json.dump(out, open(a.json, "w"), indent=1)
        return
    lm = lm_cells(a.roots or A.POOL_ROOTS, a.cut)
    out = {"lm": run(lm, "Looped LMs: one midpoint?")}
    if a.with_trm:
        trm = trm_cells(a.trm_platform)
        out["trm"] = run(trm, "TRM/HRM tasks: one midpoint?")
        out["all"] = run({**lm, **trm}, "Looped LMs + TRM/HRM: one midpoint?")
        if a.predict:
            out["predict"] = predict_section({**lm, **trm}, out["all"]["groups"])
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
