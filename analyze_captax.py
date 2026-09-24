#!/usr/bin/env python3
"""Angle B, cap vs tax, decomposed (X1; reports/edgetrm_exp/X1.md).

"Compression caps test-time scaling, it does not tax it" is a claim that weight quantization moves
the ASYMPTOTE of accuracy-vs-iterations and not its RATE. Three read-outs test it:

  fit     acc(T) = A * (1 - exp(-(T - T0) / tau)) for T > T0 (0 before): the coordinator's
          A_inf - B exp(-T/tau) with the onset made explicit (B = A exp(T0/tau)) so that
          sigmoid-looking curves that sit at 0 for the first iterations are fitted. Per quantizer
          and instance set; 95% bootstrap intervals (instances resampled, paired across quantizers)
          for A, tau, T0, and for the paired differences to fp32. Model-free companion: T90, the
          iterations to reach 90% of the curve's own maximum (linear interpolation on log T).
  solve   iterations-to-solve on instances BOTH fp32 and the quantized model solve for good
          (correct from that count through the last one): a tax shows as later solves; a pure cap
          as none.
  length  does the cap follow exact ~ (1 - h)^L (a constant per-cell hazard h amplified by the
          number of target cells L)? h fitted on the largest maze, caps predicted at every other size,
          compared with the fitted A; and h compared with the measured per-cell error.

Sources:
  dt   deep-thinking mazes 9..33, p4d `results/breadth/p4d/dt_extrap_mazes_*.json` (T = 10..500)
       and `dt_cell_mazes_{25,33}_s*.json` (per-cell accuracy, per-instance target-cell counts)
  sudoku  X1's stratified TRM/HRM sweeps `results/breadth/babel/x1/{diff,deep}_*.json` (T = ACT
       steps 1..64 or 1..512) per rating bin.

    python analyze_captax.py fit dt         # or: fit sudoku_first | sudoku | sudoku_deep
    python analyze_captax.py solve dt       # or: solve sudoku
    python analyze_captax.py length
    python analyze_captax.py json           # all fits -> results/breadth/x1_captax.json
"""
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent / "results" / "breadth"
NBOOT = 400
RNG = np.random.default_rng(0)
DT_QUANTS = ["fp32", "w8c", "w6c", "w4g32", "w4g128", "w4a", "w4c", "w4t", "w3g32", "w3a",
             "noise0.1", "noise0.15", "noise0.2", "noise0.3"]


# ------------------------------------------------------------------------ fitting
TAUS = np.exp(np.linspace(np.log(0.3), np.log(3000), 220))


def fit_curve(T, y):
    """least squares for y = A (1 - exp(-(T - T0)/tau)) (T > T0), 0 else; grid over (tau, T0),
    A closed form (clipped to [0, 1]). Returns A, tau, T0, rmse."""
    T = np.asarray(T, float)
    y = np.asarray(y, float)
    t0s = np.concatenate([[0.0], np.unique(T)[:-1] * 0.999, (np.unique(T)[:-1] + np.unique(T)[1:]) / 2])
    best = (np.inf, 0, 0, 0)
    for t0 in t0s:
        dT = np.clip(T - t0, 0, None)
        G = 1 - np.exp(-dT[None, :] / TAUS[:, None])          # (ntau, nT)
        den = (G * G).sum(1)
        A = np.where(den > 0, (G * y[None, :]).sum(1) / np.maximum(den, 1e-12), 0.0)
        A = np.clip(A, 0, 1)
        sse = ((A[:, None] * G - y[None, :]) ** 2).sum(1)
        j = int(np.argmin(sse))
        if sse[j] < best[0] - 1e-12:
            best = (sse[j], A[j], TAUS[j], t0)
    sse, A, tau, t0 = best
    return float(A), float(tau), float(t0), float(math.sqrt(sse / len(T)))


def t_frac(T, y, frac=0.9):
    """model-free rate: first T (log-interpolated) at which y reaches frac * max(y)."""
    T, y = np.asarray(T, float), np.asarray(y, float)
    tgt = frac * y.max()
    if y.max() <= 0:
        return float("nan")
    for i in range(len(T)):
        if y[i] >= tgt:
            if i == 0:
                return float(T[0])
            lo, hi = math.log(T[i - 1]), math.log(T[i])
            w = (tgt - y[i - 1]) / max(y[i] - y[i - 1], 1e-12)
            return float(math.exp(lo + w * (hi - lo)))
    return float("nan")


def fit_set(T, mats, nboot=NBOOT):
    """mats: {spec: (n, len(T)) bool}. Point fits + paired bootstrap. Returns {spec: dict}."""
    specs = list(mats)
    n = next(iter(mats.values())).shape[0]
    out = {}
    for s in specs:
        y = mats[s].mean(0)
        A, tau, t0, rmse = fit_curve(T, y)
        out[s] = {"A": A, "tau": tau, "T0": t0, "rmse": rmse, "T90": t_frac(T, y), "acc": y.tolist(),
                  "boot": {"A": [], "tau": [], "T0": [], "T90": []}}
    for _ in range(nboot):
        idx = RNG.integers(0, n, n)
        for s in specs:
            y = mats[s][idx].mean(0)
            A, tau, t0, _ = fit_curve(T, y)
            b = out[s]["boot"]
            b["A"].append(A); b["tau"].append(tau); b["T0"].append(t0); b["T90"].append(t_frac(T, y))
    for s in specs:
        b = out[s]["boot"]
        for k in ("A", "tau", "T0", "T90"):
            v = np.array(b[k], float)
            out[s][k + "_ci"] = [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
        if "fp32" in out and s != "fp32":
            f = out["fp32"]["boot"]
            for k in ("A", "tau", "T90"):
                if k == "A":
                    d = np.array(b[k]) - np.array(f[k])
                else:                                            # rate as a ratio (log scale)
                    d = np.log(np.array(b[k], float) / np.array(f[k], float))
                pt = (out[s][k] - out["fp32"][k]) if k == "A" else math.log(out[s][k] / out["fp32"][k])
                out[s]["d" + k] = [float(pt), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))]
    for s in specs:
        del out[s]["boot"]
    return out


# ------------------------------------------------------------------------ data
def _spec(r):
    return r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]


def dt_sets():
    """{size: (T, {spec: (n, nT) bool})} from the p4d extrapolation records."""
    out = {}
    for f in sorted(glob.glob(str(B / "p4d" / "dt_extrap_mazes_*.json")), key=lambda p: int(Path(p).stem.split("_")[-1])):
        recs = json.load(open(f))["records"]
        T = sorted({r["iters"] for r in recs})
        by = {(_spec(r), r["iters"]): np.array(r["correct"], bool) for r in recs}
        mats = {}
        for q in DT_QUANTS:
            if all((q, t) in by for t in T):
                mats[q] = np.stack([by[(q, t)] for t in T], 1)
        out[Path(f).stem.split("_")[-1]] = (T, mats, recs[0].get("mean_target_cells"))
    return out


def sudoku_sets(kind="diff"):
    """{(task, bin): (T, {spec: (n, K) bool})} from X1's stratified sweeps; noise draws kept apart."""
    meta = json.load(open(B / "sudoku_extreme_test_meta.json"))
    row_bin = dict(zip(meta["strat"]["rows"], meta["strat"]["bin"]))
    names = ["0", "1-10", "11-30", "31-60", "61-120", ">120"]
    out = {}
    for task in ("sudoku_mlp", "sudoku_attn", "hrm_sudoku"):
        files = sorted(glob.glob(str(B / ("p4d" if kind == "deep" else "babel") / "x1" / f"{kind}_{task}_*.json")))
        if not files:
            continue
        mats, rows, K = {}, None, None
        for f in files:
            d = json.load(open(f))
            rows = rows or d["rows"]
            assert d["rows"] == rows
            K = d["K"]
            for r in d["records"]:
                if r["spec"] in mats:
                    continue
                v = np.array([int(h, 16) for h in r["bits"]], dtype=object)
                mats[r["spec"]] = np.array([[(x >> k) & 1 for k in range(K)] for x in v], bool)
        bins = np.array([row_bin[r] for r in rows])
        # log-spaced read-out steps (half-octaves) keep the fits and bootstraps cheap
        T = sorted({int(round(2 ** (i / 2))) for i in range(0, int(2 * math.log2(K)) + 1)})
        cols = [t - 1 for t in T]
        for b, nm in enumerate(names):
            m = bins == b
            out[(task, nm)] = (T, {s: M[m][:, cols] for s, M in mats.items()})
    return out


def sudoku_first_sets():
    """{(task, bin): (T, mats)} from the existing p4d records on the first 1000 test rows
    (nsup 1,2,4,8,16 from *_all.json, 32/64 from *_deep.json; default H)."""
    meta = json.load(open(B / "sudoku_extreme_test_meta.json"))
    r = np.array(meta["first1000"]["rating"])
    bins = [("easy 0-4", (r >= 0) & (r <= 4)), ("medium 5-30", (r >= 5) & (r <= 30)), ("hard >=31", r >= 31)]
    T = [1, 2, 4, 8, 16, 32, 64]
    out = {}
    for task, H in (("sudoku_mlp", 3), ("sudoku_attn", 3), ("hrm_sudoku", 2)):
        by = {}
        for f in (f"{task}_all.json", f"{task}_deep.json"):
            for rec in json.load(open(B / f))["records"]:
                if rec["H_cycles"] == H:
                    by[(rec["quant"]["spec"], rec["nsup"])] = np.array(rec["correct"], bool)
        specs = [q for q in ("fp32", "w8c", "w4g32", "w4a", "w4c", "w4t", "w3g32", "w3a", "w3c", "w2a", "w2g32")
                 if all((q, t) in by for t in T)]
        for nm, m in bins:
            mats = {q: np.stack([by[(q, t)][m] for t in T], 1) for q in specs}
            # drop quantizers that never solve anything in this bin (nothing to fit)
            mats = {q: M for q, M in mats.items() if M.any()}
            out[(task, nm)] = (T, mats)
    return out


# ------------------------------------------------------------------------ read-outs
def fmt_ci(v, ci, pct=False, nd=1):
    f = (lambda x: f"{100*x:.{nd}f}") if pct else (lambda x: f"{x:.{nd}f}")
    return f"{f(v)} [{f(ci[0])},{f(ci[1])}]"


def print_fits(title, T, fits):
    print(f"\n### {title} (T = {T[0]}..{T[-1]})")
    print("| quant | acc@Tmax | A (cap) [95%] | tau [95%] | T0 | T90 [95%] | rmse | ΔA vs fp32 [95%] | log(tau/tau_fp32) [95%] | log(T90/T90_fp32) [95%] |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s, f in fits.items():
        dA = "" if "dA" not in f else f"{100*f['dA'][0]:+.1f} [{100*f['dA'][1]:+.1f},{100*f['dA'][2]:+.1f}]"
        dt = "" if "dtau" not in f else f"{f['dtau'][0]:+.2f} [{f['dtau'][1]:+.2f},{f['dtau'][2]:+.2f}]"
        d9 = "" if "dT90" not in f else f"{f['dT90'][0]:+.2f} [{f['dT90'][1]:+.2f},{f['dT90'][2]:+.2f}]"
        print(f"| {s} | {100*f['acc'][-1]:.1f} | {fmt_ci(f['A'], f['A_ci'], pct=True)} | {fmt_ci(f['tau'], f['tau_ci'])} | "
              f"{f['T0']:.0f} | {fmt_ci(f['T90'], f['T90_ci'], nd=0)} | {100*f['rmse']:.1f} | {dA} | {dt} | {d9} |")


def fit(which):
    res = {}
    if which == "dt":
        for size, (T, mats, _) in dt_sets().items():
            fits = fit_set(T, mats)
            res[f"dt_maze_{size}"] = {"T": T, "fits": fits}
            print_fits(f"deep-thinking maze {size} (n=1000, p4d)", T, fits)
    elif which == "sudoku_first":
        for (task, nm), (T, mats) in sudoku_first_sets().items():
            fits = fit_set(T, mats, nboot=300)
            res[f"first_{task}_{nm}"] = {"T": T, "fits": fits}
            print_fits(f"{task} {nm} (first 1000 rows, n={next(iter(mats.values())).shape[0]}, p4d)", T, fits)
    else:
        kind = "deep" if which == "sudoku_deep" else "diff"
        for (task, nm), (T, mats) in sudoku_sets(kind).items():
            fits = fit_set(T, mats, nboot=200)
            res[f"{kind}_{task}_{nm}"] = {"T": T, "fits": fits}
            print_fits(f"{task} rating {nm} (n={next(iter(mats.values())).shape[0]}, Babel L40S, {kind})", T, fits)
    return res


def stable_solve(M):
    """(n, nT) bool -> index of the first count from which the instance stays correct; -1 if wrong at the end."""
    tail = np.flip(np.logical_and.accumulate(np.flip(M, 1), axis=1), 1)
    first = np.where(tail.any(1), tail.argmax(1), -1)
    return first


def solve(which):
    if which == "dt":
        sets = {f"dt maze {k}": (T, m) for k, (T, m, _) in dt_sets().items()}
    elif which == "sudoku_first":
        sets = {f"{t} {b} (first 1000)": v for (t, b), v in sudoku_first_sets().items()}
    else:
        sets = {f"{t} rating {b}": v for (t, b), v in sudoku_sets("deep" if which == "sudoku_deep" else "diff").items()}
    print("## Iterations-to-solve on instances both fp32 and the quantized model solve for good at the last count")
    print("(stable solve = first count from which the instance stays correct; later = the quantized model needs more)")
    for name, (T, mats) in sets.items():
        if "fp32" not in mats:
            continue
        Tn = np.array(T, float)
        sf = stable_solve(mats["fp32"])
        print(f"\n### {name}")
        print("| quant | both solve | later | same | earlier | median T fp32 -> quant | mean log2(T_q/T_fp32) [95%] |")
        print("|---|---|---|---|---|---|---|")
        for s, M in mats.items():
            if s == "fp32":
                continue
            sq = stable_solve(M)
            both = (sf >= 0) & (sq >= 0)
            nb = int(both.sum())
            if nb < 5:
                print(f"| {s} | {nb} | | | | | |")
                continue
            a, b = Tn[sf[both]], Tn[sq[both]]
            lr = np.log2(b / a)
            boots = [lr[RNG.integers(0, nb, nb)].mean() for _ in range(1000)]
            print(f"| {s} | {nb} | {100*(b > a).mean():.1f}% | {100*(b == a).mean():.1f}% | {100*(b < a).mean():.1f}% | "
                  f"{np.median(a):.0f} -> {np.median(b):.0f} | {lr.mean():+.3f} [{np.percentile(boots, 2.5):+.3f},{np.percentile(boots, 97.5):+.3f}] |")


def length():
    """exact ~ (1 - h)^L. Per quantizer: fit h to the size-33 cap (fitted A), predict every other
    size's cap from its mean target-cell count; also the per-instance version on sizes 25/33 by
    path-length quartile; and h vs the measured per-cell target error (dt_cell)."""
    sets = dt_sets()
    fits = {size: fit_set(T, mats, nboot=0) for size, (T, mats, _) in sets.items()}
    L = {size: v[2] for size, v in sets.items()}
    print("## Cap vs output length: exact = (1 - h)^L with one per-cell hazard h per quantizer, fitted at size 33")
    print("mean target cells L per size: " + ", ".join(f"{s}: {L[s]:.0f}" for s in L))
    print("| quant | h (per 1000 cells) | " + " | ".join(f"size {s}: A fit / pred" for s in L) + " |")
    print("|---|---|" + "---|" * len(L))
    for q in DT_QUANTS:
        if q not in fits["33"]:
            continue
        A33 = min(fits["33"][q]["A"], 0.999999)
        fA33 = min(fits["33"]["fp32"]["A"], 0.999999)
        # relative cap (quantizer A / fp32 A) so fp32's own shortfall is not counted as hazard
        rel = min(A33 / fA33, 0.999999)
        h = 1 - rel ** (1 / L["33"])
        cells = []
        for s in L:
            if q in fits[s]:
                pred = (1 - h) ** L[s] * fits[s]["fp32"]["A"]
                cells.append(f"{100*fits[s][q]['A']:.1f} / {100*pred:.1f}")
            else:
                cells.append("--")
        print(f"| {q} | {1000*h:.2f} | " + " | ".join(cells) + " |")
    # per-cell error measured on the reruns vs the hazard implied by exact accuracy
    print("\n### measured per-cell target error vs the hazard implied by exact accuracy (dt_cell, 300 iterations)")
    print("| size | seed | quant | exact | cell acc (target) | 1-exact implied by i.i.d. cells: 1-(p_t)^L | mean wrong target cells per failing instance |")
    print("|---|---|---|---|---|---|---|")
    for size in ("25", "33"):
        for f in sorted(glob.glob(str(B / "p4d" / f"dt_cell_mazes_{size}_s*.json"))):
            recs = json.load(open(f))["records"]
            for r in recs:
                if r["iters"] != 300:
                    continue
                pt, ex, Lm = r["cell_acc_target"], r["acc"], r["mean_target_cells"]
                iid = 1 - pt ** Lm
                per_fail = ((1 - pt) * Lm / (1 - ex)) if ex < 1 else float("nan")
                print(f"| {size} | {Path(f).stem[-2:]} | {_spec(r)} | {100*ex:.1f} | {100*pt:.2f} | {100*iid:.1f} | {per_fail:.1f} |")
    # within-size: failure rate by path-length quartile vs the (1-h)^L prediction
    print("\n### within size: failure by path-length quartile at 500 iterations, observed vs 1-(1-h)^L_quartile (h fitted to that size's failure rate)")
    for size in ("25", "33"):
        files = sorted(glob.glob(str(B / "p4d" / f"dt_cell_mazes_{size}_s*.json")))
        nt = None
        for fpath in files:
            for r in json.load(open(fpath))["records"]:
                if "n_target_cells" in r:
                    nt = np.array(r["n_target_cells"], float)
                    break
            if nt is not None:
                break
        T, mats, _ = sets[size]
        qs = np.quantile(nt, [0.25, 0.5, 0.75])
        bins = np.digitize(nt, qs)
        print(f"size {size}: target-cell quartile means " + ", ".join(f"{nt[bins==b].mean():.0f}" for b in range(4)))
        print("| quant | observed Q1..Q4 | predicted Q1..Q4 |")
        print("|---|---|---|")
        for q in ("w4c", "w4t", "w3a", "w3g32", "noise0.2"):
            if q not in mats:
                continue
            ok = mats[q][:, -1] | ~mats["fp32"][:, -1]          # failures among the instances fp32 solves
            fsolved = mats["fp32"][:, -1]
            fail = ~mats[q][:, -1] & fsolved
            rate = fail.sum() / fsolved.sum()
            # solve mean((1-h)^L) = 1 - rate for h by bisection
            lo, hi = 0.0, 0.5
            for _ in range(60):
                mid = (lo + hi) / 2
                if np.mean((1 - mid) ** nt[fsolved]) > 1 - rate:
                    lo = mid
                else:
                    hi = mid
            h = lo
            obs = [fail[(bins == b) & fsolved].sum() / max(((bins == b) & fsolved).sum(), 1) for b in range(4)]
            pred = [1 - np.mean((1 - h) ** nt[(bins == b) & fsolved]) for b in range(4)]
            print(f"| {q} | " + " / ".join(f"{100*x:.1f}" for x in obs) + " | " + " / ".join(f"{100*x:.1f}" for x in pred) + " |")


def to_json():
    res = {"dt": fit("dt"), "sudoku_first": fit("sudoku_first")}
    s = fit("sudoku")
    if s:
        res["sudoku"] = s
    d = fit("sudoku_deep")
    if d:
        res["sudoku_deep"] = d
    p = B / "x1_captax.json"
    json.dump(res, open(p, "w"))
    print("wrote", p)


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = sys.argv[2] if len(sys.argv) > 2 else "dt"
    {"fit": lambda: fit(arg), "solve": lambda: solve(arg), "length": length, "json": to_json}[cmd]()
