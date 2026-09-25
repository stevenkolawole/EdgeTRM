#!/usr/bin/env python3
"""X1 read-out: does compression cap test-time scaling on Sudoku-Extreme, per difficulty bin?
(angle B on a second family; reports/edgetrm_exp/X1.md).

Difficulty = the test puzzle's `rating` (tdoku backtracks), from
results/breadth/sudoku_extreme_test_meta.json (built from sapientinc/sudoku-extreme test.csv).

    (append --axis residual to any command to bin by blanks left after singles instead of the rating)
    python analyze_difficulty.py first    # existing p4d records, first 1000 test rows, nsup 1..64, H 1..4
    python analyze_difficulty.py strat    # stratified Babel sweep (difficulty_sweep.py), 6 bins x 1000, k = 1..64
    python analyze_difficulty.py deep     # deep Babel sweep, 300 per bin, k = 1..512
    python analyze_difficulty.py halt     # halting-head AUROC per rating bin and quantizer (angle C)
    python analyze_difficulty.py gapdepth # deep cells: gap at 16/64/512 with CIs, depth trend test, draws pooled
    python analyze_difficulty.py glink    # which puzzles fail: fp32's solve step vs rating
    python analyze_difficulty.py json     # figure-ready JSON -> results/breadth/x1_difficulty_readout.json
    python analyze_difficulty.py fig [first1000|strat|deep]   # draft figure -> plots/x1_difficulty_<src>.{png,pdf}

Every accuracy comes with n and a Wilson 95% interval; every gap to fp32 is paired (same
puzzles) with a normal interval on the paired difference. Noise specs with several draws
(`noise0.15@s0/@s1/@s2`) are shown per draw and as the median draw.
"""
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parent / "results" / "breadth"
META = json.load(open(B / "sudoku_extreme_test_meta.json"))
TASKS = [("sudoku_mlp", "TRM Sudoku-MLP", 3), ("sudoku_attn", "TRM Sudoku-attn", 3), ("hrm_sudoku", "HRM Sudoku", 2)]
# bins for the first-1000 read-out (value-based, fixed before looking at accuracies by bin)
# difficulty axis: --axis rating (default; the dataset's tdoku backtrack count, planned axis) or
# --axis residual (blanks left after naked + hidden singles; sudoku_difficulty.py)
AXIS = "rating"
if "--axis" in sys.argv:
    AXIS = sys.argv[sys.argv.index("--axis") + 1]
    del sys.argv[sys.argv.index("--axis"):sys.argv.index("--axis") + 2]
RATING_FIRST = [(0, 4, "easy 0-4"), (5, 30, "medium 5-30"), (31, 10 ** 6, "hard >=31")]
RES_BINS = [(0, 0, "res 0"), (1, 45, "res 1-45"), (46, 50, "res 46-50"), (51, 55, "res 51-55"), (56, 81, "res >=56")]
FIRST_BINS = RATING_FIRST if AXIS == "rating" else RES_BINS
STRAT_NAMES = ["0", "1-10", "11-30", "31-60", "61-120", ">120"]
NSUP = (1, 2, 4, 8, 16, 32, 64)
# which quantizers to show, per model, and their side of the basin (from the Sep 14-23 sweeps)
SHOW_FIRST = ["fp32", "w8c", "w4g32", "w4c", "w4a", "w4t", "w3g32", "w3c", "w2g32"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def paired_gap(a, b):
    """mean(a) - mean(b) for paired 0/1 vectors, with a 95% normal interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    d = a - b
    m = d.mean()
    se = d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return m, m - 1.96 * se, m + 1.96 * se


def fmt(p, lo=None, hi=None):
    if p != p:
        return "--"
    return f"{100*p:.1f}" if lo is None else f"{100*p:.1f} [{100*lo:.0f},{100*hi:.0f}]"


# ------------------------------------------------------------------------ existing records (first 1000, p4d)
def first_bins():
    r = np.array(META["first1000"]["rating" if AXIS == "rating" else "singles_residual"])
    return [(name, (r >= lo) & (r <= hi)) for lo, hi, name in FIRST_BINS], r


def load_first(task, H):
    """{(spec, H, nsup): correct[1000]} from the p4d breadth grids (+ deep nsup 32/64, + noise seed 0)."""
    out = {}
    for f in (f"{task}_all.json", f"{task}_deep.json", f"{task}_noise.json"):
        p = B / f
        if not p.exists():
            continue
        for rec in json.load(open(p))["records"]:
            key = (rec["quant"]["spec"], rec["H_cycles"], rec["nsup"])
            if key in out and f.endswith("noise.json"):
                continue                      # fp32 appears in both; keep the grid's copy
            out[key] = np.array(rec["correct"], bool)
    return out


def first():
    bins, _ = first_bins()
    print("## Read-out A: existing p4d records, first 1000 test rows (n per bin: "
          + ", ".join(f"{name} {int(m.sum())}" for name, m in bins) + ")")
    for task, label, H in TASKS:
        rec = load_first(task, H)
        specs = [s for s in SHOW_FIRST if (s, H, 16) in rec]
        noise = sorted({k[0] for k in rec if k[0].startswith("noise") and k[1] == H},
                       key=lambda s: float(s[5:]))
        print(f"\n### {label} (H={H}); accuracy % [Wilson 95%] by nsup")
        print("| quant | bin | " + " | ".join(f"n={k}" for k in NSUP) + " | gap@16 | gap@64 | own gain 16->64 |")
        print("|---|---|" + "---|" * (len(NSUP) + 3))
        for s in specs + noise:
            for name, m in bins:
                row = []
                for k in NSUP:
                    c = rec.get((s, H, k))
                    row.append("--" if c is None else fmt(*wilson(int(c[m].sum()), int(m.sum()))))
                g = []
                for k in (16, 64):
                    c, f = rec.get((s, H, k)), rec.get(("fp32", H, k))
                    if c is None or f is None or s == "fp32":
                        g.append("")
                    else:
                        d, lo, hi = paired_gap(c[m], f[m])
                        g.append(f"{100*d:+.1f} [{100*lo:+.0f},{100*hi:+.0f}]")
                c16, c64 = rec.get((s, H, 16)), rec.get((s, H, 64))
                gain = "" if c16 is None or c64 is None else f"{100*(c64[m].mean()-c16[m].mean()):+.1f}"
                print(f"| {s} | {name} | " + " | ".join(row) + f" | {g[0]} | {g[1]} | {gain} |")
        # H axis at nsup 16
        print(f"\n{label}: accuracy % by H at nsup 16 (easy/medium/hard)")
        for s in specs:
            cells = []
            for h in (1, 2, 3, 4):
                c = rec.get((s, h, 16))
                cells.append("--" if c is None else "/".join(f"{100*c[m].mean():.0f}" for _, m in bins))
            print(f"  {s:7s} " + "  ".join(f"H{h}: {x}" for h, x in zip((1, 2, 3, 4), cells)))


# ------------------------------------------------------------------------ stratified sweep (Babel)
def unpack(hexes, K):
    x = np.array([int(h, 16) for h in hexes], dtype=object)
    return np.array([[(v >> k) & 1 for k in range(K)] for v in x], dtype=bool)


def load_strat(task, kind="diff"):
    """{spec: record with 'B' (n x K bool)}; merges every file of this task (fp32 from the first file,
    other fp32 copies are checked for identity)."""
    recs, fp32s = {}, []
    # kind: "deep" (K=512, p4d) | "diff" (K=64, 6000 puzzles, p4d) | "diff_babel" (the Babel K=64 MLP file)
    plat, pref = {"deep": ("p4d", "deep"), "diff": ("p4d", "diff"), "diff_babel": ("babel", "diff")}[kind]
    files = sorted(glob.glob(str(B / plat / "x1" / f"{pref}_{task}_*.json")))
    if kind == "deep":                                   # robustness draws share the deep protocol and fp32
        files += sorted(glob.glob(str(B / plat / "x1" / f"rb_{task}_*.json")))
    rows = None
    for f in files:
        d = json.load(open(f))
        if rows is None:
            rows = d["rows"]
        assert d["rows"] == rows, f"{f}: different puzzle set"
        file_fp32 = None
        for r in d["records"]:
            r = dict(r)
            r["B"] = unpack(r["bits"], r["K"])
            r["file"] = Path(f).name
            if r["spec"] == "fp32":
                file_fp32 = r["B"]
                fp32s.append(r)
            r["F"] = file_fp32                    # the fp32 run of the same job (same node, same process);
            #                                       None for a --ref_wait file, filled in below
            if r["spec"] == "fp32" and "fp32" in recs:
                continue
            recs.setdefault(r["spec"], r)
    for s_, r_ in recs.items():
        if isinstance(r_, dict) and r_.get("F") is None and "fp32" in recs:
            r_["F"] = recs["fp32"]["B"]
    if fp32s:
        ref = fp32s[0]["B"]
        same = [float((f["B"] == ref).all(1).mean()) for f in fp32s[1:]]
        recs["_fp32_identity"] = same
    return recs, rows


ROW_BIN = dict(zip(META["strat"]["rows"], META["strat"]["bin"]))
ROW_RATING = dict(zip(META["strat"]["rows"], META["strat"]["rating"]))
ROW_RES = dict(zip(META["strat"]["rows"], META["strat"].get("singles_residual", [0] * len(META["strat"]["rows"]))))


def strat_bins(rows=None):
    """bins as masks over `rows` (the puzzle order of a sweep file); default = the full stratified set."""
    rows = META["strat"]["rows"] if rows is None else rows
    if AXIS == "residual":
        v = np.array([ROW_RES[r] for r in rows])
        return [(nm, (v >= lo) & (v <= hi)) for lo, hi, nm in RES_BINS]
    b = np.array([ROW_BIN[r] for r in rows])
    return [(STRAT_NAMES[i], b == i) for i in range(len(STRAT_NAMES))]


def _groups(recs):
    """spec -> list of draw specs (noise0.15 -> [noise0.15@s0, @s1, @s2]); plain specs map to themselves."""
    g = defaultdict(list)
    for s in recs:
        if s.startswith("_"):
            continue
        g[s.split("@s")[0]].append(s)
    return g


def strat(kind="diff"):
    for task, label, H in TASKS:
        recs, rows = load_strat(task, kind)
        if not recs:
            print(f"\n### {label}: no {kind} records yet")
            continue
        bins = strat_bins(rows)
        K = recs["fp32"]["K"]
        ks = [k for k in NSUP + (128, 256, 512) if k <= K]
        mid = 16 if K <= 64 else 64
        f = recs["fp32"]["B"]
        print(f"\n### {label} ({kind}, n per bin {[int(m.sum()) for _, m in bins]}, K={K}; fp32 identity across files: "
              f"{recs.get('_fp32_identity')})")
        print("| quant | φ_H | bin | " + " | ".join(f"k={k}" for k in ks) + f" | gap@{mid} | gap@K | own gain {mid}->K | fp32 gain {mid}->K |")
        print("|---|---|---|" + "---|" * (len(ks) + 4))
        for base, draws in _groups(recs).items():
            for s in sorted(draws):
                r = recs[s]
                Bq, f = r["B"], r["F"]
                for name, m in bins:
                    row = [fmt(*wilson(int(Bq[m, k - 1].sum()), int(m.sum()))) for k in ks]
                    g16 = paired_gap(Bq[m, mid - 1], f[m, mid - 1]) if K >= mid else (float("nan"),) * 3
                    gK = paired_gap(Bq[m, K - 1], f[m, K - 1])
                    own = Bq[m, K - 1].mean() - Bq[m, mid - 1].mean()
                    fg = f[m, K - 1].mean() - f[m, mid - 1].mean()
                    gs = ["" if s == "fp32" else f"{100*g[0]:+.1f} [{100*g[1]:+.0f},{100*g[2]:+.0f}]" for g in (g16, gK)]
                    print(f"| {s} | {r['fid_H_mean']:.3f} | {name} | " + " | ".join(row)
                          + f" | {gs[0]} | {gs[1]} | {100*own:+.1f} | {100*fg:+.1f} |")


# ------------------------------------------------------------------------ G link
def solve_step(Bm):
    """first step from which the puzzle stays correct through the last step; K+1 if wrong at K."""
    n, K = Bm.shape
    out = np.full(n, K + 1)
    ok_tail = np.flip(np.logical_and.accumulate(np.flip(Bm, 1), axis=1), 1)   # correct from k to K
    for i in range(n):
        idx = np.nonzero(ok_tail[i])[0]
        if len(idx):
            out[i] = idx[0] + 1
    return out


def spearman(x, y):
    def rank(v):
        v = np.asarray(v, float)
        o = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[o] = np.arange(len(v))
        # average ties
        _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.bincount(inv, weights=r)
        return (sums / cnt)[inv]
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


STEP_BINS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 32), (33, 64)]


def glink():
    print("## G link: among puzzles fp32 solves at the last step, failure rate of each quantizer at the "
          "last step, by fp32's stable solve step and by rating bin")
    # (a) existing p4d residual records, first 1000, per-step k = 1..16
    r1000 = np.array(META["first1000"]["rating"])
    for task, label, H in TASKS:
        p = B / "p4d" / f"residual_{task}.json"
        if not p.exists():
            continue
        recs = {r["quant"]["spec"]: r for r in json.load(open(p))["records"]}
        K = recs["fp32"]["nsup"]
        Bf = np.array([recs["fp32"]["per_step"][str(k)]["correct"] for k in range(1, K + 1)], bool).T
        st = solve_step(Bf)
        solved = st <= K
        print(f"\n### {label}, first 1000 (p4d residual, K={K}); fp32 solves {solved.sum()}; "
              f"Spearman(fp32 solve step, rating) over solved = {spearman(st[solved], r1000[solved]):+.2f}")
        sb = [(a, b) for a, b in STEP_BINS if a <= K]
        rb = [(lo, hi, nm) for lo, hi, nm in FIRST_BINS]
        r1000 = first_bins()[1]
        print("| quant | acc@K | " + " | ".join(f"step {a}-{b}" for a, b in sb) + " | " + " | ".join(nm for _, _, nm in rb) + " |")
        print("|---|---|" + "---|" * (len(sb) + len(rb)))
        cnt = ["n=" + str(int((solved & (st >= a) & (st <= b)).sum())) for a, b in sb] + \
              ["n=" + str(int((solved & (r1000 >= lo) & (r1000 <= hi)).sum())) for lo, hi, _ in rb]
        print("| (fp32-solved count) | | " + " | ".join(cnt) + " |")
        for s, r in recs.items():
            if s == "fp32":
                continue
            c = np.array(r["per_step"][str(K)]["correct"], bool)
            cells = []
            for a, b in sb:
                m = solved & (st >= a) & (st <= b)
                cells.append(fmt(1 - c[m].mean()) if m.sum() else "--")
            for lo, hi, _ in rb:
                m = solved & (r1000 >= lo) & (r1000 <= hi)
                cells.append(fmt(1 - c[m].mean()) if m.sum() else "--")
            print(f"| {s} | {100*c.mean():.1f} | " + " | ".join(cells) + " |")
    # (b) stratified Babel records
    for task, label, H in TASKS:
        recs, rows = load_strat(task)
        if not recs:
            continue
        bins = strat_bins(rows)
        rs = np.array([ROW_RATING[r] for r in rows])
        K = recs["fp32"]["K"]
        st = solve_step(recs["fp32"]["B"])
        solved = st <= K
        print(f"\n### {label}, stratified (K={K}); fp32 solves {solved.sum()} of {len(st)}; "
              f"Spearman(fp32 solve step, rating) over solved = {spearman(st[solved], rs[solved]):+.2f}")
        sb = [(a, b) for a, b in STEP_BINS if a <= K]
        print("| quant | acc@K | " + " | ".join(f"step {a}-{b}" for a, b in sb) + " | " + " | ".join(nm for nm, _ in bins) + " |")
        print("|---|---|" + "---|" * (len(sb) + len(bins)))
        cnt = ["n=" + str(int((solved & (st >= a) & (st <= b)).sum())) for a, b in sb] + \
              ["n=" + str(int((solved & m).sum())) for _, m in bins]
        print("| (fp32-solved count) | | " + " | ".join(cnt) + " |")
        for s, r in recs.items():
            if s in ("fp32", "_fp32_identity"):
                continue
            c = r["B"][:, K - 1]
            cells = []
            for a, b in sb:
                m = solved & (st >= a) & (st <= b)
                cells.append(fmt(1 - c[m].mean()) if m.sum() else "--")
            for _, bm in bins:
                m = solved & bm
                cells.append(fmt(1 - c[m].mean()) if m.sum() else "--")
            print(f"| {s} | {100*c.mean():.1f} | " + " | ".join(cells) + " |")


def _auroc(score, positive):
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    npos, nneg = positive.sum(), (~positive).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def halt(kind="diff"):
    """angle C per difficulty bin: AUROC of -q_halt (after the last step) for 'answer wrong'."""
    for task, label, H in TASKS:
        recs, rows = load_strat(task, kind)
        if not recs:
            continue
        bins = strat_bins(rows)
        K = recs["fp32"]["K"]
        print(f"\n### {label} ({kind}, K={K}): AUROC of -q_halt for a wrong answer at step {K}, by rating bin "
              "(accuracy in parentheses)")
        print("| quant | all | " + " | ".join(nm for nm, _ in bins) + " |")
        print("|---|---|" + "---|" * len(bins))
        for s, r in recs.items():
            if s.startswith("_"):
                continue
            q = np.array(r["qhalt"], float)
            c = r["B"][:, K - 1]
            cells = [f"{_auroc(-q, ~c):.2f} ({100*c.mean():.0f})"]
            for _, m in bins:
                cells.append(f"{_auroc(-q[m], ~c[m]):.2f} ({100*c[m].mean():.0f})")
            print(f"| {s} | " + " | ".join(cells) + " |")


def _ncdf(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def gapdepth():
    """Per deep cell: paired gap to fp32 at 16 / 64 / 512 steps (95% CI), and the depth trend
    D = gap(512) - gap(16) per puzzle, with CI and one-sided p for D > 0 (the gap shrinks = catch-up)
    and for D < 0 (it grows). Noise specs are also pooled over draws: mean over draws with a t-interval
    across draws (draw-to-draw spread is the dominant variance at the edge)."""
    ks = (16, 64, 512)
    tcrit = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}
    for task, label, H in TASKS:
        recs, rows = load_strat(task, "deep")
        if not recs:
            continue
        v = np.array([ROW_RES[r] for r in rows])
        masks = [("all", np.ones(len(v), bool)), ("hard", v >= 51)]
        F = recs["fp32"]["B"]
        print(f"\n### {label}: gap to fp32 (points) at 16 / 64 / 512 steps [95% CI], trend D = gap512 - gap16 "
              f"(n all {len(v)}, hard {int((v >= 51).sum())})")
        print("| quant | φ_H | subset | gap16 | gap64 | gap512 | D [95%] | p(D>0) | p(D<0) |")
        print("|---|---|---|---|---|---|---|---|---|")
        pooled = defaultdict(lambda: defaultdict(list))
        for s_, r in sorted(recs.items(), key=lambda kv: kv[0]):
            if s_.startswith("_") or s_ == "fp32":
                continue
            Bq = r["B"]
            for nm, m in masks:
                cells = []
                for k in ks:
                    g = paired_gap(Bq[m, k - 1], F[m, k - 1])
                    cells.append(f"{100*g[0]:+.1f} [{100*g[1]:+.1f},{100*g[2]:+.1f}]")
                d = (Bq[m, 511].astype(float) - F[m, 511]) - (Bq[m, 15].astype(float) - F[m, 15])
                mu, se = d.mean(), d.std(ddof=1) / math.sqrt(len(d))
                pp = 1 - _ncdf(mu / se) if se > 0 else float("nan")
                print(f"| {s_} | {r['fid_H_mean']:.3f} | {nm} | " + " | ".join(cells)
                      + f" | {100*mu:+.1f} [{100*(mu-1.96*se):+.1f},{100*(mu+1.96*se):+.1f}] | {pp:.3g} | {1-pp:.3g} |")
                if "@s" in s_:
                    for k in ks:
                        pooled[(s_.split("@s")[0], nm)][k].append(Bq[m, k - 1].mean() - F[m, k - 1].mean())
                    pooled[(s_.split("@s")[0], nm)]["D"].append(mu)
                    pooled[(s_.split("@s")[0], nm)]["phi"].append(r["fid_H_mean"])
        if pooled:
            print(f"\n{label}: noise specs pooled over draws (mean over draws [95% t-interval across draws]; n draws)")
            print("| spec | subset | draws | φ_H mean | gap16 | gap64 | gap512 | D | draws with D>0 |")
            print("|---|---|---|---|---|---|---|---|---|")
            for (base, nm), dd in sorted(pooled.items()):
                nd = len(dd["D"])
                def ci(x):
                    x = np.array(x)
                    if len(x) < 2:
                        return f"{100*x.mean():+.1f}"
                    h = tcrit.get(len(x) - 1, 2.0) * x.std(ddof=1) / math.sqrt(len(x))
                    return f"{100*x.mean():+.1f} [{100*(x.mean()-h):+.1f},{100*(x.mean()+h):+.1f}]"
                print(f"| {base} | {nm} | {nd} | {np.mean(dd['phi']):.3f} | {ci(dd[16])} | {ci(dd[64])} | {ci(dd[512])} | "
                      f"{ci(dd['D'])} | {sum(x > 0 for x in dd['D'])}/{nd} |")


# ------------------------------------------------------------------------ figure-ready JSON and draft figure
def build_json():
    out = {"meta": {"difficulty": "sapientinc/sudoku-extreme test.csv `rating` = tdoku backtracks",
                    "interval": "Wilson 95%", "script": "EdgeTRM/analyze_difficulty.py"}, "first1000": {}, "strat": {}}
    bins, _ = first_bins()
    for task, label, H in TASKS:
        rec = load_first(task, H)
        t = {}
        specs = sorted({k[0] for k in rec if k[1] == H})
        for s in specs:
            t[s] = {}
            for name, m in bins:
                pts = [(k, rec[(s, H, k)]) for k in NSUP if (s, H, k) in rec]
                t[s][name] = {"n": int(m.sum()), "iters": [k for k, _ in pts],
                              "acc": [], "lo": [], "hi": []}
                for k, c in pts:
                    p, lo, hi = wilson(int(c[m].sum()), int(m.sum()))
                    t[s][name]["acc"].append(round(100 * p, 2))
                    t[s][name]["lo"].append(round(100 * lo, 2))
                    t[s][name]["hi"].append(round(100 * hi, 2))
        out["first1000"][task] = {"label": label, "H": H, "platform": "p4d (A100-40GB)", "quants": t}
    out["deep"] = {}
    for kind, key in (("diff", "strat"), ("deep", "deep")):
        for task, label, H in TASKS:
            recs, rows_ = load_strat(task, kind)
            if not recs:
                continue
            sb = strat_bins(rows_)
            K = recs["fp32"]["K"]
            steps = list(range(1, K + 1)) if K <= 64 else \
                sorted({int(round(2 ** (i / 4))) for i in range(0, int(4 * math.log2(K)) + 1)})
            t = {}
            for s, r in recs.items():
                if s.startswith("_"):
                    continue
                t[s] = {"fid_H_mean": r["fid_H_mean"], "seed": r["seed"]}
                for name, m in sb:
                    acc, lo, hi = [], [], []
                    for k in steps:
                        p, a_, b_ = wilson(int(r["B"][m, k - 1].sum()), int(m.sum()))
                        acc.append(round(100 * p, 2)); lo.append(round(100 * a_, 2)); hi.append(round(100 * b_, 2))
                    t[s][name] = {"n": int(m.sum()), "iters": steps, "acc": acc, "lo": lo, "hi": hi}
            out[key][task] = {"label": label, "H": H, "K": K, "platform": "Babel L40S",
                              "fp32_identity_across_files": recs.get("_fp32_identity"), "quants": t}
    out["meta"]["axis"] = AXIS
    p = B / ("x1_difficulty_readout.json" if AXIS == "rating" else "x1_difficulty_readout_residual.json")
    json.dump(out, open(p, "w"))
    print("wrote", p)
    return out


def fig():
    sys.path.insert(0, str(Path(__file__).resolve().parent / "plots"))
    import figstyle as fs
    import matplotlib.pyplot as plt
    fs.use()
    d = build_json()
    src = sys.argv[2] if len(sys.argv) > 2 else ("strat" if d["strat"] else "first1000")
    tasks = [t for t, _, _ in TASKS if t in d[src]]
    bins = list(next(iter(d[src][next(iter(d[src]))]["quants"].values())).keys())
    bins = [b for b in bins if b not in ("fid_H_mean", "seed")]
    # per model: fp32, an in-basin quantizer, and the lossy ones that do not collapse
    pick = {"sudoku_mlp": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w3g32", fs.AMBER), ("noise0.15@s0", fs.RED)],
            "sudoku_attn": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w4t", fs.AMBER), ("noise0.7@s0", fs.RED)],
            "hrm_sudoku": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w2g32", fs.AMBER), ("noise1.0@s0", fs.RED)]}
    if src == "deep":
        pick = {"sudoku_mlp": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w3g32", fs.AMBER), ("noise0.15@s0", fs.RED)],
                "sudoku_attn": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w3c", fs.AMBER), ("noise0.7@s0", fs.RED)],
                "hrm_sudoku": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w2g32", fs.AMBER), ("noise1.0@s0", fs.RED)]}
    if src == "first1000":
        pick = {"sudoku_mlp": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w3g32", fs.AMBER), ("w3c", fs.RED)],
                "sudoku_attn": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w4t", fs.AMBER), ("w3c", fs.RED)],
                "hrm_sudoku": [("fp32", fs.INK), ("w4c", fs.BLUE), ("w3c", fs.AMBER), ("w2g32", fs.RED)]}
    fig_, axes = plt.subplots(len(tasks), len(bins), figsize=(fs.TEXT_W * 1.3, 1.25 * len(tasks) + 0.4),
                              sharex=True, sharey=True, squeeze=False)
    for i, t in enumerate(tasks):
        q = d[src][t]["quants"]
        for j, b in enumerate(bins):
            ax = axes[i][j]
            for s, col in pick[t]:
                if s not in q:
                    continue
                c = q[s][b]
                ax.plot(c["iters"], c["acc"], color=col, lw=1.1, label=s)
                ax.fill_between(c["iters"], c["lo"], c["hi"], color=col, alpha=0.15, lw=0)
            ax.set_xscale("log", base=2)
            if i == 0:
                ax.set_title((f"rating {b}" if AXIS == "rating" else b) + (f" (n={q['fp32'][b]['n']})" if 'fp32' in q else ""), fontsize=7)
            if j == 0:
                ax.set_ylabel(d[src][t]["label"] + "\nacc. %", fontsize=7)
            if i == len(tasks) - 1:
                ax.set_xlabel("ACT steps")
            if j == len(bins) - 1:
                ax.legend(fontsize=5.5, frameon=False, loc="lower right")
    fig_.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / (f"x1_difficulty_{src}" + ("" if AXIS == "rating" else "_residual"))
    fig_.savefig(str(out) + ".png", dpi=200)
    fig_.savefig(str(out) + ".pdf")
    print("wrote", out)


if __name__ == "__main__":
    {"first": first, "strat": strat, "strat_babel": lambda: strat("diff_babel"), "deep": lambda: strat("deep"), "glink": glink, "halt": halt, "gapdepth": gapdepth, "json": build_json,
     "fig": fig}[sys.argv[1]]()
