#!/usr/bin/env python3
"""M5 read-out: the halting head as an early exit for compressed recursive reasoners.

Every exit rule is evaluated offline on per-step records (the answer the model would emit if it
stopped after step k is the argmax at step k: TRM/HRM evaluation never resets the carry before
halt_max_steps, so a trajectory stopped at k equals the first k steps of a longer one).

Sources (all Babel L40S, first 1000 test rows, default H: TRM 3, HRM 2):
  det    X8 detector_sweep  det_<model>__*.json   per_step q_all / c_all, 16 steps
  resid  residual_sweep     results/breadth/babel/residual_<model>.json   q, exact, resid_H, 16 steps
  resc   X1 maze_rescore    results/breadth/babel/x1/maze_rescore_<model>_{a,b}.json  q, exact/valid/shortest, 16 steps
  m5     exit_sweep (M5)    <m5dir>/<model>/<spec>.npz   H=1 granularity, q_halt, q_continue, resid, msp,
                            prediction change, up to 512 default-H steps (Sudoku) / 16 (maze)
Priority when a (model, spec) exists in several sources: m5 > det > resid > resc (same trajectories;
the loader checks that fp32 agrees across sources).

Exit rules (t_i = exit step of instance i; if the rule never fires, t_i = K):
  fixed        all K steps (TRM/HRM test-time protocol)
  native       TRM: sigmoid(q_halt) > 0.5; HRM: q_halt > q_continue (the training-time ACT rule)
               (HRM without q_continue in the record: q_halt > 0, labelled native0)
  pat<p>       native for p consecutive steps
  min<m>+nat   native, but not before step m
  thr[b]       q_halt > tau, tau tuned on fp32 (loss budget b points vs fixed K) and frozen
  res[b]       convergence residual ||dz_H||/||z_H|| < eps, eps tuned on fp32, frozen
  msp[b]       min over answer cells of max-softmax > tau, tuned on fp32, frozen
  stab<p>      argmax answer unchanged for p consecutive steps (label-free, no tuning)
  nat|stab<p>  native OR stable (fallback when the head stays silent)
  nat&same     native AND the answer did not change since the previous step (consistency guard)
  oracle       first step at which the answer is correct (upper bound)
Tuning is two-fold cross-fitted on puzzle parity: tuned on the even fp32 half, applied to the odd
half of every cell, and vice versa. Baseline "fixed@s": fixed depth matched to the rule's mean
steps (linear interpolation between the neighbouring integer depths = a random mixture of them).

    python analyze_m5.py rules [--K 16] [--models ...]      # main table
    python analyze_m5.py bins                              # Sudoku difficulty bins (singles residual)
    python analyze_m5.py device                            # latency per solved instance
    python analyze_m5.py json                              # results/breadth/m5_summary.json
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RB = HERE / "results" / "breadth"
DET_DIR = Path(os.environ.get("M5_DET_DIR", RB / "babel" / "x8"))
M5_DIR = Path(os.environ.get("M5_DIR", RB / "babel" / "m5"))
MODELS = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze_attn", "hrm_maze"]
HDEF = {"sudoku_mlp": 3, "sudoku_attn": 3, "maze_attn": 3, "hrm_sudoku": 2, "hrm_maze": 2, "arc_attn": 3}
IS_HRM = {m: m.startswith("hrm") for m in MODELS + ["arc_attn"]}
BUDGETS = (0.5, 1.0)


# ----------------------------------------------------------------------------- loading
class Cell:
    def __init__(self, model, spec, src, c, q, qc=None, res=None, msp=None, chg=None, extra=None, h1=None):
        self.model, self.spec, self.src = model, spec, src
        self.c, self.q, self.qc, self.res, self.msp, self.chg = c, q, qc, res, msp, chg
        self.extra = extra or {}          # alternative correctness arrays (maze: valid, vshort)
        self.h1 = h1                      # dict of H=1-granularity arrays (m5 only)
        self.n, self.K = c.shape

    def trunc(self, K):
        f = lambda a: None if a is None else a[:, :K]
        return Cell(self.model, self.spec, self.src, f(self.c), f(self.q), f(self.qc), f(self.res), f(self.msp),
                    f(self.chg), {k: f(v) for k, v in self.extra.items()}, self.h1)

    def with_correct(self, key):
        cc = self.c if key == "exact" else self.extra[key]
        return Cell(self.model, self.spec, self.src, cc, self.q, self.qc, self.res, self.msp, self.chg, self.extra, self.h1)


def _unpack(hexes, K):
    return np.array([[(int(h, 16) >> j) & 1 for j in range(K)] for h in hexes], bool)


def load_det(model):
    out = {}
    for f in sorted(glob.glob(str(DET_DIR / f"det_{model}__*.json"))):
        for r in json.load(open(f))["records"]:
            if r["spec"] in out:
                continue
            ps = r["per_step"]
            out[r["spec"]] = Cell(model, r["spec"], "det", np.array(ps["c_all"], bool), np.array(ps["q_all"], np.float32))
    return out


def load_resid(model):
    f = RB / "babel" / f"residual_{model}.json"
    out = {}
    if not f.exists():
        return out
    for r in json.load(open(f))["records"]:
        ps = r["per_step"]
        K = r["nsup"]
        g = lambda key: np.stack([np.array(ps[str(k)][key]) for k in range(1, K + 1)], 1)
        out[r["quant"]["spec"]] = Cell(model, r["quant"]["spec"], "resid", g("correct").astype(bool),
                                       g("qhalt").astype(np.float32), res=g("resid_H").astype(np.float32))
    return out


def load_resc(model):
    out = {}
    for f in sorted(glob.glob(str(RB / "babel" / "x1" / f"maze_rescore_{model}_*.json"))):
        d = json.load(open(f))
        for r in d["records"]:
            if r["spec"] in out:
                continue
            K = r["nsup"]
            fl = {k: _unpack(v, K) for k, v in r["flags"].items()}
            out[r["spec"]] = Cell(model, r["spec"], "resc", fl["exact"], np.array(r["qhalt"], np.float32),
                                  extra={"valid": fl["simple"], "vshort": fl["shortest"]})
    return out


def load_m5(model, sub=None):
    out = {}
    files = sorted(glob.glob(str(M5_DIR / (sub or model) / "*.npz")))
    if sub is None:
        files += sorted(glob.glob(str(M5_DIR / f"{model}_deep" / "*.npz")))
    for f in files:
        if ".tmp" in f:
            continue
        d = np.load(f)
        info = json.loads(str(d["info"]))
        h = info["H_default"]
        h1 = {k: d[k] for k in d.files if k != "info"}
        sel = lambda a: a[:, h - 1::h]
        extra = {k: sel(h1[k]) for k in ("valid", "vshort") if k in h1}
        # default-H residual: the last H=1 sub-step's residual (the change made by the final H cycle)
        g = lambda k: sel(h1[k]).astype(np.float32) if k in h1 else None
        out[info["spec"]] = Cell(model, info["spec"], "m5", sel(h1["correct"]).astype(bool), g("qh"), qc=g("qc"), res=g("resH"),
                                 msp=g("msp"), chg=_chg_default(h1["chg"], h), extra=extra, h1=h1)
        if "qh32" in h1 and info["spec"] != "fp32" and not info.get("keep_head"):
            # the same trajectory read by the unquantized fp32 halting head (head is read-only)
            out[info["spec"] + "|h32"] = Cell(model, info["spec"] + "|h32", "m5", sel(h1["correct"]).astype(bool), g("qh32"),
                                              qc=g("qc32"), res=g("resH"), msp=g("msp"), chg=_chg_default(h1["chg"], h),
                                              extra=extra, h1={**h1, "qh": h1["qh32"], "qc": h1["qc32"]})
    return out


def _chg_default(chg1, h):
    """changes between consecutive default-H steps = sum of the H=1 changes inside the step is an upper
    bound; use 'any change inside the step' (0 means the answer was identical at the two boundaries or
    changed and changed back, which the sum cannot distinguish; we take the sum)."""
    n, K1 = chg1.shape
    return chg1.reshape(n, K1 // h, h).astype(np.int32).sum(2)


def load_all(model):
    """one Cell per spec; later sources replace earlier ones, inheriting signals the newer record
    lacks (maze validity flags, residuals) when the older record covers the same puzzles and steps."""
    cells = {}
    for loader in (load_resc, load_resid, load_det, load_m5):
        for s, c in loader(model).items():
            old = cells.get(s)
            if old is not None and old.n == c.n and old.K >= c.K:
                if not c.extra and old.extra:
                    c.extra = {k: v[:, :c.K] for k, v in old.extra.items()}
                if c.res is None and old.res is not None:
                    c.res = old.res[:, :c.K]
            elif old is not None and old.n != c.n:
                continue                                  # keep the larger-n record
            cells[s] = c
    return cells


def check_agreement(model):
    """fp32 per-step exact correctness across sources (same platform, same trajectories)."""
    srcs = {name: fn(model).get("fp32") for name, fn in (("resc", load_resc), ("resid", load_resid), ("det", load_det), ("m5", load_m5))}
    srcs = {k: v for k, v in srcs.items() if v is not None}
    base = None
    rows = []
    for k, v in srcs.items():
        if base is None:
            base = (k, v)
            continue
        K, n = min(base[1].K, v.K), min(base[1].n, v.n)
        mism = int((base[1].c[:n, :K] != v.c[:n, :K]).sum())
        dq = float(np.abs(base[1].q[:n, :K] - v.q[:n, :K]).max())
        rows.append(f"{model} fp32 {base[0]} vs {k} (first {n}): per-step exact mismatches {mism} of {n * K}; max |dq| {dq:.3f}")
    return rows


# ----------------------------------------------------------------------------- rules
def first_true(mask):
    """index of the first True per row, or K-1 if none."""
    K = mask.shape[1]
    any_ = mask.any(1)
    return np.where(any_, mask.argmax(1), K - 1)


def native_mask(cell):
    if IS_HRM[cell.model] and cell.qc is not None:
        return cell.q > cell.qc
    return cell.q > 0


def run_len(mask):
    """number of consecutive True ending at each step."""
    out = np.zeros(mask.shape, np.int32)
    run = np.zeros(mask.shape[0], np.int32)
    for k in range(mask.shape[1]):
        run = np.where(mask[:, k], run + 1, 0)
        out[:, k] = run
    return out


def stable_mask(cell, p):
    if cell.chg is None:
        return None
    m = cell.chg == 0
    m[:, 0] = False
    return run_len(m) >= p


def exit_steps(cell, rule, param=None):
    K = cell.K
    if rule == "fixed":
        return np.full(cell.n, K - 1)
    if rule == "native":
        return first_true(native_mask(cell))
    if rule == "pat":
        return first_true(run_len(native_mask(cell)) >= param)
    if rule == "min":
        m = native_mask(cell).copy()
        m[:, :param - 1] = False
        return first_true(m)
    if rule == "thr":
        return first_true(cell.q > param)
    if rule == "res":
        if cell.res is None:
            return None
        m = cell.res < param
        m[:, 0] = False
        return first_true(m)
    if rule == "msp":
        if cell.msp is None:
            return None
        return first_true(cell.msp > param)
    if rule in ("stab", "stabt"):
        m = stable_mask(cell, param)
        return None if m is None else first_true(m)
    if rule == "natcons":                                 # native AND answer unchanged since the previous step
        m = stable_mask(cell, 1)
        return None if m is None else first_true(native_mask(cell) & m)
    if rule == "natstab":
        m = stable_mask(cell, param)
        return None if m is None else first_true(native_mask(cell) | m)
    if rule == "oracle":
        return first_true(cell.c)
    if rule == "depth":                                   # fixed depth d (1-based)
        return np.full(cell.n, param - 1)
    raise ValueError(rule)


def score(cell, t, idx=None):
    c = cell.c
    if idx is not None:
        c, t = c[idx], t[idx]
    ok = c[np.arange(len(t)), t]
    return ok, t + 1


GRIDS = {"thr": np.round(np.arange(-4, 10.01, 0.25), 2),
         "res": np.array([10 ** e for e in np.arange(-4, 0.01, 0.1)]),
         "stabt": np.array([1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80]),
         "msp": np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.99975])}   # msp stored in fp16: 0.99975 means == 1.0


def tune(ref, rule, budget, idx):
    """smallest mean steps on ref[idx] with accuracy >= fixed-K accuracy - budget (points)."""
    target = ref.c[idx, -1].mean() * 100 - budget
    best = None
    for p in GRIDS[rule]:
        t = exit_steps(ref, rule, p)
        if t is None:
            return None
        ok, st = score(ref, t, idx)
        if ok.mean() * 100 >= target - 1e-9 and (best is None or st.mean() < best[1]):
            best = (p, st.mean())
    return None if best is None else best[0]


def crossfit(ref, cell, rule, budget):
    """two-fold cross-fitted tuned rule: returns per-instance exit steps on cell and the two params."""
    n = cell.n
    halves = [np.arange(0, n, 2), np.arange(1, n, 2)]
    t = np.full(n, cell.K - 1)
    params = []
    for a, b in ((0, 1), (1, 0)):
        p = tune(ref, rule, budget, halves[a])
        if p is None:
            return None, None
        params.append(float(p))
        tb = exit_steps(cell, rule, p)
        if tb is None:
            return None, None
        t[halves[b]] = tb[halves[b]]
    return t, params


def fixed_curve(cell):
    return cell.c.mean(0)


def matched_fixed(cell, s):
    """accuracy of a fixed depth equal to mean steps s (linear mix of floor/ceil depths)."""
    acc = fixed_curve(cell)
    lo = int(np.floor(s))
    hi = min(lo + 1, cell.K)
    lo = max(lo, 1)
    w = s - np.floor(s)
    return (1 - w) * acc[lo - 1] + w * acc[hi - 1]


def paired_ci(ok, ref_ok):
    d = ok.astype(float) - ref_ok.astype(float)
    return 100 * d.mean(), 100 * 1.96 * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0


RULESET = [("native", None), ("pat", 2), ("pat", 3), ("min", 2), ("min", 4),
           ("thr", 0.5), ("thr", 1.0), ("res", 0.5), ("res", 1.0), ("msp", 0.5), ("msp", 1.0),
           ("stab", 1), ("stab", 2), ("natstab", 2), ("natcons", None), ("oracle", None)]


def rule_name(rule, p):
    return {"native": "native", "pat": f"pat{p}", "min": f"min{p}+nat", "thr": f"thr[{p}]", "res": f"res[{p}]",
            "msp": f"msp[{p}]", "stab": f"stab{p}", "natstab": f"nat|stab{p}", "natcons": "nat&same", "oracle": "oracle", "stabt": f"stab*[{p}]", "depth": "depth"}[rule]


SEL_CANDIDATES = [("native", None), ("pat", 2), ("pat", 3), ("min", 2), ("min", 4)]   # need only q (all sources)


def select_rule(ref, budget=0.5):
    """recalibration-free rule chosen on fp32 only: the candidate with the fewest mean steps whose
    fp32 accuracy is within `budget` points of fixed K. Applied unchanged to every compressed cell."""
    best = None
    acc_k = ref.c[:, -1].mean() * 100
    for rule, p in SEL_CANDIDATES:
        ok, st = score(ref, exit_steps(ref, rule, p))
        if ok.mean() * 100 >= acc_k - budget - 1e-9 and (best is None or st.mean() < best[2]):
            best = (rule, p, st.mean())
    return best[:2]


def fp32_depth(ref, budget=0.5):
    """fixed-depth baseline chosen on fp32 with the same budget as the selected exit rule: the
    smallest depth whose fp32 accuracy is within `budget` points of fixed K."""
    acc = ref.c.mean(0) * 100
    return int(np.nonzero(acc >= acc[-1] - budget - 1e-9)[0][0]) + 1


def basin_group(cell, ref):
    r = cell.c[:, -1].mean() / max(ref.c[:, -1].mean(), 1e-9)
    return "in" if r >= 0.9 else ("edge" if r >= 0.5 else "out")


def evaluate(cell, ref, rules=RULESET):
    """one row per rule: acc, delta vs fixed K [95% CI], mean steps, matched fixed-depth acc."""
    full_ok = cell.c[:, -1]
    out = {"model": cell.model, "spec": cell.spec, "src": cell.src, "K": cell.K, "n": cell.n,
           "acc_fixed": 100 * full_ok.mean(), "rules": {}}
    if ref is not None:
        out["group"] = basin_group(cell, ref)
        sel = select_rule(ref)
        out["sel_rule"] = rule_name(*sel)
        out["fp32_depth"] = fp32_depth(ref)
        rules = list(rules) + [("sel", sel), ("depth", out["fp32_depth"])]
    for rule, p in rules:
        if rule in ("thr", "res", "msp", "stabt"):
            if ref is None:
                continue
            t, params = crossfit(ref, cell, rule, p)
        elif rule == "sel":
            t, params = exit_steps(cell, *p), None
        else:
            t, params = exit_steps(cell, rule, p), None
        if t is None:
            continue
        ok, st = score(cell, t)
        d, ci = paired_ci(ok, full_ok)
        s = st.mean()
        out["rules"]["sel" if rule == "sel" else ("depth" if rule == "depth" else rule_name(rule, p))] = {"acc": 100 * ok.mean(), "d": d, "ci": ci, "steps": s, "speedup": cell.K / s,
                                            "matched": 100 * matched_fixed(cell, s), "params": params,
                                            "at_K": float((t == cell.K - 1).mean()),
                                            "lost": int((full_ok & ~ok).sum()), "rescued": int((~full_ok & ok).sum())}
    return out


# ----------------------------------------------------------------------------- read-outs
def spec_order(s):
    fam = 0 if s == "fp32" else (1 if s.startswith("w") else (2 if s.startswith("noise") else 3))
    return (fam, s)


def cells_for(model, K=None, corr="exact"):
    cells = load_all(model)
    if K is not None:
        cells = {s: (c.trunc(K) if c.K >= K else None) for s, c in cells.items()}
        cells = {s: c for s, c in cells.items() if c is not None}
    if corr != "exact":
        cells = {s: c.with_correct(corr) for s, c in cells.items() if corr in c.extra}
    return cells


def cmd_rules(a):
    for model in a.models:
        for row in check_agreement(model):
            print(row)
        for corr in (["exact", "vshort", "valid"] if "maze" in model else ["exact"]):
            cells = cells_for(model, a.K, corr)
            if "fp32" not in cells:
                continue
            ref = cells["fp32"]
            print(f"\n=== {model}  K={a.K}  correctness={corr}  (acc %, delta vs fixed-K [95% CI], mean steps; matched fixed depth)")
            names = [rule_name(r, p) for r, p in RULESET]
            show = [nm for nm in names if nm in a.show.split(",")] if a.show else ["native", "pat2", "thr[0.5]", "res[0.5]", "stab2", "nat|stab2", "oracle"]
            print(f"{'spec':14s} {'src':5s} {'fixed':>6s} | " + " | ".join(f"{nm:>24s}" for nm in show))
            for s in sorted(cells, key=spec_order):
                r = evaluate(cells[s], ref, [x for x in RULESET if rule_name(*x) in show])
                parts = []
                for nm in show:
                    v = r["rules"].get(nm)
                    parts.append(f"{v['acc']:5.1f} {v['d']:+5.1f}±{v['ci']:3.1f} {v['steps']:5.2f} ({v['matched']:4.1f})" if v else f"{'-':>24s}")
                print(f"{s:14s} {cells[s].src:5s} {r['acc_fixed']:6.1f} | " + " | ".join(parts))


def base_spec(s):
    b = s.split("@s")[0]
    return b + ("|h32" if s.endswith("|h32") and not b.endswith("|h32") else "")


def summarize(model, K=16, corr="exact", rules=("native", "sel", "pat2", "min4+nat", "thr[0.5]", "res[0.5]", "stab2", "nat|stab2", "oracle")):
    """evaluate every cell; merge noise draws (@s0..2 and the unseeded draw-0 records) per base spec."""
    cells = cells_for(model, K, corr)
    if "fp32" not in cells:
        return None
    ref = cells["fp32"]
    wanted = [x for x in RULESET if rule_name(*x) in rules]
    rows = {s: evaluate(c, ref, wanted) for s, c in cells.items()}
    return {"model": model, "K": K, "corr": corr, "sel_rule": rows["fp32"]["sel_rule"], "rows": rows}


def group_rows(summ):
    """per base spec: mean (and range) over draws of acc_fixed, and per rule acc / d / steps."""
    groups = {}
    for s, r in summ["rows"].items():
        b = base_spec(s)
        if b != s and b in summ["rows"] and "@" not in b:
            pass
        groups.setdefault(b, []).append(r)
    out = {}
    for b, rs in groups.items():
        # prefer seeded draws when both seeded and unseeded (residual-file draw 0) records exist
        seeded = [r for r in rs if "@s" in r["spec"]]
        rs = seeded or rs
        g = {"spec": b, "draws": len(rs), "group": rs[0].get("group"), "n": rs[0]["n"],
             "acc_fixed": float(np.mean([r["acc_fixed"] for r in rs])),
             "acc_fixed_range": [float(min(r["acc_fixed"] for r in rs)), float(max(r["acc_fixed"] for r in rs))], "rules": {}}
        for nm in rs[0]["rules"]:
            vals = [r["rules"][nm] for r in rs if nm in r["rules"]]
            g["rules"][nm] = {k: float(np.mean([v[k] for v in vals])) for k in ("acc", "d", "steps", "speedup", "matched")}
            g["rules"][nm]["d_range"] = [float(min(v["d"] for v in vals)), float(max(v["d"] for v in vals))]
            g["rules"][nm]["ci"] = float(np.mean([v["ci"] for v in vals]))
        if len({r.get("group") for r in rs}) > 1:
            g["group"] = "/".join(sorted({r.get("group") for r in rs}))
        out[b] = g
    return out


def cmd_summary(a):
    for model in a.models:
        for corr in (["exact", "vshort"] if "maze" in model else ["exact"]):
            summ = summarize(model, a.K, corr)
            if summ is None:
                continue
            G = group_rows(summ)
            show = ["native", "sel", "depth", "res[0.5]", "oracle"]
            print(f"\n=== {model} K={a.K} {corr}; rule chosen on fp32 = {summ['sel_rule']}, fixed depth chosen on fp32 = {summ['rows']['fp32']['fp32_depth']}. "
                  f"Per rule: acc (delta vs fixed [range over draws]) mean steps -> speedup; [matched fixed depth]")
            print(f"{'spec':10s} {'grp':5s} {'dr':>2s} {'fixed':>5s} | " + " | ".join(f"{nm:>30s}" for nm in show))
            for b in sorted(G, key=lambda b: ({"in": 0, "edge": 1, "out": 2}.get(G[b]["group"], 1), spec_order(b))):
                g = G[b]
                parts = []
                for nm in show:
                    v = g["rules"].get(nm)
                    if v is None:
                        parts.append(f"{'-':>30s}")
                        continue
                    rng = f"[{v['d_range'][0]:+.1f},{v['d_range'][1]:+.1f}]" if g["draws"] > 1 else f"±{v['ci']:.1f}"
                    parts.append(f"{v['acc']:5.1f} ({v['d']:+5.1f}{rng:>12s}) {v['steps']:5.2f}->{v['speedup']:4.1f}x [{v['matched']:4.1f}]")
                print(f"{b:10s} {g['group']:5s} {g['draws']:2d} {g['acc_fixed']:5.1f} | " + " | ".join(parts))


RES_BINS = [(0, 0, "res 0"), (1, 45, "1-45"), (46, 50, "46-50"), (51, 55, "51-55"), (56, 999, ">=56")]


def sudoku_bins():
    meta = json.load(open(RB / "sudoku_extreme_test_meta.json"))["first1000"]
    r = np.array(meta["singles_residual"])
    return [(lab, np.nonzero((r >= lo) & (r <= hi))[0]) for lo, hi, lab in RES_BINS]


def cmd_bins(a):
    """per difficulty bin (blanks left after naked + hidden singles, X1's fixed bins; first 1000 rows):
    fixed-K accuracy, the fp32-selected rule's accuracy and mean steps, native's mean steps."""
    bins = sudoku_bins()
    out = {}
    for model in [m for m in a.models if "sudoku" in m]:
        cells = cells_for(model, a.K)
        if "fp32" not in cells:
            continue
        ref = cells["fp32"]
        sel = select_rule(ref)
        print(f"\n=== {model} K={a.K}: per residual bin (n = {', '.join(str(len(ix)) for _, ix in bins)}); "
              f"fixed acc / {rule_name(*sel)} acc, mean steps / native mean steps / oracle steps")
        for spec in sorted(cells, key=spec_order):
            if spec not in a.specs.split(",") and a.specs != "all":
                continue
            c = cells[spec]
            ts, tn, to = exit_steps(c, *sel), exit_steps(c, "native"), exit_steps(c, "oracle")
            row = []
            for lab, ix in bins:
                ok_s, st_s = score(c, ts, ix)
                row.append({"bin": lab, "n": int(len(ix)), "fixed": 100 * c.c[ix, -1].mean(), "sel_acc": 100 * ok_s.mean(),
                            "sel_steps": float(st_s.mean()), "nat_steps": float((tn[ix] + 1).mean()),
                            "nat_acc": 100 * score(c, tn, ix)[0].mean(), "oracle_steps": float((to[ix] + 1).mean())})
            out[(model, spec)] = row
            print(f"{spec:14s} " + " | ".join(f"{r['fixed']:5.1f}/{r['sel_acc']:5.1f} {r['sel_steps']:5.2f} {r['nat_steps']:5.2f}" for r in row))
    return out


DEV_TASK = {"sudoku_mlp": "sudoku_mlp", "sudoku_attn": "sudoku_attn", "maze_attn": "maze"}
# device precision -> harness cell whose (accuracy, steps) the device row borrows
# Only the FP16 (fp32 export) graph is paired with harness fp32 accuracy across devices: the S24 fp32 chain
# equals the host fp32 per-step correctness on every puzzle (cmd_s24). INT8 rows use on-device accuracy only
# (cmd_s24); every INT4 graph computes the wrong function on the HTP (X3, REWRITE_NOTES 47-49): no INT4 rows.
DEV_PREC = {"fp32": "fp32"}
DEV_SHOW = ["Samsung Galaxy S24", "Samsung Galaxy S26", "Snapdragon 8 Elite QRD", "Samsung Galaxy S21",
            "Dragonwing RB3 Gen 2 Vision Kit", "SA8295P ADP", "Snapdragon X Elite CRD"]
# Arm Ethos-U INT8 step, Vela 5.2 at 500 MHz (results/devices/mcu_h1.md headline table), ms per step
MCU = {("sudoku_mlp", "U55-256", 1): 103.99, ("sudoku_mlp", "U55-256", 3): 311.66,
       ("sudoku_mlp", "U85-256", 1): 81.40, ("sudoku_mlp", "U85-256", 3): 243.87,
       ("sudoku_mlp", "U55-256 FVP", 1): 110.77, ("sudoku_mlp", "U55-256 FVP", 3): 332.56,
       ("sudoku_attn", "U55-256", 1): 325.08, ("sudoku_attn", "U55-256", 3): 974.91,
       ("sudoku_attn", "U85-256", 1): 110.23, ("sudoku_attn", "U85-256", 3): 330.54,
       ("maze", "U55-256", 1): 11793.86, ("maze", "U55-256", 3): 35378.44,
       ("maze", "U85-256", 1): 761.67, ("maze", "U85-256", 3): 2283.42}


def aihub_h3():
    """Sep-10 AI Hub profiles of the H=3 supervision-step graph (one step = the default-H step of the
    TRM checkpoints), NPU runtimes only: {(task, precision, device): ms}."""
    rows = json.load(open(RB / "aihub_status.json"))
    out = {}
    for r in rows:
        if r["kind"] == "profile" and r["status"] == "SUCCESS" and r["runtime"].startswith("qnn") and r.get("ms"):
            out[(r["task"], r["precision"], r["device"])] = r["ms"]
    return out


def cmd_device(a):
    """latency per answer and per solved instance = step latency x mean steps (/ accuracy), fixed 16
    vs the fp32-selected exit rule and native, TRM tasks with a device graph (HRM has none)."""
    lat = aihub_h3()
    res = {}
    for model, dtask in DEV_TASK.items():
        cells = cells_for(model, 16)
        if "fp32" not in cells:
            continue
        ref = cells["fp32"]
        sel = select_rule(ref)
        stats = {}
        for prec, spec in DEV_PREC.items():
            c = cells.get(spec)
            if c is None:                                     # no +a8 record: fall back to weight-only
                spec = spec.replace("+a8", "")
                c = cells.get(spec)
            if c is None:
                continue
            d = {}
            for nm, rule in (("fixed", ("fixed", None)), ("native", ("native", None)), ("sel", sel)):
                ok, st = score(c, exit_steps(c, *rule))
                d[nm] = (ok.mean(), st.mean())
            stats[prec] = (spec, d)
        print(f"\n=== {model} (device graph task '{dtask}', H=3 step; exit rule chosen on fp32 = {rule_name(*sel)})")
        for prec, (spec, d) in stats.items():
            fa, fs = d["fixed"]; sa, ss = d["sel"]; na, ns = d["native"]
            print(f"  {prec} graph <- harness {spec}: acc fixed {100*fa:.1f} / sel {100*sa:.1f} / native {100*na:.1f}; "
                  f"steps 16 / {ss:.2f} / {ns:.2f}; per-solved speedup sel {(16/fa)/(ss/sa):.2f}x native {(16/fa)/(ns/na):.2f}x")
        print(f"  {'device':32s} {'prec':5s} {'ms/step':>8s} {'ms/ans 16':>10s} {'ms/ans sel':>10s} {'ms/solved 16':>12s} {'ms/solved sel':>13s} {'ms/solved nat':>13s}")
        for dev in DEV_SHOW + ["MCU"]:
            for prec, (spec, d) in stats.items():
                if dev == "MCU":             # INT8 step, latency only (output correctness unchecked, X3 item 33)
                    items = [(f"Ethos-{k[1]} INT8 (Vela est.; latency only)" if "FVP" not in k[1] else f"Ethos-{k[1]} INT8 (latency only)", v)
                             for k, v in MCU.items() if k[0] == dtask and k[2] == 3]
                else:
                    ms = lat.get((dtask, prec, dev))
                    items = [(dev, ms)] if ms else []
                for name, ms in items:
                    fa, fs = d["fixed"]; sa, ss = d["sel"]; na, ns = d["native"]
                    row = {"ms_step": ms, "ms_ans_fixed": ms * 16, "ms_ans_sel": ms * ss, "ms_solved_fixed": ms * 16 / max(fa, 1e-9),
                           "ms_solved_sel": ms * ss / max(sa, 1e-9), "ms_solved_native": ms * ns / max(na, 1e-9),
                           "acc_fixed": 100 * fa, "acc_sel": 100 * sa, "steps_sel": ss, "harness_cell": spec}
                    res[(model, name, prec)] = row
                    print(f"  {name[:32]:32s} {prec:5s} {ms:8.1f} {row['ms_ans_fixed']:10.0f} {row['ms_ans_sel']:10.0f} "
                          f"{row['ms_solved_fixed']:12.0f} {row['ms_solved_sel']:13.0f} {row['ms_solved_native']:13.0f}")
    # H=1 graphs (the deployable carry-in/carry-out step, X3): exit read after every H cycle
    h1prof = json.load(open(RB / "m5_aihub_h1_profiles.json")) if (RB / "m5_aihub_h1_profiles.json").exists() else []
    print("\n=== H=1 step graph, fp32 model, exit at H-cycle granularity (M5 records)")
    for model, dtask in DEV_TASK.items():
        cells = {s: c for s, c in load_m5(model).items() if c.h1 is not None}
        if "fp32" not in cells:
            continue
        h = HDEF[model]
        c1 = h1_cell(cells["fp32"]).trunc(16 * h)
        sel1 = select_rule(c1)
        ok, st = score(c1, exit_steps(c1, *sel1))
        fa = c1.c[:, -1].mean()
        lat = [(f"{r['device']} ({r['prec']}, AI Hub)", r["ms"]) for r in h1prof if r["task"] == dtask and r["prec"] == "fp32"]
        lat += [(f"Ethos-{k[1]}" + (" INT8 (Vela est., latency only)" if "FVP" not in k[1] else " INT8 (latency only)"), v)
                for k, v in MCU.items() if k[0] == dtask and k[2] == 1]
        print(f"  {model}: fixed {16 * h} H=1 steps acc {100 * fa:.1f}; {rule_name(*sel1)} at H=1: acc {100 * ok.mean():.1f}, {st.mean():.1f} H=1 steps")
        for name, ms in lat:
            row = {"ms_step_h1": ms, "ms_solved_fixed": ms * 16 * h / fa, "ms_solved_exit_h1": ms * st.mean() / ok.mean()}
            res[(model, name, "H1")] = row
            print(f"    {name:40s} {ms:8.2f} ms/H=1 step: ms/solved fixed {row['ms_solved_fixed']:8.0f} -> exit {row['ms_solved_exit_h1']:8.0f}")
    return res


def h1_cell(cell):
    """the same trajectory at H=1 ("inner step") granularity: one step = one H cycle."""
    h1 = cell.h1
    f = lambda k: h1[k].astype(np.float32) if k in h1 else None
    extra = {k: h1[k] for k in ("valid", "vshort") if k in h1}
    return Cell(cell.model, cell.spec + "@H1", "m5", h1["correct"].astype(bool), f("qh"), qc=f("qc"), res=f("resH"),
                msp=f("msp"), chg=h1["chg"].astype(np.int32), extra=extra)


def cmd_inner(a):
    """exit at H=1 granularity (the q head is read after every H cycle) vs at the default-H step,
    both in default-H step units, on the m5 records (K default-H steps)."""
    for model in a.models:
        cells = {s: c for s, c in load_m5(model).items() if c.h1 is not None}
        if "fp32" not in cells:
            continue
        h = HDEF[model]
        K = min(a.K, cells["fp32"].K)
        ref = cells["fp32"].trunc(K)
        ref1 = h1_cell(cells["fp32"]).trunc(K * h)
        sel, sel1 = select_rule(ref), select_rule(ref1)
        print(f"\n=== {model} K={K} default-H steps (H={h}); rule chosen on fp32: default-H {rule_name(*sel)}, H=1 {rule_name(*sel1)}")
        print(f"{'spec':18s} {'fixed':>5s} | {'default-H: native':>22s} {'sel':>18s} | {'H=1: native':>22s} {'sel':>18s}")
        for spec in sorted(cells, key=spec_order):
            c = cells[spec].trunc(K)
            c1 = h1_cell(cells[spec]).trunc(K * h)
            parts = []
            for cc, unit, sl in ((c, 1, sel), (c1, h, sel1)):
                for rule in (("native", None), sl):
                    ok, st = score(cc, exit_steps(cc, *rule))
                    parts.append(f"{100*ok.mean():5.1f} {(100*ok.mean()-100*c.c[:, -1].mean()):+5.1f} {st.mean()/unit:5.2f}")
            print(f"{spec:18s} {100*c.c[:, -1].mean():5.1f} | {parts[0]:>22s} {parts[1]:>18s} | {parts[2]:>22s} {parts[3]:>18s}")


BUDGETS_DEEP = (8, 16, 32, 64, 128, 256, 512)


def deep_frontier(cell, rule, budgets=BUDGETS_DEEP):
    """accuracy vs mean steps for (a) a fixed budget B and (b) the exit rule capped at B."""
    t = exit_steps(cell, *rule)
    rows = []
    for B in budgets:
        if B > cell.K:
            continue
        tb = np.minimum(t, B - 1)
        ok = cell.c[np.arange(cell.n), tb]
        rows.append({"B": B, "fixed_acc": 100 * cell.c[:, B - 1].mean(), "exit_acc": 100 * ok.mean(), "exit_steps": float((tb + 1).mean()),
                     "fixed_steps_per_solved": B / max(cell.c[:, B - 1].mean(), 1e-9),
                     "exit_steps_per_solved": float((tb + 1).mean()) / max(ok.mean(), 1e-9)})
    return rows


def cmd_deep(a):
    """test-time scaling to 64 / 512 default-H steps (Sudoku): the exit rule (chosen on fp32 at K=16)
    capped at a budget B vs the fixed budget B; steps per solved instance."""
    out = {}
    for model in [m for m in a.models if "sudoku" in m]:
        cells = {s: c for s, c in load_all(model).items() if c.K >= 64}
        if "fp32" not in cells:
            continue
        sel = select_rule(cells["fp32"].trunc(16))
        print(f"\n=== {model}: rule chosen on fp32 at 16 steps = {rule_name(*sel)}; per budget B: fixed acc | exit acc, mean steps | steps per solved fixed -> exit")
        for spec in sorted(cells, key=spec_order):
            fr = deep_frontier(cells[spec], sel)
            out[(model, spec)] = fr
            print(f"{spec:16s} " + " | ".join(f"B{r['B']}: {r['fixed_acc']:4.1f}|{r['exit_acc']:4.1f} {r['exit_steps']:5.1f} ({r['fixed_steps_per_solved']:5.0f}->{r['exit_steps_per_solved']:4.0f})" for r in fr if r["B"] >= 16))
    return out


M2_DIR = Path(os.environ.get("M5_M2_DIR", RB / "babel" / "m2" / "cells"))


def seq_sample(corr, q, rule="pat", p=2, Kmax=None):
    """M5 x M2, sample until the head is confident: rollouts k = 1, 2, ... (M2's PTRM noise, each D
    steps), each stopped by the exit rule; the first rollout whose rule fires gives the answer; if none
    fires within Kmax rollouts, answer = the tried rollout with the highest final q. corr, q: [K, n, D].
    Returns per-puzzle correctness, steps spent, rollouts used."""
    K, n, D = corr.shape
    Kmax = Kmax or K
    qf = q.astype(np.float32)
    fired = qf > 0
    if rule == "pat":
        rl = np.zeros_like(fired, dtype=np.int32)
        run = np.zeros((K, n), np.int32)
        for t in range(D):
            run = np.where(fired[:, :, t], run + 1, 0)
            rl[:, :, t] = run
        fired = rl >= p
    any_f = fired.any(2)                                   # [K, n]
    t_f = np.where(any_f, fired.argmax(2), D - 1)          # exit step per rollout
    ok = np.zeros(n, bool); steps = np.zeros(n); used = np.zeros(n, int)
    for i in range(n):
        for k in range(Kmax):
            steps[i] += t_f[k, i] + 1
            if any_f[k, i]:
                ok[i] = corr[k, i, t_f[k, i]]; used[i] = k + 1
                break
        else:
            kb = int(np.argmax(qf[:Kmax, i, D - 1]))
            ok[i] = corr[kb, i, D - 1]; used[i] = Kmax
    return ok, steps, used


def cmd_sample(a):
    for f in sorted(glob.glob(str(M2_DIR / "*.npz"))):
        name = Path(f).stem
        d = np.load(f)
        corr, q = d["corr"].astype(bool), d["q"]
        K, n, D = corr.shape
        if K < 4:
            continue
        single = corr[:, :, -1].mean() * 100
        bestq = corr[np.argmax(q[:, :, -1].astype(np.float32), 0), np.arange(n), -1].mean() * 100
        passk = corr[:, :, -1].any(0).mean() * 100
        print(f"\n{name}: K={K} n={n} D={D}; single rollout {single:.1f}, best-Q@{K} {bestq:.1f} (cost {K * D} steps), pass@{K} {passk:.1f}")
        for rule, p in (("nat", None), ("pat", 2)):
            for Km in sorted({1, 2, 4, 8, K}):
                if Km > K:
                    continue
                ok, st, used = seq_sample(corr, q, rule, p or 1, Km)
                bq = corr[np.argmax(q[:Km, :, -1].astype(np.float32), 0), np.arange(n), -1].mean() * 100
                print(f"  seq {rule}{p or ''} Kmax={Km:2d}: acc {100 * ok.mean():5.1f}  mean steps {st.mean():6.1f}  rollouts {used.mean():4.2f}"
                      f"   | best-Q@{Km} {bq:5.1f} at {Km * D} steps")


DT_DIR = Path(os.environ.get("M5_DT_DIR", RB / "babel" / "m5"))
DT_RULES = [("res", 0.5), ("res", 1.0), ("msp", 0.5), ("stabt", 0.5), ("stabt", 1.0), ("stab", 10), ("stab", 30), ("oracle", None)]


def load_dt(size):
    out = {}
    for f in sorted(glob.glob(str(DT_DIR / f"dt_maze{size}" / "*.npz"))):
        if ".tmp" in f:
            continue
        d = np.load(f)
        info = json.loads(str(d["info"]))
        c = d["correct"].astype(bool)
        out[info["spec"]] = Cell(f"dt_maze{size}", info["spec"], "m5", c, np.zeros(c.shape, np.float32),
                                 res=d["res"].astype(np.float32), msp=d["conf"].astype(np.float32), chg=d["chg"].astype(np.int32))
    return out


def cmd_dt(a):
    """deep-thinking mazes (no halting head): label-free exits tuned on fp32, frozen, vs the fixed budget."""
    res = {}
    for size in (13, 25, 33):
        cells = load_dt(size)
        if "fp32" not in cells:
            continue
        ref = cells["fp32"]
        d0 = fp32_depth(ref)
        print(f"\n=== DT maze {size}x{size}, T={ref.K}, n={ref.n}; fixed depth chosen on fp32 (0.5 pt) = {d0}")
        print(f"{'spec':14s} {'fixed':>5s} {'@d0':>5s} | " + " | ".join(f"{rule_name(r, p):>22s}" for r, p in DT_RULES))
        for spec in sorted(cells, key=spec_order):
            c = cells[spec]
            r = evaluate(c, None, [])
            parts = []
            for rule, p in DT_RULES:
                if rule in ("res", "msp", "stabt"):
                    t, prm = crossfit(ref, c, rule, p)
                else:
                    t, prm = exit_steps(c, rule, p), None
                ok, st = score(c, t)
                parts.append(f"{100*ok.mean():5.1f} {100*ok.mean()-r['acc_fixed']:+5.1f} {st.mean():6.1f}")
                res[(size, spec, rule_name(rule, p))] = {"acc": 100 * ok.mean(), "steps": float(st.mean()), "params": prm}
            print(f"{spec:14s} {r['acc_fixed']:5.1f} {100*c.c[:, d0-1].mean():5.1f} | " + " | ".join(f"{x:>22s}" for x in parts))
    return res


POL_DIR = Path(os.environ.get("M5_POL_DIR", RB / "babel" / "m5" / "polish"))


def load_polish(model):
    out = {}
    for f in sorted(glob.glob(str(POL_DIR / model / "*.npz"))):
        if ".tmp" in f:
            continue
        d = np.load(f)
        info = json.loads(str(d["info"]))
        out[(info["cheap"], info["polish"])] = {k: d[k] for k in d.files if k != "info"} | {"info": info}
    return out


def polish_rows(model, d, sel):
    """exit-then-polish variants for one (cheap, polish) record; returns list of (name, acc, cheap steps, polish steps)."""
    c = d["c_cheap"].astype(bool); n, K = c.shape
    cp = d["c_pol"].astype(bool); qp = d["q_pol"].astype(np.float32); J = cp.shape[2]
    qc_ = d["q_cheap"].astype(np.float32)
    cell = Cell(model, d["info"]["cheap"], "pol", c, qc_, qc=d["qc_cheap"].astype(np.float32) if IS_HRM[model] else None)
    rows = [("cheap fixed", 100 * c[:, -1].mean(), K, 0)]
    idx = np.arange(n)
    for nm, rule in (("native", ("native", None)), ("sel", sel)):
        t = exit_steps(cell, *rule)
        rows.append((f"cheap {nm} exit", 100 * c[idx, t].mean(), (t + 1).mean(), 0))
        for j in (1, 2, 4):
            if j <= J:
                rows.append((f"{nm} exit + P{j}", 100 * cp[idx, t, j - 1].mean(), (t + 1).mean(), j))
        # polish until the polish head halts (q > 0), at most J polish steps
        f = qp[idx, t] > 0
        jj = np.where(f.any(1), f.argmax(1), J - 1)
        rows.append((f"{nm} exit + P(head)", 100 * cp[idx, t, jj].mean(), (t + 1).mean(), (jj + 1).mean()))
    for j in (1, 2, 4):
        if j <= J:
            rows.append((f"M1 append c{K}.P{j}", 100 * cp[:, -1, j - 1].mean(), K, j))
            rows.append((f"M1 replace c{K - j}.P{j}", 100 * cp[:, K - j - 1, j - 1].mean(), K - j, j))
    return rows


def cmd_polish(a):
    out = {}
    for model in a.models:
        recs = load_polish(model)
        if not recs:
            continue
        ref = cells_for(model, 16).get("fp32")
        sel = select_rule(ref) if ref is not None else ("native", None)
        print(f"\n=== {model}: exit (compressed head) then polish; fp32 fixed-16 acc {100 * ref.c[:, -1].mean():.1f}; sel = {rule_name(*sel)}")
        for (cs, ps), d in sorted(recs.items()):
            rows = polish_rows(model, d, sel)
            out[(model, cs, ps)] = rows
            print(f"  {cs} -> polish {ps} (n={d['info']['n']}):  " + "; ".join(f"{nm} {acc:.1f} [{cst:.1f}+{pst:.1f}]" for nm, acc, cst, pst in rows))
    return out


DEV_CHAINS = RB.parent / "devices" / "infer_m5"      # X3's S24 chains (aihub_infer.py): per step preds + q_halt
S24_H1_MS = {("sudoku_mlp", "fp32"): (3.738, "H=1 profile j5qll10op"), ("sudoku_mlp", "w8a8"): (3.514, "H=1 profile jp3zz7oz5"),
             # no H=1 profile of these graphs yet: Sep-10 H=3 step / 3 (H3/H1 measured 3.04 fp32, 2.92 w8a8 on Sudoku-MLP)
             ("sudoku_attn", "w8a8"): (11.384 / 3, "approx: H=3 profile jp8e0v1xp / 3"),
             ("maze", "fp32"): (150.171 / 3, "approx: H=3 profile jgk271myg / 3")}
DEV_MODEL = {"sudoku_mlp": "sudoku_mlp", "sudoku_attn": "sudoku_attn", "maze": "maze_attn"}


def load_chain(task, prec, device="Samsung_Galaxy_S24"):
    """slim copy of X3's chain (per-step exact correctness vs the test labels, q_halt), written from
    p4d ~/aihub_artifacts/h1/<task>/infer/dev_<prec>_<device>.npz; unfinished chains are skipped."""
    f = DEV_CHAINS / f"{task}__dev_{prec}_{device}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    if not json.loads(str(d["info"]))["done"]:
        return None
    return Cell(DEV_MODEL[task], f"S24 {prec}", "device", d["correct"].T.astype(bool), d["q"].T.astype(np.float32))


def cmd_s24(a):
    """exit rules scored on X3's on-device S24 chains (QNN HTP, H=1 stepc graph, 8 H=1 steps, carry through the
    quantized I/O). Rule chosen on the host fp32 M5 record (n=1000) at the same budget and granularity."""
    out = {}
    for task in ("sudoku_mlp", "sudoku_attn", "maze"):
        model = DEV_MODEL[task]
        host = load_m5(model).get("fp32")
        if host is None:
            continue
        K = 8
        ref = h1_cell(host).trunc(K)
        sel = select_rule(ref)
        okh, sth = score(ref, exit_steps(ref, *sel))
        print(f"\n=== {task}: budget {K} H=1 steps; rule chosen on host fp32 (n=1000) = {rule_name(*sel)} "
              f"(host fp32: {100*ref.c[:, -1].mean():.1f} -> {100*okh.mean():.1f} at {sth.mean():.2f} steps)")
        hc = ref
        for prec in ("fp32", "w8a8", "w8a16", "w4a16"):
            c = load_chain(task, prec)
            if c is None:
                continue
            n = c.n
            agree = None
            if prec == "fp32":
                agree = float((c.c == hc.c[:n]).mean())
            fixed = 100 * c.c[:, -1].mean()
            rows = {}
            for nm, rule in (("native", ("native", None)), ("sel", sel), ("pat2", ("pat", 2))):
                ok, st = score(c, exit_steps(c, *rule))
                d, ci = paired_ci(ok, c.c[:, -1])
                rows[nm] = (100 * ok.mean(), d, ci, st.mean())
            ms = S24_H1_MS.get((task, prec))
            lat = ""
            if ms and fixed > 0:
                a_, d_, ci_, s_ = rows["sel"]
                lat = (f"; S24 {ms[0]:.2f} ms/H=1 step ({ms[1]}): ms/solved fixed {ms[0] * K / (fixed / 100):.1f} -> exit "
                       f"{ms[0] * s_ / max(a_ / 100, 1e-9):.1f}")
            out[(task, prec)] = {"n": n, "fixed": fixed, "rules": rows, "ms": ms, "agree_host_fp32": agree}
            print(f"  S24 {prec:6s} n={n}: fixed {fixed:5.1f} | " + " | ".join(
                f"{nm} {v[0]:5.1f} ({v[1]:+.1f}±{v[2]:.1f}) {v[3]:.2f} st" for nm, v in rows.items()) + lat +
                  (f"; per-step exact agreement with host fp32 {agree:.3f}" if agree is not None else ""))
    return out


HYB_DIR = Path(os.environ.get("M5_HYB_DIR", RB / "babel" / "m5" / "hybrid"))


def cmd_hybrid(a):
    """exit x M1 depth-for-precision (exit_hybrid.py records): per rule accuracy, mean cheap + polish steps,
    paired difference vs M1's fixed schedule with the same k; plus the exit-only rows."""
    out = {}
    for f in sorted(glob.glob(str(HYB_DIR / "*.json"))):
        for r in json.load(open(f))["records"]:
            c = np.array(r["correct"], bool); nc = np.array(r["cheap_steps"]); npl = np.array(r["polish_steps"])
            name = f"{Path(f).stem} {r['cheap']}->{r['polish']} (T={r['T']}, n={r['n']})"
            print(f"\n=== {name}")
            base = {}
            for i, rule in enumerate(r["rules"]):
                if rule.startswith("fix"):
                    base[int(rule[3:])] = c[i]
            for i, rule in enumerate(r["rules"]):
                k = parse_k(rule)
                ref = base.get(k)
                dd = paired_ci(c[i], ref) if ref is not None and not rule.startswith("fix") else None
                tot = nc[i].mean() + npl[i].mean()
                out[(Path(f).stem, r["cheap"], rule)] = {"acc": 100 * c[i].mean(), "cheap": float(nc[i].mean()),
                                                         "polish": float(npl[i].mean()), "total": float(tot),
                                                         "d_vs_fixed": dd}
                print(f"  {rule:16s} acc {100 * c[i].mean():5.1f}  cheap {nc[i].mean():6.1f} + polish {npl[i].mean():5.1f} = {tot:6.1f} "
                      f"({r['T'] / tot:4.1f}x fewer than T)" + (f"  vs fix{k}: {dd[0]:+.1f} +-{dd[1]:.1f}" if dd else ""))
    return out


def cmd_var(a):
    """trained variants (X5 checkpoints, 3 seeds per mixer): rule chosen on each variant's own fp32 (and, as a
    transfer test, the rule chosen on the released checkpoint), applied to its compressed cells; 16 and 64 steps."""
    out = {}
    released = {m: select_rule(cells_for(m, 16)["fp32"]) for m in ("sudoku_attn", "sudoku_mlp")}
    for d in sorted(glob.glob(str(M5_DIR / "var" / "*"))):
        tag = Path(d).name
        model = "sudoku_attn" if tag.startswith("attn") else "sudoku_mlp"
        cells = load_m5(model, sub=f"var/{tag}")
        if "fp32" not in cells:
            continue
        for K in (16, 64):
            ref = cells["fp32"].trunc(K)
            sel = select_rule(ref)
            print(f"\n=== {tag} K={K}: own-fp32 rule {rule_name(*sel)}; released-checkpoint rule {rule_name(*released[model])}; "
                  f"fixed depth on fp32 {fp32_depth(ref)}")
            for spec in sorted(cells, key=spec_order):
                c = cells[spec].trunc(K)
                parts = []
                for nm, rule in (("own", sel), ("released", released[model]), ("native", ("native", None)), ("oracle", ("oracle", None))):
                    ok, st = score(c, exit_steps(c, *rule))
                    d_, ci = paired_ci(ok, c.c[:, -1])
                    parts.append(f"{nm} {100*ok.mean():5.1f} ({d_:+5.1f}) {st.mean():5.2f}")
                    out[(tag, K, spec, nm)] = {"acc": 100 * ok.mean(), "d": d_, "ci": ci, "steps": float(st.mean())}
                print(f"  {spec:14s} fixed {100*c.c[:, -1].mean():5.1f} | " + " | ".join(parts))
    return out


def parse_k(rule):
    if rule.startswith("fix"):
        return int(rule[3:])
    rest = rule.split("->")[1]
    return int(rest.split("_")[1]) if rest.startswith("h") else int(rest)


def cmd_json(a):
    """figure-ready summary: per model / K / correctness definition, every cell's rule table (draws
    merged), plus Sudoku bins and device rows."""
    out = {"note": "M5 exit-rule read-out; see analyze_m5.py docstring and reports/edgetrm_exp/M5.md",
           "summaries": [], "bins": {}, "device": {}}
    for model in a.models + ["arc_attn"]:
        for K in (16, 64, 512):
            for corr in (["exact", "vshort", "valid"] if "maze" in model else ["exact"]):
                try:
                    summ = summarize(model, K, corr, rules=tuple(rule_name(*r) for r in RULESET))
                except Exception as e:                      # a model without records at this K
                    summ = None
                if summ is None or len(summ["rows"]) == 0:
                    continue
                out["summaries"].append({"model": model, "K": K, "corr": corr, "sel_rule": summ["sel_rule"],
                                         "fp32_depth": summ["rows"]["fp32"]["fp32_depth"], "cells": summ["rows"],
                                         "merged": group_rows(summ)})
    a.specs = "all"
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        b = cmd_bins(a)
        d = cmd_device(a)
    out["bins"] = {f"{m}|{s}": v for (m, s), v in b.items()}
    out["device"] = {f"{m}|{dev}|{p}": v for (m, dev, p), v in d.items()}
    path = RB / "m5_summary.json"
    json.dump(out, open(path, "w"), indent=1, default=float)
    print("wrote", path, len(out["summaries"]), "summaries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rules", "summary", "bins", "device", "json", "agree", "inner", "deep", "sample", "dt", "polish", "s24", "hybrid", "var"])
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--show", default="")
    ap.add_argument("--specs", default="fp32,w8c,w4c,w4t,w3g32,w3c,noise0.15@s0,noise0.15@s1,noise0.15@s2,noise0.7@s0,w2g32")
    a = ap.parse_args()
    if a.cmd == "rules":
        cmd_rules(a)
    elif a.cmd == "summary":
        cmd_summary(a)
    elif a.cmd == "bins":
        cmd_bins(a)
    elif a.cmd == "device":
        cmd_device(a)
    elif a.cmd == "json":
        cmd_json(a)
    elif a.cmd == "inner":
        cmd_inner(a)
    elif a.cmd == "deep":
        cmd_deep(a)
    elif a.cmd == "sample":
        cmd_sample(a)
    elif a.cmd == "dt":
        cmd_dt(a)
    elif a.cmd == "polish":
        cmd_polish(a)
    elif a.cmd == "s24":
        cmd_s24(a)
    elif a.cmd == "hybrid":
        cmd_hybrid(a)
    elif a.cmd == "var":
        cmd_var(a)
    elif a.cmd == "agree":
        for m in a.models:
            for row in check_agreement(m):
                print(row)


if __name__ == "__main__":
    main()
