#!/usr/bin/env python3
"""X8 read-out (angles C and D; reports/edgetrm_exp/X8.md). Pure numpy.

  python analyze_x8.py auroc     # detectors side by side: AUROC for "answer wrong", per definition
  python analyze_x8.py select    # defer rule: threshold set once on fp32, coverage / risk everywhere
  python analyze_x8.py basin     # head calibration vs the basin (risk, ECE, q separation by class)
  python analyze_x8.py early     # per-step read-out: step-k signals predicting the final answer
  python analyze_x8.py checker   # does the head add anything over the hand-coded checker?
  python analyze_x8.py fid       # fidelity as a filter, not a ranker (existing breadth records)
  python analyze_x8.py indep     # cell independence: observed exact vs prod_i p_cell^L_i
  python analyze_x8.py ptrm      # PTRM reconciliation: single rollout vs best-of-K
  python analyze_x8.py json      # everything -> results/breadth/x8_summary.json
  python analyze_x8.py all       # every table

Inputs: results/breadth/babel/x8/det_<task>.json (detector_sweep.py), ptrm_<task>_<noise>.json
(ptrm_sweep.py), and the Sep 13-23 breadth records under results/breadth/ for `fid`.
Conventions: every detector score is oriented "higher = more likely wrong"; AUROC CIs are
Hanley-McNeil 95%; coverage / risk CIs are Wilson 95%; noise cells with several draws are shown
as mean [min, max] over draws.  Basin class of a cell from its accuracy retained r = acc / acc_fp32
(same run, same definition): in r >= 0.9, edge 0.3 <= r < 0.9, out r < 0.3.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE / "results" / "breadth"
X8 = Path(os.environ["X8_DIR"]) if os.environ.get("X8_DIR") else B / "babel" / "x8"
TASKS = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze_attn", "hrm_maze"]
KIND = {t: ("maze" if "maze" in t else "sudoku") for t in TASKS}
DEFS = {"sudoku": ["exact"], "maze": ["exact", "valid", "vshort"]}
DET_ORDER = ["head", "combo", "halt_time", "checker", "msp_min", "msp_mean", "mrg_min", "resid_H", "resid_L",
             "pi_dis", "fid_H", "agree_fp32"]
REF_BASED = {"fid_H", "agree_fp32"}


# ----------------------------------------------------------------------------- stats
def auroc(s, pos):
    s = np.asarray(s, float)
    pos = np.asarray(pos, bool)
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    _, start, cnt = np.unique(ss, return_index=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(start + (cnt + 1) / 2.0, cnt)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def hm_ci(a, n1, n0):
    if a is None or n1 == 0 or n0 == 0:
        return (None, None)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt(max(a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a), 0) / (n1 * n0))
    return (max(0.0, a - 1.96 * se), min(1.0, a + 1.96 * se))


def wilson(k, n):
    if n == 0:
        return (None, None)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_ci(x, y, reps=2000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    rho = spearman(x, y)
    if rho is None:
        return None, (None, None)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(reps):
        i = rng.integers(0, len(x), len(x))
        r = spearman(x[i], y[i])
        if r is not None:
            bs.append(r)
    return rho, (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))


def fmt(v, d=2):
    return "--" if v is None else f"{v:.{d}f}"


# ----------------------------------------------------------------------------- loading
def load_det(task):
    """Merge det_<task>.json and det_<task>__*.json (one file per job line). The fp32 record of
    the first file is kept as the reference (every file reran fp32 in its own process; the
    per-file fp32 is what that file's fid_H / agree_fp32 were computed against)."""
    files = sorted(glob.glob(str(X8 / f"det_{task}.json"))) + sorted(glob.glob(str(X8 / f"det_{task}__*.json")))
    if not files:
        return None
    meta, recs, seen = None, [], set()
    for f in files:
        j = json.load(open(f))
        meta = meta or j.get("meta", {})
        for r in j["records"]:
            if r["spec"] in seen:
                continue
            seen.add(r["spec"])
            r["_file"] = Path(f).name
            recs.append(r)
    fp = [r for r in recs if r["spec"] == "fp32"]
    return meta, fp + [r for r in recs if r["spec"] != "fp32"]


def base_spec(spec):
    return spec.split("@s")[0]


def wrong_of(rec, d):
    f = rec["final"]
    if d == "exact":
        return 1 - np.asarray(f["correct"], int)
    if d == "valid":
        return 1 - np.asarray(f["valid"], int)
    return 1 - np.asarray(f["vshort"], int)


def acc_of(rec, d):
    return 1 - wrong_of(rec, d).mean()


def basin_class(r):
    if r is None:
        return "?"
    return "in" if r >= 0.9 else ("edge" if r >= 0.3 else "out")


COMBO_FEATS = ("qhalt", "resid_H", "resid_L", "msp_min", "msp_mean", "mrg_min")


def combo_features(f):
    eps = 1e-6
    q = np.asarray(f["qhalt"], float)
    rh = np.log(np.asarray(f["resid_H"], float) + eps)
    rl = np.log(np.asarray(f["resid_L"], float) + eps)
    pm = np.clip(np.asarray(f["msp_min"], float), eps, 1 - eps)
    pa = np.clip(np.asarray(f["msp_mean"], float), eps, 1 - eps)
    mm = np.asarray(f["mrg_min"], float)
    return np.stack([q, rh, rl, np.log(pm / (1 - pm)), np.log(pa / (1 - pa)), mm], 1)


def fit_logreg(X, y, lam=1e-2, iters=50):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w, -30, 30)))
        g = Z.T @ (p - y) + lam * np.r_[w[:-1], 0]
        Hs = (Z * (p * (1 - p))[:, None]).T @ Z + lam * np.diag(np.r_[np.ones(len(w) - 1), 0])
        w -= np.linalg.solve(Hs + 1e-9 * np.eye(len(w)), g)
    return mu, sd, w


def apply_logreg(model, X):
    mu, sd, w = model
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    return Z @ w


def detector_scores(rec, meta, kind, d, combo_models=None, nsup=16):
    """dict name -> score (higher = wrong) at the final step, for correctness definition d."""
    f = rec["final"]
    s = {"head": -np.asarray(f["qhalt"], float)}
    if kind == "sudoku":
        s["checker"] = np.asarray(f["chk"], float)
    else:
        chk = np.asarray(f["chk"], float)
        if d == "valid":
            s["checker"] = chk
        else:
            ex = np.asarray(f["plen"], float) - np.asarray(meta["bfs"], float)
            s["checker"] = chk * 1000 + np.where(chk == 0, np.maximum(ex, 0), 0)
    for k in ("msp_min", "msp_mean", "mrg_min", "mrg_mean"):
        s[k] = -np.asarray(f[k], float)
    s["resid_H"] = np.asarray(f["resid_H"], float)
    s["resid_L"] = np.asarray(f["resid_L"], float)
    if "pi_dis" in f:
        s["pi_dis"] = np.asarray(f["pi_dis"], float) + 1e-3 * (1 - np.asarray(f["pi_cos"], float))
    if "fid_H" in f:
        s["fid_H"] = -np.asarray(f["fid_H"], float)
        s["agree_fp32"] = -np.asarray(f["agree_fp32"], float)
    if "per_step" in rec:
        q = np.asarray(rec["per_step"]["q_all"], float)                  # [n, nsup]
        pos = q > 0
        first = np.where(pos.any(1), pos.argmax(1) + 1, q.shape[1] + 1).astype(float)
        s["halt_time"] = first - 1e-3 * q[:, -1]                          # later first halt -> wrong
    if combo_models is not None:
        X = combo_features(f)
        idx = np.arange(len(X))
        out = np.zeros(len(X))
        for fold in (0, 1):
            m = idx % 2 == fold
            out[m] = apply_logreg(combo_models[1 - fold], X[m])          # fitted on the other half of fp32
        s["combo"] = out
    return s


def fit_combo(fp32_rec, d):
    X = combo_features(fp32_rec["final"])
    y = wrong_of(fp32_rec, d).astype(float)
    idx = np.arange(len(X))
    return {fold: fit_logreg(X[idx % 2 == fold], y[idx % 2 == fold]) for fold in (0, 1)}


def group_specs(recs):
    """base spec -> list of records (noise draws grouped), fp32 first, input order kept."""
    g = {}
    for r in recs:
        g.setdefault(base_spec(r["spec"]), []).append(r)
    return g


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None
    return float(np.mean(vals)), float(min(vals)), float(max(vals))


def fmt_agg(vals, d=2):
    m, lo, hi = agg(vals)
    if m is None:
        return "--"
    if len([v for v in vals if v is not None]) == 1:
        return f"{m:.{d}f}"
    return f"{m:.{d}f} [{lo:.{d}f},{hi:.{d}f}]"


# ----------------------------------------------------------------------------- 1. AUROC
def compute_auroc():
    out = {}
    for task in TASKS:
        L = load_det(task)
        if L is None:
            continue
        meta, recs = L
        kind = KIND[task]
        fp = next(r for r in recs if r["spec"] == "fp32")
        tout = {}
        for d in DEFS[kind]:
            cm = fit_combo(fp, d)
            acc_fp = acc_of(fp, d)
            rows = []
            for r in recs:
                w = wrong_of(r, d).astype(bool)
                sc = detector_scores(r, meta, kind, d, cm)
                a = {}
                for k, v in sc.items():
                    A = auroc(v, w)
                    a[k] = {"auroc": A, "ci": hm_ci(A, int(w.sum()), int((~w).sum()))}
                acc = 1 - w.mean()
                rows.append({"spec": r["spec"], "base": base_spec(r["spec"]), "seed": r.get("seed"),
                             "acc": float(acc), "retained": float(acc / acc_fp) if acc_fp > 0 else None,
                             "class": basin_class(acc / acc_fp if acc_fp > 0 else None) if r["spec"] != "fp32" else "fp32",
                             "n": len(w), "n_wrong": int(w.sum()), "fid_H": float(np.mean(r["final"]["fid_H"])) if "fid_H" in r["final"] else 1.0,
                             "auroc": a})
            tout[d] = rows
        out[task] = tout
    return out


def print_auroc(res):
    for task, tout in res.items():
        for d, rows in tout.items():
            print(f"\n### {task} — wrong = not {d}  (n per cell = {rows[0]['n']}; AUROC, noise draws mean [min,max])")
            dets = [k for k in DET_ORDER if any(k in r["auroc"] for r in rows)]
            print("| spec | class | acc | φ_H | " + " | ".join(dets) + " |")
            print("|---|---|---|---|" + "---|" * len(dets))
            g = defaultdict(list)
            for r in rows:
                g[r["base"]].append(r)
            for base, rr in g.items():
                cells = [fmt_agg([x["auroc"].get(k, {}).get("auroc") for x in rr]) for k in dets]
                cls = "/".join(sorted({x["class"] for x in rr}))
                print(f"| {base}{' x'+str(len(rr)) if len(rr) > 1 else ''} | {cls} | {fmt_agg([100*x['acc'] for x in rr], 1)} | "
                      f"{fmt_agg([x['fid_H'] for x in rr], 3)} | " + " | ".join(cells) + " |")


# ----------------------------------------------------------------------------- 2. selective prediction
def calibrate(score, wrong, rstar):
    """Largest threshold t (an observed score) such that risk among {score <= t} <= rstar."""
    o = np.argsort(score, kind="mergesort")
    s, w = score[o], wrong[o]
    cw = np.cumsum(w)
    cn = np.arange(1, len(s) + 1)
    last_of_tie = np.r_[s[1:] != s[:-1], True]
    ok = (cw / cn <= rstar) & last_of_tie
    if ok[-1]:
        return np.inf                                   # the whole calibration set meets the target
    if not ok.any():
        return -np.inf
    return float(s[np.nonzero(ok)[0].max()])


SEL_DETS = ["head_rule", "head", "combo", "checker", "msp_min", "msp_mean", "resid_H", "pi_dis"]


def compute_select(rstars=(0.01, 0.05)):
    out = {}
    for task in TASKS:
        L = load_det(task)
        if L is None:
            continue
        meta, recs = L
        kind = KIND[task]
        fp = next(r for r in recs if r["spec"] == "fp32")
        tout = {}
        for d in DEFS[kind]:
            cm = fit_combo(fp, d)
            sfp = detector_scores(fp, meta, kind, d, cm)
            wfp = wrong_of(fp, d)
            acc_fp = 1 - wfp.mean()
            idx = np.arange(len(wfp))
            thr = {}
            for rs in rstars:
                for k in SEL_DETS:
                    if k == "head_rule":
                        thr[(k, rs)] = {0: 0.0, 1: 0.0}                    # accept iff q_halt > 0 (sigmoid > 0.5)
                        continue
                    if k not in sfp:
                        continue
                    thr[(k, rs)] = {fold: calibrate(sfp[k][idx % 2 != fold], wfp[idx % 2 != fold], rs) for fold in (0, 1)}
            rows = []
            for r in recs:
                w = wrong_of(r, d)
                sc = detector_scores(r, meta, kind, d, cm)
                sc["head_rule"] = sc["head"]
                acc = 1 - w.mean()
                row = {"spec": r["spec"], "base": base_spec(r["spec"]), "acc": float(acc),
                       "retained": float(acc / acc_fp) if acc_fp > 0 else None,
                       "class": "fp32" if r["spec"] == "fp32" else basin_class(acc / acc_fp if acc_fp > 0 else None),
                       "sel": {}}
                for (k, rs), t in thr.items():
                    if k not in sc:
                        continue
                    acc_mask = np.zeros(len(w), bool)
                    for fold in (0, 1):
                        m = idx % 2 == fold
                        if k == "head_rule":
                            acc_mask[m] = sc[k][m] < 0                       # -q < 0  <=> q > 0
                        else:
                            acc_mask[m] = sc[k][m] <= t[fold]
                    nacc = int(acc_mask.sum())
                    kw = int(w[acc_mask].sum())
                    row["sel"][f"{k}@{rs}"] = {"coverage": nacc / len(w), "risk": (kw / nacc) if nacc else None,
                                               "risk_ci": wilson(kw, nacc), "n_acc": nacc, "n_wrong_acc": kw,
                                               "cov_of_correct": float(acc_mask[w == 0].mean()) if (w == 0).any() else None}
                rows.append(row)
            tout[d] = {"thresholds": {f"{k}@{rs}": v for (k, rs), v in thr.items()}, "rows": rows}
        out[task] = tout
    return out


def print_select(res, rs=None):
    for task, tout in res.items():
        for d, T in tout.items():
            rlist = [0.01, 0.05] if rs is None else [rs]
            for rr in rlist:
                print(f"\n### {task} — wrong = not {d}; threshold set on fp32 for risk <= {rr:.0%} (two-fold cross-fit), "
                      f"applied unchanged. cell = coverage% / risk% [Wilson 95%]; head_rule = the model's own sigmoid(q) > 0.5")
                dets = [k for k in SEL_DETS if any(f"{k}@{rr}" in r["sel"] for r in T["rows"])]
                print("| spec | class | acc | " + " | ".join(dets) + " |")
                print("|---|---|---|" + "---|" * len(dets))
                for r in T["rows"]:
                    cells = []
                    for k in dets:
                        v = r["sel"].get(f"{k}@{rr}")
                        if v is None:
                            cells.append("--")
                        elif v["risk"] is None:
                            cells.append(f"{100*v['coverage']:.0f} / --")
                        else:
                            cells.append(f"{100*v['coverage']:.0f} / {100*v['risk']:.1f} [{100*v['risk_ci'][0]:.1f},{100*v['risk_ci'][1]:.1f}]")
                    print(f"| {r['spec']} | {r['class']} | {100*r['acc']:.1f} | " + " | ".join(cells) + " |")


# ----------------------------------------------------------------------------- 3. head calibration vs the basin
def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    e = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def compute_basin(sel=None):
    """per cell: retained, class, fidelity, head AUROC, ECE of sigmoid(q) as P(correct),
    mean q for correct / wrong, and the fp32-threshold risk of the head rule and head@1%/5%."""
    sel = sel or compute_select()
    out = {}
    for task in TASKS:
        L = load_det(task)
        if L is None:
            continue
        meta, recs = L
        kind = KIND[task]
        tout = {}
        for d in DEFS[kind]:
            fp = next(r for r in recs if r["spec"] == "fp32")
            acc_fp = acc_of(fp, d)
            selrows = {r["spec"]: r for r in sel[task][d]["rows"]}
            rows = []
            for r in recs:
                w = wrong_of(r, d).astype(bool)
                q = np.asarray(r["final"]["qhalt"], float)
                p = 1 / (1 + np.exp(-q))
                acc = 1 - w.mean()
                sr = selrows[r["spec"]]["sel"]
                rows.append({"spec": r["spec"], "base": base_spec(r["spec"]), "acc": float(acc),
                             "retained": float(acc / acc_fp) if acc_fp > 0 else None,
                             "class": "fp32" if r["spec"] == "fp32" else basin_class(acc / acc_fp if acc_fp > 0 else None),
                             "fid_H": float(np.mean(r["final"]["fid_H"])) if "fid_H" in r["final"] else 1.0,
                             "auroc_head": auroc(-q, w), "ece": ece(p, ~w), "mean_p": float(p.mean()),
                             "q_correct": float(q[~w].mean()) if (~w).any() else None,
                             "q_wrong": float(q[w].mean()) if w.any() else None,
                             "rule": sr.get("head_rule@0.01"), "h01": sr.get("head@0.01"), "h05": sr.get("head@0.05")})
            tout[d] = rows
        out[task] = tout
    return out


def print_basin(res):
    for task, tout in res.items():
        for d, rows in tout.items():
            print(f"\n### {task} — wrong = not {d}: halting head vs the basin")
            print("| spec | class | acc | retained | φ_H | AUROC | ECE | mean σ(q) | q̄ correct | q̄ wrong | rule cov/risk | rule: correct kept | head@1% cov/risk | head@5% cov/risk |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            for r in sorted(rows, key=lambda x: -(x["retained"] or 0)):
                def cr(v):
                    if v is None:
                        return "--"
                    return f"{100*v['coverage']:.0f}/{fmt(None if v['risk'] is None else 100*v['risk'], 1)}"
                print(f"| {r['spec']} | {r['class']} | {100*r['acc']:.1f} | {fmt(r['retained'])} | {r['fid_H']:.3f} | {fmt(r['auroc_head'])} | "
                      f"{r['ece']:.3f} | {r['mean_p']:.2f} | {fmt(r['q_correct'], 1)} | {fmt(r['q_wrong'], 1)} | {cr(r['rule'])} | "
                      f"{fmt(None if r['rule'] is None or r['rule']['cov_of_correct'] is None else 100*r['rule']['cov_of_correct'], 0)} | {cr(r['h01'])} | {cr(r['h05'])} |")


# ----------------------------------------------------------------------------- 4. per-step read-out
def compute_early(specs_per_task=None):
    out = {}
    for task in TASKS:
        L = load_det(task)
        if L is None:
            continue
        meta, recs = L
        kind = KIND[task]
        tout = {}
        for d in DEFS[kind]:
            rows = []
            for r in recs:
                if specs_per_task and base_spec(r["spec"]) not in specs_per_task.get(task, []):
                    continue
                w = wrong_of(r, d).astype(bool)
                q = np.asarray(r["per_step"]["q_all"], float)
                c = np.asarray(r["per_step"]["c_all"], bool)
                head = [auroc(-q[:, k], w) for k in range(q.shape[1])]
                # checker / max-softmax / residual at the recorded steps
                other = {}
                for ks, st in sorted(r["steps"].items(), key=lambda x: int(x[0])):
                    other[ks] = {"checker": auroc(np.asarray(st["chk"], float), w),
                                 "msp_min": auroc(-np.asarray(st["msp_min"], float), w),
                                 "resid_H": auroc(np.asarray(st["resid_H"], float), w),
                                 "wrong_now": auroc((~np.asarray(st["correct"], bool)).astype(float), w)}
                # arrived-then-left: correct at some step, wrong at the end
                left = float((c.any(1) & w).mean()) if kind == "sudoku" or d == "exact" else None
                rows.append({"spec": r["spec"], "acc": float(1 - w.mean()), "head_by_step": head, "other": other,
                             "arrived_then_left": left})
            tout[d] = rows
        out[task] = tout
    return out


def print_early(res):
    for task, tout in res.items():
        for d, rows in tout.items():
            print(f"\n### {task} — AUROC of step-k signals for the step-16 answer being wrong (not {d})")
            ks = sorted({int(k) for r in rows for k in r["other"]})
            print("| spec | acc | head k=1/2/4/8/12/16 | checker@k | msp_min@k | resid_H@k | wrong-now@k |")
            print("|---|---|---|---|---|---|---|")
            for r in rows:
                h = "/".join(fmt(r["head_by_step"][k - 1]) for k in ks)
                o = lambda key: "/".join(fmt(r["other"][str(k)][key]) for k in ks if str(k) in r["other"])
                print(f"| {r['spec']} | {100*r['acc']:.1f} | {h} | {o('checker')} | {o('msp_min')} | {o('resid_H')} | {o('wrong_now')} |")


# ----------------------------------------------------------------------------- 5. head vs checker
def compute_checker():
    """Sudoku: the checker is exact correctness (unique solutions) -> AUROC 1 by construction; we
    report (a) that identity, (b) Spearman(q, -conflicts) among wrong answers (is the head a graded
    conflict count?), (c) step-k head vs step-k checker for the final answer (early warning).
    Maze: (a) head AUROC inside the checker-passing set (valid-shortest paths: does the head know
    which one matches the reference?), (b) head vs checker AUROC under each definition."""
    out = {}
    for task in TASKS:
        L = load_det(task)
        if L is None:
            continue
        meta, recs = L
        kind = KIND[task]
        rows = []
        for r in recs:
            f = r["final"]
            q = np.asarray(f["qhalt"], float)
            chk = np.asarray(f["chk"], float)
            row = {"spec": r["spec"], "acc": float(np.mean(f["correct"]))}
            ex_w = 1 - np.asarray(f["correct"], int)
            if kind == "sudoku":
                row["checker_equals_exact"] = bool(((chk == 0) == (ex_w == 0)).all())
                row["n_checker_disagree"] = int(((chk == 0) != (ex_w == 0)).sum())
                wm = ex_w == 1
                row["rho_q_conflicts_wrong"] = spearman(q[wm], -chk[wm]) if wm.sum() > 5 else None
                row["rho_q_nwrong_wrong"] = spearman(q[wm], -np.asarray(f["n_wrong"], float)[wm]) if wm.sum() > 5 else None
            else:
                vs = np.asarray(f["vshort"], bool)
                va = np.asarray(f["valid"], bool)
                row["n_vshort"] = int(vs.sum())
                row["n_vshort_not_exact"] = int((vs & (ex_w == 1)).sum())
                row["n_valid_not_vshort"] = int((va & ~vs).sum())
                row["n_invalid"] = int((~va).sum())
                row["head_auroc_exact_within_vshort"] = auroc(-q[vs], ex_w[vs].astype(bool)) if vs.any() else None
                row["head_auroc_invalid"] = auroc(-q, ~va)
                row["q_exact"] = float(q[ex_w == 0].mean()) if (ex_w == 0).any() else None
                row["q_vshort_not_exact"] = float(q[vs & (ex_w == 1)].mean()) if (vs & (ex_w == 1)).any() else None
                row["q_invalid"] = float(q[~va].mean()) if (~va).any() else None
            rows.append(row)
        out[task] = rows
    return out


def print_checker(res):
    for task, rows in res.items():
        print(f"\n### {task}: head vs hand-coded checker")
        keys = [k for k in rows[0] if k not in ("spec",)]
        print("| spec | " + " | ".join(keys) + " |")
        print("|---|" + "---|" * len(keys))
        for r in rows:
            print(f"| {r['spec']} | " + " | ".join(fmt(r[k]) if isinstance(r[k], float) or r[k] is None else str(r[k]) for k in keys) + " |")


# ----------------------------------------------------------------------------- 6. fidelity as a filter
FID_SOURCES = {
    "sudoku_mlp": ["sudoku_mlp_all.json", "sudoku_mlp_noise.json", "sudoku_mlp_noiseH.json", "sudoku_mlp_deep.json",
                   "babel/noise_sudoku_mlp_seed1.json", "babel/noise_sudoku_mlp_seed2.json", "babel/anoise_sudoku_mlp.json"],
    "sudoku_attn": ["sudoku_attn_all.json", "sudoku_attn_noise.json", "sudoku_attn_noiseH.json", "sudoku_attn_deep.json",
                    "babel/noise_sudoku_attn_seed1.json", "babel/noise_sudoku_attn_seed2.json", "babel/anoise_sudoku_attn.json"],
    "hrm_sudoku": ["hrm_sudoku_all.json", "hrm_sudoku_noise.json", "hrm_sudoku_deep.json",
                   "babel/noise_hrm_sudoku_seed1.json", "babel/noise_hrm_sudoku_seed2.json", "p4d/anoise_hrm_sudoku.json"],
    "maze_attn": ["maze_attn_n816.json", "maze_attn_n124.json", "maze_attn_noise.json", "maze_attn_deep.json",
                  "babel/noise_maze_attn_seed1.json", "babel/noise_maze_attn_seed2.json", "babel/anoise_maze_attn.json"],
    "hrm_maze": ["hrm_maze_n816.json", "hrm_maze_n124.json", "hrm_maze_noise.json", "hrm_maze_deep.json",
                 "babel/noise_hrm_maze_seed1.json", "babel/noise_hrm_maze_seed2.json", "p4d/anoise_hrm_maze.json"],
}
OUT_LEN = {"sudoku_mlp": 81, "sudoku_attn": 81, "hrm_sudoku": 81, "maze_attn": 900, "hrm_maze": 900}


def fid_cells(task):
    """Every (file, nsup, spec) cell at the model's default H (TRM 3, HRM 2; the HRM files hold
    H 1-4 and hrm_sudoku_noise.json is H=3, so it drops out) with phi_H, exact, cell accuracy,
    retained vs the same file's fp32 at the same (H, nsup).  Duplicate (H, nsup, spec) across
    files: first file wins, except noise draws from the seed files (kept as separate draws)."""
    Hdef = 2 if task.startswith("hrm") else 3
    cells, seen = [], set()
    for fn in FID_SOURCES[task]:
        f = B / fn
        if not f.exists():
            continue
        recs = json.load(open(f))["records"]
        fpr = {(r["H_cycles"], r["nsup"]): r for r in recs if (r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]) == "fp32"}
        tag = "seed1" if "seed1" in fn else ("seed2" if "seed2" in fn else "seed0")
        for r in recs:
            spec = r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
            if spec == "fp32" or (r["H_cycles"], r["nsup"]) not in fpr or r["H_cycles"] != Hdef:
                continue
            key = (r["H_cycles"], r["nsup"], spec, tag)
            if key in seen:
                continue
            seen.add(key)
            f0 = fpr[(r["H_cycles"], r["nsup"])]
            if f0["pexact"] <= 0:
                continue
            cells.append({"file": fn, "H": r["H_cycles"], "nsup": r["nsup"], "spec": spec, "draw": tag,
                          "phi": r["fidelity_H"], "exact": r["pexact"], "cell": r["cell"],
                          "ret": r["pexact"] / f0["pexact"], "ret_cell": r["cell"] / f0["cell"]})
    return cells


def dt_cells():
    """Deep-thinking maze cells (p4d dt_extrap_mazes_<size>.json): whole-map phi, exact; and
    dt_cell_mazes_{25,33}_s*.json with per-cell accuracy."""
    out = []
    for f in sorted(glob.glob(str(B / "p4d" / "dt_extrap_mazes_*.json"))):
        size = int(Path(f).stem.split("_")[-1])
        recs = json.load(open(f))["records"]
        spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
        fp = {r["iters"]: r for r in recs if spec(r) == "fp32"}
        for r in recs:
            if spec(r) == "fp32" or r["iters"] not in fp or fp[r["iters"]]["acc"] <= 0:
                continue
            out.append({"size": size, "iters": r["iters"], "spec": spec(r), "phi": r["fidelity"], "exact": r["acc"],
                        "ret": r["acc"] / fp[r["iters"]]["acc"], "phi_target": r.get("fid_target")})
    cellrows = []
    for f in sorted(glob.glob(str(B / "p4d" / "dt_cell_mazes_*.json"))):
        size = int(Path(f).stem.split("_")[3])
        recs = json.load(open(f))["records"]
        spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
        fp = {r["iters"]: r for r in recs if spec(r) == "fp32"}
        ntc = next((r.get("n_target_cells") for r in recs if r.get("n_target_cells")), None)
        for r in recs:
            if r["iters"] not in fp:
                continue
            cellrows.append({"size": size, "iters": r["iters"], "spec": spec(r), "seed": r.get("seed"), "phi": r["fidelity"],
                             "exact": r["acc"], "cell": r["cell_acc"], "cell_target": r.get("cell_acc_target"),
                             "ret": r["acc"] / fp[r["iters"]]["acc"] if fp[r["iters"]]["acc"] > 0 else None,
                             "ret_cell": r["cell_acc"] / fp[r["iters"]]["cell_acc"], "n_target_cells": r.get("n_target_cells") or ntc,
                             "correct": r.get("correct")})
    return out, cellrows


def best_threshold(phi, damaged):
    """threshold t: predict damaged iff phi < t. Returns (t, accuracy, balanced accuracy)."""
    phi, damaged = np.asarray(phi, float), np.asarray(damaged, bool)
    cands = np.unique(np.r_[phi, phi.max() + 1e-6])
    best = (None, -1, -1)
    for t in cands:
        pred = phi < t
        acc = (pred == damaged).mean()
        tp = (pred & damaged).sum() / max(damaged.sum(), 1)
        tn = (~pred & ~damaged).sum() / max((~damaged).sum(), 1)
        bal = (tp + tn) / 2
        if bal > best[2] + 1e-12:
            best = (float(t), float(acc), float(bal))
    return best


def compute_fid():
    res = {"per_model": {}, "filter": {}}
    allcells = {}
    for task in TASKS:
        cells = fid_cells(task)
        allcells[task] = cells
        if not cells:
            continue
        phi = np.array([c["phi"] for c in cells])
        ret = np.array([c["ret"] for c in cells])
        inb = ret >= 0.9
        rho_all, ci_all = spearman_ci(phi, ret)
        rho_in, ci_in = spearman_ci(phi[inb], ret[inb])
        # default depth only: H = the checkpoint default (TRM 3, HRM 2), nsup = 16
        Hdef = 2 if task.startswith("hrm") else 3
        dd = [c for c in cells if c["nsup"] == 16 and c["H"] == Hdef]
        phid = np.array([c["phi"] for c in dd]); retd = np.array([c["ret"] for c in dd])
        inbd = retd >= 0.9
        rho_in_d, ci_in_d = spearman_ci(phid[inbd], retd[inbd]) if inbd.sum() >= 3 else (None, (None, None))
        nb = ret >= 0.97
        nbd = retd >= 0.97
        rho_nb, ci_nb = spearman_ci(phi[nb], ret[nb]) if nb.sum() >= 3 else (None, (None, None))
        rho_nb_d, ci_nb_d = spearman_ci(phid[nbd], retd[nbd]) if nbd.sum() >= 3 else (None, (None, None))
        res["per_model"][task] = {"n_cells": len(cells), "n_in": int(inb.sum()), "rho_all": rho_all, "ci_all": ci_all,
                                  "rho_in": rho_in, "ci_in": ci_in, "auroc_damaged": auroc(-phi, ~inb),
                                  "n_near": int(nb.sum()), "rho_near": rho_nb, "ci_near": ci_nb,
                                  "default_depth": {"n": len(dd), "n_in": int(inbd.sum()), "rho_in": rho_in_d, "ci_in": ci_in_d,
                                                    "rho_all": spearman(phid, retd), "n_near": int(nbd.sum()), "rho_near": rho_nb_d, "ci_near": ci_nb_d},
                                  "cells": [{k: c[k] for k in ("H", "nsup", "spec", "draw", "phi", "exact", "cell", "ret", "ret_cell")} for c in cells]}
    # filter: one threshold per output length (exact-retained unit) vs the per-cell unit
    groups = {"sudoku (81)": ["sudoku_mlp", "sudoku_attn", "hrm_sudoku"], "maze (900)": ["maze_attn", "hrm_maze"]}
    dtc, dtcell = dt_cells()
    filt = {}
    for gname, ts in groups.items():
        cs = [c for t in ts for c in allcells.get(t, [])]
        if not cs:
            continue
        phi = np.array([c["phi"] for c in cs])
        dmg_exact = np.array([c["ret"] < 0.9 for c in cs])
        dmg_cell = np.array([c["ret_cell"] < 0.99 for c in cs])
        t_e = best_threshold(phi, dmg_exact)
        t_c = best_threshold(phi, dmg_cell)
        filt[gname] = {"n": len(cs), "n_damaged_exact": int(dmg_exact.sum()), "n_damaged_cell": int(dmg_cell.sum()),
                       "auroc_exact": auroc(-phi, dmg_exact), "thr_exact": t_e, "auroc_cell": auroc(-phi, dmg_cell), "thr_cell": t_c,
                       "per_model_thr_exact": {t: best_threshold([c["phi"] for c in allcells[t]], [c["ret"] < 0.9 for c in allcells[t]]) for t in ts if allcells.get(t)}}
    for size in sorted({c["size"] for c in dtc}):
        cs = [c for c in dtc if c["size"] == size]
        phi = np.array([c["phi"] for c in cs])
        dmg = np.array([c["ret"] < 0.9 for c in cs])
        filt[f"DT maze {size}"] = {"n": len(cs), "n_damaged_exact": int(dmg.sum()), "auroc_exact": auroc(-phi, dmg),
                                   "thr_exact": best_threshold(phi, dmg)}
    for size in sorted({c["size"] for c in dtcell}):
        cs = [c for c in dtcell if c["size"] == size and c["spec"] != "fp32"]
        phi = np.array([c["phi"] for c in cs])
        de = np.array([(c["ret"] or 0) < 0.9 for c in cs]); dc = np.array([c["ret_cell"] < 0.99 for c in cs])
        filt[f"DT maze {size} (per-cell runs)"] = {"n": len(cs), "n_damaged_exact": int(de.sum()), "n_damaged_cell": int(dc.sum()),
                                                   "auroc_exact": auroc(-phi, de), "thr_exact": best_threshold(phi, de),
                                                   "auroc_cell": auroc(-phi, dc), "thr_cell": best_threshold(phi, dc)}
    # one global threshold across every group, exact-retained unit
    allc = [c for t in TASKS for c in allcells.get(t, [])] + [{"phi": c["phi"], "ret": c["ret"]} for c in dtc]
    if allc:
        phi = np.array([c["phi"] for c in allc]); dmg = np.array([c["ret"] < 0.9 for c in allc])
        g = best_threshold(phi, dmg)
        filt["global (all groups)"] = {"n": len(allc), "n_damaged_exact": int(dmg.sum()), "auroc_exact": auroc(-phi, dmg), "thr_exact": g}
        # accuracy of the global threshold inside each group vs the group's own threshold
        for gname, v in filt.items():
            if gname.startswith("global"):
                continue
            if gname.startswith("DT") and "per-cell" not in gname:
                cs = [c for c in dtc if f"DT maze {c['size']}" == gname]
            elif gname in groups:
                cs = [c for t in groups[gname] for c in allcells.get(t, [])]
            else:
                continue
            ph = np.array([c["phi"] for c in cs]); dm = np.array([c["ret"] < 0.9 for c in cs])
            pred = ph < g[0]
            v["global_thr_acc"] = float((pred == dm).mean())
            v["global_thr_bal"] = float(((pred & dm).sum() / max(dm.sum(), 1) + (~pred & ~dm).sum() / max((~dm).sum(), 1)) / 2)
    res["filter"] = filt
    return res


def print_fid(res):
    print("\n### Fidelity vs accuracy retained, per model (breadth cells at the default H (TRM 3, HRM 2), nsup 1-64, quantizers, weight and activation noise, draws; each paired with fp32 at the same H and nsup)")
    print("| model | cells | in-basin (ret ≥ 0.9) | Spearman all [95% boot] | Spearman in-basin [95% boot] | near-baseline (ret ≥ 0.97): n / ρ | AUROC φ for damaged | nsup 16 only: n / in / ρ in-basin / near n, ρ |")
    print("|---|---|---|---|---|---|---|---|")
    for t, v in res["per_model"].items():
        dd = v["default_depth"]
        print(f"| {t} | {v['n_cells']} | {v['n_in']} | {fmt(v['rho_all'])} [{fmt(v['ci_all'][0])},{fmt(v['ci_all'][1])}] | "
              f"{fmt(v['rho_in'])} [{fmt(v['ci_in'][0])},{fmt(v['ci_in'][1])}] | {v['n_near']} / {fmt(v['rho_near'])} [{fmt(v['ci_near'][0])},{fmt(v['ci_near'][1])}] | {fmt(v['auroc_damaged'])} | "
              f"{dd['n']} / {dd['n_in']} / {fmt(dd['rho_in'])} [{fmt(dd['ci_in'][0])},{fmt(dd['ci_in'][1])}] / {dd['n_near']}, {fmt(dd['rho_near'])} [{fmt(dd['ci_near'][0])},{fmt(dd['ci_near'][1])}] |")
    print("\n### Filter: one φ threshold per output length (damaged = exact retained < 0.9; cell unit: per-cell retained < 0.99)")
    print("| group | cells | damaged (exact) | AUROC | best t / acc / bal.acc | global t: acc / bal | damaged (cell) | AUROC cell | best t cell / bal |")
    print("|---|---|---|---|---|---|---|---|---|")
    for g, v in res["filter"].items():
        te = v.get("thr_exact"); tc = v.get("thr_cell")
        print(f"| {g} | {v['n']} | {v.get('n_damaged_exact')} | {fmt(v.get('auroc_exact'))} | "
              f"{fmt(te[0], 3)} / {fmt(te[1])} / {fmt(te[2])} | {fmt(v.get('global_thr_acc'))} / {fmt(v.get('global_thr_bal'))} | "
              f"{v.get('n_damaged_cell', '--')} | {fmt(v.get('auroc_cell'))} | {fmt(tc[0], 3) if tc else '--'} / {fmt(tc[2]) if tc else '--'} |")
        if "per_model_thr_exact" in v:
            print("|  per model | " + "; ".join(f"{m}: t={fmt(x[0], 3)} bal={fmt(x[2])}" for m, x in v["per_model_thr_exact"].items()) + " |||||||||")


# ----------------------------------------------------------------------------- 7. independence
def length_bins(L, nb=4):
    qs = np.quantile(L, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(qs, L, side="right") - 1, 0, nb - 1)
    return b, qs


def compute_indep():
    out = {}
    for task in TASKS:
        Ld = load_det(task)
        if Ld is None:
            continue
        meta, recs = Ld
        kind = KIND[task]
        n = len(meta["n_open"] if kind == "maze" else meta["n_blank"])
        if kind == "sudoku":
            nb = np.asarray(meta["n_blank"])
            variants = [("blanks", nb, nb, "n_wrong"), ("blanks; all 81 cells", nb, np.full(n, 81), "n_wrong_all")]
        else:
            no, pl = np.asarray(meta["n_open"]), np.asarray(meta["ref_plen"])
            variants = [("open cells", no, no, "n_wrong"), ("path length", pl, no, "n_wrong"),
                        ("path length; all 900 cells", pl, np.full(n, 900), "n_wrong_all")]
        tout = {}
        for lname, Lv, Lans, nwkey in variants:
            # product model: exact = prod over the L answer cells of p_cell (independent errors)
            bins, qs = length_bins(Lv)
            rows = []
            for r in recs:
                f = r["final"]
                nw = np.asarray(f[nwkey], float)
                corr = np.asarray(f["correct"], bool)
                row = {"spec": r["spec"], "bins": []}
                for b in range(len(qs) - 1):
                    m = bins == b
                    if not m.any():
                        continue
                    p = 1 - nw[m].sum() / Lans[m].sum()
                    pred = float(np.mean(p ** Lans[m]))
                    obs = float(corr[m].mean())
                    # dispersion of wrong-cell counts vs binomial(L, 1-p)
                    var_b = float(np.mean(Lans[m] * p * (1 - p)))
                    disp = float(nw[m].var() / var_b) if var_b > 0 else None
                    frac_wrong_cells_in_top10 = None
                    if nw[m].sum() > 0:
                        srt = np.sort(nw[m])[::-1]
                        k = max(1, int(round(0.1 * m.sum())))
                        frac_wrong_cells_in_top10 = float(srt[:k].sum() / srt.sum())
                    row["bins"].append({"range": [float(qs[b]), float(qs[b + 1])], "n": int(m.sum()), "p_cell": float(p),
                                        "obs_exact": obs, "pred_exact": pred, "ratio": obs / pred if pred > 0 else None,
                                        "dispersion": disp, "wrong_cells_in_worst10pct": frac_wrong_cells_in_top10,
                                        "mean_wrong_given_wrong": float(nw[m][~corr[m]].mean()) if (~corr[m]).any() else None})
                rows.append(row)
            tout[lname] = rows
        out[task] = tout
    # deep-thinking mazes (per-cell accuracy only in aggregate): prediction with the run's p_cell
    _, dtcell = dt_cells()
    dt = []
    for c in dtcell:
        if c["correct"] is None or c["n_target_cells"] is None:
            continue
        Lt = np.asarray(c["n_target_cells"], float)
        corr = np.asarray(c["correct"], bool)
        bins, qs = length_bins(Lt)
        rowb = []
        for b in range(len(qs) - 1):
            m = bins == b
            p = c["cell_target"]
            rowb.append({"range": [float(qs[b]), float(qs[b + 1])], "n": int(m.sum()), "obs_exact": float(corr[m].mean()),
                         "pred_exact_target": float(np.mean(p ** Lt[m])) if p is not None else None})
        dt.append({"size": c["size"], "iters": c["iters"], "spec": c["spec"], "seed": c["seed"], "cell": c["cell"],
                   "cell_target": c["cell_target"], "exact": c["exact"], "bins": rowb})
    out["dt_mazes"] = dt
    return out


def print_indep(res, specs=None):
    for task, tout in res.items():
        if task == "dt_mazes":
            print("\n### Deep-thinking mazes: observed exact vs p_target^L (L = target cells), by L quartile")
            print("| size | iters | spec | seed | cell | cell target | obs / pred per quartile |")
            print("|---|---|---|---|---|---|---|")
            for r in tout:
                print(f"| {r['size']} | {r['iters']} | {r['spec']} | {r['seed']} | {100*r['cell']:.2f} | {fmt(None if r['cell_target'] is None else 100*r['cell_target'])} | "
                      + "; ".join(f"{100*b['obs_exact']:.1f}/{fmt(None if b['pred_exact_target'] is None else 100*b['pred_exact_target'], 1)}" for b in r["bins"]) + " |")
            continue
        for lname, rows in tout.items():
            print(f"\n### {task}: observed exact vs Π p_cell^L (L = answer cells), by {lname} quartile. cell = obs% / pred% (dispersion; share of wrong cells in worst 10% puzzles)")
            print("| spec | Q1 | Q2 | Q3 | Q4 |")
            print("|---|---|---|---|---|")
            for r in rows:
                if specs and base_spec(r["spec"]) not in specs:
                    continue
                print(f"| {r['spec']} | " + " | ".join(
                    f"{100*b['obs_exact']:.1f} / {100*b['pred_exact']:.1f} ({fmt(b['dispersion'], 1)}; {fmt(b['wrong_cells_in_worst10pct'])})" for b in r["bins"]) + " |")


# ----------------------------------------------------------------------------- 8. PTRM
def compute_ptrm():
    out = {}
    for f in sorted(glob.glob(str(X8 / "ptrm_*.json"))):
        if "smoke" in f:
            continue
        j = json.load(open(f))
        key = f"{j['task']}|{j['noise']}|{j.get('wspec', 'fp32')}"
        rows = []
        for r in sorted(j["records"], key=lambda x: x["sigma"]):
            s = r["summary"]
            K = r["K"]
            rows.append({"sigma": r["sigma"], "K": K, "single_mean": s["single_mean"], "single_sd": s["single_sd"],
                         "single_min": s["single_min"], "single_max": s["single_max"],
                         "pass": s[str(K)]["pass"], "bestq": s[str(K)]["bestq"], "mode": s[str(K)]["mode"],
                         "curve": {k: v for k, v in s.items() if k.isdigit()}, "zL_rms": r.get("zL_rms")})
        out[key] = {"n": j["n"], "nsup": j["nsup"], "rows": rows}
    return out


def print_ptrm(res):
    for key, v in res.items():
        print(f"\n### {key}  (n={v['n']}, D={v['nsup']})")
        print("| σ | K | single rollout mean ± sd [min,max] | pass@K | best-Q@K | mode@K | z_L rms |")
        print("|---|---|---|---|---|---|---|")
        for r in v["rows"]:
            print(f"| {r['sigma']} | {r['K']} | {100*r['single_mean']:.1f} ± {100*r['single_sd']:.1f} [{100*r['single_min']:.1f},{100*r['single_max']:.1f}] | "
                  f"{100*r['pass']:.1f} | {100*r['bestq']:.1f} | {100*r['mode']:.1f} | {fmt(r['zL_rms'], 3)} |")


# ----------------------------------------------------------------------------- main
def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("auroc", "all"):
        print_auroc(compute_auroc())
    if cmd in ("select", "all"):
        print_select(compute_select())
    if cmd in ("basin", "all"):
        print_basin(compute_basin())
    if cmd in ("early", "all"):
        print_early(compute_early())
    if cmd in ("checker", "all"):
        print_checker(compute_checker())
    if cmd in ("fid", "all"):
        print_fid(compute_fid())
    if cmd in ("indep", "all"):
        print_indep(compute_indep())
    if cmd in ("ptrm", "all"):
        print_ptrm(compute_ptrm())
    if cmd == "json":
        sel = compute_select()
        doc = {"auroc": compute_auroc(), "select": sel, "basin": compute_basin(sel), "early": compute_early(),
               "checker": compute_checker(), "fid": compute_fid(), "indep": compute_indep(), "ptrm": compute_ptrm(),
               "conventions": __doc__}
        p = B / "x8_summary.json"
        json.dump(doc, open(p, "w"), default=lambda o: None if o is None else (float(o) if isinstance(o, (np.floating,)) else int(o) if isinstance(o, np.integer) else str(o)))
        print("wrote", p)


if __name__ == "__main__":
    main()
