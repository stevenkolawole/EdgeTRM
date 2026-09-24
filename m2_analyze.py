#!/usr/bin/env python3
"""M2 read-out: selection rules over K rollouts per cell, matched compute, calibration.

Cells are the .npz files written by m2_bestofk.py (index: JSON lines).  Selection rules, applied at
step d (<= D) over the first k rollouts:
  random   expected accuracy of one rollout picked uniformly (= mean single-rollout accuracy)
  maxq     rollout with the highest halting logit (PTRM's best-Q@K)
  vote     majority vote over answer hashes (ties -> first rollout with a modal answer, as X8)
  qvote    answer with the largest sum of sigmoid(q) over the rollouts that produced it
  hfirst   halt-first (Dillon 2601.19085): the rollout whose q first crosses 0, its answer at that
           step; ties by q; if none crosses, maxq at d
  chk      checker oracle: a rollout whose label-free checker passes (max-q among them), else maxq.
           Sudoku: valid completed grid (= exact on unique solutions).  Maze: valid simple S-G path.
  pass     any rollout correct (upper bound)
Maze answers are scored three ways (exact / valid / vshort = valid and as short as BFS).

    python m2_analyze.py table   --root results/m2   [--task sudoku_mlp]
    python m2_analyze.py compute --root results/m2
    python m2_analyze.py calib   --root results/m2
    python m2_analyze.py json    --root results/m2   (figure-ready results/m2/m2_summary.json)
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

RULES = ("random", "maxq", "vote", "qvote", "hfirst", "chk", "pass")


def load_index(root):
    recs = []
    for f in sorted(Path(root).glob("index*.jsonl")):
        for line in open(f):
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    # de-duplicate by file name (first wins)
    seen, out = set(), []
    for r in recs:
        if r["file"] in seen:
            continue
        seen.add(r["file"])
        out.append(r)
    return out


def load_cell(root, rec):
    for sub in ("cells", "."):
        p = Path(root) / sub / rec["file"]
        if p.exists():
            return dict(np.load(p))
    raise FileNotFoundError(rec["file"])


def _mode_pick(h):
    """h: [k, n] hashes -> index of first rollout carrying a modal answer, per puzzle."""
    k, n = h.shape
    eq = (h[:, None, :] == h[None, :, :]).sum(1)                      # [k, n] votes for each rollout's answer
    top = eq.max(0)
    return np.argmax(eq == top[None, :], axis=0)


def _qvote_pick(h, q):
    k, n = h.shape
    w = 1.0 / (1.0 + np.exp(-q.astype(np.float64)))
    same = (h[:, None, :] == h[None, :, :])                           # [k, k, n]
    score = (same * w[None, :, :]).sum(1)                             # summed weight of each rollout's answer
    return np.argmax(score, axis=0)


def select(cell, kind, k, d, rules=RULES, src="main", polish_step=None, sel_by="own"):
    """Returns {rule: {metric: per-puzzle bool array}} for the first k rollouts at step d.
    src='pol': use the fp32-polished arrays at polish_step (1-based); sel_by 'own' uses the polished
    (fp32 head) q, 'comp' uses the compressed model's q at d for choosing, then scores the polished answer."""
    if src == "main":
        C = cell["corr"][:k, :, d - 1].astype(bool)
        Q = cell["q"][:k, :, d - 1].astype(np.float32)
        Hh = cell[f"hash_{d}"][:k]
        CH = cell[f"chk_{d}"][:k]
        VS = cell[f"vshort_{d}"][:k].astype(bool) if kind == "maze" else None
        Qsel = Q
    else:
        p = polish_step - 1
        C = cell["pol_corr"][:k, :, p].astype(bool)
        Q = cell["pol_q"][:k, :, p].astype(np.float32)
        Hh = cell["pol_hash"][:k, :, p]
        CH = cell["pol_chk"][:k, :, p]
        VS = cell["pol_vshort"][:k, :, p].astype(bool) if kind == "maze" else None
        Qsel = Q if sel_by == "own" else cell["q"][:k, :, d - 1].astype(np.float32)
    n = C.shape[1]
    ar = np.arange(n)
    valid = CH == 0
    metrics = {"exact": C}
    if kind == "maze":
        metrics.update({"valid": valid, "vshort": VS})
    out = {}
    picks = {}
    if "maxq" in rules or "chk" in rules or "hfirst" in rules:
        picks["maxq"] = Qsel.argmax(0)
    if "vote" in rules:
        picks["vote"] = _mode_pick(Hh)
    if "qvote" in rules:
        picks["qvote"] = _qvote_pick(Hh, Qsel)
    if "chk" in rules:
        qq = np.where(valid, Qsel, -np.inf)
        pk = qq.argmax(0)
        picks["chk"] = np.where(valid.any(0), pk, picks["maxq"])
    for r in rules:
        if r == "random":
            out[r] = {mname: M.mean(0) for mname, M in metrics.items()}          # expected, float per puzzle
        elif r == "pass":
            out[r] = {mname: M.any(0) for mname, M in metrics.items()}
        elif r == "hfirst":
            if src != "main":
                continue
            qs = cell["q"][:k, :, :d].astype(np.float32)                         # [k, n, d]
            crossed = qs > 0
            first = np.where(crossed.any(2), crossed.argmax(2), 10 ** 6)          # [k, n]
            best_t = first.min(0)
            cand = first == best_t[None, :]
            qd = np.where(cand, Q, -np.inf)
            pk = qd.argmax(0)
            pk = np.where(best_t < 10 ** 6, pk, picks["maxq"])
            tt = np.where(best_t < 10 ** 6, best_t, d - 1)
            ex = cell["corr"][pk, ar, tt].astype(bool)
            out[r] = {"exact": ex, "mean_step": float(np.mean(tt + 1))}
        else:
            pk = picks[r]
            out[r] = {mname: M[pk, ar] for mname, M in metrics.items()}
    return out


def boot_ci(x, B=1000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def auroc(score, label):
    """P(score of a positive > score of a negative), ties 1/2; label bool."""
    s = np.asarray(score, float)
    y = np.asarray(label, bool)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ss = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def within_puzzle_auroc(C, Q):
    """Selection AUROC: over puzzles with both correct and wrong rollouts, P(q_correct > q_wrong)."""
    num, den = 0.0, 0
    for i in range(C.shape[1]):
        c, q = C[:, i], Q[:, i]
        if c.all() or (~c).all():
            continue
        qp, qn = q[c], q[~c]
        gt = (qp[:, None] > qn[None, :]).sum() + 0.5 * (qp[:, None] == qn[None, :]).sum()
        num += gt
        den += len(qp) * len(qn)
    return float(num / den) if den else float("nan"), den


def ece(p, y, bins=10):
    p = np.asarray(p, float); y = np.asarray(y, float)
    e = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & (p < hi) if b < bins - 1 else (p >= lo) & (p <= hi)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def kind_of(task):
    return "maze" if "maze" in task else "sudoku"


def group_specs(recs):
    """Group noise draws: spec 'noise0.15@s1' -> base 'noise0.15'."""
    g = defaultdict(list)
    for r in recs:
        base = r["spec"].split("@s")[0]
        g[(r["task"], base, r["noise"], r["sigma"], r["D"], r["K"], r.get("polish", 0))].append(r)
    return g


def fmt(x):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:5.1f}"


def cmd_table(a):
    recs = [r for r in load_index(a.root) if (not a.task or r["task"] == a.task)]
    g = group_specs(recs)
    # deterministic baselines (sigma 0, K 1) per (task, spec base)
    rows = []
    for key in sorted(g, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5])):
        task, base, noise, sg, D, K, P = key
        if K == 1 or D != a.D:
            continue
        kind = kind_of(task)
        per = defaultdict(list)
        for r in g[key]:
            cell = load_cell(a.root, r)
            for k in [kk for kk in (1, 2, 4, 8, 16, 32, 64) if kk <= K]:
                s = select(cell, kind, k, D)
                for rule, mm in s.items():
                    mk = a.metric if kind == "maze" else "exact"
                    per[(k, rule)].append(float(np.mean(mm[mk])) if mk in mm else float("nan"))
        det = [r2 for r2 in recs if r2["task"] == task and r2["spec"].split("@s")[0] == base and r2["K"] == 1 and r2["D"] >= D and r2["n"] == g[key][0]["n"]]
        detacc = []
        for r2 in det:
            c2 = load_cell(a.root, r2)
            if kind == "maze" and a.metric != "exact":
                if f"chk_{D}" in c2:
                    detacc.append(float((c2[f"chk_{D}"][0] == 0).mean() if a.metric == "valid" else c2[f"vshort_{D}"][0].mean()))
            else:
                detacc.append(float(c2["corr"][0, :, D - 1].mean()))
        ks = [kk for kk in (1, 2, 4, 8, 16, 32, 64) if kk <= K]
        line = f"{task:11s} {base:11s} {noise}{sg:<4g} draws={len(g[key])} det={fmt(np.mean(detacc)) if detacc else '  -  '} | "
        line += " ".join(f"K{k}:" + "/".join(fmt(np.mean(per[(k, r)])) for r in ("maxq", "vote", "qvote", "pass")) for k in ks if k in (1, 4, 16, 32))
        line += f" | rand {fmt(np.mean(per[(K, 'random')]))} hfirst {fmt(np.mean(per[(K, 'hfirst')]))} chk {fmt(np.mean(per[(K, 'chk')]))}"
        rows.append(line)
    print(f"metric={a.metric} D={a.D}; per K: maxQ/vote/qvote/pass")
    print("\n".join(rows))


def cmd_calib(a):
    recs = [r for r in load_index(a.root) if (not a.task or r["task"] == a.task) and r["K"] > 1 and r["D"] == a.D]
    print("task spec noise sigma | single, AUROC(q) pooled, within-puzzle sel. AUROC (pairs), ECE, frac q>0 among wrong, sel.eff = (maxq-rand)/(pass-rand)")
    for r in recs:
        cell = load_cell(a.root, r)
        kind = kind_of(r["task"])
        D = r["D"]
        C = cell["corr"][:, :, D - 1].astype(bool)
        Q = cell["q"][:, :, D - 1].astype(np.float32)
        if kind == "maze" and a.metric != "exact":
            C = (cell[f"chk_{D}"] == 0) if a.metric == "valid" else cell[f"vshort_{D}"].astype(bool)
        au = auroc(Q.ravel(), C.ravel())
        wp, pairs = within_puzzle_auroc(C, Q)
        e = ece(1 / (1 + np.exp(-Q.ravel())), C.ravel())
        fp = float((Q[~C] > 0).mean()) if (~C).any() else float("nan")
        s = select(cell, kind, r["K"], D, rules=("random", "maxq", "pass"))
        mk = a.metric if kind == "maze" else "exact"
        rnd, mq, ps = (float(np.mean(s[x][mk])) for x in ("random", "maxq", "pass"))
        eff = (mq - rnd) / (ps - rnd) if ps > rnd else float("nan")
        print(f"{r['task']:11s} {r['spec']:13s} {r['noise']}{r['sigma']:<4g} K={r['K']:2d} | single {fmt(C.mean())} AUROC {au:.3f} "
              f"within {wp:.3f} ({pairs}) ECE {e:.3f} FP {fmt(fp)} eff {eff:.2f}")


def cmd_compute(a):
    """Matched compute: seq (noise none, K 1, D big) vs par (D16, K=S/16) vs hybrid (D64 or D32)."""
    recs = load_index(a.root)
    by = defaultdict(list)
    for r in recs:
        by[(r["task"], r["spec"])].append(r)
    for (task, spec), rs in sorted(by.items()):
        seq = [r for r in rs if r["noise"] == "none" and r["D"] >= 64]
        par = [r for r in rs if r["K"] > 1 and r["D"] == 16 and r.get("polish", 0) == 0 and (a.sigma is None or abs(r["sigma"] - a.sigma) < 1e-9)]
        hyb = [r for r in rs if r["K"] > 1 and r["D"] >= 32 and (a.sigma is None or abs(r["sigma"] - a.sigma) < 1e-9)]
        if not seq or not par:
            continue
        kind = kind_of(task)
        sc = load_cell(a.root, seq[0])
        pc = load_cell(a.root, par[0])
        hc = load_cell(a.root, hyb[0]) if hyb else None
        Dseq = seq[0]["D"]
        print(f"{task} {spec}  (seq D={Dseq}; par sigma={par[0]['sigma']} K={par[0]['K']}; hybrid {'D%d K%d' % (hyb[0]['D'], hyb[0]['K']) if hyb else '-'})")
        print("   S   | seq last  seq maxq-step | par maxq  par vote  par pass | hyb(D32) maxq | hyb(D64) maxq")
        for S in (16, 32, 64, 128, 256, 512):
            if S > Dseq:
                break
            c = sc["corr"][0, :, :S].astype(bool)
            q = sc["q"][0, :, :S].astype(np.float32)
            seq_last = c[:, S - 1].mean()
            seq_q = c[np.arange(c.shape[0]), q.argmax(1)].mean()
            K = S // 16
            ptxt = "   -        -        -   "
            if K <= par[0]["K"]:
                s = select(pc, kind, K, 16, rules=("maxq", "vote", "pass"))
                ptxt = f"{fmt(np.mean(s['maxq']['exact']))}    {fmt(np.mean(s['vote']['exact']))}    {fmt(np.mean(s['pass']['exact']))}"
            h32 = h64 = "  -  "
            if hc is not None:
                for dd in (32, 64):
                    kk = S // dd
                    if dd <= hyb[0]["D"] and 1 <= kk <= hyb[0]["K"] and f"hash_{dd}" in hc:
                        s = select(hc, kind, kk, dd, rules=("maxq",))
                        if dd == 32:
                            h32 = fmt(np.mean(s["maxq"]["exact"]))
                        else:
                            h64 = fmt(np.mean(s["maxq"]["exact"]))
            print(f"  {S:4d} | {fmt(seq_last)}     {fmt(seq_q)}     | {ptxt} | {h32}         | {h64}")


def cmd_json(a):
    recs = load_index(a.root)
    out = []
    for r in recs:
        cell = load_cell(a.root, r)
        kind = kind_of(r["task"])
        D = r["D"]
        e = {kk: r[kk] for kk in ("task", "spec", "wseed", "noise", "sigma", "K", "D", "polish", "n", "gpu", "host")}
        e["zl_rms"] = cell["zl_rms"].tolist()
        e["per_step_single"] = cell["corr"].mean((0, 1)).tolist()
        if r["K"] > 1 and f"hash_{D}" in cell:
            e["sel"] = {}
            for k in [kk for kk in (1, 2, 4, 8, 16, 32, 64) if kk <= r["K"]]:
                s = select(cell, kind, k, D)
                e["sel"][k] = {rule: {m: float(np.mean(v)) for m, v in mm.items() if m != "mean_step"} for rule, mm in s.items()}
            if r.get("polish"):
                e["polish_sel"] = {}
                for p in range(1, r["polish"] + 1):
                    for sb in ("own", "comp"):
                        s = select(cell, kind, r["K"], D, src="pol", polish_step=p, sel_by=sb,
                                   rules=("random", "maxq", "vote", "qvote", "chk", "pass"))
                        e["polish_sel"][f"P{p}_{sb}"] = {rule: {m: float(np.mean(v)) for m, v in mm.items()} for rule, mm in s.items()}
        out.append(e)
    dst = Path(a.root) / "m2_summary.json"
    json.dump(out, open(dst, "w"))
    print("wrote", dst, len(out), "cells")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("table", "calib", "compute", "json"))
    ap.add_argument("--root", default="results/m2")
    ap.add_argument("--task", default="")
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--metric", default="exact", choices=("exact", "valid", "vshort"))
    ap.add_argument("--sigma", type=float, default=None)
    a = ap.parse_args()
    {"table": cmd_table, "calib": cmd_calib, "compute": cmd_compute, "json": cmd_json}[a.cmd](a)


if __name__ == "__main__":
    main()
