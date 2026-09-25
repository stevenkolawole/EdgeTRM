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
  pass     any rollout correct at step d (upper bound for rules that read step d)
  maxq_any the (rollout, step <= d) pair with the highest halting logit, its answer at that step
           (selection over steps as well as rollouts; K = 1 is sequential selection over steps)
  pass_any any rollout correct at any step <= d (upper bound for maxq_any / hfirst)
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

RULES = ("random", "maxq", "vote", "qvote", "hfirst", "maxq_any", "chk", "pass", "pass_any")


OFFSET = 0          # test-row offset of the cells a command reads (set from --offset; held-out sets use 1000)


def load_index(root, any_offset=False):
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
        if r.get("tta") and "_v2" not in r["file"]:
            continue                      # TTA cells before the symmetry fix (Sep 25 10:03 UTC) are invalid
        if any_offset or r.get("offset", 0) == OFFSET:
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
        elif r in ("maxq_any", "pass_any"):
            if src != "main":
                continue
            ca = cell["corr"][:k, :, :d].astype(bool)
            if r == "pass_any":
                out[r] = {"exact": ca.any((0, 2))}
            else:
                qa = cell["q"][:k, :, :d].astype(np.float32)
                flat = qa.transpose(1, 0, 2).reshape(n, -1).argmax(1)
                kk, tt = np.unravel_index(flat, (k, d))
                out[r] = {"exact": ca[kk, ar, tt], "mean_step": float(np.mean(tt + 1))}
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
        g[(r["task"], base, r["noise"], r["sigma"], r["D"], r["K"], r.get("polish", 0), r["n"])].append(r)
    return g


def fmt(x):
    return "  -  " if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:5.1f}"


def cmd_table(a):
    recs = [r for r in load_index(a.root) if (not a.task or r["task"] == a.task)]
    g = group_specs(recs)
    # deterministic baselines (sigma 0, K 1) per (task, spec base)
    rows = []
    for key in sorted(g, key=lambda k: (k[0], k[1], k[2], k[3], k[4], k[5])):
        task, base, noise, sg, D, K, P, n_ = key
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
        line = f"{task:11s} {base:11s} {noise}{sg:<4g} n={n_} draws={len(g[key])} det={fmt(np.mean(detacc)) if detacc else '  -  '} | "
        line += " ".join(f"K{k}:" + "/".join(fmt(np.mean(per[(k, r)])) for r in ("maxq", "vote", "qvote", "pass")) for k in ks if k in (1, 4, 16, 32))
        line += f" | rand {fmt(np.mean(per[(K, 'random')]))} hfirst {fmt(np.mean(per[(K, 'hfirst')]))} maxq_any {fmt(np.mean(per[(K, 'maxq_any')]))} pass_any {fmt(np.mean(per[(K, 'pass_any')]))} chk {fmt(np.mean(per[(K, 'chk')]))}"
        line += f" || K1 hfirst {fmt(np.mean(per[(1, 'hfirst')]))} maxq_any {fmt(np.mean(per[(1, 'maxq_any')]))} pass_any {fmt(np.mean(per[(1, 'pass_any')]))}"
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


def slice_cell(cell, n):
    """First n puzzles of a cell (every cell evaluates the first rows of the test set)."""
    out = {}
    for k, v in cell.items():
        if k in ("zl_rms",):
            out[k] = v
        elif k == "bfs":
            out[k] = v[:n]
        elif v.ndim >= 2:
            out[k] = v[:, :n]
        else:
            out[k] = v
    return out


def cmd_compute(a):
    """Matched compute S = K x D supervision steps per puzzle, on the same puzzles (first n of the smaller cell):
    seq = one deterministic trajectory of S steps (answer at S, or at the max-q step <= S);
    par = K = S/16 noisy rollouts of 16 steps (best-Q; best-Q over (rollout, step); pass@K);
    hyb = K = S/D rollouts of D = 32 or 64 steps (best-Q over (rollout, step))."""
    recs = load_index(a.root)
    by = defaultdict(list)
    for r in recs:
        by[(r["task"], r["spec"])].append(r)
    for (task, spec), rs in sorted(by.items()):
        if a.task and task != a.task:
            continue
        seqs = sorted([r for r in rs if r["noise"] == "none" and r["K"] == 1], key=lambda r: -r["D"])
        pars = sorted([r for r in rs if r["K"] > 1 and r["D"] == 16 and abs(r["sigma"] - a.sigma_main) < 1e-9 and r.get("wsample", "none") == "none"], key=lambda r: -r["K"])
        hybs = sorted([r for r in rs if r["K"] > 1 and r["D"] > 16 and abs(r["sigma"] - a.sigma_main) < 1e-9 and r.get("wsample", "none") == "none"], key=lambda r: -r["D"])
        if not seqs or not pars:
            continue
        kind = kind_of(task)
        mk = a.metric if kind == "maze" else "exact"
        sq, pr = seqs[0], pars[0]
        n = min(sq["n"], pr["n"], *(h["n"] for h in hybs)) if hybs else min(sq["n"], pr["n"])
        sc, pc = slice_cell(load_cell(a.root, sq), n), slice_cell(load_cell(a.root, pr), n)
        hc = slice_cell(load_cell(a.root, hybs[0]), n) if hybs else None
        Dseq = sq["D"]
        print(f"{task} {spec}: n={n}; seq D={Dseq}; par K={pr['K']} (sigma {pr['sigma']}); hyb {'D%d K%d' % (hybs[0]['D'], hybs[0]['K']) if hybs else '-'}; metric {mk}")
        print("     S | seq@S  seq maxq<=S | par bestQ  par maxq_any  par pass | hyb D32 maxq_any | hyb D64 maxq_any")
        ar = np.arange(n)
        for S in (16, 32, 64, 128, 256, 512):
            if S > max(Dseq, 16 * pr["K"]):
                break
            if S <= Dseq:
                c = sc["corr"][0, :, :S].astype(bool)
                q = sc["q"][0, :, :S].astype(np.float32)
                s1, s2 = fmt(c[:, S - 1].mean()), fmt(c[ar, q.argmax(1)].mean())
                if kind == "maze" and mk != "exact" and f"chk_{S}" in sc:
                    s1 = fmt(((sc[f"chk_{S}"][0] == 0) if mk == "valid" else sc[f"vshort_{S}"][0].astype(bool)).mean())
                    s2 = "  -  "
            else:
                s1 = s2 = "  -  "
            K = S // 16
            ptxt = "  -        -          -   "
            if K <= pr["K"]:
                rules = ("maxq", "maxq_any", "pass") if mk == "exact" else ("maxq", "pass")
                sel = select(pc, kind, K, 16, rules=rules)
                ptxt = f"{fmt(np.mean(sel['maxq'][mk]))}      {fmt(np.mean(sel['maxq_any'][mk])) if 'maxq_any' in sel else '  -  '}     {fmt(np.mean(sel['pass'][mk]))}"
            hv = {}
            if hc is not None and mk == "exact":
                for dd in (32, 64):
                    kk = S // dd
                    if dd <= hybs[0]["D"] and 1 <= kk <= hybs[0]["K"]:
                        hv[dd] = fmt(np.mean(select(hc, kind, kk, dd, rules=("maxq_any",))["maxq_any"]["exact"]))
            print(f"  {S:4d} | {s1}  {s2}      | {ptxt} | {hv.get(32, '  -  ')}            | {hv.get(64, '  -  ')}")


def _det_stats(root, recs, task, spec, n):
    """Deterministic cell (noise none, D >= 16): exact at step 16, at the max-q step <= 16, any step <= 16, at step 64."""
    for r in recs:
        if r["task"] == task and r["spec"] == spec and r["noise"] == "none" and r["D"] >= 16 and r["n"] == n and r["K"] == 1:
            c = load_cell(root, r)
            C = c["corr"][0].astype(bool); Q = c["q"][0].astype(np.float32)
            ar = np.arange(C.shape[0])
            out = {"s16": C[:, 15].mean(), "maxq16": C[ar, Q[:, :16].argmax(1)].mean(), "any16": C[:, :16].any(1).mean()}
            if r["D"] >= 64:
                out.update({"s64": C[:, 63].mean(), "maxq64": C[ar, Q[:, :64].argmax(1)].mean()})
            return out
    return None


def cmd_main(a):
    """Headline table: per model and spec base, draws averaged (range in brackets).
    det = deterministic step 16 | det max-Q step | best-Q@K | maxq_any@K | pass@K | vote@K | fp32 polish."""
    recs = load_index(a.root)
    main = [r for r in recs if r["noise"] == "ptrm" and abs(r["sigma"] - a.sigma_main) < 1e-9 and r["D"] == 16 and r["n"] == a.n
            and r.get("polish", 0) == a.main_polish and r.get("polish_from", "last") == "last" and r.get("wsample", "none") == "none"]
    tasks = sorted({r["task"] for r in main}) if not a.task else [a.task]
    Ks = (1, 4, 8, 16, 32)
    for task in tasks:
        kind = kind_of(task)
        mk = a.metric if kind == "maze" else "exact"
        bases = []
        for r in main:
            if r["task"] == task and r["spec"].split("@s")[0] not in bases:
                bases.append(r["spec"].split("@s")[0])
        print(f"== {task} (sigma {a.sigma_main}, D 16, n {a.n}, metric {mk}); cells: det16 | det maxQ-step16 || best-Q@1/4/8/16/32 || maxq_any@16/32 | pass@32 | vote@16 | P2-polish best-Q (own head)@max K")
        for base in bases:
            rs = [r for r in main if r["task"] == task and r["spec"].split("@s")[0] == base]
            per = defaultdict(list)
            dets = defaultdict(list)
            for r in rs:
                cell = load_cell(a.root, r)
                for k in Ks:
                    if k <= r["K"]:
                        sel = select(cell, kind, k, 16, rules=("maxq", "maxq_any", "pass", "vote"))
                        for rule in ("maxq", "maxq_any", "pass", "vote"):
                            if mk in sel.get(rule, {}):
                                per[(rule, k)].append(float(np.mean(sel[rule][mk])))
                if r.get("polish") and "pol_corr" in cell:
                    sp = select(cell, kind, r["K"], 16, src="pol", polish_step=r["polish"], sel_by="own", rules=("maxq",))
                    per[("pol", r["K"])].append(float(np.mean(sp["maxq"][mk])))
                d = _det_stats(a.root, recs, task, r["spec"], r["n"]) if kind == "sudoku" else None
                if d:
                    for kk, vv in d.items():
                        dets[kk].append(vv)

            def f(v):
                if not v:
                    return "  -  "
                if len(v) == 1:
                    return f"{100*v[0]:5.1f}"
                return f"{100*np.mean(v):5.1f}[{100*min(v):.0f}-{100*max(v):.0f}]"
            polk = max([k for (rule, k) in per if rule == "pol"], default=None)
            print(f"  {base:11s} n_draws={len(rs)} | {f(dets['s16'])} | {f(dets['maxq16'])} || " +
                  " ".join(f(per[('maxq', k)]) for k in Ks) + f" || {f(per[('maxq_any', 16)])} {f(per[('maxq_any', 32)])} | {f(per[('pass', 32)])} | {f(per[('vote', 16)])} | {f(per[('pol', polk)]) if polk else '  -  '}")


def cmd_polish(a):
    """M1 x M2: per polish cell, exact % at step 16 (single / best-Q / max-Q (rollout, step) / pass) and after p fp32
    polish steps (single+polish / best-Q by the fp32 head / best-Q chosen by the compressed head at 16 / pass)."""
    recs = [r for r in load_index(a.root) if r.get("polish") and (not a.task or r["task"] == a.task)]
    print("task spec from K | step16: single bestQ maxq_any pass | per polish step p: single / bestQ(fp32 head) / bestQ(comp head) / pass")
    for r in sorted(recs, key=lambda r: (r["task"], r.get("polish_from", "last"), r["spec"], r["K"])):
        c = load_cell(a.root, r)
        kind = kind_of(r["task"])
        if kind != "sudoku":
            continue
        K = r["K"]
        m = select(c, kind, K, 16, rules=("random", "maxq", "maxq_any", "pass"))
        f = lambda d, rr: "%5.1f" % (100 * np.mean(d[rr]["exact"]))
        line = f"{r['task']:11s} {r['spec']:13s} {r.get('polish_from', 'last'):4s} K{K:2d} | {f(m, 'random')} {f(m, 'maxq')} {f(m, 'maxq_any')} {f(m, 'pass')} |"
        for pp in range(1, r["polish"] + 1):
            po = select(c, kind, K, 16, src="pol", polish_step=pp, rules=("random", "maxq", "pass"))
            pc = select(c, kind, K, 16, src="pol", polish_step=pp, sel_by="comp", rules=("maxq",))
            line += f" P{pp}: {f(po, 'random')}/{f(po, 'maxq')}/{f(pc, 'maxq')}/{f(po, 'pass')}"
        print(line)


def sample_until_confident(C, Q, thr, Kmax=None):
    """M2 x M5. C, Q: [K, n, D] exact and halting logit per rollout and step. Rollouts are run one after another;
    within a rollout the model exits at the first step with q > thr (M5's early exit) and that answer is returned;
    if the rollout never crosses, the next rollout starts (fresh noise). If no rollout crosses within Kmax, the answer
    with the highest q over every (rollout, step) seen is returned. Returns (exact per puzzle, steps used per puzzle)."""
    K, n, D = C.shape
    K = min(K, Kmax or K)
    C, Q = C[:K], Q[:K]
    crossed = Q > thr
    anyc = crossed.any(2)                                   # [K, n]
    first_t = np.where(anyc, crossed.argmax(2), D)          # [K, n] (D = never)
    first_k = np.where(anyc.any(0), anyc.argmax(0), K)      # [n]
    ar = np.arange(n)
    ok = first_k < K
    kk = np.minimum(first_k, K - 1)
    tt = np.minimum(first_t[kk, ar], D - 1)
    exact = np.where(ok, C[kk, ar, tt], False)
    steps = np.where(ok, first_k * D + tt + 1, K * D).astype(float)
    # fallback: max-q over everything
    flat = Q.transpose(1, 0, 2).reshape(n, -1).argmax(1)
    fk, ft = np.unravel_index(flat, (K, D))
    exact = np.where(ok, exact, C[fk, ar, ft])
    return exact.astype(bool), steps


def cmd_exit(a):
    """Sample until confident (M2 x M5): accuracy and mean supervision steps per puzzle vs threshold and K_max,
    against fixed best-Q@K (16 K steps) and the deterministic 16-step model."""
    recs = [r for r in load_index(a.root) if r["noise"] == "ptrm" and abs(r["sigma"] - a.sigma_main) < 1e-9 and r["D"] == 16
            and r["n"] == a.n and r.get("polish", 0) == a.main_polish and r.get("polish_from", "last") == "last"
            and (not a.task or r["task"] == a.task) and kind_of(r["task"]) == "sudoku" and r.get("wsample", "none") == "none"]
    print("task spec K | best-Q@K (steps 16K) | exit thr=0: K_max 1 / 4 / 16 / K  acc (mean steps) | thr=-2 K_max K | thr=+2 K_max K")
    for r in sorted(recs, key=lambda r: (r["task"], r["spec"])):
        c = load_cell(a.root, r)
        C = c["corr"].astype(bool); Q = c["q"].astype(np.float32)
        K = r["K"]
        bq = float(np.mean(select(c, "sudoku", K, 16, rules=("maxq",))["maxq"]["exact"]))
        parts = []
        for km in (1, 4, 16, K):
            if km > K:
                continue
            e, st = sample_until_confident(C, Q, 0.0, km)
            parts.append(f"{100*e.mean():5.1f} ({st.mean():5.1f})")
        e1, s1 = sample_until_confident(C, Q, -2.0, K)
        e2, s2 = sample_until_confident(C, Q, 2.0, K)
        print(f"{r['task']:11s} {r['spec']:13s} K{K:2d} | {100*bq:5.1f} ({16*K}) | " + " / ".join(parts) +
              f" | {100*e1.mean():5.1f} ({s1.mean():5.1f}) | {100*e2.mean():5.1f} ({s2.mean():5.1f})")


def cmd_wsample(a):
    """Quantization as the sampler: K stochastic-rounding draws of one master (draw 0 = round-to-nearest), each run
    deterministically (or with kicks), selected by the halting head. Matched-compute references at S = 16 K steps:
    fp32 one trajectory of S steps (seq), fp32 PTRM kicks best-Q@K (par), and PTRM kicks on the RTN model."""
    recs = load_index(a.root)
    sr = [r for r in recs if r.get("wsample", "none") != "none" and (not a.task or r["task"] == a.task)]
    Ks = (1, 2, 4, 8, 16)

    def ref_seq(task, S):
        for r in recs:
            if r["task"] == task and r["spec"] == "fp32" and r["noise"] == "none" and r["K"] == 1 and r["D"] >= S and r["n"] == a.n:
                c = load_cell(a.root, r)
                if "maze" in task and a.metric != "exact":
                    if f"chk_{S}" not in c:
                        return None
                    return float(((c[f"chk_{S}"][0] == 0) if a.metric == "valid" else c[f"vshort_{S}"][0].astype(bool)).mean())
                return float(c["corr"][0, :, S - 1].mean())
        return None

    def ref_par(task, spec, K):
        for r in recs:
            if (r["task"] == task and r["spec"] == spec and r["noise"] == "ptrm" and abs(r["sigma"] - 0.3) < 1e-9 and r["D"] == 16
                    and r["K"] >= K and r["n"] == a.n and r.get("wsample", "none") == "none" and r.get("polish_from", "last") == "last"):
                c = load_cell(a.root, r)
                return float(np.mean(select(c, kind_of(task), K, 16, rules=("maxq",))["maxq"][a.metric if "maze" in task else "exact"]))
        return None
    print(f"offset {OFFSET}, n {a.n}. per K: best-Q (max-Q over (draw, step)) [pass]; refs at S = 16K: fp32 seq / fp32 kicks best-Q / RTN-model kicks best-Q")
    for r in sorted(sr, key=lambda r: (r["task"], r["noise"], r["spec"])):
        if r["n"] != a.n:
            continue
        c = load_cell(a.root, r)
        kind = kind_of(r["task"])
        mk = a.metric if kind == "maze" else "exact"
        C = c["corr"][:, :, 15].astype(bool)
        rtn = C[0].mean() if r["wsample"] == "sr0" else float("nan")
        srs = C[1:].mean() if r["wsample"] == "sr0" else C.mean()
        print(f"{r['task']:11s} {r['spec']:6s} {r['wsample']} {r['noise']}{r['sigma']:g}: RTN det {100*rtn:5.1f}, SR single {100*srs:5.1f} [{100*C.mean(1).min():.1f}-{100*C.mean(1).max():.1f}]")
        for k in Ks:
            if k > r["K"]:
                continue
            rules = ("maxq", "maxq_any", "pass") if mk == "exact" else ("maxq", "pass")
            sel = select(c, kind, k, 16, rules=rules)
            mqa = f" ({100*np.mean(sel['maxq_any']['exact']):5.1f})" if "maxq_any" in sel else ""
            sq, pr, pq = ref_seq(r["task"], 16 * k), ref_par(r["task"], "fp32", k), ref_par(r["task"], r["spec"], k)
            f = lambda v: "  -  " if v is None else f"{100*v:5.1f}"
            print(f"    K={k:2d} S={16*k:3d}: {100*np.mean(sel['maxq'][mk]):5.1f}{mqa} [{100*np.mean(sel['pass'][mk]):5.1f}]   refs: {f(sq)} / {f(pr)} / {f(pq)}")


def abc_scores(cell, k, d=16):
    """Per puzzle over the first k draws at step d: the max-Q answer (the system's answer) and its exactness,
    agreement a = share of draws matching the majority answer, a_sel = share matching the max-Q answer, max q."""
    H = cell[f"hash_{d}"][:k]
    C = cell["corr"][:k, :, d - 1].astype(bool)
    Q = cell["q"][:k, :, d - 1].astype(np.float32)
    n = H.shape[1]; ar = np.arange(n)
    same = (H[:, None, :] == H[None, :, :]).sum(1)          # [k, n] votes for each draw's answer
    a_maj = same.max(0) / k
    pick = Q.argmax(0)
    a_sel = same[pick, ar] / k
    return {"exact": C[pick, ar], "exact_vote": C[_mode_pick(H), ar], "a": a_maj, "a_sel": a_sel, "q": Q.max(0)}


def risk_at_coverage(score, correct, covs=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
    """Answer the top-cov fraction by score (higher = more confident); risk = error rate among answered."""
    order = np.argsort(-score, kind="mergesort")
    out = []
    for c in covs:
        m = max(1, int(round(c * len(score))))
        out.append(1 - correct[order[:m]].mean())
    return out


def _pct(x):
    r = np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort")
    return r / max(1, len(x) - 1)


def cmd_abc(a):
    """ABC agreement over SR draws (Agreement-Based Cascading, 2407.02348) vs the halting head, and the cascade to fp32."""
    recs = load_index(a.root)
    want = [("sudoku_mlp", "w4c", "sr0"), ("sudoku_attn", "w3c", "sr0"), ("hrm_sudoku", "w3c", "sr0"), ("sudoku_mlp", "w3g32", "sr0"),
            ("sudoku_mlp", "fp32", "kick"), ("sudoku_attn", "fp32", "kick"), ("hrm_sudoku", "fp32", "kick")]
    covs = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
    print("risk (% wrong among answered) at coverage " + " / ".join(f"{int(100*c)}" for c in covs) + "; AUROC of the score for 'answer correct'")
    for task, spec, kind_ in want:
        if kind_ == "sr0":
            rs = [r for r in recs if r["task"] == task and r["spec"] == spec and r.get("wsample") == "sr0" and r["noise"] == "none" and r["n"] == a.n
                  and r.get("master", "fp32") == "fp32" and not r.get("tta")]
        else:
            rs = [r for r in recs if r["task"] == task and r["spec"] == spec and r["noise"] == "ptrm" and abs(r["sigma"] - 0.3) < 1e-9 and r["D"] == 16
                  and r["n"] == a.n and r.get("wsample", "none") == "none" and r.get("polish", 0) == 2]
        if not rs:
            continue
        cell = load_cell(a.root, rs[0])
        det = [r for r in recs if r["task"] == task and r["spec"] == "fp32" and r["noise"] == "none" and r["K"] == 1 and r["n"] == a.n and r["D"] >= 16]
        fp = load_cell(a.root, det[0])["corr"][0, :, 15].astype(bool) if det else None
        for k in (4, 8, 16):
            if k > rs[0]["K"]:
                continue
            sc = abc_scores(cell, k)
            corr = sc["exact"]
            combo = np.minimum(_pct(sc["a_sel"] + 1e-3 * sc["q"]), _pct(sc["q"]))
            line = f"{task:11s} {spec:5s} {kind_:4s} K={k:2d} acc {100*corr.mean():5.1f} (vote {100*sc['exact_vote'].mean():5.1f}) |"
            for nm, score in (("a", sc["a"]), ("a_sel", sc["a_sel"] + 1e-3 * sc["q"]), ("maxq", sc["q"]), ("a&q", combo)):
                rk = risk_at_coverage(score, corr, covs)
                line += f" {nm}: " + "/".join(f"{100*x:4.1f}" for x in rk) + f" AU {auroc(score, corr):.3f} |"
            print(line)
            if fp is not None and k == 16:
                # cascade: answer on device when score > theta, else defer to fp32 (16 steps); theta from rows 0-99
                cal, ev = np.arange(100), np.arange(100, len(corr))
                for nm, score in (("a", sc["a"]), ("maxq", sc["q"]), ("a&q", combo)):
                    parts = []
                    for target in (0.1, 0.2, 0.3):
                        th = np.quantile(score[cal], target)
                        dfr = score[ev] <= th
                        acc = np.where(dfr, fp[ev], corr[ev]).mean()
                        parts.append(f"defer {100*dfr.mean():4.1f}% -> {100*acc:5.1f}")
                    print(f"      cascade to fp32 (fp32 alone {100*fp[ev].mean():5.1f}, device alone {100*corr[ev].mean():5.1f}) by {nm}: " + "; ".join(parts))


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
    ap.add_argument("cmd", choices=("table", "calib", "compute", "json", "main", "polish", "exit", "wsample", "abc"))
    ap.add_argument("--sigma_main", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--main_polish", type=int, default=2, help="main grid = the P2 cells (rescue cells with P4 excluded)")
    ap.add_argument("--root", default="results/m2")
    ap.add_argument("--task", default="")
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--metric", default="exact", choices=("exact", "valid", "vshort"))
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    global OFFSET
    OFFSET = a.offset
    {"table": cmd_table, "calib": cmd_calib, "compute": cmd_compute, "json": cmd_json, "main": cmd_main, "polish": cmd_polish, "exit": cmd_exit, "wsample": cmd_wsample, "abc": cmd_abc}[a.cmd](a)


if __name__ == "__main__":
    main()
