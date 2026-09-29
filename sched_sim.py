#!/usr/bin/env python3
"""E9: loop scheduling on the existing p4d controller records (results/m5/ctl/p4d; no model is run).

A looped model is an anytime algorithm: after every supervision loop it has an answer, and its halting head says whether
that answer is final. Serving N puzzles on one device under a total loop budget B is then a loop-allocation problem:
each loop goes to one puzzle, and a puzzle whose head fired leaves the queue and frees its loops for the others.
Every policy below is a per-puzzle PROGRAM, a sequence of recorded loops (cost, correctness of the state after it),
and a scheduler is a rule for which program advances next. Round-robin sharing of one budget is a water-fill over the
programs (every live program advances one loop per round until the budget is spent); the last round is partial and
served in random order, scored as its expectation. Batches, sub-batches and bootstrap draws are puzzle weights.

Records (m5_ctl_ho.Split, w8c polish as in m5_w8.py): dc[n, T] correctness after each loop of the deterministic
trajectory (T = 64 Sudoku, 16 Maze); dq/dqc halting logits; td first loop at which the frozen head rule fires (-1 never);
pc[n, 16, 4] correctness after j = 1..4 w8c finishing loops from loop t = 1..16; rc/rq[K, n, 16] z_L-noise rollouts with
their own head, tr first firing loop per rollout; rpol = rollout end state (no w8c finishing record exists for it).
Cost model (m5_addons.b_eff): a compressed loop costs 1; a w8c finishing loop costs r = 8 / b_eff (w4: 2, w3: 2.67,
w3g32: 2.29, w2g32: 3.2, w2a: 4). Rows: Sudoku dev 0-999 / held out 1000-1999; Maze dev 0-499 / held out 500-999.
Every configuration is chosen on the dev rows at each budget and scored on the held-out rows.

Policies (budget B = b x N, b = mean loops per puzzle):
  fixed      every puzzle gets b loops (b capped at T)                       fixed_dev  fixed depth D <= b chosen on dev
  rr         round-robin anytime over the full trajectory, answer = last state (equals fixed at integer b <= T)
  head       round-robin, a puzzle leaves when its head fires, cap D0 (dev)  [E]
  ep         head + w8c finishing loops jE (fired) / jF (capped) at cost r   [E+P, config set of m5_ctl_ho]
  esp        the full ICLR controller E+S+P (rollouts for unfired puzzles, best-q fallback, finishing), replayed under
             the shared budget; while a rollout runs, the puzzle's answer is the best-q candidate completed so far
  oracle     knows the cheapest recorded path that solves each puzzle (loop t, or loop t + j finishing: the E+P family),
             serves cheapest first, gives unsolvable puzzles nothing (upper bound on fixed / rr / head / ep)
  oracle_all the same with rollouts admitted (one loop, then rollout k to its correct loop): the bound for esp
  *_alone    head / ep / esp run per puzzle with no shared budget (the m5_ctl_ho protocol: configuration with the best dev
             accuracy among those with dev MEAN honest cost <= b, every puzzle runs its program to the end); the
             scheduled minus alone difference is what the shared, truncatable budget buys
Deadline setting: Poisson arrivals at load rho (arrivals per 16 loop-times), deadline delta loops after arrival; policies
fifo16 (fixed depth 16, answer only at loop 16, admission: a puzzle that cannot finish by its deadline is dropped),
rr (processor sharing, cap 16, answer = last state), head, ep (dev configs at b = 16), oracle (shortest job first over
solvable puzzles). A loop is started only if it finishes before the puzzle's deadline. Hit = answered correctly in time.

    python sched_sim.py            # tables + results/breadth/sched_sim.json (~2 min on the CPU)
    python sched_sim.py --quick    # two named cells, fewer draws
"""
import argparse
import json
import time
from collections import deque

import numpy as np

import analyze_m5 as A
import m5_addons as AD
import m5_ctl_ho as C
import m5_w8 as W

BUD = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
NS = [64, 256, 1000]
NB = 1000
NSUB = 20
LOADS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
DELTAS = [16, 32]
N_ARR = 1000
NAMED = {"sudoku_mlp|w4c": "TRM Sudoku-MLP w4c", "hrm_maze|w2g32": "HRM Maze w2g32"}
POLICIES = ["fixed", "fixed_dev", "rr", "head", "ep", "esp", "head_alone", "ep_alone", "esp_alone", "oracle", "oracle_all"]
MAIN = ["fixed", "rr", "head", "ep", "esp", "oracle", "oracle_all"]
PAIRS = [("head", "fixed"), ("ep", "fixed"), ("esp", "fixed"), ("ep", "head"), ("esp", "head"), ("esp", "ep"), ("oracle", "ep"),
         ("oracle_all", "esp"), ("head", "fixed_dev"), ("esp", "fixed_dev"), ("head", "head_alone"), ("ep", "ep_alone"),
         ("esp", "esp_alone"), ("esp_alone", "fixed")]
ALONE = 1e12                                                        # budget that never truncates: the controller run per puzzle


# ----------------------------------------------------------------------------- programs
class Prog:
    """cost[n, L], cor[n, L], ln[n]: per-puzzle loop costs and the correctness after each loop; S/A padded cumulative tables."""

    def __init__(self, cost, cor, valid):
        order = np.argsort(~valid, axis=1, kind="stable")           # valid loops first, order kept
        cost = np.take_along_axis(np.where(valid, cost, 0.0), order, 1)
        cor = np.take_along_axis(cor & valid, order, 1)
        ln = valid.sum(1)
        L = int(ln.max()) if len(ln) else 0
        self.cost, self.cor, self.ln = cost[:, :L], cor[:, :L], ln
        n = cost.shape[0]
        self.S = np.zeros((n, L + 1))
        self.A = np.zeros((n, L + 1))
        if L:
            self.S[:, 1:] = np.cumsum(self.cost, 1)                   # spend after l rounds (flat once the program ends)
            idx = np.minimum(np.arange(1, L + 1)[None, :], np.maximum(ln, 1)[:, None]) - 1
            self.A[:, 1:] = np.take_along_axis(self.cor, idx, 1) & (ln > 0)[:, None]   # answer after l rounds


def run(prog, w, B):
    """water-fill: w[R, n] puzzle weights (counts), B[R] budgets -> acc[R] (fraction of N = w.sum(1)), cost per puzzle[R]."""
    spend = w @ prog.S                                                # [R, L+1], nondecreasing in l
    accs = w @ prog.A
    N = w.sum(1)
    B = np.broadcast_to(np.asarray(B, float), (w.shape[0],))
    l = (spend <= B[:, None] + 1e-9).sum(1) - 1                       # last full round affordable
    r = np.arange(w.shape[0])
    s0, a0 = spend[r, l], accs[r, l]
    nxt = np.minimum(l + 1, spend.shape[1] - 1)
    ds = spend[r, nxt] - s0
    frac = np.where(ds > 0, np.clip((B - s0) / np.where(ds > 0, ds, 1), 0, 1), 0.0)
    acc = a0 + frac * (accs[r, nxt] - a0)
    return acc / N, (s0 + frac * ds) / N


def build(sp, cfg, r):
    """program of one m5_ctl_ho configuration on a Split: ('fixed', D) | ('m1', t, j) | ('bestq', K, j) | ('ctl', D0, K, jE, jF)."""
    n, i = sp.n, np.arange(sp.n)
    kind = cfg[0]
    if kind == "fixed":
        D = min(cfg[1], sp.T)
        return Prog(np.ones((n, D)), sp.dc[:, :D].copy(), np.ones((n, D), bool))
    if kind == "m1":
        t, j = cfg[1], cfg[2]
        cost = np.concatenate([np.ones((n, t)), np.full((n, j), r)], 1)
        cor = np.concatenate([sp.dc[:, :t], sp.pc[:, t - 1, :j]], 1)
        return Prog(cost, cor.astype(bool), np.ones_like(cost, bool))
    if kind == "bestq":
        K, j = cfg[1], cfg[2]
        cost = np.concatenate([np.ones((n, 16 * K)), np.full((n, j), r)], 1)
        cor = np.zeros((n, 16 * K + j), bool)
        bq, bc = np.full(n, -np.inf), np.zeros(n, bool)
        for k in range(K):
            cor[:, 16 * k:16 * (k + 1)] = bc[:, None]                 # answer while rollout k runs: best-q so far
            q = sp.rq[k, :, -1]
            upd = q > bq
            bq, bc = np.where(upd, q, bq), np.where(upd, sp.rc[k, :, -1], bc)
            cor[:, 16 * k + 15] = bc                                  # rollout k complete: it is a candidate now
        if j:
            kb = sp.rq[:K, :, -1].argmax(0)
            cor[:, 16 * K:] = sp.rpol[kb, i, min(j, sp.rpol.shape[2]) - 1][:, None]
        return Prog(cost, cor, np.ones_like(cost, bool))
    assert kind == "ctl"
    D0, K, jE, jF = cfg[1:]
    td = sp.td
    fired = (td >= 0) & (td < D0)
    m = np.where(fired, td + 1, D0)                                   # deterministic loops run
    L = D0 + 4 + 16 * K + 4
    col = np.arange(L)[None, :]
    cost = np.ones((n, L))
    cor = np.zeros((n, L), bool)
    valid = np.zeros((n, L), bool)
    # deterministic block
    valid[:, :D0] = col[:, :D0] < m[:, None]
    cor[:, :D0] = sp.dc[:, :D0]
    # finishing after a head fire
    if jE:
        c0 = D0
        cost[:, c0:c0 + 4] = r
        valid[:, c0:c0 + 4] = fired[:, None] & (np.arange(4)[None, :] < jE)
        cor[:, c0:c0 + 4] = sp.pc[i, np.clip(np.where(fired, td, 0), 0, sp.P - 1), :4]
    rest = ~fired
    bq = sp.dq[:, D0 - 1].copy()
    bc = sp.dc[:, D0 - 1].copy()
    use_roll = np.zeros(n, bool)
    kb = np.zeros(n, int)
    alive = rest.copy()
    for k in range(K):
        c0 = D0 + 4 + 16 * k
        tr = sp.tr[k]
        fires = alive & (tr >= 0)
        stop = np.where(fires, tr + 1, 16)
        blk = np.arange(16)[None, :]
        valid[:, c0:c0 + 16] = alive[:, None] & (blk < stop[:, None])
        cor[:, c0:c0 + 16] = np.where(fires[:, None] & (blk == tr[:, None]), sp.rc[k, i][:, :16], bc[:, None])
        q = sp.rq[k, :, -1]
        upd = alive & ~fires & (q > bq)
        bq, bc = np.where(upd, q, bq), np.where(upd, sp.rc[k, :, -1], bc)
        use_roll, kb = use_roll | upd, np.where(upd, k, kb)
        cor[:, c0 + 15] = np.where(alive & ~fires, bc, cor[:, c0 + 15])   # rollout k complete without a fire: candidate
        alive = alive & ~fires
    if jF:
        c0 = D0 + 4 + 16 * K
        ok = alive & (use_roll | (D0 <= sp.P))
        cost[:, c0:c0 + 4] = r
        valid[:, c0:c0 + 4] = ok[:, None] & (np.arange(4)[None, :] < jF)
        fin_roll = sp.rpol[kb, i, min(jF, sp.rpol.shape[2]) - 1]
        fin_det = sp.pc[i, min(D0, sp.P) - 1, :4]
        cor[:, c0:c0 + 4] = np.where(use_roll[:, None], fin_roll[:, None], fin_det)
    return Prog(cost, cor, valid)


def oracle_cost(sp, r, rollouts=False):
    """cheapest recorded path to a correct state per puzzle (inf if none): loop t of the deterministic trajectory, or loop
    t <= 16 plus j finishing loops (the E+P family); with rollouts=True also one loop then rollout k to its correct loop."""
    n, T = sp.n, sp.T
    c = np.full(n, np.inf)
    t = np.arange(T)[None, :] + 1.0
    c = np.minimum(c, np.where(sp.dc, t, np.inf).min(1))
    tj = (np.arange(16)[:, None] + 1.0) + r * (np.arange(sp.J)[None, :] + 1.0)      # [16, J]
    c = np.minimum(c, np.where(sp.pc, tj[None], np.inf).reshape(n, -1).min(1))
    if rollouts:
        tr = 1.0 + (np.arange(16)[None, None, :] + 1.0)                              # one loop, then rollout k to loop t
        c = np.minimum(c, np.where(sp.rc, tr, np.inf).transpose(1, 0, 2).reshape(n, -1).min(1))
    return c


def run_oracle(c, w, B):
    order = np.argsort(c)
    cs, ws = c[order], w[:, order]
    fin = np.isfinite(cs)
    cum = np.cumsum(ws * np.where(fin, cs, 0), 1)
    ok = (cum <= np.asarray(B, float)[:, None] + 1e-9) & fin[None, :]
    N = w.sum(1)
    return (ws * ok).sum(1) / N, (cum * ok).max(1, initial=0) / N


# ----------------------------------------------------------------------------- selection
def configs(sp, pol, b):
    D0s = [d for d in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64) if d <= sp.T]
    if pol == "fixed":
        return [("fixed", min(b, sp.T))]
    if pol == "fixed_dev":
        return [("fixed", d) for d in range(1, min(b, sp.T) + 1)]
    if pol == "rr":
        return [("fixed", sp.T)]
    return sp.configs({"head": "E", "ep": "E+P", "esp": "E+S+P"}[pol.replace("_alone", "")])


def select(dev, r, pol, cache):
    """per budget: the config with the best dev accuracy under the shared budget b x n_dev (tie: less spend). For the
    *_alone policies (the controller run per puzzle, m5_ctl_ho protocol) the config runs untruncated and must have a dev
    mean honest cost <= b."""
    out = {}
    w = np.ones((1, dev.n))
    alone = pol.endswith("_alone")
    for b in BUD:
        best = None
        for cfg in configs(dev, pol, b):
            if cfg not in cache:
                cache[cfg] = build(dev, cfg, r)
            a, c = run(cache[cfg], w, [ALONE if alone else b * dev.n])
            if alone and c[0] > b + 1e-9:
                continue
            key = (round(float(a[0]), 9), -float(c[0]))
            if best is None or key > best[0]:
                best = (key, cfg)
        out[b] = best[1]
    return out


# ----------------------------------------------------------------------------- deadline simulation
def deadline_sim(prog, pids, arr, delta, mode, rng):
    """one queue. prog: Prog (per held-out puzzle). mode: 'rr' (processor sharing, anytime), 'fifo' (answer only when the
    program completes; drop on admission if it cannot), 'sjf' (oracle: shortest remaining cost first, unsolvable skipped).
    Returns hit[n_arr], loops spent per arrival."""
    n_arr = len(pids)
    dl = arr + delta
    hit = np.zeros(n_arr, bool)
    spent = np.zeros(n_arr)
    k = np.zeros(n_arr, int)
    ln = prog.ln[pids]
    cost, cor = prog.cost, prog.cor
    live = deque()
    now, nxt = 0.0, 0

    def finalize(j):
        p = pids[j]
        if mode == "fifo":
            hit[j] = k[j] == ln[j] and ln[j] > 0 and cor[p, ln[j] - 1]
        else:
            hit[j] = k[j] > 0 and cor[p, k[j] - 1]

    while nxt < n_arr or live:
        while nxt < n_arr and arr[nxt] <= now + 1e-9:
            j = nxt
            nxt += 1
            if ln[j] == 0 or (mode in ("fifo", "sjf") and now + cost[pids[j], :ln[j]].sum() > dl[j] + 1e-9):
                finalize(j)                                           # nothing to run, or cannot finish in time
            else:
                live.append(j)
        # expire
        if live:
            keep = deque()
            for j in live:
                if dl[j] <= now + 1e-9:
                    finalize(j)
                else:
                    keep.append(j)
            live = keep
        if not live:
            if nxt < n_arr:
                now = max(now, arr[nxt])
                continue
            break
        if mode == "sjf":
            rem = [(cost[pids[j], k[j]:ln[j]].sum(), j) for j in live]
            j = min(rem)[1]
            live.remove(j)
        else:
            j = live.popleft()
        p = pids[j]
        c = cost[p, k[j]]
        if now + c > dl[j] + 1e-9:                                    # next loop would not finish in time
            finalize(j)
            continue
        now += c
        spent[j] += c
        k[j] += 1
        if k[j] >= ln[j]:
            finalize(j)
        else:
            if mode == "sjf":
                live.appendleft(j)
            else:
                live.append(j)
    return hit, spent


# ----------------------------------------------------------------------------- driver
def boot_pairs(accb, pairs):
    out = {}
    for a, b in pairs:
        if a in accb and b in accb:
            d = accb[a] - accb[b]
            out[f"{a}-{b}"] = [100 * float(d.mean()), 100 * float(np.percentile(d, 2.5)), 100 * float(np.percentile(d, 97.5))]
    return out


def cell_list(quick):
    cells = []
    for fam in C.families():
        for spec in fam["cells"]:
            if spec == "fp32" or "noise" in spec:
                continue
            key = f"{fam['name']}|{spec}"
            if quick and key not in NAMED:
                continue
            cells.append((fam, spec))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=str(A.RB / "sched_sim.json"))
    a = ap.parse_args()
    nb = 200 if a.quick else NB
    rng = np.random.default_rng(0)
    out = {"meta": {"budgets": BUD, "N": NS, "NB": nb, "nsub": NSUB, "loads": LOADS, "deltas": DELTAS, "n_arr": N_ARR,
                    "policies": POLICIES, "cost_model": "compressed loop 1, w8c finishing loop 8/b_eff (m5_addons.b_eff)",
                    "split": "Sudoku dev 0-999 / held out 1000-1999; Maze dev 0-499 / held out 500-999"},
           "cells": {}}
    t0 = time.time()
    for fam, spec in cell_list(a.quick):
        key = f"{fam['name']}|{spec}"
        fp = C.cell_splits(fam["task"], "fp32", fam)
        rule, rrule = C.rules_for(fp[0][0], fam["task"])
        cs = W.splits(fam, spec, "w8c")
        if cs is None:
            print(key, "w8c records missing, skipped")
            continue
        det, m2, pc, hrm = cs
        dev, ho = [C.Split(*det[k], *m2[k], pc[k], hrm, rule, rrule) for k in (0, 1)]
        r = 8.0 / AD.b_eff(spec)
        n = ho.n
        cell = {"kind": fam["kind"], "r": r, "rule": A.rule_name(*rule), "n_dev": dev.n, "n_ho": n, "T": ho.T,
                "fire_ho": float((ho.td >= 0).mean()), "solvable_ho": None, "budget": {}, "ci": {}, "deadline": {}}
        # dev selection
        cache = {}
        picks = {p: select(dev, r, p, cache) for p in POLICIES if not p.startswith("oracle")}
        oc, oc_all = oracle_cost(ho, r), oracle_cost(ho, r, rollouts=True)
        cell["solvable_ho"] = float(np.isfinite(oc).mean())
        cell["solvable_ho_all"] = float(np.isfinite(oc_all).mean())
        # held-out programs per budget
        progs = {}
        for p, pk in picks.items():
            for b, cfg in pk.items():
                if (p, cfg) not in progs:
                    progs[(p, cfg)] = build(ho, cfg, r)
        # batches: sub-batches of N puzzles (mean over NSUB draws), full held-out, and the bootstrap
        wsub = {}
        for N in NS:
            Ne = min(N, n)
            if Ne == n:
                wsub[N] = np.ones((1, n))
            else:
                ws = np.zeros((NSUB, n))
                for d in range(NSUB):
                    ws[d, rng.choice(n, Ne, replace=False)] = 1
                wsub[N] = ws
        wboot = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(float)
        for b in BUD:
            row = {}
            accb = {}
            for p in POLICIES:
                if p.startswith("oracle"):
                    f = lambda w, B, c=(oc if p == "oracle" else oc_all): run_oracle(c, w, B)  # noqa: E731
                    cfg = None
                else:
                    cfg = picks[p][b]
                    pr = progs[(p, cfg)]
                    f = lambda w, B, pr=pr: run(pr, w, B)  # noqa: E731
                ent = {"cfg": None if cfg is None else [str(x) for x in cfg], "acc": {}, "cost": {}}
                bb = lambda w: np.full(w.shape[0], ALONE) if p.endswith("_alone") else b * w.sum(1)  # noqa: E731
                for N, ws in wsub.items():
                    acc, cost = f(ws, bb(ws))
                    ent["acc"][str(N)] = 100 * float(acc.mean())
                    ent["cost"][str(N)] = float(cost.mean())
                accb[p], _ = f(wboot, bb(wboot))
                row[p] = ent
            cell["budget"][str(b)] = row
            cell["ci"][str(b)] = boot_pairs(accb, PAIRS)
            cell.setdefault("_boot", {})[b] = accb
        # deadline setting: configs chosen at b = 16
        dprog = {"fifo16": (build(ho, ("fixed", min(16, ho.T)), r), "fifo"),
                 "rr": (build(ho, ("fixed", min(16, ho.T)), r), "rr"),
                 "head": (progs[("head", picks["head"][16])], "rr"),
                 "ep": (progs[("ep", picks["ep"][16])], "rr")}
        ocp = Prog(np.where(np.isfinite(oc), oc, 0)[:, None], np.isfinite(oc)[:, None], np.isfinite(oc)[:, None])
        dprog["oracle"] = (ocp, "sjf")
        drng = np.random.default_rng(1)
        pids = drng.integers(0, n, N_ARR)
        gaps = drng.exponential(1.0, N_ARR)
        for delta in DELTAS:
            cell["deadline"][str(delta)] = {}
            for rho in LOADS:
                arr = np.cumsum(gaps) * (16.0 / rho)
                res = {}
                hits = {}
                for name, (pr, mode) in dprog.items():
                    h, s = deadline_sim(pr, pids, arr, delta, mode, drng)
                    hits[name] = h.astype(float)
                    res[name] = {"hit": 100 * float(h.mean()), "loops_per_arrival": float(s.mean())}
                idx = rng.integers(0, N_ARR, (nb, N_ARR))
                hb = {k: v[idx].mean(1) for k, v in hits.items()}
                cell["deadline"][str(delta)][str(rho)] = {"hit": res, "ci": boot_pairs(hb, [("head", "fifo16"), ("head", "rr"), ("ep", "head"),
                                                                                             ("ep", "fifo16"), ("oracle", "ep")])}
                cell.setdefault("_dboot", {})[(delta, rho)] = hb
        out["cells"][key] = cell
        f16 = cell["budget"]["16"]
        print(f"{key:22s} r={r:.2f} fire={cell['fire_ho']:.2f} solv={cell['solvable_ho']:.2f} | b=8: "
              + " ".join(f"{p}={cell['budget']['8'][p]['acc']['1000']:.1f}" for p in MAIN)
              + f" | b=16: " + " ".join(f"{p}={f16[p]['acc']['1000']:.1f}" for p in MAIN)
              + f" | dl16 rho1: " + " ".join(f"{k}={v['hit']:.1f}" for k, v in cell["deadline"]["16"]["1.0"]["hit"].items())
              + f"  [{time.time() - t0:.0f}s]")
    # ------------------------------------------------------------------ summaries
    groups = {"sudoku20": [k for k, v in out["cells"].items() if v["kind"] == "sudoku"],
              "maze4": [k for k, v in out["cells"].items() if v["kind"] == "maze"]}
    out["summary"] = {}
    for g, keys in groups.items():
        if not keys:
            continue
        cells = [out["cells"][k] for k in keys]
        s = {"cells": keys, "budget": {}, "deadline": {}}
        for b in BUD:
            row = {}
            for p in POLICIES:
                row[p] = {N: {"acc": float(np.mean([c["budget"][str(b)][p]["acc"][str(N)] for c in cells])),
                              "cost": float(np.mean([c["budget"][str(b)][p]["cost"][str(N)] for c in cells]))} for N in NS}
            pooled = {}
            for pa, pb in PAIRS:
                d = np.mean([c["_boot"][b][pa] - c["_boot"][b][pb] for c in cells], 0)
                pooled[f"{pa}-{pb}"] = [100 * float(d.mean()), 100 * float(np.percentile(d, 2.5)), 100 * float(np.percentile(d, 97.5))]
            s["budget"][str(b)] = {"mean": row, "ci": pooled}
        for delta in DELTAS:
            s["deadline"][str(delta)] = {}
            for rho in LOADS:
                names = list(cells[0]["deadline"][str(delta)][str(rho)]["hit"].keys())
                mean = {nm: {"hit": float(np.mean([c["deadline"][str(delta)][str(rho)]["hit"][nm]["hit"] for c in cells])),
                             "loops_per_arrival": float(np.mean([c["deadline"][str(delta)][str(rho)]["hit"][nm]["loops_per_arrival"] for c in cells]))}
                        for nm in names}
                pooled = {}
                for pa, pb in [("head", "fifo16"), ("head", "rr"), ("ep", "head"), ("ep", "fifo16"), ("oracle", "ep")]:
                    d = np.mean([c["_dboot"][(delta, rho)][pa] - c["_dboot"][(delta, rho)][pb] for c in cells], 0)
                    pooled[f"{pa}-{pb}"] = [100 * float(d.mean()), 100 * float(np.percentile(d, 2.5)), 100 * float(np.percentile(d, 97.5))]
                s["deadline"][str(delta)][str(rho)] = {"mean": mean, "ci": pooled}
        out["summary"][g] = s
    for c in out["cells"].values():
        c.pop("_boot", None)
        c.pop("_dboot", None)
    # ------------------------------------------------------------------ print
    for g, s in out["summary"].items():
        print(f"\n=== {g} ({len(s['cells'])} cells), held out, N = 1000 (all held-out rows): acc @ mean cost; pooled paired CIs")
        print(f"  {'b':>3s} | " + " | ".join(f"{p:>14s}" for p in MAIN) + " | head-fixed | ep-head | esp-head | oracle-ep")
        for b in BUD:
            m, ci = s["budget"][str(b)]["mean"], s["budget"][str(b)]["ci"]
            f = lambda k: f"{ci[k][0]:+5.1f} [{ci[k][1]:+.1f},{ci[k][2]:+.1f}]"  # noqa: E731
            print(f"  {b:3d} | " + " | ".join(f"{m[p][1000]['acc']:5.1f} @{m[p][1000]['cost']:5.1f}" for p in MAIN)
                  + f" | {f('head-fixed')} | {f('ep-head')} | {f('esp-head')} | {f('oracle-ep')}")
        print(f"\n  scheduled (shared budget, anytime truncation) vs the controller alone (m5_ctl_ho protocol: dev mean cost <= b, no truncation)")
        print(f"  {'b':>3s} | " + " | ".join(f"{p:>14s}" for p in ("head", "head_alone", "ep", "ep_alone", "esp", "esp_alone")) + " | head-alone | ep-alone | esp-alone | esp_alone-fixed")
        for b in BUD:
            m, ci = s["budget"][str(b)]["mean"], s["budget"][str(b)]["ci"]
            f = lambda k: f"{ci[k][0]:+5.1f} [{ci[k][1]:+.1f},{ci[k][2]:+.1f}]"  # noqa: E731
            print(f"  {b:3d} | " + " | ".join(f"{m[p][1000]['acc']:5.1f} @{m[p][1000]['cost']:5.1f}" for p in ("head", "head_alone", "ep", "ep_alone", "esp", "esp_alone"))
                  + f" | {f('head-head_alone')} | {f('ep-ep_alone')} | {f('esp-esp_alone')} | {f('esp_alone-fixed')}")
        for delta in DELTAS:
            print(f"\n  deadline {delta} loops: hit rate (%) vs load (arrivals per 16 loop-times)")
            names = list(s["deadline"][str(delta)][str(LOADS[0])]["mean"].keys())
            print(f"  {'rho':>5s} | " + " | ".join(f"{nm:>7s}" for nm in names) + " | head-fifo16 | head-rr | ep-head")
            for rho in LOADS:
                m, ci = s["deadline"][str(delta)][str(rho)]["mean"], s["deadline"][str(delta)][str(rho)]["ci"]
                f = lambda k: f"{ci[k][0]:+5.1f} [{ci[k][1]:+.1f},{ci[k][2]:+.1f}]"  # noqa: E731
                print(f"  {rho:5.2f} | " + " | ".join(f"{m[nm]['hit']:7.1f}" for nm in names) + f" | {f('head-fifo16')} | {f('head-rr')} | {f('ep-head')}")
    for key, nm in NAMED.items():
        if key not in out["cells"]:
            continue
        c = out["cells"][key]
        print(f"\n=== {nm} ({key}; r = {c['r']:.2f}, head fires on {100 * c['fire_ho']:.0f}% of held-out puzzles, {100 * c['solvable_ho']:.0f}% solvable by some recorded path)")
        cols = MAIN + ["esp_alone"]
        print(f"  {'b':>3s} | " + " | ".join(f"{p:>14s}" for p in cols) + " | head-fixed | ep-head | esp-head | esp-esp_alone")
        for b in BUD:
            m, ci = c["budget"][str(b)], c["ci"][str(b)]
            f = lambda k: f"{ci[k][0]:+5.1f} [{ci[k][1]:+.1f},{ci[k][2]:+.1f}]"  # noqa: E731
            print(f"  {b:3d} | " + " | ".join(f"{m[p]['acc']['1000']:5.1f} @{m[p]['cost']['1000']:5.1f}" for p in cols)
                  + f" | {f('head-fixed')} | {f('ep-head')} | {f('esp-head')} | {f('esp-esp_alone')}")
        print(f"  N effect at b = 8 (acc, N = 64 / 256 / 1000): " + "  ".join(f"{p} " + "/".join(f"{m[p]['acc'][str(N)]:.1f}" for N in NS) for p in POLICIES for m in [c['budget']['8']]))
        for delta in DELTAS:
            names = list(c["deadline"][str(delta)][str(LOADS[0])]["hit"].keys())
            print(f"  deadline {delta}: " + " | ".join(f"rho {rho:g}: " + " ".join(f"{nm}={c['deadline'][str(delta)][str(rho)]['hit'][nm]['hit']:.0f}" for nm in names) for rho in LOADS))
    json.dump(out, open(a.json, "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("\nwrote", a.json, f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
