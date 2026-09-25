#!/usr/bin/env python3
"""M5 unified budgeted test-time controller: exit (M5) + sample-until-confident (M2) + polish (M1).

For one model and one (compressed) weight cell, per puzzle:
  1. deterministic trajectory with the exit rule (chosen on fp32 at 16 steps), capped at D0 steps;
  2. if the head never fires: stochastic rollouts k = 1..K (M2's PTRM z_L noise, 16 steps each), each stopped
     by the same exit rule; the first rollout whose rule fires gives the answer;
  3. if no rollout fires: the candidate with the highest final q (deterministic end state or a tried
     rollout), and, if j > 0 and that candidate is a rollout, j fp32 polish steps from its end state (M2's
     P2 records, M1's append schedule).
Cost = supervision steps actually run (cheap and polish steps counted alike; polish-weighted cost printed too).
Baselines on the same puzzles: fixed depth T; exit only capped at B; best-Q@K (K x 16); sequential sampling alone.
Configs are selected per budget on one half of the puzzles (even/odd rows) and scored on the other half,
then the halves are swapped (two-fold cross-fit); the frontier reports the pooled held-out accuracy and cost.

Data: deterministic trajectories = M5 exit_sweep records (n=1000, default H, up to 512 steps) or M2's
none0 records (mazes, n=250, 128 steps); rollouts = M2 cells `<model>__<spec>__ptrm<s>__D16_K<k>[_P2].npz`.

    python m5_controller.py [--models sudoku_mlp sudoku_attn hrm_sudoku maze_attn hrm_maze]
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

import analyze_m5 as A

M2_DIR = Path(os.environ.get("M5_M2_DIR", A.RB / "babel" / "m2" / "cells"))
BUDGETS = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def det_arrays(model, spec):
    """deterministic per-step exact + q_halt (+ q_continue for HRM) on the first n rows."""
    cells = A.load_m5(model)
    if spec in cells:
        c = cells[spec]
        return c.c, c.q, c.qc
    f = M2_DIR / f"{model}__{spec}__none0__D128_K1.npz"
    if f.exists():
        d = np.load(f)
        return d["corr"][0].astype(bool), d["q"][0].astype(np.float32), None
    return None


def roll_arrays(model, spec):
    """M2 rollouts: prefer K32_P2 (with polish), else the largest K16 cell at sigma 0.3 / 1.0."""
    for pat in (f"{model}__{spec}__ptrm0.3__D16_K32_P2.npz", f"{model}__{spec}__ptrm1__D16_K32_P2.npz",
                f"{model}__{spec}__ptrm1__D16_K16.npz", f"{model}__{spec}__ptrm0.3__D16_K16*.npz"):
        fs = sorted(glob.glob(str(M2_DIR / pat)))
        if fs:
            d = np.load(fs[0])
            pol = d["pol_corr"].astype(bool) if "pol_corr" in d.files else None
            return Path(fs[0]).name, d["corr"].astype(bool), d["q"].astype(np.float32), pol
    return None


def fire_mask(q, qc, rule, hrm):
    base = (q > qc) if (hrm and qc is not None) else (q > 0)
    kind, p = rule
    if kind == "native":
        return base
    if kind == "pat":
        rl = np.zeros(base.shape, np.int32)
        run = np.zeros(base.shape[:-1], np.int32)
        for t in range(base.shape[-1]):
            run = np.where(base[..., t], run + 1, 0)
            rl[..., t] = run
        return rl >= p
    if kind == "min":
        m = base.copy()
        m[..., :p - 1] = False
        return m
    raise ValueError(rule)


def first(mask):
    anyf = mask.any(-1)
    return np.where(anyf, mask.argmax(-1), -1)


class Controller:
    def __init__(self, dc, dq, dqc, rc, rq, rpol, rule, hrm, rrule=None):
        rrule = rrule or rule
        n = min(dc.shape[0], rc.shape[1])
        self.n = n
        self.dc, self.dq = dc[:n], dq[:n]
        self.dqc = None if dqc is None else dqc[:n]
        self.rc, self.rq = rc[:, :n], rq[:, :n]
        self.rpol = None if rpol is None else rpol[:, :n]
        self.T = self.dc.shape[1]
        self.K, _, self.D = self.rc.shape
        self.td = first(fire_mask(self.dq, self.dqc, rule, hrm))           # [n]
        self.tr = first(fire_mask(self.rq, None, rrule, False))              # [K, n] (M2 stores q_halt only)

    def run(self, D0, K, j, idx=None):
        n = self.n
        i = np.arange(n) if idx is None else idx
        td = self.td[i]
        fired0 = (td >= 0) & (td < D0)
        acc = np.zeros(len(i), bool)
        cost = np.zeros(len(i), float)
        polc = np.zeros(len(i), float)
        acc[fired0] = self.dc[i[fired0], td[fired0]]
        cost[fired0] = td[fired0] + 1
        rest = ~fired0
        cost[rest] = D0
        if K > 0 and rest.any():
            ii = i[rest]
            tr = self.tr[:K, ii]                                             # [K, m]
            F = tr >= 0
            anyf = F.any(0)
            fk = np.where(anyf, F.argmax(0), K)
            m = len(ii)
            c_roll = np.where(anyf, 16 * fk + tr[np.minimum(fk, K - 1), np.arange(m)] + 1, 16 * K)
            a_roll = np.zeros(m, bool)
            a_roll[anyf] = self.rc[fk[anyf], ii[anyf], tr[fk[anyf], np.arange(m)[anyf]]]
            # fallback: highest final q among the deterministic end state and the K tried rollouts
            qdet = self.dq[ii, D0 - 1]
            qr = self.rq[:K, ii, -1]
            kb = qr.argmax(0)
            use_roll = qr[kb, np.arange(m)] > qdet
            fb = np.where(use_roll, self.rc[kb, ii, -1], self.dc[ii, D0 - 1])
            fcost = np.zeros(m)
            if j > 0 and self.rpol is not None:
                fb = np.where(use_roll, self.rpol[kb, ii, j - 1], fb)
                fcost = np.where(use_roll, j, 0)
            a_roll = np.where(anyf, a_roll, fb)
            acc[rest] = a_roll
            cost[rest] += c_roll + np.where(anyf, 0, fcost)
            polc[rest] = np.where(anyf, 0, fcost)
        elif rest.any():
            acc[rest] = self.dc[i[rest], D0 - 1]
        return acc, cost, polc

    def configs(self, j_opts=(0, 2)):
        for D0 in [d for d in (4, 8, 16, 32, 64, 128, 256, 512) if d <= self.T]:
            for K in [k for k in (0, 1, 2, 4, 8, 16, 32) if k <= self.K]:
                for j in (j_opts if (K > 0 and self.rpol is not None) else (0,)):
                    yield ("ctrl", D0, K, j)

    def baseline(self, name, param, idx):
        i = idx
        if name == "fixed":
            return self.dc[i, param - 1], np.full(len(i), float(param))
        if name == "exit":
            a, c, _ = self.run(param, 0, 0, i)
            return a, c
        if name == "bestq":
            K = param
            kb = self.rq[:K, i, -1].argmax(0)
            return self.rc[kb, i, -1], np.full(len(i), 16.0 * K)
        if name == "seq":
            a, c, _ = self.run(1, param, 0, i)        # D0 = 1: effectively rollouts only (+1 deterministic step)
            return a, c
        raise ValueError(name)

    def method_configs(self, name):
        if name == "fixed":
            return [t for t in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512) if t <= self.T]
        if name == "exit":
            return [d for d in (4, 8, 16, 32, 64, 128, 256, 512) if d <= self.T]
        if name in ("bestq", "seq"):
            return [k for k in (1, 2, 4, 8, 16, 32) if k <= self.K]


def crossfit_frontier(ctl, evaluate, configs, budgets):
    """per budget b: pick the config with the best accuracy among those with mean cost <= b on one half,
    score it on the other half; pooled over the two folds. Returns [(b, acc, cost, chosen configs)]."""
    halves = [np.arange(0, ctl.n, 2), np.arange(1, ctl.n, 2)]
    cache = {}
    for cfg in configs:
        for h in (0, 1):
            cache[(cfg, h)] = evaluate(cfg, halves[h])
    out = []
    for b in budgets:
        accs, costs, chosen = [], [], []
        ok = True
        for tr, te in ((0, 1), (1, 0)):
            feas = [(cache[(cfg, tr)][0].mean(), cfg) for cfg in configs if cache[(cfg, tr)][1].mean() <= b + 1e-9]
            if not feas:
                ok = False
                break
            best = max(feas, key=lambda x: (x[0], -cache[(x[1], tr)][1].mean()))[1]
            a, c = cache[(best, te)][:2]
            accs.append(a); costs.append(c); chosen.append(best)
        if ok:
            out.append((b, 100 * np.concatenate(accs).mean(), float(np.concatenate(costs).mean()), chosen))
    return out


def summarize(res):
    """per cell: controller minus the best baseline at the same budget (fixed depth; best single method)."""
    rows = []
    for key, v in res.items():
        fr = v["frontier"]
        tab = {k: {b: acc for b, acc, cost, _ in lst} for k, lst in fr.items()}
        gaps_fixed, gaps_best = {}, {}
        for b, acc in tab["controller_nopolish"].items():
            if b in tab["fixed"]:
                gaps_fixed[b] = acc - tab["fixed"][b]
            others = [tab[k][b] for k in ("fixed", "exit", "bestq", "seq") if b in tab[k]]
            if others:
                gaps_best[b] = acc - max(others)
        rows.append((key, gaps_fixed, gaps_best))
    return rows


def dt_frontiers(sizes=(13, 25, 33)):
    """deep-thinking mazes (no head): answer-stability exit (argmax unchanged p iterations, p chosen per budget on
    one half) capped at B, vs the fixed budget B; cross-fitted like the controller."""
    res = {}
    for size in sizes:
        cells = A.load_dt(size)
        for spec, c in sorted(cells.items(), key=lambda kv: A.spec_order(kv[0])):
            T = c.K
            stab = {p: A.stable_mask(c, p) for p in (3, 5, 10, 20, 30)}
            tfire = {p: first(m) for p, m in stab.items()}

            class Obj:
                n = c.n

            def ev(cfg, i):
                kind, p, B = cfg
                if kind == "fixed":
                    return c.c[i, B - 1], np.full(len(i), float(B))
                t = tfire[p][i]
                t = np.where((t >= 0) & (t < B), t, B - 1)
                return c.c[i, t], (t + 1).astype(float)
            budgets = [b for b in (25, 50, 75, 100, 150, 200, 300, 400, 500) if b <= T]
            caps = [b for b in (10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500) if b <= T]
            fr_s = crossfit_frontier(Obj, ev, [("stab", p, B) for p in stab for B in caps], budgets)
            fr_f = crossfit_frontier(Obj, ev, [("fixed", 0, B) for B in caps], budgets)
            res[(size, spec)] = {"stab": fr_s, "fixed": fr_f}
            ff = {b: a for b, a, _, _ in fr_f}
            print(f"  DT {size}x{size} {spec:12s} " + " ".join(f"B{b}: {a:5.1f}@{cst:5.1f} vs {ff.get(b, float('nan')):5.1f}" for b, a, cst, _ in fr_s))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze_attn", "hrm_maze"])
    ap.add_argument("--specs", default="")
    ap.add_argument("--json", default=str(A.RB / "m5_controller.json"))
    ap.add_argument("--dt", action="store_true", help="deep-thinking stability-exit frontiers only")
    a = ap.parse_args()
    if a.dt:
        print("=== DT mazes: stability exit (held-out selection) acc@mean cost vs fixed budget acc, per budget")
        r = dt_frontiers()
        json.dump({f"{k[0]}|{k[1]}": v for k, v in r.items()}, open(A.RB / "m5_controller_dt.json", "w"), indent=1, default=str)
        return
    res = {}
    for model in a.models:
        ref = A.cells_for(model, 16).get("fp32")
        rule = A.select_rule(ref) if ref is not None else ("pat", 2)
        hrm = A.IS_HRM[model]
        # rollouts carry q_halt only: for HRM choose the rollout rule on fp32 without q_continue
        rrule = A.select_rule(A.Cell(model, "fp32", "x", ref.c, ref.q)) if (hrm and ref is not None) else rule
        specs = a.specs.split(",") if a.specs else sorted({Path(f).name.split("__")[1] for f in glob.glob(str(M2_DIR / f"{model}__*__ptrm*"))})
        for spec in specs:
            det = det_arrays(model, spec)
            roll = roll_arrays(model, spec)
            if det is None or roll is None:
                continue
            rname, rc, rq, rpol = roll
            ctl = Controller(*det, rc, rq, rpol, rule, hrm, rrule)
            budgets = [b for b in BUDGETS if b <= max(ctl.T, 16 * ctl.K)]
            fr = {"controller": crossfit_frontier(ctl, lambda cfg, i: ctl.run(cfg[1], cfg[2], cfg[3], i), list(ctl.configs()), budgets),
                  # same controller without the fp32 polish (no second weight set in memory)
                  "controller_nopolish": crossfit_frontier(ctl, lambda cfg, i: ctl.run(cfg[1], cfg[2], cfg[3], i),
                                                           [c for c in ctl.configs() if c[3] == 0], budgets)}
            for name in ("fixed", "exit", "bestq", "seq"):
                fr[name] = crossfit_frontier(ctl, lambda p, i, nm=name: ctl.baseline(nm, p, i), ctl.method_configs(name), budgets)
            res[(model, spec)] = {"rule": A.rule_name(*rule), "rollout_rule": A.rule_name(*rrule), "rollouts": rname, "n": ctl.n, "T": ctl.T, "K": ctl.K, "frontier": fr}
            print(f"\n=== {model} {spec}: n={ctl.n}, deterministic to {ctl.T} steps, rollouts {rname}; exit rule {A.rule_name(*rule)}")
            print(f"{'budget':>6s} | {'controller':>18s} | {'no polish':>16s} | {'fixed depth':>16s} | {'exit only':>16s} | {'best-Q@K':>16s} | {'seq sampling':>16s} | controller config (fold A / B)")
            tabs = {k: {b: (acc, cost) for b, acc, cost, _ in v} for k, v in fr.items()}
            for b, acc, cost, ch in fr["controller"]:
                cells = []
                for name in ("controller", "controller_nopolish", "fixed", "exit", "bestq", "seq"):
                    v = tabs[name].get(b)
                    cells.append(f"{v[0]:5.1f} @{v[1]:6.1f}" if v else f"{'-':>12s}")
                print(f"{b:6d} | " + " | ".join(f"{c:>16s}" for c in cells) + f" | {ch[0][1:]} / {ch[1][1:]}")
    print("\n=== summary: controller WITHOUT polish minus fixed depth / minus best single method, at budgets 16 / 64 / 256 (held-out halves)")
    for (m, sp), gf, gb in summarize(res):
        f = lambda g, b: f"{g[b]:+5.1f}" if b in g else "    -"
        print(f"  {m:12s} {sp:14s} fixed: {f(gf,16)} {f(gf,64)} {f(gf,256)}   best single: {f(gb,16)} {f(gb,64)} {f(gb,256)}   "
              f"min over budgets vs fixed {min(gf.values()):+5.1f}, vs best single {min(gb.values()):+5.1f}")
    out = {f"{m}|{s}": v for (m, s), v in res.items()}
    json.dump(out, open(a.json, "w"), indent=1, default=lambda o: o if not isinstance(o, np.generic) else o.item())
    print("wrote", a.json)


if __name__ == "__main__":
    main()
