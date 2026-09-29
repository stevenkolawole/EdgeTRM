#!/usr/bin/env python3
"""E_COMBINED read-out: the controller (exit + 8-bit finishing) with and without per-step dither on the cheap loops,
at the same honest cost. Predictions and protocol: reports/edgetrm_exp/E_COMBINED.md (written before any record).

Records (exit_polish.py --K 64 --J 4 Sudoku, --K 32 Maze; p4d A100; results/m5/comb/p4d/polish_o{0,1000|500}/<task>/):
  <cheap>__w8c.npz                        RTN cheap trajectory, w8c finishing after every cheap step (deterministic)
  <cheap>__w8c__dither_mw8c_s{k}.npz      cheap weights re-drawn (SR onto the cheap grid from the INT8 copy) before every
                                          cheap step, seed k = M2 arm 6's rollout k; the same w8c finishing
each with c_cheap[n, T], q_cheap[n, T], qc_cheap[n, T], c_pol[n, T, J].
Exit rule: m5_ctl_ho.rules_for on the dev fp32 exit_sweep record (results/m5/ctl/p4d/det_o0/<task>/fp32.npz), frozen.
Arms (configuration chosen on dev rows 0-999 at each budget, scored on held-out rows 1000-1999):
  fixed      RTN, d cheap steps
  ctl        controller alone: RTN cheap, exit at the first firing step t < D0 then jE finishing, else D0 then jF
  dit        dither alone: dither cheap, fixed depth d
  dit+E      dither + exit, no finishing
  dit+P      dither + finishing, no exit (fixed d then j)
  comb       dither + exit + finishing (the ctl configuration set on the dither trajectory)
  (ctl's configuration set includes E-only and P-only configurations, as m5_ctl_ho's E+P does)
Honest cost: cheap step 1, finishing step r = 8 / b_eff(cheap) (m5_addons.b_eff); the re-draw is charged 0 (cheap-step
counts are reported so a per-step re-draw cost can be added). Stochastic arms: per-puzzle mean over seeds; dev selection
on the seed mean; CIs: puzzle bootstrap, 2000 draws, paired on the held-out puzzles.
Checks: the RTN c_cheap equals the det_o record at default-H steps and the polish_o record's c_pol at t <= 16.

    python m5_comb.py [--root DIR] [--budgets 8,12,16,24,32,48,64]     # tables + results/breadth/m5_comb.json
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

import analyze_m5 as A
import m5_ctl_ho as C
from m5_addons import b_eff
from m5_controller import fire_mask, first

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("M5_COMB_DIR", HERE / "results" / "m5" / "comb" / "p4d"))
CELLS = {"sudoku_mlp": ["w4c", "w3g32"], "sudoku_attn": ["w3c", "w4t"], "hrm_sudoku": ["w3c"],
         "maze_attn": ["w2a", "w4t"], "hrm_maze": ["w2a", "w2g32"]}
# row sets: Sudoku dev 0-999 / held out 1000-1999 (polish_o0 / polish_o1000); Maze dev 0-499 / held out 500-999
# (polish_o0 / polish_o500; the det reference is the one det_o0 record, sliced). Maze records: K = 32 cheap steps.
SPLITS = {"sudoku": ((0, "dev", None), (1000, "ho", None)), "maze": ((0, "dev", slice(0, 500)), (500, "ho", slice(500, 1000)))}
BUD = [8, 12, 16, 24, 32, 48, 64]
NB = 2000
D0S = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
JS = (0, 1, 2, 4)
ARMS = ["fixed", "ctl", "dit", "dit+E", "dit+P", "comb"]
# re-draw charge c_d per dithered cheap step, as a share of a cheap step: the measured L40S batch-1 shares of
# reports/edgetrm_exp/E1_DITHER.md sec. 3 (Sudoku-MLP 0.067, Sudoku-attn 0.045, HRM Sudoku 0.218, Maze-Hard attention
# 0.040); the attention Maze cells take the Sudoku-attn value, HRM Maze the HRM Sudoku value (coordinator, Sep 29).
CD = {"sudoku_mlp": 0.067, "sudoku_attn": 0.045, "hrm_sudoku": 0.218, "maze_attn": 0.045, "hrm_maze": 0.218}


def load(path):
    d = np.load(path)
    info = json.loads(str(d["info"]))
    return {"c": d["c_cheap"].astype(bool), "q": d["q_cheap"].astype(np.float32), "qc": d["qc_cheap"].astype(np.float32),
            "pc": d["c_pol"].astype(bool), "info": info}


class Arm:
    """one trajectory family (RTN: one record; dither: one record per seed) on one row set."""

    def __init__(self, recs, hrm, rule, r, cd=0.0):
        self.cd = cd                                          # re-draw charge per cheap step (0 for RTN)
        self.S = len(recs)
        self.c = np.stack([x["c"] for x in recs])            # [S, n, T]
        self.pc = np.stack([x["pc"] for x in recs])          # [S, n, T, J]
        self.q = np.stack([x["q"] for x in recs])
        self.qc = np.stack([x["qc"] for x in recs]) if hrm else None
        self.S, self.n, self.T = self.c.shape
        self.J = self.pc.shape[3]
        self.td = first(fire_mask(self.q, self.qc, rule, hrm))     # [S, n], -1 = never
        self.r = r

    def eval(self, cfg):
        """per-puzzle (seed-mean) correctness, honest cost, cheap steps, finishing steps."""
        kind = cfg[0]
        S, n = self.S, self.n
        i = np.arange(n)
        if kind == "fixed":
            d, j = cfg[1], cfg[2]
            acc = self.c[:, :, d - 1] if j == 0 else self.pc[:, :, d - 1, j - 1]
            cheap = np.full((S, n), float(d))
            fin = np.full((S, n), float(j))
        else:
            D0, jE, jF = cfg[1:]
            acc = np.zeros((S, n), bool)
            cheap = np.zeros((S, n))
            fin = np.zeros((S, n))
            for s in range(S):
                td = self.td[s]
                f = (td >= 0) & (td < D0)
                t = td[f]
                acc[s, f] = self.pc[s, i[f], t, jE - 1] if jE else self.c[s, i[f], t]
                cheap[s, f] = t + 1
                fin[s, f] = jE
                acc[s, ~f] = self.pc[s, i[~f], D0 - 1, jF - 1] if jF else self.c[s, i[~f], D0 - 1]
                cheap[s, ~f] = D0
                fin[s, ~f] = jF
        cost = cheap * (1 + self.cd) + self.r * fin
        return acc.mean(0), cost.mean(0), cheap.mean(0), fin.mean(0)

    def head(self, cap=16):
        """firing fraction and mean first-firing step (fired puzzles) under the cap, seed-averaged."""
        f = (self.td >= 0) & (self.td < cap)
        return float(f.mean()), float((self.td[f] + 1).mean()) if f.any() else float("nan")


def configs(kind):
    if kind == "fixed":
        return [("fixed", d, 0) for d in D0S]
    if kind == "fixedP":
        return [("fixed", d, j) for d in D0S for j in JS]
    if kind == "E":
        return [("ctl", d, 0, 0) for d in D0S]
    return [("fixed", d, j) for d in D0S for j in JS] + [("ctl", d, jE, jF) for d in D0S for jE in JS for jF in JS]


ARM_CFG = {"fixed": "fixed", "ctl": "EP", "dit": "fixed", "dit+E": "E", "dit+P": "fixedP", "comb": "EP"}
ARM_SRC = {"fixed": "rtn", "ctl": "rtn", "dit": "dit", "dit+E": "dit", "dit+P": "dit", "comb": "dit"}


def select(dev, ho, cfgs, budgets):
    cfgs = [c for c in cfgs if c[1] <= min(dev.T, ho.T)]        # Maze records hold K = 32 cheap steps
    res = {c: dev.eval(c) for c in cfgs}
    out = {}
    for b in budgets:
        feas = [(v[0].mean(), -v[1].mean(), c) for c, v in res.items() if v[1].mean() <= b + 1e-9]
        if not feas:
            continue
        c = max(feas, key=lambda z: (z[0], z[1]))[2]
        a, cost, cheap, fin = ho.eval(c)
        out[b] = {"cfg": c, "dev_acc": 100 * res[c][0].mean(), "acc_i": a, "cost": float(cost.mean()),
                  "cheap": float(cheap.mean()), "fin": float(fin.mean())}
    return out


def boot(x, rng, y=None):
    x = np.asarray(x, float) - (0 if y is None else np.asarray(y, float))
    idx = rng.integers(0, len(x), (NB, len(x)))
    b = x[idx].mean(1)
    return 100 * x.mean(), 100 * np.percentile(b, 2.5), 100 * np.percentile(b, 97.5)


def pooled(ds, rng):
    bs = np.mean([d[rng.integers(0, len(d), (NB, len(d)))].mean(1) for d in ds], 0)
    return 100 * np.mean([d.mean() for d in ds]), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def fmt(t):
    return f"{t[0]:+5.1f} [{t[1]:+.1f},{t[2]:+.1f}]"


def checks(task, spec, off, rtn, rows=None):
    """RTN record vs the existing exit_sweep and polish records of the controller platform."""
    msg = []
    p = C.ROOT / f"det_o{0 if rows else off}" / task / f"{spec}.npz"
    if p.exists():
        dc, _, _ = C.load_det(p, rows)
        T = min(dc.shape[1], rtn["c"].shape[1])
        n = min(dc.shape[0], rtn["c"].shape[0])
        mm = int((dc[:n, :T] != rtn["c"][:n, :T]).sum())
        msg.append(f"det: {mm} mismatches of {n * T} (puzzle, step) cells")
    p = C.ROOT / f"polish_o{off}" / task / f"{spec}__w8c.npz"
    if p.exists():
        pc = np.load(p)["c_pol"].astype(bool)
        n = min(pc.shape[0], rtn["pc"].shape[0])
        mm = int((pc[:n] != rtn["pc"][:n, :pc.shape[1], :pc.shape[2]]).sum())
        msg.append(f"polish: {mm} mismatches of {pc[:n].size}")
    return "; ".join(msg) or "no reference records"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--budgets", default=",".join(map(str, BUD)))
    ap.add_argument("--json", default=str(A.RB / "m5_comb.json"))
    a = ap.parse_args()
    root = Path(a.root)
    all_budgets = [int(b) for b in a.budgets.split(",")]
    out = {}
    for charged in (False, True):
        cells, pooled = run(root, all_budgets, charged)
        out["cells" if not charged else "cells_cd"] = cells
        out["pooled" if not charged else "pooled_cd"] = pooled
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.json, "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else (list(o) if isinstance(o, tuple) else str(o)))
    print("\nwrote", a.json)


def run(root, all_budgets, charged):
    """one full read-out; charged: dithered cheap steps cost 1 + CD[task] instead of 1."""
    rng = np.random.default_rng(0)
    out = {"cells": {}, "pooled": {}}
    rows = []
    print(f"\n##### re-draw {'charged (c_d = ' + ', '.join(f'{k} {v}' for k, v in CD.items()) + ')' if charged else 'charged 0'}")
    for task, specs in CELLS.items():
        cd = CD[task] if charged else 0.0
        hrm = task.startswith("hrm")
        splits = SPLITS["maze" if "maze" in task else "sudoku"]
        d0 = C.load_det(C.ROOT / "det_o0" / task / "fp32.npz", splits[0][2])
        rule, _ = C.rules_for(d0, task)
        for spec in specs:
            r = 8.0 / b_eff(spec)
            arms = {}
            ok = True
            for off, name, rsl in splits:
                d = root / f"polish_o{off}" / task
                rtn_p = d / f"{spec}__w8c.npz"
                dit_p = sorted(glob.glob(str(d / f"{spec}__w8c__dither_mw8c_s*.npz")))
                if not rtn_p.exists() or not dit_p:
                    print(f"  {task} {spec} {name}: records missing (rtn {rtn_p.exists()}, dither {len(dit_p)})")
                    ok = False
                    break
                rtn = load(rtn_p)
                arms[(name, "rtn")] = Arm([rtn], hrm, rule, r)
                arms[(name, "dit")] = Arm([load(p) for p in dit_p], hrm, rule, r, cd)
                arms[(name, "check")] = checks(task, spec, off, rtn, rsl)
            if not ok:
                continue
            n_dev, n_ho = arms[("dev", "rtn")].n, arms[("ho", "rtn")].n
            S = arms[("ho", "dit")].S
            T = min(arms[("dev", "rtn")].T, arms[("ho", "rtn")].T)
            budgets = [b for b in all_budgets if b <= T]          # Maze: budgets 8-32 (K = 32)
            print(f"\n=== {task} {spec}  rule {A.rule_name(*rule)}  r = {r:.2f}  c_d = {cd}  dev n={n_dev} held-out n={n_ho}  dither seeds {S}")
            print(f"  checks dev: {arms[('dev', 'check')]} | held out: {arms[('ho', 'check')]}")
            sel = {arm: select(arms[("dev", ARM_SRC[arm])], arms[("ho", ARM_SRC[arm])], configs(ARM_CFG[arm]), budgets) for arm in ARMS}
            hd = {k: {"dev": arms[("dev", k)].head(), "ho": arms[("ho", k)].head()} for k in ("rtn", "dit")}
            fp = C.load_det(C.ROOT / f"det_o{0 if splits[1][2] else splits[1][0]}" / task / "fp32.npz", splits[1][2])
            if fp is not None:                                   # fp32 head on the held-out rows, for reference
                tf = first(fire_mask(fp[1][None], fp[2][None] if hrm else None, rule, hrm))
                ff = (tf >= 0) & (tf < 16)
                hd["fp32"] = {"ho": (float(ff.mean()), float((tf[ff] + 1).mean()) if ff.any() else float("nan"))}
            print(f"  head at cap 16 (fired fraction, mean first step): RTN dev {hd['rtn']['dev'][0]:.3f} {hd['rtn']['dev'][1]:.2f} | "
                  f"held out {hd['rtn']['ho'][0]:.3f} {hd['rtn']['ho'][1]:.2f}; dither dev {hd['dit']['dev'][0]:.3f} {hd['dit']['dev'][1]:.2f} | "
                  f"held out {hd['dit']['ho'][0]:.3f} {hd['dit']['ho'][1]:.2f}"
                  + (f"; fp32 held out {hd['fp32']['ho'][0]:.3f} {hd['fp32']['ho'][1]:.2f}" if "fp32" in hd else ""))
            print(f"  {'budget':>6s} | " + " | ".join(f"{m:>18s}" for m in ARMS) + " | comb-ctl [CI] | comb-dit [CI] | comb-ditE [CI] | ctl-fixed [CI] | dit-fixed [CI]")
            cell = {"rule": A.rule_name(*rule), "r": r, "c_d": cd, "n_dev": n_dev, "n_ho": n_ho, "seeds": S, "T": T, "head": hd,
                    "checks": {"dev": arms[("dev", "check")], "ho": arms[("ho", "check")]}, "budgets": {}}
            for b in budgets:
                if not all(b in sel[m] for m in ARMS):
                    continue
                parts = [f"{100 * sel[m][b]['acc_i'].mean():5.1f} @{sel[m][b]['cost']:5.1f} ({sel[m][b]['fin']:.2f})" for m in ARMS]
                g = lambda m: sel[m][b]["acc_i"]  # noqa: E731
                diffs = {"comb-ctl": boot(g("comb"), rng, g("ctl")), "comb-dit": boot(g("comb"), rng, g("dit")),
                         "comb-ditE": boot(g("comb"), rng, g("dit+E")), "comb-ditP": boot(g("comb"), rng, g("dit+P")),
                         "ditE-ctl": boot(g("dit+E"), rng, g("ctl")), "ditP-ctl": boot(g("dit+P"), rng, g("ctl")),
                         "ctl-fixed": boot(g("ctl"), rng, g("fixed")), "dit-fixed": boot(g("dit"), rng, g("fixed"))}
                print(f"  {b:6d} | " + " | ".join(f"{p:>18s}" for p in parts) + " | " + " | ".join(fmt(diffs[k]) for k in ("comb-ctl", "comb-dit", "comb-ditE", "ctl-fixed", "dit-fixed")))
                cell["budgets"][b] = {m: {"cfg": list(sel[m][b]["cfg"]), "dev_acc": sel[m][b]["dev_acc"], "acc": boot(g(m), rng),
                                          "cost": sel[m][b]["cost"], "cheap": sel[m][b]["cheap"], "fin": sel[m][b]["fin"]} for m in ARMS}
                cell["budgets"][b]["diffs"] = diffs
                rows.append((task, spec, b, sel))
            out["cells"][f"{task}|{spec}"] = cell
    # pooled over the cells of each family (Sudoku: 5 cells; Maze: 4 cells) at each budget
    for fam in ("sudoku", "maze"):
      print(f"\n=== {fam}: pooled over cells, held out: mean acc @ mean honest cost; paired pooled CIs")
      out["pooled"][fam] = {}
      for b in all_budgets:
        sub = [x for x in rows if x[2] == b and (fam in x[0])]
        if not sub:
            continue
        line = " | ".join(f"{m} {100 * np.mean([x[3][m][b]['acc_i'].mean() for x in sub]):5.1f} @{np.mean([x[3][m][b]['cost'] for x in sub]):5.1f}" for m in ARMS)
        d = lambda m1, m2: pooled([x[3][m1][b]["acc_i"].astype(float) - x[3][m2][b]["acc_i"] for x in sub], rng)  # noqa: E731
        dd = {"comb-ctl": d("comb", "ctl"), "comb-dit": d("comb", "dit"), "ctl-fixed": d("ctl", "fixed"), "dit-fixed": d("dit", "fixed"),
              "ditE-ctl": d("dit+E", "ctl"), "ditP-ctl": d("dit+P", "ctl"), "comb-ditE": d("comb", "dit+E"), "comb-ditP": d("comb", "dit+P")}
        print(f"  {b:4d} ({len(sub)} cells): {line}\n       " + "  ".join(f"{k} {fmt(v)}" for k, v in dd.items()))
        out["pooled"][fam][b] = {"cells": len(sub), "diffs": dd,
                            "mean": {m: (100 * np.mean([x[3][m][b]["acc_i"].mean() for x in sub]), float(np.mean([x[3][m][b]["cost"] for x in sub]))) for m in ARMS}}
    return out["cells"], out["pooled"]


if __name__ == "__main__":
    main()
