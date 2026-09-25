#!/usr/bin/env python3
"""M5 unified test-time controller, selected on dev rows and scored on held-out rows (p5 exhibit, one platform).

The method, per puzzle, for a compressed recursive reasoner:
  E  exit: deterministic trajectory stopped by the model's own halting head (rule chosen on the dev fp32 cell,
     frozen), capped at D0 supervision steps; optionally jE fp32 polish steps from the exit state (M1);
  S  if the head never fires: PTRM z_L-noise rollouts (M2), 16 steps each, each stopped by the same head, k <= K;
  P  if no rollout fires: the candidate with the highest final q_halt (deterministic end state or a tried rollout),
     followed by jF fp32 polish steps from that state.
Cost = supervision steps actually run; a polish step counts as one step (its fp32 share is reported separately).

Ablations on the same puzzles, every one with its own configuration chosen on dev at each budget:
  fixed     fixed depth T (the test-time protocol)
  E         exit only                          (D0)
  P  (M1)   t cheap steps then j fp32 steps     (t <= 16, j <= 4)
  S  (M2)   best-Q@K over K rollouts (16 steps each) and sequential sampling stopped by the head
  E+P, E+S, S+P, E+S+P (full controller); a method may switch any of its components off (its configuration set is
  the union over its component subsets), so "contribution of X" = full minus the best method without X, at the budget
Selection: for each method and budget b, the configuration with the best dev accuracy among those with dev mean cost <= b;
it is then scored on the held-out rows. CIs: puzzle bootstrap (2000 draws) on the held-out rows; differences between methods
are paired (same held-out puzzles).
Row sets: released Sudoku models and trained seeds: dev = test rows 0-999, held out = 1000-1999. Mazes (the test set has
1000 rows): dev 0-499, held out 500-999. Deep-thinking mazes (no head, no sampler): answer-stability exit vs fixed budget,
dev rows 0-499, held out 500-999.
Records (all p5, H200): results/m5/ctl/p5/ = p5 results/m5ctl/ (tools/m5_p5_ctl_lines.py).

    python m5_ctl_ho.py                     # tables + results/breadth/m5_ctl_ho.json
"""
import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

import analyze_m5 as A
from m5_controller import fire_mask, first

ROOT = Path(os.environ.get("M5_CTL_DIR", A.HERE / "results" / "m5" / "ctl" / "p5"))
BUDGETS = [4, 8, 16, 32, 64, 128, 256, 512]
NB = 2000
RELEASED = {"sudoku_mlp": "fp32,noise0.15@s0,noise0.15@s2,noise0.1@s0,w3a,w3c,w3g32,w4c,w4t",
            "sudoku_attn": "fp32,noise0.5@s0,noise0.5@s1,noise0.5@s2,noise0.7@s0,noise0.7@s1,noise0.7@s2,w3a,w3c,w3g32,w4c,w4t",
            "hrm_sudoku": "fp32,noise1.0@s0,noise1.0@s1,noise1.0@s2,w2a,w2g32,w3c,w3g32,w4c,w4t"}
MAZES = {"maze_attn": "fp32,noise1.0@s0,noise1.5@s0,w2a,w4t", "hrm_maze": "fp32,noise1.0@s0,noise1.5@s0,w2a,w2g32"}
H = {"sudoku_mlp": 3, "sudoku_attn": 3, "hrm_sudoku": 2, "maze_attn": 3, "hrm_maze": 2}
METHODS = ["fixed", "E", "P", "S", "E+P", "E+S", "S+P", "E+S+P"]


# ----------------------------------------------------------------------------- records
def load_det(path, rows=None):
    """exit_sweep npz -> correct, q_halt, q_continue at default-H steps (rows: optional slice)."""
    if not Path(path).exists():
        return None
    d = np.load(path)
    info = json.loads(str(d["info"]))
    h = info["H_default"]
    sl = rows or slice(None)
    g = lambda k: d[k][sl, h - 1::h]  # noqa: E731
    return g("correct").astype(bool), g("qh").astype(np.float32), g("qc").astype(np.float32)


def load_m2(task, spec, sigma, K, suffix):
    fs = sorted(glob.glob(str(ROOT / "m2cells" / f"{task}__{spec}__ptrm{sigma:g}__D16_K{K}_P2{suffix}.npz")))
    if not fs:
        return None
    d = np.load(fs[0])
    return d["corr"].astype(bool), d["q"].astype(np.float32), d["pol_corr"].astype(bool)


def load_pol(path):
    if not Path(path).exists():
        return None
    d = np.load(path)
    return d["c_pol"].astype(bool)                        # [n, 16, J]


class Split:
    """one (cell, row set): deterministic trajectory, rollouts, fp32 polish."""

    def __init__(self, dc, dq, dqc, rc, rq, rpol, pc, hrm, rule, rrule):
        n = min(dc.shape[0], rc.shape[1], pc.shape[0])
        self.n = n
        self.dc, self.dq, self.dqc = dc[:n], dq[:n], (None if dqc is None or not hrm else dqc[:n])
        self.rc, self.rq, self.rpol, self.pc = rc[:, :n], rq[:, :n], rpol[:, :n], pc[:n]
        self.T = self.dc.shape[1]
        self.K = self.rc.shape[0]
        self.J = self.pc.shape[2]
        self.P = self.pc.shape[1]                            # polish available after cheap steps 1..P
        self.td = first(fire_mask(self.dq, self.dqc, rule, hrm))
        self.tr = first(fire_mask(self.rq, None, rrule, False))

    def ctl(self, D0, K, jE, jF):
        n = self.n
        i = np.arange(n)
        td = self.td
        fired = (td >= 0) & (td < D0)
        acc = np.zeros(n, bool)
        cost = np.zeros(n)
        pol = np.zeros(n)
        t = td[fired]
        if jE > 0:
            acc[fired] = self.pc[i[fired], t, jE - 1]
            cost[fired] = t + 1 + jE
            pol[fired] = jE
        else:
            acc[fired] = self.dc[i[fired], t]
            cost[fired] = t + 1
        rest = ~fired
        ii = i[rest]
        m = len(ii)
        cost[rest] = D0
        if m == 0:
            return acc, cost, pol
        if K > 0:
            tr = self.tr[:K, ii]
            F = tr >= 0
            anyf = F.any(0)
            fk = np.where(anyf, F.argmax(0), K)
            c_roll = np.where(anyf, 16 * fk + tr[np.minimum(fk, K - 1), np.arange(m)] + 1, 16 * K)
            a = np.zeros(m, bool)
            a[anyf] = self.rc[fk[anyf], ii[anyf], tr[fk[anyf], np.arange(m)[anyf]]]
            qdet = self.dq[ii, D0 - 1]
            qr = self.rq[:K, ii, -1]
            kb = qr.argmax(0)
            use_roll = qr[kb, np.arange(m)] > qdet
            fb = np.where(use_roll, self.rc[kb, ii, -1], self.dc[ii, D0 - 1])
            fcost = np.zeros(m)
            if jF > 0:
                fb = np.where(use_roll, self.rpol[kb, ii, min(jF, self.rpol.shape[2]) - 1],
                              self.pc[ii, D0 - 1, jF - 1] if D0 <= self.P else fb)
                fcost = np.where(use_roll | (D0 <= self.P), jF, 0)
            acc[rest] = np.where(anyf, a, fb)
            cost[rest] += c_roll + np.where(anyf, 0, fcost)
            pol[rest] = np.where(anyf, 0, fcost)
        else:
            if jF > 0 and D0 <= self.P:
                acc[rest] = self.pc[ii, D0 - 1, jF - 1]
                cost[rest] += jF
                pol[rest] = jF
            else:
                acc[rest] = self.dc[ii, D0 - 1]
        return acc, cost, pol

    def eval(self, cfg):
        kind = cfg[0]
        n = self.n
        if kind == "fixed":
            return self.dc[:, cfg[1] - 1], np.full(n, float(cfg[1])), np.zeros(n)
        if kind == "m1":
            t, j = cfg[1], cfg[2]
            return self.pc[:, t - 1, j - 1], np.full(n, float(t + j)), np.full(n, float(j))
        if kind == "bestq":
            K, j = cfg[1], cfg[2]
            kb = self.rq[:K, :, -1].argmax(0)
            a = self.rc[kb, np.arange(n), -1] if j == 0 else self.rpol[kb, np.arange(n), j - 1]
            return a, np.full(n, 16.0 * K + j), np.full(n, float(j))
        if kind == "ctl":
            return self.ctl(*cfg[1:])
        raise ValueError(cfg)

    def configs(self, method):
        D0s = [d for d in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256) if d <= self.T]
        Ks = [k for k in (1, 2, 4, 8, 16, 32) if k <= self.K]
        Js = [j for j in (1, 2) if j <= min(self.J, self.rpol.shape[2])]
        Jp = [j for j in (1, 2, 4) if j <= self.J]
        base = {"fixed": [("fixed", t) for t in D0s],
                "E": [("ctl", d, 0, 0, 0) for d in D0s],
                "P": [("m1", t, j) for t in range(1, self.P + 1) for j in Jp],
                "S": [("bestq", k, 0) for k in Ks] + [("ctl", 1, k, 0, 0) for k in Ks],
                "EP": [("ctl", d, 0, jE, jF) for d in D0s if d <= self.P for jE in [0] + Jp for jF in [0] + Jp if jE or jF],
                "ES": [("ctl", d, k, 0, 0) for d in D0s for k in Ks],
                "SP": [("bestq", k, j) for k in Ks for j in Js],
                "ESP": [("ctl", d, k, jE, jF) for d in D0s for k in Ks for jE in ([0] + Jp if d <= self.P else [0]) for jF in [0] + Js]}
        # a method with components X may switch any of them off: its configuration set is the union over subsets of X
        parts = {"fixed": ["fixed"], "E": ["E"], "P": ["P"], "S": ["S"], "E+P": ["E", "P", "EP"], "E+S": ["E", "S", "ES"],
                 "S+P": ["S", "P", "SP"], "E+S+P": ["E", "P", "S", "EP", "ES", "SP", "ESP"]}
        if method in parts:
            return list(dict.fromkeys(c for p in parts[method] for c in base[p]))
        raise ValueError(method)


# ----------------------------------------------------------------------------- selection and CIs
def boot_ci(x, rng, paired=None):
    x = np.asarray(x, float) if paired is None else np.asarray(x, float) - np.asarray(paired, float)
    n = len(x)
    idx = rng.integers(0, n, (NB, n))
    bs = x[idx].mean(1)
    return 100 * x.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def select_score(dev, ho, method):
    """per budget: config chosen on dev, per-instance held-out correctness and cost."""
    cfgs = dev.configs(method)
    cfgs = [c for c in cfgs if c in set(ho.configs(method))]
    dres = {c: dev.eval(c) for c in cfgs}
    out = {}
    for b in BUDGETS:
        feas = [(r[0].mean(), -r[1].mean(), c) for c, r in dres.items() if r[1].mean() <= b + 1e-9]
        if not feas:
            continue
        best = max(feas, key=lambda x: (x[0], x[1]))[2]
        a, c, p = ho.eval(best)
        out[b] = {"cfg": best, "dev_acc": 100 * dres[best][0].mean(), "acc_i": a, "cost": float(c.mean()), "pol": float(p.mean())}
    return out


def cell_splits(task, spec, fam):
    """(dev Split, held-out Split) for one cell, or None when records are missing."""
    hrm = task.startswith("hrm")
    if fam["kind"] == "maze":
        det = [load_det(ROOT / "det_o0" / task / f"{spec}.npz", slice(0, 500)), load_det(ROOT / "det_o0" / task / f"{spec}.npz", slice(500, 1000))]
        m2 = [load_m2(task, spec, 1, 16, ""), load_m2(task, spec, 1, 16, "_o500")]
        pol = [load_pol(ROOT / f"polish_o{o}" / task / f"{spec}__fp32.npz") for o in (0, 500)]
    else:
        sub = fam.get("sub", task)
        det = [load_det(ROOT / f"det_o{o}" / sub / f"{spec}.npz") for o in (0, 1000)]
        tag = fam.get("tag", "")
        m2 = [load_m2(task, spec, 0.3, fam["K"], f"{('_o1000' if o else '')}{tag}") for o in (0, 1000)]
        pol = [load_pol(ROOT / f"polish_o{o}" / sub / f"{spec}__fp32.npz") for o in (0, 1000)]
    if any(x is None for x in det + m2):
        return None
    if spec == "fp32":                         # fp32 polish of an fp32 trajectory = more fp32 steps
        # (index clipped at the last recorded step: the answer then uses fewer steps than charged, a conservative cost)
        pol = [np.stack([d[0][:, np.minimum(np.arange(t + 1, t + 5), d[0].shape[1] - 1)] for t in range(16)], 1) for d in det]
    if any(x is None for x in pol):
        return None
    return det, m2, pol, hrm


def rules_for(fam_ref_dev, task):
    dc, dq, dqc = fam_ref_dev
    hrm = task.startswith("hrm")
    ref = A.Cell(task, "fp32", "p5", dc[:, :16], dq[:, :16], qc=dqc[:, :16] if hrm else None)
    rule = A.select_rule(ref)
    rrule = A.select_rule(A.Cell(task, "fp32", "p5", dc[:, :16], dq[:, :16])) if hrm else rule
    return rule, rrule


def families():
    fams = []
    for task, cells in RELEASED.items():
        fams.append({"name": task, "task": task, "kind": "sudoku", "K": 32, "cells": cells.split(",")})
    for arch, task in (("attn512", "sudoku_attn"), ("mlp512", "sudoku_mlp")):
        for s in range(3):
            fams.append({"name": f"{arch}_s{s}", "task": task, "kind": "sudoku", "K": 16, "sub": f"var/{arch}_s{s}",
                         "tag": f"_{arch}s{s}", "cells": ["fp32", "w4c", "w3c", "w3g32"]})
    for task, cells in MAZES.items():
        fams.append({"name": task, "task": task, "kind": "maze", "K": 16, "cells": cells.split(",")})
    return fams


def run_family(fam, rng):
    task = fam["task"]
    res = {}
    fp = cell_splits(task, "fp32", fam)
    if fp is None:
        print(f"  {fam['name']}: fp32 records missing, skipped")
        return res
    rule, rrule = rules_for(fp[0][0], task)
    for spec in fam["cells"]:
        cs = cell_splits(task, spec, fam)
        if cs is None:
            print(f"  {fam['name']} {spec}: records missing")
            continue
        det, m2, pol, hrm = cs
        dev, ho = [Split(*det[k], *m2[k], pol[k], hrm, rule, rrule) for k in (0, 1)]
        sel = {m: select_score(dev, ho, m) for m in METHODS}
        # practical claim: exit (frozen rule) capped at the default depth vs the default fixed depth, held out
        a_e, c_e, _ = ho.ctl(16, 0, 0, 0)
        a_f = ho.dc[:, 15]
        claim = {"exit16": boot_ci(a_e, rng), "fixed16": boot_ci(a_f, rng), "diff": boot_ci(a_e, rng, a_f), "steps": float(c_e.mean())}
        res[spec] = {"rule": A.rule_name(*rule), "n_dev": dev.n, "n_ho": ho.n, "sel": sel, "claim": claim}
    return res


def fmt(t):
    return f"{t[0]:5.1f} [{t[1]:5.1f},{t[2]:5.1f}]"


def report(allres, rng):
    print("\n=== practical overthinking claim (held out): exit with the fp32-chosen head rule, capped at 16 steps, vs fixed 16 steps")
    rows = []
    for fam, res in allres.items():
        for spec, r in res.items():
            c = r["claim"]
            rows.append((fam, spec, c))
            print(f"  {fam:12s} {spec:14s} exit {fmt(c['exit16'])} @{c['steps']:5.2f} | fixed16 {fmt(c['fixed16'])} | diff {fmt(c['diff'])}")
    ni = sum(c["diff"][1] > -1.0 for _, _, c in rows)
    sup = sum(c["diff"][1] > 0 for _, _, c in rows)
    inf = sum(c["diff"][2] < 0 for _, _, c in rows)
    print(f"  cells {len(rows)}: non-inferior (CI low > -1 pt) {ni}, better (CI low > 0) {sup}, worse (CI high < 0) {inf}")
    print("\n=== held-out accuracy at matched budget (config chosen on dev); full = E+S+P")
    for fam, res in allres.items():
        for spec, r in res.items():
            print(f"\n  {fam} {spec} (rule {r['rule']}, dev n={r['n_dev']}, held-out n={r['n_ho']})")
            print(f"  {'budget':>6s} | " + " | ".join(f"{m:>15s}" for m in METHODS) + " | full-fixed [CI] | full-best single [CI]")
            for b in BUDGETS:
                if b not in r["sel"]["E+S+P"]:
                    continue
                cells = []
                for m in METHODS:
                    v = r["sel"][m].get(b)
                    cells.append(f"{100 * v['acc_i'].mean():5.1f} @{v['cost']:6.1f}" if v else f"{'-':>13s}")
                full = r["sel"]["E+S+P"][b]["acc_i"]
                d_fix = boot_ci(full, rng, r["sel"]["fixed"][b]["acc_i"]) if b in r["sel"]["fixed"] else None
                singles = [(r["sel"][m][b]["acc_i"].mean(), m) for m in ("fixed", "E", "P", "S") if b in r["sel"][m]]
                bm = max(singles)[1]
                d_best = boot_ci(full, rng, r["sel"][bm][b]["acc_i"])
                print(f"  {b:6d} | " + " | ".join(f"{c:>15s}" for c in cells) + f" | {fmt(d_fix) if d_fix else '-':>20s} | {fmt(d_best)} ({bm})")
    print("\n=== ablation: mean over cells of held-out accuracy, and component contributions (full minus full-without-X), pooled paired CI")
    for b in (16, 64, 256):
        accs = {m: [] for m in METHODS}
        contrib = {"E": [], "S": [], "P": []}
        for fam, res in allres.items():
            for spec, r in res.items():
                if all(b in r["sel"][m] for m in METHODS):
                    for m in METHODS:
                        accs[m].append(r["sel"][m][b]["acc_i"])
                    full = r["sel"]["E+S+P"][b]["acc_i"].astype(float)
                    for x, wo in (("E", "S+P"), ("S", "E+P"), ("P", "E+S")):
                        contrib[x].append(full - r["sel"][wo][b]["acc_i"])
        if not accs["fixed"]:
            continue
        line = " ".join(f"{m} {100 * np.mean([a.mean() for a in accs[m]]):5.1f}" for m in METHODS)
        con = []
        for x, lst in contrib.items():
            means = np.array([d.mean() for d in lst])
            bs = np.mean([d[rng.integers(0, len(d), (NB, len(d)))].mean(1) for d in lst], 0)
            con.append(f"{x}: {100 * means.mean():+5.2f} [{100 * np.percentile(bs, 2.5):+5.2f},{100 * np.percentile(bs, 97.5):+5.2f}]")
        print(f"  budget {b:4d} ({len(accs['fixed'])} cells): {line}\n      contribution " + "  ".join(con))


def dt_ho(rng):
    """deep-thinking mazes: stability exit (p, cap B) vs fixed B, chosen on rows 0-499, scored on 500-999."""
    out = {}
    print("\n=== DT mazes (p5): answer-stability exit vs fixed budget, chosen on rows 0-499, held out 500-999")
    for size in (13, 25, 33):
        for f in sorted(glob.glob(str(ROOT / f"dt_maze{size}" / "*.npz"))):
            if ".tmp" in f:
                continue
            d = np.load(f)
            spec = json.loads(str(d["info"]))["spec"]
            c = d["correct"].astype(bool)
            T = c.shape[1]
            cell = A.Cell(f"dt_maze{size}", spec, "p5", c, np.zeros(c.shape, np.float32), chg=d["chg"].astype(np.int32))
            tf = {p: first(A.stable_mask(cell, p)) for p in (3, 5, 10, 20, 30)}
            caps = [b for b in (10, 20, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500) if b <= T]

            def ev(cfg, rows):
                kind, p, B = cfg
                if kind == "fixed":
                    return c[rows, B - 1], np.full(len(rows), float(B))
                t = tf[p][rows]
                t = np.where((t >= 0) & (t < B), t, B - 1)
                return c[rows, t], (t + 1).astype(float)
            dev, ho = np.arange(0, 500), np.arange(500, 1000)
            row = {}
            for b in [b for b in (25, 50, 100, 200, 300, 500) if b <= T]:
                pick = {}
                for m, cf in (("stab", [("stab", p, B) for p in tf for B in caps]), ("fixed", [("fixed", 0, B) for B in caps])):
                    feas = [(ev(x, dev)[0].mean(), -ev(x, dev)[1].mean(), x) for x in cf if ev(x, dev)[1].mean() <= b]
                    if feas:
                        pick[m] = max(feas, key=lambda z: (z[0], z[1]))[2]
                if len(pick) < 2:
                    continue
                a_s, c_s = ev(pick["stab"], ho)
                a_f, _ = ev(pick["fixed"], ho)
                row[b] = {"stab": boot_ci(a_s, rng), "cost": float(c_s.mean()), "fixed": boot_ci(a_f, rng), "diff": boot_ci(a_s, rng, a_f),
                          "cfg": pick["stab"]}
            out[f"dt{size}|{spec}"] = row
            print(f"  {size}x{size} {spec:12s} " + " ".join(f"B{b}: {v['stab'][0]:5.1f}@{v['cost']:5.1f} vs {v['fixed'][0]:5.1f} (d {v['diff'][0]:+4.1f} [{v['diff'][1]:+4.1f},{v['diff'][2]:+4.1f}])"
                                                           for b, v in row.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(A.RB / "m5_ctl_ho.json"))
    ap.add_argument("--fams", default="")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    allres = {}
    for fam in families():
        if a.fams and fam["name"] not in a.fams.split(","):
            continue
        allres[fam["name"]] = run_family(fam, rng)
    report(allres, rng)
    dt = dt_ho(rng)

    def ser(r):
        return {spec: {"rule": v["rule"], "n_dev": v["n_dev"], "n_ho": v["n_ho"], "claim": v["claim"],
                       "sel": {m: {str(b): {"cfg": list(x["cfg"]), "dev_acc": x["dev_acc"], "acc": boot_ci(x["acc_i"], rng),
                                            "cost": x["cost"], "fp32_steps": x["pol"]} for b, x in s.items()} for m, s in v["sel"].items()}}
                for spec, v in r.items()}
    json.dump({"families": {k: ser(v) for k, v in allres.items()}, "dt": dt}, open(a.json, "w"), indent=1,
              default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("wrote", a.json)


if __name__ == "__main__":
    main()
