#!/usr/bin/env python3
"""E9B: label-free stop signals against the halting head, on the existing p4d controller records (no model is run).

E9 (sched_sim.py) found an oracle that stops every puzzle at its cheapest correct recorded state 8.8 points above the
head-gated controller with finishing on the 20 Sudoku cells: puzzles whose deterministic trajectory is right at some loop
and wrong later, which the head never flags. This script asks whether a signal computable without labels catches them.

Signals per puzzle and supervision step t (T = 64 Sudoku, 16 Maze), all from records already on disk:
  unch[t]   read-out unchanged since step t-1: the exit_sweep `chg` array (answer cells whose argmax changed since the
            previous H=1 sub-step) summed over the H sub-steps ending at t is zero (sufficient for hash equality)
  chk[t]    checker satisfied: on Sudoku the constraint checker equals `correct` exactly (unique solutions; verified on
            every m2 cell file), so chk = dc; on mazes no per-step checker was recorded and dc stands in (optimistic)
  q[t]      halting logit (q_halt - q_continue for HRM), the quantity the frozen head rule thresholds
Rules -> first firing step ta (-1 never) and extra loops spent after the answer state (qpeak observes the drop one loop
later and keeps the previous read-out):
  head            the frozen rule of m5_ctl_ho (Split.td)
  stab(s)         unch for s consecutive steps, s = 1..4
  chk             first step with chk
  qpeak(m)        first local maximum of q at t >= m, answer state t, cost t + 2
  qstall(delta,m) first t >= m with q_t - q_{t-1} < delta (delta = dev quantiles of the increments)
  head|stab(s)    whichever fires first;   head&chk  both must have fired
A rule replaces the head inside the E and E+P configuration families of m5_ctl_ho (D0, jE, jF chosen on dev with the
rule's own parameter); w8c finishing at cost r = 8 / b_eff as in m5_w8 / sched_sim. Two protocols on the same held-out
rows: A = per puzzle, honest cost, m5 selection (dev mean cost <= b); B = E9's shared budget (water-fill at b loops per
puzzle). Baselines: fixed depth, the head, E9's oracle (E+P family) and oracle_det (deterministic path only).
Rows: Sudoku dev 0-999 / held out 1000-1999; Maze dev 0-499 / held out 500-999. CIs: paired puzzle bootstrap per cell,
1000 draws, averaged over the cells of a family (the m5_w8.pooled construction).

    python stop_signal.py            # tables + results/breadth/stop_signal.json (a few minutes on the CPU)
    python stop_signal.py --quick    # two named cells, fewer draws
"""
import argparse
import copy
import json
import time

import numpy as np

import analyze_m5 as A
import m5_addons as AD
import m5_ctl_ho as C
import m5_w8 as W
import sched_sim as S
from m5_controller import first

BUD = S.BUD
NB = 1000
FAMS = ("E", "E+P")
RULES = ["head", "stab", "head|stab", "chk", "head&chk", "qpeak", "qstall"]
QUANT = (0.1, 0.25, 0.5, 0.75)
SHOW_B = (8, 16, 32)


# ----------------------------------------------------------------------------- signals
def load_unch(fam, spec, k):
    """unch[n, T]: read-out unchanged between supervision steps t-1 and t (chg summed over the H sub-steps is 0)."""
    sub = fam.get("sub", fam["task"])
    if fam["kind"] == "maze":
        path, sl = C.ROOT / "det_o0" / sub / f"{spec}.npz", slice(500 * k, 500 * (k + 1))
    else:
        path, sl = C.ROOT / f"det_o{1000 * k}" / sub / f"{spec}.npz", slice(None)
    d = np.load(path)
    h = json.loads(str(d["info"]))["H_default"]
    chg = d["chg"][sl].astype(np.int32)
    n, K = chg.shape
    T = K // h
    s = np.zeros((n, T), np.int32)
    for t in range(1, T):
        s[:, t] = chg[:, t * h:t * h + h].sum(1)
    unch = s == 0
    unch[:, 0] = False
    return unch


def conf(sp):
    return sp.dq - sp.dqc if sp.dqc is not None else sp.dq


def t_stab(unch, s):
    return first(A.run_len(unch) >= s)


def t_qpeak(cf, m):
    T = cf.shape[1]
    peak = np.zeros(cf.shape, bool)
    peak[:, :T - 1] = cf[:, :T - 1] > cf[:, 1:]
    peak[:, 1:T - 1] &= cf[:, 1:T - 1] >= cf[:, :T - 2]
    peak[:, :m] = False
    return first(peak)


def t_qstall(cf, delta, m):
    st = np.zeros(cf.shape, bool)
    st[:, 1:] = (cf[:, 1:] - cf[:, :-1]) < delta
    st[:, :max(m, 1)] = False
    return first(st)


def combine(ta, tb, how):
    a, b = ta >= 0, tb >= 0
    if how == "or":
        return np.where(a & b, np.minimum(ta, tb), np.where(a, ta, np.where(b, tb, -1)))
    return np.where(a & b, np.maximum(ta, tb), -1)


def variants(rule, sp, unch, deltas):
    """list of (label, ta, extra) for one rule on one Split."""
    cf = conf(sp)
    if rule == "head":
        return [("head", sp.td, 0)]
    if rule == "stab":
        return [(f"stab{s}", t_stab(unch, s), 0) for s in (1, 2, 3, 4)]
    if rule == "head|stab":
        return [(f"head|stab{s}", combine(sp.td, t_stab(unch, s), "or"), 0) for s in (1, 2, 3, 4)]
    if rule == "chk":
        return [("chk", first(sp.dc), 0)]
    if rule == "head&chk":
        return [("head&chk", combine(sp.td, first(sp.dc), "and"), 0)]
    if rule == "qpeak":
        return [(f"qpeak{m}", t_qpeak(cf, m), 1) for m in (0, 1, 2, 4)]
    if rule == "qstall":
        return [(f"qstall[{q:g}]m{m}", t_qstall(cf, d, m), 0) for q, d in zip(QUANT, deltas) for m in (1, 2, 4)]
    raise ValueError(rule)


def with_rule(sp, ta, extra):
    """Split whose head is replaced by the rule: td = ta (+ extra); with extra = 1 the answer (and the finishing start
    state) at loop ta + 1 is the kept read-out of state ta, so cost = ta + 2 and the answer is state ta."""
    sp2 = copy.copy(sp)
    if extra == 0:
        sp2.td = ta
        return sp2
    n, i = sp.n, np.arange(sp.n)
    f = (ta >= 0) & (ta + 1 < sp.T)
    dc, pc = sp.dc.copy(), sp.pc.copy()
    dc[i[f], ta[f] + 1] = sp.dc[i[f], ta[f]]
    g = f & (ta + 1 < sp.P)
    pc[i[g], ta[g] + 1] = sp.pc[i[g], ta[g]]
    sp2.dc, sp2.pc, sp2.td = dc, pc, np.where(f, ta + 1, -1)
    return sp2


# ----------------------------------------------------------------------------- protocol A (per puzzle, honest cost)
def eval_alone(sp, cfg, r):
    a, c, p = sp.eval(cfg)
    return a.astype(bool), c + (r - 1.0) * p


def select_alone(dev_sps, ho_sps, base, fam, r):
    """dev_sps / ho_sps: {label: Split}; base: (dev, ho) unmodified Splits for the rule-independent configs (fixed, m1).
    Returns per budget (label, cfg, acc_i held out, cost held out)."""
    res = {}
    for lab, dsp in dev_sps.items():
        cfgs = dsp.configs(fam) if fam != "fixed" else [("fixed", d) for d in range(1, dsp.T + 1)]
        for cfg in cfgs:
            if cfg[0] != "ctl":
                if ("_", cfg) in res:
                    continue                                          # rule-independent configs once, on the base split
                a, c = eval_alone(base[0], cfg, r)
                res[("_", cfg)] = (a.mean(), c.mean())
            else:
                a, c = eval_alone(dsp, cfg, r)
                res[(lab, cfg)] = (a.mean(), c.mean())
    out = {}
    for b in BUD:
        feas = [(v[0], -v[1], k) for k, v in res.items() if v[1] <= b + 1e-9]
        if not feas:
            continue
        lab, cfg = max(feas, key=lambda z: (z[0], z[1]))[2]
        a, c = eval_alone(base[1] if lab == "_" else ho_sps[lab], cfg, r)
        out[b] = {"lab": lab, "cfg": cfg, "acc_i": a, "cost": float(c.mean())}
    return out


# ----------------------------------------------------------------------------- protocol B (E9 shared budget)
def select_sched(dev_sps, base, fam, r, cache):
    """per budget (label, cfg): best dev accuracy under the shared budget b x n_dev (tie: less spend); label "_" marks a
    rule-independent config (fixed, m1) evaluated on the base split."""
    out = {}
    for b in BUD:
        best = None
        for lab, dsp in dev_sps.items():
            w = np.ones((1, dsp.n))
            cfgs = dsp.configs(fam) if fam != "fixed" else [("fixed", min(b, dsp.T))]
            for cfg in cfgs:
                key = (lab if cfg[0] == "ctl" else "_", cfg)
                if key not in cache:
                    cache[key] = S.build(dsp if key[0] != "_" else base, cfg, r)
                a, c = S.run(cache[key], w, [b * dsp.n])
                k = (round(float(a[0]), 9), -float(c[0]))
                if best is None or k > best[0]:
                    best = (k, key[0], cfg)
        out[b] = best[1:]
    return out


# ----------------------------------------------------------------------------- driver
def pooled(ds):
    m = float(np.mean([d.mean() for d in ds]))
    return [100 * m, 100 * float(np.percentile(np.mean(ds, 0), 2.5)), 100 * float(np.percentile(np.mean(ds, 0), 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=str(A.RB / "stop_signal.json"))
    a = ap.parse_args()
    nb = 200 if a.quick else NB
    rng = np.random.default_rng(0)
    t0 = time.time()
    out = {"meta": {"budgets": BUD, "NB": nb, "rules": RULES, "families": FAMS, "quantiles": QUANT,
                    "cost_model": "compressed loop 1, w8c finishing loop 8/b_eff", "split": S.__doc__.split("Rows: ")[1].split("\n")[0]},
           "cells": {}}
    for fam, spec in S.cell_list(a.quick):
        key = f"{fam['name']}|{spec}"
        fp = C.cell_splits(fam["task"], "fp32", fam)
        rule, rrule = C.rules_for(fp[0][0], fam["task"])
        cs = W.splits(fam, spec, "w8c")
        if cs is None:
            print(key, "w8c records missing, skipped")
            continue
        det, m2, pc, hrm = cs
        dev, ho = [C.Split(*det[k], *m2[k], pc[k], hrm, rule, rrule) for k in (0, 1)]
        unch = [load_unch(fam, spec, k)[:sp.n] for k, sp in ((0, dev), (1, ho))]
        r = 8.0 / AD.b_eff(spec)
        inc = np.diff(conf(dev), axis=1).ravel()
        deltas = [float(np.quantile(inc, q)) for q in QUANT]
        n = ho.n
        cell = {"kind": fam["kind"], "r": r, "rule": A.rule_name(*rule), "n_dev": dev.n, "n_ho": n, "T": ho.T, "deltas": deltas,
                "fire_ho": {}, "A": {}, "B": {}, "ci": {"A": {}, "B": {}}}
        # rule variants as Splits
        dsp, hsp = {}, {}
        for rl in RULES:
            dv, hv = variants(rl, dev, unch[0], deltas), variants(rl, ho, unch[1], deltas)
            dsp[rl] = {lab: with_rule(dev, ta, ex) for lab, ta, ex in dv}
            hsp[rl] = {lab: with_rule(ho, ta, ex) for lab, ta, ex in hv}
            for lab, ta, ex in hv:
                cell["fire_ho"][lab] = {"rate": float((ta >= 0).mean()), "mean_step": float(np.where(ta >= 0, ta + 1 + ex, ho.T).mean()),
                                        "acc_at_fire": float(ho.dc[np.arange(n), np.maximum(ta, 0)][ta >= 0].mean()) if (ta >= 0).any() else None}
        oc, ocd = S.oracle_cost(ho, r), np.where(ho.dc, np.arange(ho.T)[None, :] + 1.0, np.inf).min(1)
        wboot = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(nb)]).astype(float)
        w1 = np.ones((1, n))
        # ---------------- protocol A
        accA = {}
        selA = {"fixed": select_alone({"_": dev}, {"_": ho}, (dev, ho), "fixed", r)}
        for fm in FAMS:
            for rl in RULES:
                selA[f"{rl}/{fm}"] = select_alone(dsp[rl], hsp[rl], (dev, ho), fm, r)
        for b in BUD:
            row = {}
            accA[b] = {}
            for name, sel in selA.items():
                if b not in sel:
                    continue
                v = sel[b]
                row[name] = {"lab": v["lab"], "cfg": [str(x) for x in v["cfg"]], "acc": 100 * float(v["acc_i"].mean()), "cost": v["cost"]}
                accA[b][name] = (wboot @ v["acc_i"]) / n
            for nm, c in (("oracle", oc), ("oracle_det", ocd)):
                acc, cost = S.run_oracle(c, w1, [b * n])
                row[nm] = {"acc": 100 * float(acc[0]), "cost": float(cost[0])}
                accA[b][nm], _ = S.run_oracle(c, wboot, b * wboot.sum(1))
            cell["A"][str(b)] = row
        # ---------------- protocol B
        cache = {}
        selB = {"fixed": select_sched({"_": dev}, dev, "fixed", r, cache)}
        for fm in FAMS:
            for rl in RULES:
                selB[f"{rl}/{fm}"] = select_sched(dsp[rl], dev, fm, r, cache)
        hprog = {}
        accB = {}
        for b in BUD:
            row = {}
            accB[b] = {}
            for name, sel in selB.items():
                lab, cfg = sel[b]
                rl = name.split("/")[0]
                sp = ho if lab == "_" else hsp[rl][lab]
                k = (lab, cfg)
                if k not in hprog:
                    hprog[k] = S.build(sp, cfg, r)
                acc, cost = S.run(hprog[k], w1, [b * n])
                row[name] = {"lab": lab, "cfg": [str(x) for x in cfg], "acc": 100 * float(acc[0]), "cost": float(cost[0])}
                accB[b][name], _ = S.run(hprog[k], wboot, b * wboot.sum(1))
            for nm, c in (("oracle", oc), ("oracle_det", ocd)):
                acc, cost = S.run_oracle(c, w1, [b * n])
                row[nm] = {"acc": 100 * float(acc[0]), "cost": float(cost[0])}
                accB[b][nm], _ = S.run_oracle(c, wboot, b * wboot.sum(1))
            cell["B"][str(b)] = row
        # per-s stability (each s alone, and head|stab with that s), E+P family, protocol B at SHOW_B
        cell["per_s"] = {}
        for lab in [f"stab{s}" for s in (1, 2, 3, 4)] + [f"head|stab{s}" for s in (1, 2, 3, 4)]:
            rl = "stab" if lab.startswith("stab") else "head|stab"
            sel = select_sched({lab: dsp[rl][lab]}, dev, "E+P", r, cache)
            cell["per_s"][lab] = {}
            for b in SHOW_B:
                l2, cfg = sel[b]
                sp = ho if l2 == "_" else hsp[rl][l2]
                k = (l2, cfg)
                if k not in hprog:
                    hprog[k] = S.build(sp, cfg, r)
                acc, cost = S.run(hprog[k], w1, [b * n])
                accB[b][f"{lab}/E+P"], _ = S.run(hprog[k], wboot, b * wboot.sum(1))
                cell["per_s"][lab][str(b)] = {"cfg": [str(x) for x in cfg], "acc": 100 * float(acc[0]), "cost": float(cost[0])}
        cell["_boot"] = {"A": accA, "B": accB}
        out["cells"][key] = cell
        gB = cell["B"]["16"]
        print(f"{key:22s} r={r:.2f} T={ho.T} | B b=16: fixed={gB['fixed']['acc']:.1f} head/E+P={gB['head/E+P']['acc']:.1f}@{gB['head/E+P']['cost']:.1f} "
              f"head|stab={gB['head|stab/E+P']['acc']:.1f}@{gB['head|stab/E+P']['cost']:.1f} ({gB['head|stab/E+P']['lab']}) "
              f"stab={gB['stab/E+P']['acc']:.1f} ({gB['stab/E+P']['lab']}) chk={gB['chk/E+P']['acc']:.1f} qpeak={gB['qpeak/E+P']['acc']:.1f} "
              f"qstall={gB['qstall/E+P']['acc']:.1f} oracle={gB['oracle']['acc']:.1f} | A b=16: head={cell['A']['16'].get('head/E+P', {}).get('acc', float('nan')):.1f} "
              f"head|stab={cell['A']['16'].get('head|stab/E+P', {}).get('acc', float('nan')):.1f}  [{time.time() - t0:.0f}s]")
    # ------------------------------------------------------------------ summaries
    groups = {"sudoku20": [k for k, v in out["cells"].items() if v["kind"] == "sudoku"],
              "maze4": [k for k, v in out["cells"].items() if v["kind"] == "maze"]}
    names = ["fixed"] + [f"{rl}/{fm}" for fm in FAMS for rl in RULES] + ["oracle", "oracle_det"]
    out["summary"] = {}
    for g, keys in groups.items():
        if not keys:
            continue
        cells = [out["cells"][k] for k in keys]
        s = {"cells": keys, "A": {}, "B": {}}
        for prot in ("A", "B"):
            for b in BUD:
                have = [nm for nm in names if all(nm in c[prot][str(b)] for c in cells)]
                mean = {nm: {"acc": float(np.mean([c[prot][str(b)][nm]["acc"] for c in cells])),
                             "cost": float(np.mean([c[prot][str(b)][nm]["cost"] for c in cells])),
                             "labs": sorted(set(c[prot][str(b)][nm].get("lab", "") for c in cells))} for nm in have}
                ci = {}
                bt = lambda nm: [c["_boot"][prot][b][nm] for c in cells]  # noqa: E731
                for fm in FAMS:
                    h = f"head/{fm}"
                    if h not in have:
                        continue
                    for rl in RULES:
                        nm = f"{rl}/{fm}"
                        if nm in have and rl != "head":
                            d = [x - y for x, y in zip(bt(nm), bt(h))]
                            ci[f"{nm}-{h}"] = pooled(d)
                            gap = np.mean([x - y for x, y in zip(bt("oracle"), bt(h))], 0)
                            num = np.mean(d, 0)
                            rec = num / np.where(gap > 0, gap, np.nan)
                            pt = (mean[nm]["acc"] - mean[h]["acc"]) / max(mean["oracle"]["acc"] - mean[h]["acc"], 1e-9)
                            ci[f"rec:{nm}"] = [100 * pt, 100 * float(np.nanpercentile(rec, 2.5)), 100 * float(np.nanpercentile(rec, 97.5))]
                    ci[f"oracle-{h}"] = pooled([x - y for x, y in zip(bt("oracle"), bt(h))])
                    ci[f"{h}-fixed"] = pooled([x - y for x, y in zip(bt(h), bt("fixed"))])
                if prot == "B" and b in SHOW_B:
                    for lab in cells[0]["per_s"]:
                        nm = f"{lab}/E+P"
                        mean[nm] = {"acc": float(np.mean([c["per_s"][lab][str(b)]["acc"] for c in cells])),
                                    "cost": float(np.mean([c["per_s"][lab][str(b)]["cost"] for c in cells]))}
                        ci[f"{nm}-head/E+P"] = pooled([x - y for x, y in zip(bt(nm), bt("head/E+P"))])
                s[prot][str(b)] = {"mean": mean, "ci": ci}
        # signal statistics on the held-out rows (mean over cells): fire rate, mean loops at fire, accuracy of the state fired on
        labs = ["head", "stab1", "stab2", "stab3", "stab4", "chk", "head&chk", "qpeak1", "qstall[0.5]m1"]
        s["fire"] = {lab: {k: float(np.mean([c["fire_ho"][lab][k] for c in cells if c["fire_ho"][lab][k] is not None]))
                           for k in ("rate", "mean_step", "acc_at_fire")} for lab in labs if all(lab in c["fire_ho"] for c in cells)}
        out["summary"][g] = s
    for c in out["cells"].values():
        c.pop("_boot")
    # ------------------------------------------------------------------ print
    for g, s in out["summary"].items():
        for prot, title in (("B", "E9 shared budget"), ("A", "per puzzle, honest cost, m5 selection")):
            for fm in FAMS:
                print(f"\n=== {g} ({len(s['cells'])} cells), protocol {prot} ({title}), family {fm}: held-out acc @ mean loops; diff vs head [CI]; recovery of oracle gap %")
                cols = [f"{rl}/{fm}" for rl in RULES]
                print(f"  {'b':>3s} | {'fixed':>12s} | " + " | ".join(f"{rl:>12s}" for rl in RULES) + f" | {'oracle':>12s} | " + " | ".join(f"{rl:>22s}" for rl in RULES[1:]))
                for b in BUD:
                    m, ci = s[prot][str(b)]["mean"], s[prot][str(b)]["ci"]
                    if any(cn not in m for cn in cols):
                        continue
                    f = lambda nm: f"{m[nm]['acc']:5.1f} @{m[nm]['cost']:5.1f}"  # noqa: E731
                    dd = []
                    for rl in RULES[1:]:
                        k = f"{rl}/{fm}-head/{fm}"
                        dd.append(f"{ci[k][0]:+5.1f} [{ci[k][1]:+.1f},{ci[k][2]:+.1f}] {ci['rec:' + rl + '/' + fm][0]:4.0f}%")
                    print(f"  {b:3d} | {f('fixed'):>12s} | " + " | ".join(f"{f(cn):>12s}" for cn in cols) + f" | {f('oracle'):>12s} | " + " | ".join(f"{x:>22s}" for x in dd))
    for g, s in out["summary"].items():
        print(f"\n=== {g}: stability per s, protocol B, family E+P: acc @ loops, diff vs head [CI]")
        print(f"  {'b':>3s} | {'head':>12s} | " + " | ".join(f"{lab:>32s}" for lab in cells_labs(s)))
        for b in SHOW_B:
            m, ci = s["B"][str(b)]["mean"], s["B"][str(b)]["ci"]
            print(f"  {b:3d} | {m['head/E+P']['acc']:5.1f} @{m['head/E+P']['cost']:5.1f} | " + " | ".join(
                f"{m[lab + '/E+P']['acc']:5.1f} @{m[lab + '/E+P']['cost']:5.1f} {ci[lab + '/E+P-head/E+P'][0]:+5.1f} [{ci[lab + '/E+P-head/E+P'][1]:+.1f},{ci[lab + '/E+P-head/E+P'][2]:+.1f}]"
                for lab in cells_labs(s)))
        print(f"\n=== {g}: signals on the held-out rows (mean over cells): fire rate, mean loops at fire (T if never), accuracy of the fired state")
        for lab, v in s["fire"].items():
            print(f"  {lab:14s} rate {v['rate']:.3f}  loops {v['mean_step']:5.2f}  acc@fire {v['acc_at_fire']:.3f}")
        print(f"\n=== {g}: oracle decomposition at b = 16 (protocol B): oracle (E+P paths) {s['B']['16']['mean']['oracle']['acc']:.1f}, "
              f"oracle_det (deterministic path only) {s['B']['16']['mean']['oracle_det']['acc']:.1f}, head/E+P {s['B']['16']['mean']['head/E+P']['acc']:.1f}, "
              f"chk/E+P {s['B']['16']['mean']['chk/E+P']['acc']:.1f}")
    json.dump(out, open(a.json, "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    print("\nwrote", a.json, f"({time.time() - t0:.0f}s)")


def cells_labs(s):
    return [f"stab{k}" for k in (1, 2, 3, 4)] + [f"head|stab{k}" for k in (1, 2, 3, 4)]


if __name__ == "__main__":
    main()
