#!/usr/bin/env python3
"""Read-out of the looped-LM return test, exactly as pre-registered (prereg/X4_return_test_prereg.md, 8a14e1f).

    python ret_analyze.py [results/looped/babel/ret]

Per cell: bf16 / compressed / hybrid greedy accuracy, retained, collapse (< 50% of bf16), φ (prompt final-latent
cosine), cell rho (median over examples of 1/t_c), return rates (teacher forced at t=1; answer level).
P1: pooled AUROC of rho and of (1-φ) for collapse; leave-one-model-out accuracy of ONE cut (fit on 3 models by
    minimising errors, midpoint of the optimal interval; scored on the 4th), for rho and for φ.
    P1 holds iff AUROC(rho) >= 0.95 and LOMO acc(rho) >= 0.95 and LOMO acc(rho) > LOMO acc(φ).
P2: among problems bf16 solves, in cells whose compressed correctness is mixed, AUROC of answer-level return and
    of 1/t_c... (higher t_c = more room = expected correct) for predicting compressed-model correctness, pooled.
    P2 holds iff pooled AUROC >= 0.8.
Scope check: return rate at t=1 of the held cells (w8c, w4g32).
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def auroc(s_pos, s_neg):
    s_pos, s_neg = np.asarray(s_pos, float), np.asarray(s_neg, float)
    if not len(s_pos) or not len(s_neg):
        return float("nan")
    return float(((s_pos[:, None] > s_neg[None, :]).sum() + 0.5 * (s_pos[:, None] == s_neg[None, :]).sum())
                 / (len(s_pos) * len(s_neg)))


def best_cut(x, lab):
    """Cut on score x (predict collapse iff x > t) minimising errors; midpoint of the optimal interval."""
    o = np.argsort(x)
    xs, lb = np.asarray(x)[o], np.asarray(lab)[o].astype(int)
    held_after = np.r_[(1 - lb).sum() - np.r_[0, np.cumsum(1 - lb)][:-1], 0]   # held cells above position i
    col_before = np.r_[0, np.cumsum(lb)]                                         # collapsed at or below
    err = held_after + col_before
    e = err.min()
    ids = np.where(err == e)[0]

    def at(i):
        if i == 0:
            return xs[0] - 1e-6
        if i == len(xs):
            return xs[-1] + 1e-6
        return (xs[i - 1] + xs[i]) / 2
    return (at(ids[0]) + at(ids[-1])) / 2


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results/looped/babel/ret"
    cells = []
    for f in sorted(root.glob("*_k*_*.json")):
        if f.name.endswith("_bf16.json"):
            continue
        d = json.load(open(f))
        s, m = d["summary"], d["meta"]
        rec = d["records"]
        rho = [min(r["rho"], 1e6) for r in rec]
        cells.append({"model": m["model"], "quant": m["quant"]["spec"], "k": m["k"], "acc_fp": s["acc_fp"],
                      "acc_q": s["acc_q"], "acc_h": s["acc_hybrid"], "ret": s["retained"], "fid": s["fid"],
                      "rho": float(np.median(rho)), "ret_tf": s["return_tf_rate"], "ret_ans": s["return_answer_rate"],
                      "tc": s["tc_median"], "records": rec})
    if not cells:
        print("no records under", root)
        return
    for c in cells:
        c["collapse"] = (c["ret"] is not None) and c["ret"] < 0.5
    print("## Return test on looped LMs (Babel L40S, GSM8K n=100, batch 16)\n")
    print("| model | k | cell | bf16 | compressed | hybrid | retained | collapse | φ | cell ρ (median 1/t_c) | "
          "return (tf, t=1) | return (answer) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in sorted(cells, key=lambda c: (c["model"], -c["fid"])):
        print(f"| {c['model']} | {c['k']} | {c['quant']} | {100 * c['acc_fp']:.0f} | {100 * c['acc_q']:.0f} | "
              f"{100 * c['acc_h']:.0f} | {100 * c['ret']:.0f}% | {'yes' if c['collapse'] else ''} | {c['fid']:.3f} | "
              f"{c['rho']:.2f} | {100 * c['ret_tf']:.0f}% | {100 * c['ret_ans']:.0f}% |")
    lab = np.array([c["collapse"] for c in cells])
    rho = np.array([c["rho"] for c in cells])
    phi = np.array([c["fid"] for c in cells])
    models = sorted(set(c["model"] for c in cells))
    out = {}
    print("\n### P1: one pooled threshold?\n")
    for name, score in (("rho", rho), ("1-φ", 1 - phi)):
        au = auroc(score[lab], score[~lab])
        correct, n = 0, 0
        per = {}
        for mdl in models:
            te = np.array([c["model"] == mdl for c in cells])
            tr = ~te
            if lab[tr].all() or not lab[tr].any():
                t = np.median(score[tr])
            else:
                t = best_cut(score[tr], lab[tr])
            ok = ((score[te] > t) == lab[te])
            per[mdl] = float(ok.mean())
            correct += int(ok.sum())
            n += int(te.sum())
        lomo = correct / n
        out[name] = {"auroc": au, "lomo_acc": lomo, "per_model": per}
        print(f"- {name}: pooled AUROC {au:.3f}; leave-one-model-out accuracy of one cut {lomo:.3f} "
              f"({', '.join(f'{m} {v:.2f}' for m, v in per.items())})")
    p1 = out["rho"]["auroc"] >= 0.95 and out["rho"]["lomo_acc"] >= 0.95 and out["rho"]["lomo_acc"] > out["1-φ"]["lomo_acc"]
    print(f"\n**P1 {'HOLDS' if p1 else 'FAILS'}** (criterion: AUROC(ρ) ≥ 0.95, LOMO(ρ) ≥ 0.95, LOMO(ρ) > LOMO(φ)).")
    print("\n### P2: per-example return predicts per-example correctness\n")
    pos_a, neg_a, pos_t, neg_t = [], [], [], []
    used = []
    for c in cells:
        rs = [r for r in c["records"] if r["fp_correct"]]
        qc = [r["q_correct"] for r in rs]
        if not rs or all(qc) or not any(qc):
            continue
        used.append(f"{c['model']}/{c['quant']}")
        for r in rs:
            (pos_a if r["q_correct"] else neg_a).append(r["return_answer"])
            (pos_t if r["q_correct"] else neg_t).append(min(r["t_c"], 1e6))
    au_a, au_t = auroc(pos_a, neg_a), auroc(pos_t, neg_t)
    print(f"cells with mixed compressed correctness among bf16-solved problems: {len(used)} ({', '.join(used)})")
    print(f"- answer-level return -> compressed correct: AUROC {au_a:.3f} (n = {len(pos_a)} correct, {len(neg_a)} wrong)")
    print(f"- t_c (distance to the edge in units of the compression error) -> compressed correct: AUROC {au_t:.3f}")
    p2 = max(au_a, au_t) >= 0.8 if not math.isnan(au_a) else False
    print(f"\n**P2 {'HOLDS' if p2 else 'FAILS'}** (criterion: pooled AUROC ≥ 0.8).")
    held = [c for c in cells if c["quant"] in ("w8c", "w4g32")]
    if held:
        print("\n### Scope check: do held cells return?\n")
        for c in held:
            print(f"- {c['model']} {c['quant']}: return (tf, t=1) {100 * c['ret_tf']:.0f}%, answer-level {100 * c['ret_ans']:.0f}%, "
                  f"median t_c {c['tc']:.2f}")
    json.dump({"cells": [{k: v for k, v in c.items() if k != "records"} for c in cells], "P1": out, "P1_holds": p1,
               "P2": {"auroc_answer": au_a, "auroc_tc": au_t, "cells": used}, "P2_holds": p2},
              open(root / "ret_readout.json", "w"), indent=1)


if __name__ == "__main__":
    main()
