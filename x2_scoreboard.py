#!/usr/bin/env python3
"""X2 blind width-prediction scoreboard: X2's predictions (x2_predict.py, written to Babel predictions.jsonl from X2's own
S before any width is read) against the measured sigma_1/2 of X5's new mixer seeds (results/q7/babel/var_<m>/
eval_step_65100.json, Babel, n=1000, 3 noise draws per sigma) and of the QAT / fine-tune models (X2 width sweeps,
results/breadth/babel/x2/sens/wid_<name>_seed*.json). Band: leave-one-out error of the five Babel TRM/HRM models that
fixed delta* = 0.4033 (median x1.46, worst x1.92).

    python x2_scoreboard.py [--pred results/breadth/babel/x2/pred/predictions.jsonl]
"""
import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sens_analyze as A

HERE = Path(__file__).resolve().parent
BAND = (1.46, 1.92)


def x5_width(m):
    f = HERE / "results" / "q7" / "babel" / f"var_{m}" / "eval_step_65100.json"
    if not f.exists():                                         # staged copy of X5's Babel record (DONE marker seen)
        f = HERE / "results" / "breadth" / "babel" / "x2" / "pred" / "x5_eval" / f"var_{m}_eval_step_65100.json"
    if not f.exists():
        return None
    d = json.load(open(f))
    sa, a0 = defaultdict(list), None
    for r in d["records"]:
        s = r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
        if s == "fp32":
            a0 = r["pexact"]
        elif s.startswith("noise"):
            sa[float(s[5:])].append(r["pexact"])
    sh, note = A.sigma_half(sa, a0)
    return {"sigma_half": sh, "bound": note, "fp32": a0, "source": str(f.relative_to(HERE)), "n": d["n"]}


def x2_width(name):
    fs = sorted(glob.glob(str(HERE / "results" / "breadth" / "babel" / "x2" / "sens" / f"wid_{name}_seed*.json")))
    if len(fs) < 3:
        return None
    sa, a0 = defaultdict(list), None
    for f in fs:
        for r in json.load(open(f))["records"]:
            s = r["quant"]["spec"]
            if s == "fp32":
                a0 = a0 or r["pexact"]
            elif s.startswith("noise"):
                sa[float(s[5:])].append(r["pexact"])
    sh, note = A.sigma_half(sa, a0)
    return {"sigma_half": sh, "bound": note, "fp32": a0, "source": "X2 width sweep " + name, "n": 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(HERE / "results" / "breadth" / "babel" / "x2" / "pred" / "predictions.jsonl"))
    a = ap.parse_args()
    first = {}
    for line in open(a.pred):
        p = json.loads(line)
        first.setdefault(p["model"], p)                       # the earliest prediction counts
    print("| model | S (X2) | predicted sigma_1/2 (time) | measured sigma_1/2 (source) | ratio | within x1.46 / x1.92 |")
    print("|---|---|---|---|---|---|")
    rat = []
    for m, p in first.items():
        if m.endswith("_babel"):
            continue                                          # post-hoc (variants), scored elsewhere
        w = (x5_width(m[3:]) or x2_width(m)) if m.startswith("x5_") else x2_width(m)
        if not w:
            print(f"| {m} | {p['S']:.3f} | {p['sigma_half_pred']:.3f} ({p['time_utc']}) | pending | | |")
            continue
        if w["sigma_half"] is None:
            print(f"| {m} | {p['S']:.3f} | {p['sigma_half_pred']:.3f} ({p['time_utc']}) | {w['bound']} ({w['source']}) | | |")
            continue
        r = p["sigma_half_pred"] / w["sigma_half"]
        rat.append(r)
        L = abs(np.log(r))
        print(f"| {m} | {p['S']:.3f} | {p['sigma_half_pred']:.3f} ({p['time_utc']}) | {w['sigma_half']:.3f} (fp32 {w['fp32']:.3f}; "
              f"{w['source']}) | {r:.2f} | {'yes' if L <= np.log(BAND[0]) else 'no'} / {'yes' if L <= np.log(BAND[1]) else 'NO'} |")
    if rat:
        L = np.abs(np.log(rat))
        print(f"\nscored {len(rat)}: median x{np.exp(np.median(L)):.2f}, worst x{np.exp(L.max()):.2f}; "
              f"band median x{BAND[0]}, worst x{BAND[1]}")


if __name__ == "__main__":
    main()
