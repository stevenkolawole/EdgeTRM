#!/usr/bin/env python3
"""Read-out for x2_decomp.py: per cell displacement / slowdown / path independence against X1's catch-up label.

    python x2_decomp_analyze.py [--decomp 'results/breadth/babel/x2/decomp/decomp_*.json'] [--out results/breadth/x2_decomp.json]

Label (pre-registered in reports/edgetrm_exp/X2.md): from X1's deep records (results/breadth/p4d/x1/deep_<task>_*.json,
1800 puzzles, p4d): catch-up = |gap16| >= 1.5 points and gap512 - gap16 >= |gap16| / 2; constant/widening = |gap16| >=
1.5 and not catch-up; |gap16| < 1.5 = anchor (not scored).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE / "results" / "breadth"
MIX = {"sudoku_mlp": "MLP", "sudoku_attn": "attn", "hrm_sudoku": "HRM"}


def unpack(h, K):
    v = int(h, 16)
    return np.array([(v >> k) & 1 for k in range(K)], bool)


def x1_labels():
    lab = {}
    for task in MIX:
        recs, rows = {}, None
        for f in sorted(glob.glob(str(B / "p4d" / "x1" / f"deep_{task}_*.json"))):
            d = json.load(open(f))
            rows = rows or d["rows"]
            for r in d["records"]:
                recs[r["spec"]] = r
        acc = {s: np.stack([unpack(h, 512) for h in r["bits"]]).mean(0) for s, r in recs.items()}
        f = acc["fp32"]
        for s, a_ in acc.items():
            if s == "fp32":
                continue
            g16, g64, g512 = (100 * (a_[t - 1] - f[t - 1]) for t in (16, 64, 512))
            if abs(g16) < 1.5:
                lab_ = "anchor"
            elif g512 - g16 >= abs(g16) / 2:
                lab_ = "catch-up"
            else:
                lab_ = "constant"
            lab[(task, s)] = {"gap16": g16, "gap64": g64, "gap512": g512, "label": lab_,
                              "phi": recs[s]["fid_H_mean"]}
    return lab


def auroc(pos, neg):
    if not pos or not neg:
        return None
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decomp", default=str(B / "babel" / "x2" / "decomp" / "decomp_*.json"))
    ap.add_argument("--out", default=str(B / "x2_decomp.json"))
    a = ap.parse_args()
    lab = x1_labels()
    rows, fp = [], {}
    for f in sorted(glob.glob(a.decomp)):
        d = json.load(open(f))
        task, T = d["task"], d["T"]
        for r in d["records"]:
            if r["spec"] == "fp32":
                fp[task] = r
        for r in d["records"]:
            if r["spec"] == "fp32" or any(x["task"] == task and x["spec"] == r["spec"] for x in rows):
                continue
            f0 = fp[task]
            dm = {t: float(np.median(v)) for t, v in r["disp"].items()}
            s_q = np.array(r["settle"]["0.001"], float)
            s_f = np.array(f0["settle"]["0.001"], float)
            s_q2 = np.array(r["settle"]["0.01"], float)
            s_f2 = np.array(f0["settle"]["0.01"], float)
            acc = {t: float(np.mean(v)) for t, v in r["correct"].items()}
            facc = {t: float(np.mean(v)) for t, v in f0["correct"].items()}
            pi, pif = r["pi"], f0["pi"]
            L = lab.get((task, r["spec"]), {})
            rows.append({
                "task": task, "mixer": MIX[task], "spec": r["spec"], "label": L.get("label"), "x1": L,
                "disp": dm, "P": dm["2048"] / max(dm["64"], 1e-9), "P512": dm["512"] / max(dm["64"], 1e-9),
                "settle_med": float(np.median(s_q)), "settle_fp32_med": float(np.median(s_f)),
                "frac_never_settle": float(np.mean(s_q >= T)), "frac_never_settle_fp32": float(np.mean(s_f >= T)),
                "slowdown": float(np.median(s_q) / max(np.median(s_f), 1)),
                "slowdown_1e2": float(np.median(s_q2) / max(np.median(s_f2), 1)),
                "log2_settle_ratio_med": float(np.median(np.log2(np.clip(s_q2, 1, None) / np.clip(s_f2, 1, None)))),
                "gap": {t: 100 * (acc[t] - facc[t]) for t in acc},
                "pi_same": float(np.mean(pi["frac_same_as_default"])), "pi_same_fp32": float(np.mean(pif["frac_same_as_default"])),
                "pi_distinct": float(np.mean(pi["n_distinct"])), "pi_distinct_fp32": float(np.mean(pif["n_distinct"])),
                "pi_dist": float(np.median(pi["rel_dist_to_default"])),
                "pi_dist_fp32": float(np.median(pif["rel_dist_to_default"]))})
    scored = [r for r in rows if r["label"] in ("catch-up", "constant")]
    res = {"rows": rows, "fp32": {t: {"settle_med": float(np.median(r["settle"]["0.001"])),
                                      "pi_same": float(np.mean(r["pi"]["frac_same_as_default"])),
                                      "pi_distinct": float(np.mean(r["pi"]["n_distinct"]))} for t, r in fp.items()}}
    # score: higher = more likely constant gap (displacement-dominated)
    scores = {"P (disp2048/disp64)": lambda r: r["P"], "P512 (disp512/disp64)": lambda r: r["P512"],
              "disp2048": lambda r: r["disp"]["2048"], "-slowdown (1e-2)": lambda r: -r["slowdown_1e2"],
              "-log2 settle ratio (1e-2)": lambda r: -r["log2_settle_ratio_med"],
              "PI same-as-default": lambda r: r["pi_same"], "-PI distinct": lambda r: -r["pi_distinct"]}
    res["auroc_constant_vs_catchup"] = {}
    for name, fn in scores.items():
        pos = [fn(r) for r in scored if r["label"] == "constant"]
        neg = [fn(r) for r in scored if r["label"] == "catch-up"]
        res["auroc_constant_vs_catchup"][name] = auroc(pos, neg)
    json.dump(res, open(a.out, "w"), indent=1)
    f2 = lambda x, p=3: "--" if x is None else f"{x:.{p}f}"
    print("| model | cell | X1 label (gap 16/64/512, 1800 puzzles) | disp 16 / 64 / 512 / 2048 | P = d2048/d64 | settle med q / fp32 "
          "(1e-3) | never settles q / fp32 | slowdown 1e-2 | PI same-as-default q / fp32 | PI distinct q / fp32 | gap here 16/64/512/2048 |")
    print("|---|" * 1 + "---|" * 10)
    for r in sorted(rows, key=lambda r: (r["mixer"], r["spec"])):
        x1 = r["x1"]
        print(f"| {r['mixer']} | {r['spec']} | {r['label']} ({f2(x1.get('gap16'),1)}/{f2(x1.get('gap64'),1)}/{f2(x1.get('gap512'),1)}) | "
              + " / ".join(f2(r["disp"][t]) for t in ("16", "64", "512", "2048")) + f" | {f2(r['P'],2)} | "
              f"{r['settle_med']:.0f} / {r['settle_fp32_med']:.0f} | {f2(r['frac_never_settle'],2)} / {f2(r['frac_never_settle_fp32'],2)} | "
              f"{f2(r['slowdown_1e2'],2)} | {f2(r['pi_same'],2)} / {f2(r['pi_same_fp32'],2)} | {f2(r['pi_distinct'],2)} / "
              f"{f2(r['pi_distinct_fp32'],2)} | " + " / ".join(f2(r["gap"][t], 1) for t in ("16", "64", "512", "2048")) + " |")
    n_c = sum(r["label"] == "catch-up" for r in scored)
    print(f"\nscored cells: {len(scored)} ({n_c} catch-up, {len(scored)-n_c} constant); AUROC for 'constant gap' "
          f"(higher score = constant):")
    for k, v in res["auroc_constant_vs_catchup"].items():
        print(f"   {k}: {f2(v, 2)}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
