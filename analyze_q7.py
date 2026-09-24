#!/usr/bin/env python3
"""Q7 read-out (trainable basin width): per run and checkpoint, fp32 / w4t / w4c exact accuracy, w4t
fidelity, the noise cliff sigma_1/2 (sigma at which the MEDIAN of 3 draws falls to half of fp32, linear
interpolation on the sigma grid) with the draw range at sigma 0.15 and 0.2, and ||J||_F^2/dim and the top
singular value of one recursion step. Then the equal-fp32 comparison: for every regularised checkpoint,
the same-seed-set baseline's w4t fidelity and sigma_1/2 interpolated at that checkpoint's fp32 accuracy.

    bash EdgeTRM/q7/pull_q7.sh            # eval JSONs + metrics.jsonl from p4d and Babel (no weights)
    python EdgeTRM/analyze_q7.py          # tables
    python EdgeTRM/analyze_q7.py --lambda # also the Lambda-trained baselines (fp32/w4t along training)

Only written records are read (eval_step_N.json after training wrote them); a checkpoint whose eval is
incomplete (fewer than 27 cells) is shown with its cell count and never used in a comparison.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results" / "q7"
LAMBDA = Path(__file__).resolve().parent / "results" / "breadth" / "lambda"
NCELLS = 27
SIG = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]


def sigma_half(recs, fp):
    """sigma where the median draw's exact accuracy crosses fp/2 (linear in sigma); None if never."""
    by = defaultdict(list)
    for r in recs:
        if r["quant"]["spec"].startswith("noise"):
            by[r["quant"]["sigma"]].append(r["pexact"])
    xs = [0.0] + sorted(by)
    ys = [fp] + [float(np.median(by[s])) for s in sorted(by)]
    half = fp / 2
    for i in range(1, len(xs)):
        if ys[i] < half:
            x0, x1, y0, y1 = xs[i - 1], xs[i], ys[i - 1], ys[i]
            return x0 + (y0 - half) * (x1 - x0) / max(y0 - y1, 1e-9)
    return None


def load_evals():
    rows = []
    for f in sorted(glob.glob(str(ROOT / "*" / "*" / "eval_step_*.json"))):
        host, run = Path(f).parts[-3], Path(f).parts[-2]
        step = int(re.search(r"eval_step_(\d+)", f).group(1))
        d = json.load(open(f))
        recs = d["records"]
        q = {(r["quant"]["spec"], r.get("draw", 0)): r for r in recs}
        fp = q.get(("fp32", 0))
        row = {"host": host, "run": run, "step": step, "ncells": len(recs),
               "arm": re.sub(r"_s\d+$", "", run), "seed": int(run.rsplit("_s", 1)[1]) if "_s" in run else -1,
               **{k: d.get(k) for k in ("jac_fro2", "jac_spec_mean", "jac_spec_max")}}
        if fp:
            row["fp32"] = fp["pexact"]
            for spec in ("w4t", "w4c", "w4g32", "w3g32", "w8c"):
                if (spec, 0) in q:
                    row[spec] = q[(spec, 0)]["pexact"]
                    row[spec + "_fid"] = q[(spec, 0)]["fidelity_H"]
            row["s_half"] = sigma_half(recs, fp["pexact"]) if len(recs) >= NCELLS else None
            for s in (0.15, 0.2):
                v = [r["pexact"] for r in recs if r["quant"]["spec"] == f"noise{s}"]
                row[f"n{s}"] = (min(v), float(np.median(v)), max(v)) if v else None
        rows.append(row)
    return rows


def fmt(v, p=1, pct=True):
    if v is None:
        return "  -  "
    return f"{v*100:.{p}f}" if pct else f"{v:.{p}f}"


def table(rows):
    print(f"{'host':6s} {'run':16s} {'step':>6s} {'n':>3s} {'fp32':>5s} {'w4t':>5s} {'fidH':>5s} {'w4c':>5s} "
          f"{'s1/2':>5s} {'n0.15 min/med/max':>18s} {'n0.2 med':>8s} {'J_F2':>6s} {'J_top':>6s}")
    for r in sorted(rows, key=lambda r: (r["arm"], r["seed"], r["host"], r["step"])):
        n15 = r.get("n0.15")
        n2 = r.get("n0.2")
        print(f"{r['host']:6s} {r['run']:16s} {r['step']:6d} {r['ncells']:3d} {fmt(r.get('fp32')):>5s} "
              f"{fmt(r.get('w4t')):>5s} {fmt(r.get('w4t_fid'), 3, False):>5s} {fmt(r.get('w4c')):>5s} "
              f"{fmt(r.get('s_half'), 3, False):>5s} "
              f"{(f'{n15[0]*100:.1f}/{n15[1]*100:.1f}/{n15[2]*100:.1f}' if n15 else '-'):>18s} "
              f"{(f'{n2[1]*100:.1f}' if n2 else '-'):>8s} {fmt(r.get('jac_fro2'), 3, False):>6s} "
              f"{fmt(r.get('jac_spec_mean'), 2, False):>6s}")


def equal_fp32(rows):
    """Baseline curve (fp32 -> metric) per host from the pooled base seeds; interpolate at each checkpoint."""
    print("\nEqual-fp32 comparison (baseline = pooled q7_base seeds on the same host, linear in fp32):")
    print(f"{'host':6s} {'run':16s} {'step':>6s} {'fp32':>5s} | {'fidH':>5s} vs base {'':5s} | {'s1/2':>5s} vs base")
    full = [r for r in rows if r["ncells"] >= NCELLS and r.get("fp32") is not None]
    for host in sorted({r["host"] for r in full}):
        base = sorted([r for r in full if r["host"] == host and r["arm"] == "q7_base"], key=lambda r: r["fp32"])
        if len(base) < 2:
            continue
        bx = np.array([r["fp32"] for r in base])
        for key in ("w4t_fid", "s_half"):
            pass
        for r in sorted([r for r in full if r["host"] == host and r["arm"] != "q7_base"],
                        key=lambda r: (r["arm"], r["seed"], r["step"])):
            out = []
            for key in ("w4t_fid", "s_half"):
                pts = [(b["fp32"], b[key]) for b in base if b.get(key) is not None]
                if len(pts) < 2 or r.get(key) is None:
                    out.append("  -  ")
                    continue
                pts.sort()
                bv = float(np.interp(r["fp32"], [p[0] for p in pts], [p[1] for p in pts]))
                inside = pts[0][0] <= r["fp32"] <= pts[-1][0]
                out.append(f"{r[key]:.3f} vs {bv:.3f}{'' if inside else '*'}")
            print(f"{host:6s} {r['run']:16s} {r['step']:6d} {r['fp32']*100:5.1f} | {out[0]:>16s} | {out[1]:>16s}")
    print("(* = fp32 outside the baseline's range: extrapolated, not comparable)")


def lambda_baselines():
    print("\nLambda-trained baselines (Sep 13-14; different 1k training subset; w4t from variant_*.json, n=1000):")
    for s in (0, 1, 2):
        line = []
        for f in sorted(glob.glob(str(LAMBDA / f"variant_trm_sudoku_mlp_w256_s{s}_*.json")),
                        key=lambda p: int(re.search(r"_(\d+)(?:_noise)?\.json", p).group(1))):
            if f.endswith("_noise.json"):
                continue
            step = int(re.search(r"_(\d+)\.json", f).group(1))
            recs = json.load(open(f))["records"]
            fp = next((r for r in recs if r["quant"]["spec"] == "fp32" and r["nsup"] == 16 and r["H_cycles"] == 3), None)
            w4t = next((r for r in recs if r["quant"]["spec"] == "w4t" and r["nsup"] == 16 and r["H_cycles"] == 3), None)
            if fp and w4t:
                line.append(f"{step}:{fp['pexact']*100:.1f}/{w4t['pexact']*100:.1f}({w4t['fidelity_H']:.2f})")
        print(f"  mlp_w256_s{s}: " + "  ".join(line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda", dest="lam", action="store_true")
    a = ap.parse_args()
    rows = load_evals()
    if not rows:
        print(f"no eval JSONs under {ROOT} (run q7/pull_q7.sh)")
    else:
        table(rows)
        equal_fp32(rows)
    if a.lam:
        lambda_baselines()


if __name__ == "__main__":
    main()
