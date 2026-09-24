#!/usr/bin/env python3
"""X6 read-outs: phase-boundary transfer to DEQs (MDEQ ImageNet) and a diffusion sampler (DDPM CIFAR-10).

    python analyze_x6.py deq  [results_dir]   # accuracy / residual / displacement vs sigma and solver budget
    python analyze_x6.py diff [results_dir]   # FID / paired drift vs sigma and sampler step count

Default results_dir: results/transfer/babel. Only files whose job log ended in "done rc=0" are pulled
there; noise cells are reported as mean and [min, max] over the three weight-noise seeds (n = images or
samples per cell, stated in the header).
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QORDER = ["fp32", "w8c", "w4g32", "w4g128", "w4c", "w4a", "w4t"]


def _key(spec):
    if spec.startswith("noise"):
        return (1, float(spec[5:]))
    return (0 if spec == "fp32" else 2, QORDER.index(spec) if spec in QORDER else 99)


def _agg(vals):
    v = np.array(vals, float)
    if len(v) == 1:
        return f"{v[0]:6.1f}"
    return f"{v.mean():6.1f} [{v.min():.1f},{v.max():.1f}]"


def deq(rdir=None):
    rdir = Path(rdir or HERE / "results" / "transfer" / "babel")
    files = sorted(glob.glob(str(rdir / "deq_*.json")))
    recs = []
    for f in files:
        d = json.load(open(f))
        for r in d["records"]:
            r["_file"] = Path(f).name
            recs.append(r)
    if not recs:
        print("no deq records in", rdir)
        return
    for model in sorted({r["model"] for r in recs}):
        R = [r for r in recs if r["model"] == model]
        n = sorted({r["n"] for r in R})
        for solver in ("broyden", "picard"):
            S = [r for r in R if r["solver"] == solver]
            if not S:
                continue
            budgets = S[0]["budgets"]
            show = [k for k in (2, 4, 8, 16, 26, 60, 100) if k in budgets]
            by = defaultdict(list)
            for r in S:
                if r["spec"] == "fp32" and any(x["spec"] == "fp32" and x is not r and x["seed"] == r["seed"] for x in by["fp32"]):
                    continue
                by[r["spec"]].append(r)
            print(f"\n=== {model}  solver={solver}  n={n} images/cell; top-1 % by iteration budget "
                  f"(noise: mean [min,max] over seeds)")
            print(f"{'cell':10s} {'#':>2s} " + " ".join(f"{'k='+str(k):>18s}" for k in show))
            for spec in sorted(by, key=_key):
                rs = by[spec]
                if spec == "fp32":
                    rs = rs[:1]
                row = []
                for k in show:
                    bi = budgets.index(k)
                    row.append(_agg([r["by_budget"][bi]["top1"] * 100 for r in rs]))
                print(f"{spec:10s} {len(rs):2d} " + " ".join(f"{x:>18s}" for x in row))
            kk = [k for k in (26, 100) if k in budgets]
            print(f"\n  residual / convergence / displacement at k={kk} (medians over images; mean over seeds)")
            print(f"  {'cell':10s} " + " ".join(f"{h:>12s}" for k in kk for h in
                  (f'res@{k}', f'conv1e-2@{k}', f'disp@{k}', f'cos@{k}', f'AUROC@{k}')))
            for spec in sorted(by, key=_key):
                rs = by[spec] if spec != "fp32" else by[spec][:1]
                cells = []
                for k in kk:
                    bi = budgets.index(k)
                    B = [r["by_budget"][bi] for r in rs]
                    au = [b["auroc_res_wrong"] for b in B if b["auroc_res_wrong"] is not None]
                    cells += [f"{np.mean([b['res_median'] for b in B]):.1e}",
                              f"{np.mean([b['frac_res_le']['0.01'] for b in B])*100:.1f}%",
                              f"{np.mean([b['disp_median'] for b in B]):.3f}",
                              f"{np.mean([b['cos_mean'] for b in B]):.3f}",
                              f"{np.mean(au):.2f}" if au else "-"]
                print(f"  {spec:10s} " + " ".join(f"{c:>12s}" for c in cells))


def diff(rdir=None):
    rdir = Path(rdir or HERE / "results" / "transfer" / "babel")
    recs = []
    for f in sorted(glob.glob(str(rdir / "diff_*.json"))):
        recs += json.load(open(f))["records"]
    if not recs:
        print("no diff records in", rdir)
        return
    samplers = sorted({r["sampler"] for r in recs}, key=lambda s: (s[:4], int(s[4:])))
    n = sorted({r["n"] for r in recs})
    by = defaultdict(lambda: defaultdict(list))
    for r in recs:
        if r["spec"] == "fp32" and by["fp32"][r["sampler"]]:
            continue
        by[r["spec"]][r["sampler"]].append(r)
    for metric, lab in (("fid_train", "FID vs CIFAR-10 train"), ("fid_vs_fp32", "FID vs fp32 samples"),
                        ("featcos_mean", "paired Inception-feature cosine to fp32 sample"),
                        ("rmse_mean", "paired pixel RMSE to fp32 sample")):
        print(f"\n=== DDPM CIFAR-10: {lab}; n={n} samples/cell (noise: mean [min,max] over seeds)")
        print(f"{'cell':10s} " + " ".join(f"{s:>20s}" for s in samplers))
        for spec in sorted(by, key=_key):
            row = []
            for s in samplers:
                rs = by[spec].get(s, [])
                if not rs:
                    row.append("-")
                    continue
                v = np.array([r[metric] for r in rs], float)
                fmt = "{:.3f}" if metric in ("featcos_mean", "rmse_mean") else "{:.1f}"
                row.append(fmt.format(v.mean()) + (f" [{fmt.format(v.min())},{fmt.format(v.max())}]" if len(v) > 1 else ""))
            print(f"{spec:10s} " + " ".join(f"{x:>20s}" for x in row))


if __name__ == "__main__":
    globals()[sys.argv[1]](*(sys.argv[2:]))
