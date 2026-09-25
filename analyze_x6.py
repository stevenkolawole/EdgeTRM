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


def deqlm(rdir=None):
    """DEQ-Transformer WT103 test perplexity / residual / cosine vs sigma and solver cap."""
    rdir = Path(rdir or HERE / "results" / "transfer" / "babel")
    recs = []
    for f in sorted(glob.glob(str(rdir / "deqlm_*.json"))):
        recs += json.load(open(f))["records"]
    if not recs:
        print("no deqlm records in", rdir)
        return
    for solver in ("anderson", "picard"):
        S = [r for r in recs if r["solver"] == solver]
        if not S:
            continue
        ks = sorted({r["budget"] for r in S})
        by = defaultdict(lambda: defaultdict(list))
        for r in S:
            if r["spec"] == "fp32" and by["fp32"][r["budget"]]:
                continue
            by[r["spec"]][r["budget"]].append(r)
        toks = sorted({r["tokens"] for r in S})
        for metric, lab, fmt in (("ppl", "test perplexity", "{:.2f}"), ("res_median", "median rel. residual", "{:.1e}"),
                                 ("cos_mean", "cosine to fp32 state (same cap)", "{:.3f}")):
            print(f"\n=== DEQ-Transformer WT103 {solver}: {lab} by solver cap; tokens={toks} (noise: mean [min,max] over seeds)")
            print(f"{'cell':10s} {'#':>2s} " + " ".join(f"{'k='+str(k):>24s}" for k in ks))
            for spec in sorted(by, key=_key):
                row = []
                for k in ks:
                    v = np.array([r[metric] for r in by[spec].get(k, [])], float)
                    if not len(v):
                        row.append("-")
                        continue
                    s = fmt.format(np.mean(v)) if metric != "ppl" else fmt.format(np.exp(np.mean(np.log(v))))
                    if len(v) > 1:
                        s += f" [{fmt.format(v.min())},{fmt.format(v.max())}]"
                    row.append(s)
                n = max(len(by[spec].get(k, [])) for k in ks)
                print(f"{spec:10s} {n:2d} " + " ".join(f"{x:>24s}" for x in row))


def diff(rdir=None):
    rdir = Path(rdir or HERE / "results" / "transfer" / "babel")
    recs = []
    for f in sorted(glob.glob(str(rdir / "diff*.json"))):
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


def deqfig(rdir=None, model="mdeq_small"):
    """Top-1 and residual vs sigma (log x) per solver budget, seeds as ranges; quantizers placed at their
    mean relative weight error (the noise dial's sigma equals that error by construction)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rdir = Path(rdir or HERE / "results" / "transfer" / "babel")
    recs = []
    for f in sorted(glob.glob(str(rdir / "deq_*.json"))):
        recs += [r for r in json.load(open(f))["records"] if r["model"] == model]
    if not recs:
        print("no records")
        return
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8))
    cols = {4: "#b0c4de", 8: "#6495ed", 26: "#1f4e9c", 100: "#0b1f44"}
    fp = [r for r in recs if r["spec"] == "fp32" and r["solver"] == "broyden"][0]
    budgets = fp["budgets"]
    for ax, solver, metric in ((axs[0], "broyden", "top1"), (axs[1], "picard", "top1"), (axs[2], "broyden", "res_median")):
        R = [r for r in recs if r["solver"] == solver]
        sig = sorted({float(r["spec"][5:]) for r in R if r["spec"].startswith("noise")})
        for k, c in cols.items():
            if k not in budgets:
                continue
            bi = budgets.index(k)
            m, lo, hi = [], [], []
            for s in sig:
                v = [r["by_budget"][bi][metric] for r in R if r["spec"] == f"noise{s:g}" or r["spec"] == f"noise{s}"]
                v = np.array(v) * (100 if metric == "top1" else 1)
                m.append(v.mean()); lo.append(v.min()); hi.append(v.max())
            ax.plot(sig, m, "-o", color=c, ms=3, label=f"k={k}")
            ax.fill_between(sig, lo, hi, color=c, alpha=0.2)
            f0 = [r for r in R if r["spec"] == "fp32"][0]["by_budget"][bi][metric] * (100 if metric == "top1" else 1)
            ax.axhline(f0, color=c, ls=":", lw=0.8)
        for r in R:
            if r["spec"] in QORDER[1:]:
                bi = budgets.index(26)
                y = r["by_budget"][bi][metric] * (100 if metric == "top1" else 1)
                ax.plot(r["quant"]["weight_rel_err_mean"], y, "k^", ms=5)
                ax.annotate(r["spec"], (r["quant"]["weight_rel_err_mean"], y), fontsize=6, xytext=(2, 2),
                            textcoords="offset points")
        ax.set_xscale("log")
        if metric == "res_median":
            ax.set_yscale("log")
        ax.set_title(f"{model} {solver}: {'top-1 %' if metric == 'top1' else 'median rel. residual'}", fontsize=9)
    ax = axs[3]                                                  # retained accuracy vs fidelity
    bi = budgets.index(100)
    f0 = fp["by_budget"][bi]["top1"]
    for r in recs:
        if r["solver"] != "broyden" or r["spec"] == "fp32":
            continue
        b = r["by_budget"][bi]
        isq = not r["spec"].startswith("noise")
        ax.plot(b["cos_mean"], b["top1"] / f0 * 100, "^" if isq else "o", color="k" if isq else "#1f4e9c",
                ms=5 if isq else 3.5, alpha=1 if isq else 0.7)
        if isq:
            ax.annotate(r["spec"], (b["cos_mean"], b["top1"] / f0 * 100), fontsize=6, xytext=(2, -8),
                        textcoords="offset points")
    ax.set_xlabel("cosine of equilibrium to fp32 equilibrium (k=100)")
    ax.set_title(f"{model} broyden: top-1 retained (% of fp32)", fontsize=9)
    ax.grid(alpha=0.3)
    for a_ in axs[:3]:
        a_.set_xlabel("sigma (x tensor std) / quantizer rel. weight error", fontsize=8)
    axs[0].legend(fontsize=7)
    out = HERE / "figures" / f"x6_{model}.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print("wrote", out)


def deqlmfig(rdir=None):
    """DEQ-Transformer: test ppl (log) vs sigma per Anderson cap, quantizers at their rel. weight error;
    median residual at the largest cap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rdir = Path(rdir or HERE / "results" / "transfer" / "babel")
    recs = []
    for f in sorted(glob.glob(str(rdir / "deqlm_*.json"))):
        recs += [r for r in json.load(open(f))["records"] if r["solver"] == "anderson"]
    ks = sorted({r["budget"] for r in recs})
    cols = dict(zip(ks, plt.cm.Blues(np.linspace(0.35, 1, len(ks)))))
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.6))
    sig = sorted({float(r["spec"][5:]) for r in recs if r["spec"].startswith("noise")})
    for k in ks:
        for ax, metric in ((axs[0], "ppl"), (axs[1], "res_median")):
            if metric == "res_median" and k != ks[-1]:
                continue
            m, lo, hi = [], [], []
            for sg in sig:
                v = np.array([r[metric] for r in recs if r["budget"] == k and r["spec"].startswith("noise")
                              and abs(float(r["spec"][5:]) - sg) < 1e-9])
                m.append(np.exp(np.log(v).mean())); lo.append(v.min()); hi.append(v.max())
            ax.plot(sig, m, "-o", ms=3, color=cols[k], label=f"cap {k}")
            ax.fill_between(sig, lo, hi, color=cols[k], alpha=0.2)
            f0 = [r[metric] for r in recs if r["budget"] == k and r["spec"] == "fp32"][0]
            ax.axhline(f0, color=cols[k], ls=":", lw=0.8)
            for r in recs:
                if r["budget"] == k and r["spec"] in QORDER[1:]:
                    ax.plot(r["quant"]["weight_rel_err_mean"], r[metric], "k^", ms=4)
                    if k == ks[-1]:
                        ax.annotate(r["spec"], (r["quant"]["weight_rel_err_mean"], r[metric]), fontsize=6,
                                    xytext=(2, 3), textcoords="offset points")
    for ax, t in ((axs[0], "DEQ-Transformer WT103: test perplexity (Anderson)"),
                  (axs[1], f"median rel. residual at cap {ks[-1]}")):
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t, fontsize=9)
        ax.set_xlabel("sigma (x tensor std) / quantizer rel. weight error", fontsize=8)
    axs[0].legend(fontsize=6, ncol=2)
    out = HERE / "figures" / "x6_deq_transformer.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote", out)


def difffig(rdir=None):
    """DDPM CIFAR-10: FID (vs train) and FID vs fp32 samples against sigma, one line per sampler/steps;
    DDIM (no fresh noise) and DDPM ancestral (fresh noise every step) in separate panels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rdir = Path(rdir or HERE / "results" / "transfer" / "babel")
    recs = []
    for f in sorted(glob.glob(str(rdir / "diff*.json"))):
        recs += json.load(open(f))["records"]
    samplers = sorted({r["sampler"] for r in recs}, key=lambda s: (s[:4], int(s[4:])))
    sig = sorted({float(r["spec"][5:]) for r in recs if r["spec"].startswith("noise")})
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.6))
    for j, (kind, metric) in enumerate((("ddim", "fid_train"), ("ddpm", "fid_train"), ("ddim", "fid_vs_fp32"),
                                        ("ddpm", "fid_vs_fp32"))):
        ax = axs[j]
        ss = [x for x in samplers if x.startswith(kind)]
        cols = dict(zip(ss, plt.cm.viridis(np.linspace(0, 0.85, len(ss)))))
        for sp in ss:
            m, lo, hi, xs = [], [], [], []
            for sg in sig:
                v = np.array([r[metric] for r in recs if r["sampler"] == sp and r["spec"].startswith("noise")
                              and abs(float(r["spec"][5:]) - sg) < 1e-9])
                if not len(v):
                    continue
                xs.append(sg); m.append(v.mean()); lo.append(v.min()); hi.append(v.max())
            ax.plot(xs, m, "-o", ms=3, color=cols[sp], label=f"{sp[:4].upper()} {sp[4:]} steps")
            ax.fill_between(xs, lo, hi, color=cols[sp], alpha=0.15)
            f0 = [r[metric] for r in recs if r["sampler"] == sp and r["spec"] == "fp32"]
            if f0 and metric == "fid_train":
                ax.axhline(f0[0], color=cols[sp], ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("weight noise sigma (x tensor std)", fontsize=8)
        ax.set_title(f"{kind.upper()} ({'no fresh noise' if kind == 'ddim' else 'fresh noise each step'}): "
                     f"{'FID vs train' if metric == 'fid_train' else 'FID vs fp32 samples'}", fontsize=8)
        ax.legend(fontsize=6)
    out = HERE / "figures" / "x6_ddpm_cifar10.png"
    fig.tight_layout(); fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    globals()[sys.argv[1]](*(sys.argv[2:]))
