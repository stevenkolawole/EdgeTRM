#!/usr/bin/env python3
"""E6 read-out: perturb-every-step (dither) on the third and fourth looped families (MLSys 2027).

DeepThinking maze records (dt_exit.py, Babel L40S, ~/breadth_results/e6/dt_maze<size>/<spec>[_m<master>].npz):
per-iteration correctness [n, T] per arm; arms per grid: RTN (<grid>), dither (<grid>@dither<k>), gdither
(<grid>@gdither<k>), fp32 reference. MDEQ-Small records (deq_sweep.py, ~/x6/results/e6_deq_small_<spec>.{json,npz}):
per-image top-1 by budget per (arm, solver).

Accuracy of a stochastic arm = mean over its K re-draw sequences, per item, then over items (M2 convention). Paired
intervals = per-item mean over K, bootstrap over items (4000 resamples, seed 0), reported at the last iteration /
budget and at the ICLR reading point (DeepThinking: iteration 300 and T; MDEQ: Broyden budget 26 and 100).
"RTN survives" = RTN within 10 points of fp32 at the last iteration (READOUT_SEP29B criterion).

    python EdgeTRM/e6_families.py --dt_root EdgeTRM/results/breadth/babel/e6 --deq_root EdgeTRM/results/transfer/babel/e6
    python EdgeTRM/e6_families.py --dt_root ~/breadth_results/e6 --deq_root ~/x6/results          # on Babel
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

NB, BSEED = 4000, 0


def boot_diff(a, b, nb=NB, seed=BSEED):
    """a, b: per-item accuracies (K averaged). Returns (mean diff, lo, hi) of a - b, paired over items."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (nb, len(d)))
    m = d[idx].mean(1)
    return float(d.mean() * 100), float(np.percentile(m, 2.5) * 100), float(np.percentile(m, 97.5) * 100)


def fmt_ci(t):
    return f"{t[0]:+.1f} [{t[1]:+.1f}, {t[2]:+.1f}]"


ARM_RE = re.compile(r"^(?P<base>[^@]+)(?:@(?P<arm>s|dither|gdither)(?P<k>\d+))?(?:_m(?P<master>\w+))?$")


def arm_of(spec):
    m = ARM_RE.match(spec)
    assert m, spec
    arm = m["arm"] or "rtn"
    if m["base"] == "fp32":
        arm = "fp32"
    tag = arm if not m["master"] else f"{arm}_m{m['master']}"
    return m["base"], tag, int(m["k"] or 0)


# ----------------------------------------------------------------------------- DeepThinking
def dt_readout(root, iters_at=(30, 100, 300, 500, 1000)):
    lines = []
    for d in sorted(glob.glob(os.path.join(root, "dt_maze*"))):
        size = int(re.search(r"dt_maze(\d+)", d).group(1))
        cells = defaultdict(dict)                      # base -> arm -> list of (k, correct[n, T])
        info_any = None
        for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            info = json.loads(str(z["info"]))
            info_any = info_any or info
            base, arm, k = arm_of(os.path.basename(f)[:-4])
            cells[base].setdefault(arm, []).append((k, z["correct"].astype(bool), info))
        if not cells or "fp32" not in cells:
            lines.append(f"### dt_maze{size}: no fp32 reference in {d}\n")
            continue
        fp = cells["fp32"]["fp32"][0][1]
        T = fp.shape[1]
        its = [t for t in iters_at if t <= T]
        hosts = sorted({i["gpu"] for b in cells.values() for a in b.values() for _, _, i in a})
        lines.append(f"### DeepThinking maze {size} (n = {fp.shape[0]}, T = {T}, {hosts})\n")
        lines.append("| grid | arm | " + " | ".join(f"it {t}" for t in its) + " |")
        lines.append("|---|---|" + "---|" * len(its))
        lines.append(f"| fp32 | deterministic | " + " | ".join(f"{100 * fp[:, t - 1].mean():.1f}" for t in its) + " |")
        ci_rows = []
        for base in sorted(b for b in cells if b != "fp32"):
            arms = cells[base]
            per_item = {}
            for arm, runs in sorted(arms.items()):
                C = np.stack([c for _, c, _ in sorted(runs, key=lambda r: r[0])])      # [K, n, T]
                per_item[arm] = C.mean(0)                                             # [n, T]
                label = {"rtn": "RTN", "dither": "dither", "gdither": "gdither (Gaussian control)"}.get(arm, arm)
                if arm != "rtn":
                    label += f" K={C.shape[0]}"
                lines.append(f"| {base} | {label} | " + " | ".join(f"{100 * per_item[arm][:, t - 1].mean():.1f}" for t in its) + " |")
            if "rtn" in per_item and "dither" in per_item:
                for t in sorted({300 if 300 <= T else its[-1], T}):
                    r, dd, g = per_item["rtn"][:, t - 1], per_item["dither"][:, t - 1], per_item.get("gdither")
                    f_ = fp[:, t - 1].astype(float)
                    surv = (r.mean() - f_.mean()) * 100 > -10
                    row = [base, str(t), fmt_ci(boot_diff(dd, r)), fmt_ci(boot_diff(g[:, t - 1], dd)) if g is not None else "n/a",
                           fmt_ci(boot_diff(dd, f_)), fmt_ci(boot_diff(r, f_)), "yes" if surv else "no (collapsed)"]
                    ci_rows.append(row)
        if ci_rows:
            lines.append("")
            lines.append("| grid | iteration | dither - RTN | gdither - dither | dither - fp32 | RTN - fp32 | RTN survives |")
            lines.append("|---|---|---|---|---|---|---|")
            lines.extend("| " + " | ".join(r) + " |" for r in ci_rows)
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- MDEQ
def deq_readout(root, budgets_at=(8, 16, 26, 100)):
    lines = []
    for jf in sorted(glob.glob(os.path.join(root, "e6_deq_small_*.json"))):
        d = json.load(open(jf))
        meta = d["meta"]
        npz = jf[:-5] + ".npz"
        arrs = np.load(npz) if os.path.exists(npz) else None
        budgets = meta["budgets"]
        bidx = {b: i for i, b in enumerate(budgets)}
        bs = [b for b in budgets_at if b in bidx]
        lines.append(f"### {meta['model']} {os.path.basename(jf)[13:-5]} (n = {meta['n']}, T = {meta['T']}, {meta.get('device')}, "
                     f"torch {meta.get('torch')}, master {meta.get('master', 'fp32')})\n")
        for sv in meta["solvers"]:
            recs = [r for r in d["records"] if r["solver"] == sv]
            lines.append(f"solver {sv}:\n")
            lines.append("| arm | " + " | ".join(f"k {b}" for b in bs) + " |")
            lines.append("|---|" + "---|" * len(bs))
            groups = defaultdict(list)
            for r in recs:
                base, arm, k = arm_of(r["spec"])
                groups[(base, arm)].append(r)
            per_item = {}
            for (base, arm), rs in groups.items():
                bb = [{x["budget"]: x["top1"] for x in r["by_budget"]} for r in rs]
                acc = {b: float(np.mean([x[b] for x in bb])) for b in bs}
                label = ("fp32 deterministic" if arm == "fp32" else f"{base} {arm}") + (f" K={len(rs)}" if arm not in ("rtn", "fp32") else "")
                lines.append(f"| {label} | " + " | ".join(f"{100 * acc[b]:.1f}" for b in bs) + " |")
                if arrs is not None:
                    per_item[(base, arm)] = np.mean([arrs[f"{r['spec']}|{sv}|c1"].astype(float) for r in rs], 0)   # [nb, N]
            if arrs is not None:
                rows = []
                fp = per_item.get(("fp32", "fp32"))
                for (base, arm), pi in per_item.items():
                    if arm != "dither":
                        continue
                    r = per_item.get((base, "rtn"))
                    g = per_item.get((base, "gdither"))
                    for b in sorted({26 if 26 in bidx else bs[-1], budgets[-1]}):
                        i = bidx[b]
                        surv = r is not None and (r[i].mean() - fp[i].mean()) * 100 > -10
                        rows.append([base, str(b), fmt_ci(boot_diff(pi[i], r[i])) if r is not None else "n/a",
                                     fmt_ci(boot_diff(g[i], pi[i])) if g is not None else "n/a",
                                     fmt_ci(boot_diff(pi[i], fp[i])), fmt_ci(boot_diff(r[i], fp[i])) if r is not None else "n/a",
                                     "yes" if surv else "no (collapsed)"])
                if rows:
                    lines.append("")
                    lines.append("| grid | budget | dither - RTN | gdither - dither | dither - fp32 | RTN - fp32 | RTN survives |")
                    lines.append("|---|---|---|---|---|---|---|")
                    lines.extend("| " + " | ".join(r) + " |" for r in rows)
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt_root", default="results/breadth/babel/e6")
    ap.add_argument("--deq_root", default="results/transfer/babel/e6")
    ap.add_argument("--out", default="", help="optional markdown file")
    a = ap.parse_args()
    txt = "## DeepThinking (dt_exit.py records)\n\n" + dt_readout(os.path.expanduser(a.dt_root)) + \
          "\n## MDEQ-Small (deq_sweep.py records)\n\n" + deq_readout(os.path.expanduser(a.deq_root))
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")


if __name__ == "__main__":
    main()
