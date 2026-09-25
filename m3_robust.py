#!/usr/bin/env python3
"""M3 robustness read-out (AUDIT_SEP25 item 79): every calibration method on >= 3 disjoint calibration sets per cell;
mean accuracy with a t-interval over calibration sets, and paired differences (same calibration set, same evaluation
puzzles) of the step objective (dT+gptq) against the BEST layer-wise baseline of that cell (highest mean among
omse+gptq, mse+gptq, omse+ar_layer) and against the block objective (blk+gptq).

    python m3_robust.py --dir results/m3/rob --md results/m3/robust.md
File names: <task>_<lo>_<clip>_<calib>_o<offset|block>.json (m3_mixed.py) or mdeq_small_w4t_<clip>_gptq_b<k>.json.
"""
import argparse
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

T95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36}
LAYERWISE = ["omse_gptq", "mse_gptq", "omse_ar_layer"]


def ci(x):
    x = np.asarray(x, float)
    if len(x) < 2:
        return float(x.mean()) if len(x) else float("nan"), float("nan")
    return float(x.mean()), T95.get(len(x) - 1, 2.0) * float(x.std(ddof=1)) / math.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/m3/rob")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    acc = defaultdict(dict)              # (cell) -> method -> {set: acc}
    for f in sorted(glob.glob(f"{a.dir}/*.json")):
        name = Path(f).stem
        m = re.match(r"(.+?)_(w\d+\w*?)_(dTj|dT|blk|omse|mse|none)_(gptq|ar_layer)_([ob]\d+)$", name)
        if not m:
            continue
        cell, method, cset = f"{m.group(1)} {m.group(2)}", f"{m.group(3)}_{m.group(4)}", m.group(5)
        try:
            j = json.load(open(f))
        except Exception:
            continue
        E = j.get("evals", {})
        r = next((v for k, v in E.items() if k.startswith("uniform_lo_calib") or k.startswith("lo:") and "+" in k), None)
        if r is None:
            continue
        acc[cell][method] = {**acc[cell].get(method, {}), cset: r.get("exact", r.get("top1"))}
    out = []
    for cell in sorted(acc):
        M = acc[cell]
        out.append(f"\n### {cell}")
        out.append("| method | calibration sets | mean % [95% CI over sets] |")
        out.append("|---|---|---|")
        means = {}
        for meth in sorted(M, key=lambda k: -np.mean(list(M[k].values()))):
            v = [M[meth][s] for s in sorted(M[meth])]
            mu, h = ci(v)
            means[meth] = mu
            out.append(f"| {meth} | {' / '.join(f'{100 * x:.1f}' for x in v)} | {100 * mu:.1f} [{100 * (mu - h):.1f}, {100 * (mu + h):.1f}] |")
        lw = [k for k in LAYERWISE if k in M]
        if "dT_gptq" in M and lw:
            best = max(lw, key=lambda k: means[k])
            for ref in [best] + (["blk_gptq"] if "blk_gptq" in M else []):
                sets = sorted(set(M["dT_gptq"]) & set(M[ref]))
                d = [M["dT_gptq"][s] - M[ref][s] for s in sets]
                mu, h = ci(d)
                out.append(f"- step (dT_gptq) minus {'best layer-wise ' if ref == best else ''}{ref}: "
                           f"{100 * mu:+.1f} points [{100 * (mu - h):+.1f}, {100 * (mu + h):+.1f}] over {len(sets)} paired sets")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        Path(a.md).write_text(txt + "\n")


if __name__ == "__main__":
    main()
