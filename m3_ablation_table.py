#!/usr/bin/env python3
"""Read-out of the M3 calibration ablation: objective (delta_T one-step map perturbation, one-step read-out KL,
layer-output MSE) x calibration point (fixed point s_T, first step s_0, random state at the fixed point's scale,
whole trajectory) x granularity (per-tensor, per-channel, group-32), nearest rounding, n=1000 held-out puzzles.

    python m3_ablation_table.py --dir results/m3/abl --md results/m3/ablation.md
"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/m3/abl")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    T = defaultdict(dict)
    base, fp = {}, {}
    for f in sorted(glob.glob(f"{a.dir}/*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        key = (j["task"], j["lo"])
        fp[key] = j.get("fp32", {}).get("exact")
        for k, r in j.get("evals", {}).items():
            if k.startswith("uniform:"):
                base[key] = r["exact"]
            elif k.startswith("uniform_lo_clip:"):
                c = k.split(":")[1]
                obj, _, pt = c.partition("@")
                pt = pt or ("traj" if obj == "omse" else "T")
                T[key][(obj, pt)] = (r["exact"], sum(r["correct"]), len(r["correct"]))
    out = []
    for key in sorted(T):
        C = T[key]
        out.append(f"\n### {key[0]}, {key[1]} (fp32 {100 * (fp[key] or float('nan')):.1f}; absmax {100 * base.get(key, float('nan')):.1f})")
        out.append("| objective \\ point | fixed point s_T | first step s_0 | random state | trajectory |")
        out.append("|---|---|---|---|---|")
        for obj in ("dT", "kl1", "omse", "mse"):
            row = []
            for pt in ("T", "0", "rand", "traj"):
                v = C.get((obj, pt))
                if v is None:
                    row.append("-")
                else:
                    lo, hi = wilson(v[1], v[2])
                    row.append(f"{100 * v[0]:.1f} [{100 * lo:.0f}, {100 * hi:.0f}]")
            if any(x != "-" for x in row):
                out.append(f"| {obj} | " + " | ".join(row) + " |")
    txt = "\n".join(out)
    print(txt)
    if a.md:
        Path(a.md).write_text(txt + "\n")


if __name__ == "__main__":
    main()
