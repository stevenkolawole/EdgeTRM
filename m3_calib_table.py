#!/usr/bin/env python3
"""Read-out of the M3 recursion-aware calibration grid (v3): per model, the same low format with every scale choice x
rounding, plus the uniform baselines at their own bits; accuracy with a Wilson 95% interval and the one-step delta_T
where recorded.

    python m3_calib_table.py --dir results/m3/v3 --md results/m3/v3_calibration.md
"""
import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if not n:
        return float("nan"), float("nan")
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


SCALES = ["none", "pct", "aciq", "mse", "omse", "dT", "dTj"]
ROUND = ["", "gptq", "ar_layer", "ar_block", "ar_step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/m3/v3")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    cells = defaultdict(dict)       # (task, lo) -> {(scale, rounding): (acc, k, n, file)}
    uni = defaultdict(dict)         # task -> {fmt: (acc, bits, k, n)}
    fp32 = {}
    for f in sorted(glob.glob(f"{a.dir}/*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        if "evals" not in j:
            continue
        if "model" in j and j["model"].startswith("mdeq"):          # m3_deq.py output
            task = j["model"]
            E = j["evals"]
            if "fp32" in E:
                fp32[task] = E["fp32"]["top1"]
            for k, r in E.items():
                if k.startswith("uniform:"):
                    uni[task][k[8:]] = (r["top1"], r["bits"], sum(r["correct"]), r["n"])
                elif k.startswith("lo:"):
                    sc = k[3:].split("+")[0]
                    rd = "gptq" if k.endswith("+gptq") else ""
                    cells[(task, j["lo"])][(sc, rd)] = (r["top1"], sum(r["correct"]), r["n"], Path(f).name)
            continue
        task = j["task"] + (f" [{Path(j['ckpt']).parent.name}]" if j.get("ckpt") else "")
        E = j["evals"]
        if "fp32" in j:
            fp32[task] = j["fp32"]["exact"]
        for k, r in E.items():
            n = len(r["correct"])
            if k.startswith("uniform:"):
                uni[task][k[8:]] = (r["exact"], r["bits"], sum(r["correct"]), n)
            elif k.startswith("uniform_lo_clip:"):
                sc = k.split(":")[1]
                cells[(task, j["lo"])][(sc, "")] = (r["exact"], sum(r["correct"]), n, Path(f).name)
            elif k.startswith("uniform_lo_calib:"):
                sc, rd = k.split(":")[1].split("+")
                cells[(task, j["lo"])][(sc, rd)] = (r["exact"], sum(r["correct"]), n, Path(f).name)
    out = []
    for (task, lo), C in sorted(cells.items()):
        base = uni[task].get(lo)
        if base and ("none", "") not in C:
            C[("none", "")] = (base[0], base[2], base[3], "uniform")
        out.append(f"\n### {task}, {lo} (fp32 {100 * fp32.get(task, float('nan')):.1f})")
        out.append("| scale \\ rounding | " + " | ".join(r or "nearest" for r in ROUND) + " |")
        out.append("|---|" + "---|" * len(ROUND))
        for sc in SCALES:
            if not any((sc, r) in C for r in ROUND):
                continue
            row = []
            for r in ROUND:
                v = C.get((sc, r))
                if v is None:
                    row.append("-")
                else:
                    lo_, hi_ = wilson(v[1], v[2])
                    row.append(f"{100 * v[0]:.1f} [{100 * lo_:.0f}, {100 * hi_:.0f}]")
            out.append(f"| {sc} | " + " | ".join(row) + " |")
        U = uni[task]
        if U:
            out.append("uniform formats: " + "; ".join(f"{fmt} {100 * v[0]:.1f} ({v[1]:.2f} b)"
                                                       for fmt, v in sorted(U.items(), key=lambda kv: kv[1][1])))
    txt = "\n".join(out)
    print(txt)
    if a.md:
        Path(a.md).write_text(txt + "\n")


if __name__ == "__main__":
    main()
