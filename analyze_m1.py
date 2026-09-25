#!/usr/bin/env python3
"""Read-out for M1 (precision-switching inference): tables from polish.py outputs.

    python analyze_m1.py results/breadth/babel/m1/p_*.json > results/breadth/m1_readout.md
    python analyze_m1.py --json results/breadth/m1_summary.json results/breadth/babel/m1/p_*.json

Per task: the fp32 depth curve (fp32 from scratch with k steps: the control for "the compressed phase
did work"); per compressed cell (noise: median and range over draws) and polish model: fully compressed
c16, polish head only, replace schedules c(T-k).P(k), P for all steps, append c(T).P(k), block-level
polish of the last step; the fp32-equivalent head start of each replace schedule; basin-exit and
prefix / periodic diagnostics; memory (bits per weight) and compute (weight-bit reads per inference).
"""
import argparse
import glob
import json
import re
from collections import defaultdict

import numpy as np


def load(paths):
    """task -> {"fp32": ..., "T":, "ks":, "shapes":, "recs": {(cell, draw, polish): rec}}"""
    D = {}
    for p in paths:
        b = json.load(open(p))
        t = D.setdefault(b["task"], {"fp32": b["fp32"], "T": b["T"], "ks": b["ks"], "inner": b["inner_js"],
                                     "shapes": b["shapes"], "n": b["n"], "recs": {}, "files": []})
        if abs(t["fp32"]["acc"] - b["fp32"]["acc"]) > 1e-9 or t["n"] != b["n"]:
            print(f"WARNING {p}: fp32 {b['fp32']['acc']} / n {b['n']} differ from {t['files']}")
        t["files"].append(p.split("/")[-1])
        if len(b["fp32"]["curve"]) > len(t["fp32"]["curve"]):     # the longest fp32 depth curve (fp32-only ref runs)
            t["fp32"] = b["fp32"]
        for r in b["records"]:
            key = (r["cell"], r["draw"], r["polish"])
            if key in t["recs"]:                          # same cell in another file (e.g. extra schedules): merge
                old = t["recs"][key]
                for sname, v in r["schedules"].items():
                    old["schedules"].setdefault(sname, v)
            else:
                t["recs"][key] = r
    return D


def keq(curve, acc):
    """fp32-equivalent steps: the (interpolated) step count at which fp32 from scratch reaches acc."""
    c = np.maximum.accumulate(np.asarray(curve, float))
    if acc <= c[0]:
        return acc / c[0] if c[0] > 0 else 0.0
    if acc > c[-1]:
        return float("inf")
    i = int(np.argmax(c >= acc))
    lo, hi = c[i - 1], c[i]
    return i + (acc - lo) / (hi - lo) if hi > lo else i + 1.0


def fmt(vals, pct=True):
    v = [x for x in vals if x is not None]
    if not v:
        return "--"
    s = 100 if pct else 1
    if len(v) == 1:
        return f"{s*v[0]:.1f}"
    return f"{s*np.median(v):.1f} [{s*min(v):.1f}-{s*max(v):.1f}]"


def bits_per_weight(bits_dict, shapes, names=None):
    names = names or list(shapes)
    N = sum(shapes[n][0] * shapes[n][1] for n in names)
    return sum(bits_dict[n] for n in names) / N


def cell_order(c):
    m = re.fullmatch(r"noise([\d.]+)", c)
    return (1, float(m.group(1))) if m else (0, c)


def cheap_bits(c, r, shapes):
    """bits per weight read by one compressed step (nested: only the top b bits of the stored B-bit codes)."""
    bc = bits_per_weight(r["cell_bits"], shapes)
    m = re.fullmatch(r"n(\d+)([tac])(\d+)", c)
    return bc - int(m.group(1)) + int(m.group(3)) if m else bc


def per_tensor(name):
    """weight formats a per-tensor-scale integer pipeline supports (INT8/INT4 per tensor, nested per tensor)."""
    return bool(re.fullmatch(r"w\d+t|n\d+t\d+", name))


def pol_bits(c, p, r, shapes):
    """bits per weight read by one step of the polish model."""
    if p == "fp32":
        return 32.0
    if re.fullmatch(r"(lr|lq)\d+", p):
        return bits_per_weight(r["cell_bits"], shapes) + bits_per_weight(r["polish_extra_bits"], shapes)
    if re.fullmatch(r"n\d+[tac]\d+", c):
        return bits_per_weight(r["cell_bits"], shapes)         # nested: the stored B-bit model
    return bits_per_weight(r["polish_extra_bits"], shapes)


def frontier(t, groups, cells, pols, S):
    """Each hybrid c(T-k)+P(k) against every single-precision run (fp32, the polish model, the compressed
    model, any quantized baseline in the file) with no more weight-bit traffic (bits x steps, in units of
    one fp32 step).  Noise cells are not storage formats: skipped."""
    T, shapes, fc = t["T"], t["shapes"], t["fp32"]["curve"]
    pure = [("fp32", 32.0, fc)]
    for (c, p), rs in groups.items():
        if c.startswith("noise") or c in ("fp32",):
            continue
        r = rs[0]
        bc = cheap_bits(c, r, shapes)
        pure.append((c, bc, r["c_curve"]))
        if p != "fp32" and f"P{T}" in r["schedules"] and r["schedules"][f"P{T}"]["curve"]:
            pure.append((p if not re.fullmatch(r"(lr|lq)\d+", p) else f"{c}+{p}", pol_bits(c, p, r, shapes),
                         r["schedules"][f"P{T}"]["curve"]))
    uniq = {}
    for name, b, cur in pure:
        uniq.setdefault(name, (b, cur))

    def best(cost, mem=1e9, exclude=(), only=None):
        bestv = (0.0, "--", 0)
        for name, (b, cur) in uniq.items():
            if name in exclude or b > mem + 1e-6 or (only is not None and not only(name)):
                continue
            s = int(np.floor(cost * 32 / b + 1e-9))
            s = min(s, len(cur))
            if s >= 1 and cur[s - 1] > bestv[0]:
                bestv = (cur[s - 1], name, s)
        return bestv
    print("\n### compute-matched: hybrid vs the best single-precision run at <= the same weight-bit traffic\n")
    print("traffic in fp32-step units (bits x steps / 32). Alternatives: fp32, the polish model alone, and every "
          "compressed model in these files at any step count.\n")
    print("| cell | polish | k | hybrid acc | traffic | memory (bits/w) | best at <= traffic (steps) | its acc | delta | "
          "best at <= traffic and <= memory (steps) | its acc | delta | per-tensor formats only: best (steps) | its acc | delta |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        if c.startswith("noise"):
            continue
        for p in pols:
            rs = groups.get((c, p))
            if not rs:
                continue
            r = rs[0]
            bc, bp = cheap_bits(c, r, shapes), pol_bits(c, p, r, shapes)
            for k in t["ks"]:
                sname = f"c{T-k}.P{k}"
                if k >= T or sname not in r["schedules"]:
                    continue
                acc = r["schedules"][sname]["acc"]
                cost = ((T - k) * bc + k * bp) / 32
                bstore = bits_per_weight(r["cell_bits"], shapes)
                mem = bstore if (re.fullmatch(r"n\d+[tac]\d+", c) and p != "fp32" and not re.fullmatch(r"(lr|lq)\d+", p)) \
                    else bstore + bits_per_weight(r["polish_extra_bits"], shapes)
                bv, bn, bs = best(cost)
                mv, mn, ms = best(cost, mem)
                pv, pn, ps = best(cost, mem, only=per_tensor)
                print(f"| {c} | {p} | {k} | {100*acc:.1f} | {cost:.2f} | {mem:.1f} | {bn} ({bs}) | {100*bv:.1f} | "
                      f"{100*(acc-bv):+.1f} | {mn} ({ms}) | {100*mv:.1f} | {100*(acc-mv):+.1f} | {pn} ({ps}) | {100*pv:.1f} | "
                      f"{100*(acc-pv):+.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    paths = sorted({p for g in a.paths for p in glob.glob(g)})
    D = load(paths)
    summary = {}
    for task, t in D.items():
        T, ks, f = t["T"], t["ks"], t["fp32"]
        fc = f["curve"]
        print(f"\n## {task}  (n={t['n']}, T={T}; files: {', '.join(t['files'])})\n")
        maze = "valid_acc" in f
        print(f"fp32 {100*f['acc']:.1f}" + (f" (valid path {100*f['valid_acc']:.1f}, valid shortest "
                                            f"{100*f['shortest_acc']:.1f})" if maze else "") +
              "; fp32 from scratch with k steps: " + ", ".join(f"k={k}: {100*fc[k-1]:.1f}" for k in ks + [T] if k <= T))
        groups = defaultdict(list)                       # (cell, polish) -> [rec per draw]
        for (c, d, p), r in t["recs"].items():
            groups[(c, p)].append(r)
        cells = sorted({c for c, _ in groups}, key=cell_order)
        pols = []
        for _, p in groups:
            if p not in pols:
                pols.append(p)
        rep = [f"c{T-k}.P{k}" for k in ks if k < T]
        app = f"c{T}.P{max(ks)}"
        inner = [f"c{T-1}.c|P@{j}" for j in t["inner"]]
        hdr = (["cell", "polish", f"c{T}", "head only"] + [s.replace(".P", "+P") for s in rep] + [f"P{T}"] +
               [f"append +{k}" for k in ks] + [f"last step, last {j} blocks" for j in t["inner"]])
        print("\n### accuracy (%) by schedule; noise cells: median [min-max] over draws\n")
        print("| " + " | ".join(hdr) + " |")
        print("|" + "---|" * len(hdr))
        S = summary.setdefault(task, {"fp32": f["acc"], "fp32_curve": fc, "cells": {}})
        for c in cells:
            for p in pols:
                if (c, p) not in groups:
                    continue
                rs = groups[(c, p)]
                g = lambda s: [r["schedules"][s]["acc"] if s in r["schedules"] else None for r in rs]
                row = [c, p, fmt([r["c_curve"][-1] for r in rs]), fmt(g(f"c{T}/hP"))]
                row += [fmt(g(s)) for s in rep] + [fmt(g(f"P{T}"))]
                row += [fmt([np.mean([int(ch) for ch in r["schedules"][app]["at"][str(T + k)]]) for r in rs]) for k in ks]
                row += [fmt(g(s)) for s in inner]
                print("| " + " | ".join(row) + " |")
                S["cells"].setdefault(c, {})[p] = {
                    "c16": [r["c_curve"][-1] for r in rs],
                    **{s: g(s) for s in rep + [f"P{T}", f"c{T}/hP"] + inner},
                    **{f"append{k}": [np.mean([int(ch) for ch in r["schedules"][app]["at"][str(T + k)]]) for r in rs]
                       for k in ks}}
        if maze:
            print("\n### maze: valid shortest path (%) (exact-match above)\n")
            print("| cell | polish | " + " | ".join([f"c{T}"] + rep + [f"P{T}"]) + " |")
            print("|" + "---|" * (3 + len(rep) + 1))
            for c in cells:
                for p in pols:
                    if (c, p) not in groups:
                        continue
                    rs = groups[(c, p)]
                    g = lambda s: [r["schedules"][s].get("shortest_acc") if s in r["schedules"] else None for r in rs]
                    print(f"| {c} | {p} | " + " | ".join(fmt(g(s)) for s in [f"c{T}"] + rep + [f"P{T}"]) + " |")

        # fp32-equivalent head start of the replace schedules (polish = fp32)
        print("\n### what the compressed phase is worth, in fp32 steps (polish fp32)\n")
        print("k_eq = steps fp32 needs from scratch to reach the hybrid's accuracy; head start = k_eq - k "
              "(inf: the hybrid beats fp32 at any step count <= T).\n")
        print("| cell | " + " | ".join(f"c{T-k}+f{k}: acc / k_eq / head start" for k in ks if k < T) + " |")
        print("|---|" + "---|" * len([k for k in ks if k < T]))
        for c in cells:
            rs = groups.get((c, "fp32"))
            if not rs:
                continue
            row = [c]
            for k in ks:
                if k >= T:
                    continue
                accs = [r["schedules"][f"c{T-k}.P{k}"]["acc"] for r in rs]
                kq = [keq(fc, x) for x in accs]
                row.append(f"{100*np.median(accs):.1f} / {np.median(kq):.1f} / {np.median(kq)-k:+.1f}")
            print("| " + " | ".join(row) + " |")

        # diagnostics (fp32 polish only)
        diag = sorted({s for (c, p), rs in groups.items() if p == "fp32" for s in rs[0]["schedules"]
                       if s.startswith(("c2.f", "c4.f", "c8.f", "c12.f", "f", "(")) or re.fullmatch(r"c\d+\.f\d+", s)},
                      key=lambda s: (s[0] != "c", s))
        if diag:
            print("\n### diagnostics (polish fp32): accuracy (%) and, for basin exit c(t).f(T), recovered = "
                  "fraction of fp32-solved inputs solved\n")
            print("| cell | " + " | ".join(diag) + " |")
            print("|---|" + "---|" * len(diag))
            for c in cells:
                rs = groups.get((c, "fp32"))
                if not rs:
                    continue
                row = [c]
                for s in diag:
                    acc = [r["schedules"][s]["acc"] for r in rs if s in r["schedules"]]
                    rec = [r["schedules"][s]["recovered"] for r in rs if s in r["schedules"]]
                    row.append(f"{fmt(acc)} ({fmt(rec)})" if re.fullmatch(r"c\d+\.f\d+", s) else fmt(acc))
                print("| " + " | ".join(row) + " |")

        # memory and compute
        shapes = t["shapes"]
        print("\n### memory and compute per configuration (linear weights; embeddings excluded)\n")
        print("bits/weight = stored bits per linear weight; extra = polish bits beyond the compressed model; "
              "traffic = weight bits read per inference relative to 16 steps of the compressed model "
              "(every step reads every weight; low-rank polish adds r(in+out) fp16 values per matrix).\n")
        print("| cell | polish | bits/weight (compressed) | extra bits/weight | k=1 traffic | k=4 traffic | P16 traffic |")
        print("|---|---|---|---|---|---|---|")
        for c in cells:
            for p in pols:
                rs = groups.get((c, p))
                if not rs:
                    continue
                r = rs[0]
                bc = cheap_bits(c, r, shapes)
                bx = bits_per_weight(r["polish_extra_bits"], shapes)
                mlr = re.fullmatch(r"(lr|lq)\d+", p)
                bp = bc + bx if mlr else (bits_per_weight(r["polish_extra_bits"], shapes)
                                           if not re.fullmatch(r"n\d+[tac]\d+", c) else bc)
                if p == "fp32":
                    bp = 32
                tr = lambda k: ((T - k) * bc + k * bp) / (T * bc)
                print(f"| {c} | {p} | {bc:.2f} | {bx:.2f} | {tr(1):.2f} | {tr(4):.2f} | {bp/bc:.2f} |")
        extra = sorted({s for rs in groups.values() for s in rs[0]["schedules"] if "%" in s})
        if extra:
            print("\n### mixed steps: z_H updates (%H) or z_L updates (%L) on the polish weights, accuracy (%)\n")
            print("| cell | polish | c16 | " + " | ".join(extra) + " |")
            print("|---|---|---|" + "---|" * len(extra))
            for c in cells:
                for p in pols:
                    rs = groups.get((c, p))
                    if not rs or not any(x in rs[0]["schedules"] for x in extra):
                        continue
                    g = lambda s_: [r["schedules"][s_]["acc"] if s_ in r["schedules"] else None for r in rs]
                    print(f"| {c} | {p} | {fmt([r['c_curve'][-1] for r in rs])} | " + " | ".join(fmt(g(x)) for x in extra) + " |")
        frontier(t, groups, cells, pols, S)
    if a.json:
        json.dump(summary, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
