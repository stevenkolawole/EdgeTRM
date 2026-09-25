#!/usr/bin/env python3
"""M1 frontier across step budgets: hybrids c(T-k)+P(k) (cheap steps, then k precise steps) against every single
format run for any number of steps, pooled over all budgets T of one task (and one checkpoint).

Cost models:
  bits  weight-bit traffic per inference = sum over steps of (bits per weight read by that step), in fp32-step units
  s24   Samsung Galaxy S24 NPU latency from the Sep-10 AI Hub profiles of the H=3 supervision-step graph
        (results/breadth/aihub_status.json): FP16 graph for fp32/bf16 weights, INT8 graph (w8a8) for 8-bit weights,
        INT4 graph (w4a8) for every <= 4-bit format (INT3 and nested top-bit slicing have no faster kernel: proxy).
        Latency is accounting only: the device graphs quantize activations too, the harness cells do not.
Per hybrid: the best single-format run at <= cost (and <= memory for the bits model), exact McNemar on the same
inputs when that run's per-input record exists, and fp32's step count to match the hybrid (dominance ratio).

    python m1_frontier.py --task sudoku_attn 'results/breadth/babel/m1/p_sudoku_attn_T*.json' > results/breadth/m1_frontier_attn.md
"""
import argparse
import glob
import json
import re
from math import comb

import numpy as np

S24_MS = {"sudoku_attn": {"fp": 9.062, "int8": 11.384, "int4": 7.146},
          "sudoku_mlp": {"fp": 11.37, "int8": 10.265, "int4": 6.78}}


def arr(s):
    return (np.frombuffer(s.encode(), np.uint8) - 48).astype(int)


def mcnemar(x, y):
    n10, n01 = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = n10 + n01
    if n == 0:
        return n10, n01, 1.0
    k = min(n10, n01)
    return n10, n01, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def bits_of(fmt, shapes, rec=None):
    """bits per weight read by one step of a single format (codes + scales, as polish.storage_bits)."""
    import polish as P
    N = sum(o * i for o, i in shapes.values())
    if fmt in ("fp32",):
        return 32.0
    m = re.fullmatch(r"n(\d+)([tac])(\d+)", fmt)
    if m:                                               # nested cheap step: top b bits of the stored B-bit codes
        full = sum(P.storage_bits(f"w{m.group(1)}{m.group(2)}", tuple(s)) for s in shapes.values()) / N
        return full - int(m.group(1)) + int(m.group(3))
    return sum(P.storage_bits(fmt, tuple(s)) for s in shapes.values()) / N


def mem_of(fmt, shapes):
    import polish as P
    N = sum(o * i for o, i in shapes.values())
    m = re.fullmatch(r"n(\d+)([tac])(\d+)", fmt)
    if m:
        fmt = f"w{m.group(1)}{m.group(2)}"
    return 32.0 if fmt == "fp32" else sum(P.storage_bits(fmt, tuple(s)) for s in shapes.values()) / N


def s24_of(fmt, task):
    t = S24_MS.get(task)
    if t is None:
        return None
    if fmt in ("fp32", "bf16"):
        return t["fp"]
    if fmt.startswith("w8"):
        return t["int8"]
    return t["int4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--task", required=True)
    ap.add_argument("--ckpt", default="", help="only files of this checkpoint ('' = released)")
    a = ap.parse_args()
    files = sorted({p for g in a.paths for p in glob.glob(g)})
    runs = {}                                           # single-format name -> (curve list, per-input dict step->arr)
    hyb = []
    shapes = None
    fp32_best = None
    for f in files:
        b = json.load(open(f))
        if b["task"] != a.task or b.get("ckpt", "") != a.ckpt:
            continue
        shapes = shapes or b["shapes"]
        T = b["T"]
        fc = b["fp32"]["curve"]
        fat = {int(k): arr(v) for k, v in b["fp32"].get("at", {}).items()}
        if fp32_best is None or len(fc) > len(fp32_best[0]):
            fp32_best = (fc, fat)
        for r in b["records"]:
            c, p = r["cell"], r["polish"]
            if c.startswith("noise") or re.fullmatch(r"(lr|lq)\d+", p):   # not storage formats: skipped here
                continue
            cat = {i + 1: arr(x) for i, x in enumerate(r.get("c_at", []))}
            if c not in runs or len(r["c_curve"]) > len(runs[c][0]):
                runs[c] = (r["c_curve"], cat)
            ps = r["schedules"].get(f"P{T}")
            if ps and ps.get("curve") and (p not in runs or len(ps["curve"]) > len(runs[p][0])):
                pat = {int(k): arr(v) for k, v in ps.get("at", {}).items()}
                runs[p] = ([x / len(arr(ps["correct"])) for x in ps["curve"]], pat)
            for sname, v in r["schedules"].items():
                m = re.fullmatch(r"c(\d+)\.P(\d+)", sname)
                if not m or int(m.group(1)) + int(m.group(2)) != T:
                    continue
                hyb.append(dict(T=T, cell=c, pol=p, k=int(m.group(2)), acc=v["acc"], corr=arr(v["correct"]), file=f))
    runs["fp32"] = fp32_best
    names = sorted(runs)
    bits = {n: bits_of(n, shapes) for n in names}
    mem = {n: mem_of(n, shapes) for n in names}
    print(f"# {a.task} {a.ckpt or '(released checkpoint)'}: frontier over budgets\n")
    print("single-format runs available (max steps): " + ", ".join(f"{n} ({len(runs[n][0])}, {bits[n]:.2f} b/w)" for n in names))
    fc = np.maximum.accumulate(np.asarray(runs["fp32"][0]))

    def best(cost_fn, cost, memcap):
        out = (0.0, None, 0)
        for n in names:
            if mem[n] > memcap + 1e-6:
                continue
            cur = runs[n][0]
            per = cost_fn(n)
            if per is None:
                continue
            s = min(int(np.floor(cost / per + 1e-9)), len(cur))
            if s >= 1 and cur[s - 1] > out[0]:
                out = (cur[s - 1], n, s)
        return out

    def ptest(h, n, s):
        at = runs[n][1]
        if s in at and len(at[s]) == len(h["corr"]):
            n10, n01, p = mcnemar(h["corr"], at[s])
            return f"{n10}/{n01}, p={p:.2g}"
        return "n/a"

    for model in ("bits", "s24"):
        cf = (lambda n: bits[n] / 32) if model == "bits" else (lambda n: s24_of(n, a.task))
        if model == "s24" and S24_MS.get(a.task) is None:
            continue
        unit = "fp32-step traffic" if model == "bits" else "S24 ms"
        print(f"\n## cost model: {model} ({unit})\n")
        print("| T | cheap | polish | k | acc | cost | memory b/w | best single format at <= cost, <= memory (steps) | acc | delta | paired | "
              "fp32 steps to match | fp32 cost to match / hybrid cost |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for h in sorted(hyb, key=lambda h: (h["T"], h["cell"], h["pol"], h["k"])):
            c, p, k, T = h["cell"], h["pol"], h["k"], h["T"]
            if p not in bits:
                bits[p], mem[p] = bits_of(p, shapes), mem_of(p, shapes)
            cc, pc = cf(c), cf(p)
            if cc is None or pc is None:
                continue
            cost = (T - k) * cc + k * pc
            nested = re.fullmatch(r"n(\d+)([tac])\d+", c)
            # nested cheap step + polish on the same stored codes: one copy; otherwise both copies are stored
            memh = mem[c] if (nested and p == f"w{nested.group(1)}{nested.group(2)}") else mem[c] + mem_of(p, shapes)
            bv, bn, bs = best(cf, cost, memh)
            idx = np.nonzero(fc >= h["acc"] - 1e-12)[0]
            if len(idx):
                fs = int(idx[0]) + 1
                ratio = f"{fs * cf('fp32') / cost:.2f}"
            else:
                fs, ratio = f"> {len(fc)}", "> " + f"{len(fc) * cf('fp32') / cost:.2f}"
            print(f"| {T} | {c} | {p} | {k} | {100*h['acc']:.1f} | {cost:.2f} | {memh:.1f} | {bn} ({bs}) | {100*bv:.1f} | "
                  f"{100*(h['acc']-bv):+.1f} | {ptest(h, bn, bs) if bn else ''} | {fs} | {ratio} |")


if __name__ == "__main__":
    main()
