#!/usr/bin/env python3
"""M1 robustness: frozen selection on puzzles 0-999 (dev), evaluation on puzzles 1000-1999 (held-out).

Selection rule, frozen exactly as used for the headline (sudoku_attn, released checkpoint, dev set):
  for each budget T: the hybrid c(T-k)+P(k) with INT8 polish (w8c / w8t) over every quantized cheap map and k tested,
  with the highest dev accuracy; its comparator = the single-format run (any tested format, any step count, pooled
  over every dev file of the checkpoint) with the highest dev accuracy at <= the hybrid's weight traffic and
  <= its memory (the m1_frontier.py 'bits' rule). fp32 comparator: fp32 at every depth up to 2T.
Held-out: the same (cheap map, k, polish) and the same comparator (format, steps), read from the offset-1000 files;
exact McNemar on the held-out puzzles. Multiple comparisons: m = number of hybrid configurations (cell, draw, polish,
T, schedule) compared on dev for this checkpoint before the pick (and, more conservatively, over every M1 dev
record of every task); Bonferroni p x m (= Holm for the smallest p).

    python m1_heldout.py --dev 'results/breadth/babel/m1/p_sudoku_attn_*.json' \
        --held 'results/breadth/babel/m1/p_sudoku_attn_*heldout*.json' --all 'results/breadth/babel/m1/p_*.json' 'results/m1_p4d/p_*.json'
"""
import argparse
import glob
import json
import re

import numpy as np

import m1_frontier as MF


def load(paths, offset, task, ckpt=""):
    out = []
    for f in sorted({p for g in paths for p in glob.glob(g)}):
        b = json.load(open(f))
        if b["task"] == task and b.get("offset", 0) == offset and b.get("ckpt", "") == ckpt:
            out.append((f, b))
    return out


def pool(files):
    """single-format curves (name -> (curve, {step: per-input arr})), fp32 (longest), hybrids, shapes."""
    runs, hyb, shapes, fp = {}, [], None, None
    for f, b in files:
        shapes = shapes or b["shapes"]
        T = b["T"]
        fat = {int(k): MF.arr(v) for k, v in b["fp32"].get("at", {}).items()}
        if fp is None or len(b["fp32"]["curve"]) > len(fp[0]):
            fp = (b["fp32"]["curve"], fat)
        for r in b["records"]:
            c, p = r["cell"], r["polish"]
            if re.fullmatch(r"(lr|lq)\d+", p):
                continue
            if not c.startswith("noise") and not c.startswith("e"):
                cat = {i + 1: MF.arr(x) for i, x in enumerate(r.get("c_at", []))}
                if c not in runs or len(r["c_curve"]) > len(runs[c][0]):
                    runs[c] = (r["c_curve"], cat)
                ps = r["schedules"].get(f"P{T}")
                if ps and ps.get("curve") and (p not in runs or len(ps["curve"]) > len(runs[p][0])):
                    runs[p] = ([x / len(ps["correct"]) for x in ps["curve"]],
                               {int(k): MF.arr(v) for k, v in ps.get("at", {}).items()})
            for sname, v in r["schedules"].items():
                m = re.fullmatch(r"c(\d+)\.P(\d+)", sname)
                if m and int(m.group(1)) + int(m.group(2)) == T:
                    hyb.append(dict(T=T, cell=c, draw=r["draw"], pol=p, k=int(m.group(2)), acc=v["acc"],
                                    corr=MF.arr(v["correct"]), file=f.split("/")[-1]))
    runs["fp32"] = fp
    return runs, hyb, shapes


def cost_mem(h, shapes):
    c, p, k, T = h["cell"], h["pol"], h["k"], h["T"]
    cost = ((T - k) * MF.bits_of(c, shapes) + k * MF.bits_of(p, shapes)) / 32
    nested = re.fullmatch(r"n(\d+)([tac])\d+", c)
    mem = MF.mem_of(c, shapes) if (nested and p == f"w{nested.group(1)}{nested.group(2)}") \
        else MF.mem_of(c, shapes) + MF.mem_of(p, shapes)
    return cost, mem


SHORT = []                                           # (format, steps needed, steps available) for the read-out


def best_single(runs, shapes, cost, mem):
    """Best single format at <= cost and <= memory on FULL curves: a format whose recorded curve is shorter than the
    step count its traffic allows is not silently truncated; it is flagged in SHORT (audit Sep 25, item 54) and its
    best value over the available steps is used as a lower bound only."""
    out = (0.0, None, 0)
    for n, (cur, _) in runs.items():
        if MF.mem_of(n, shapes) > mem + 1e-6:
            continue
        need = int(np.floor(cost / (MF.bits_of(n, shapes) / 32) + 1e-9))
        s = min(need, len(cur))
        if need > len(cur):
            SHORT.append((n, need, len(cur)))
        if s >= 1:
            sb = int(np.argmax(np.asarray(cur[:s]))) + 1       # best depth <= s (a format may peak before s)
            if cur[sb - 1] > out[0]:
                out = (cur[sb - 1], n, sb)
    return out


def count_configs(paths):
    """every schedule evaluated per record (cell, draw, polish, schedule name) in the dev files (offset 0)."""
    n_h, n_all, per_task = 0, 0, {}
    for f in sorted({p for g in paths for p in glob.glob(g)}):
        b = json.load(open(f))
        if b.get("offset", 0) != 0:
            continue
        for r in b["records"]:
            k = len(r["schedules"])
            n_all += k
            nh = sum(1 for s in r["schedules"] if re.fullmatch(r"c\d+\.P\d+", s))
            n_h += nh
            key = (b["task"], b.get("ckpt", ""))
            per_task[key] = per_task.get(key, 0) + k
    return n_h, n_all, per_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sudoku_attn")
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--held", nargs="+", required=True)
    ap.add_argument("--all", nargs="+", default=[], help="every M1 record file, for the conservative m")
    ap.add_argument("--exclude", default="div_|ens_|heldout|T512", help="dev files written after the pick (controls, T=512)")
    ap.add_argument("--polish", default="w8c,w8t")
    ap.add_argument("--cmp_exclude", default="div_|ens_|heldout|ctl_", help="files whose curves are not single-format references")
    a = ap.parse_args()
    all_dev = load(a.dev, 0, a.task)
    dev_files = [(f, b) for f, b in all_dev if not re.search(a.exclude, f)]          # candidates for the pick
    cmp_files = [(f, b) for f, b in all_dev if not re.search(a.cmp_exclude, f)]      # comparator curves: every run
    held_files = load(a.held, 1000, a.task)
    _, dhyb, shapes = pool(dev_files)
    druns, _, _ = pool(cmp_files)
    hruns, hhyb, _ = pool(held_files)
    pols = set(a.polish.split(","))
    print(f"# frozen selection on dev (0-999), evaluation on held-out (1000-1999): {a.task}\n")
    print("dev files: " + ", ".join(f.split("/")[-1] for f, _ in dev_files))
    print("held-out files: " + ", ".join(f.split("/")[-1] for f, _ in held_files) + "\n")
    cand = [h for h in dhyb if h["pol"] in pols and not h["cell"].startswith("noise") and not h["cell"].startswith("e")]
    m_this = len(dhyb)
    n_h, n_all, per_task = count_configs(a.all) if a.all else (None, None, {})
    print(f"configurations compared on dev before the pick: {m_this} hybrid schedules for this checkpoint "
          f"({len(cand)} with INT8 polish and a quantized cheap map); across every M1 dev record: {n_h} hybrid "
          f"schedules, {n_all} schedules of any kind.\n")
    print("| T | frozen pick (dev) | dev acc | comparator (dev pick) | dev acc | dev delta, p | dev p x m_ckpt | dev p x m_all | "
          "held-out hybrid | held-out comparator | held-out delta (discordant) | held-out p | fp32 dev max <= 2T | fp32 held-out max <= 2T | "
          "held-out vs fp32 at its best depth <= 2T |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for T in sorted({h["T"] for h in cand}):
        pick = max((h for h in cand if h["T"] == T), key=lambda h: h["acc"])
        cost, mem = cost_mem(pick, shapes)
        bv, bn, bs = best_single(druns, shapes, cost, mem)
        dcomp = druns[bn][1].get(bs)
        if dcomp is not None and len(dcomp) == len(pick["corr"]):
            n10, n01, p = MF.mcnemar(pick["corr"], dcomp)
            dp = f"{100*(pick['acc']-bv):+.1f}, p={p:.2g} ({n10}/{n01})"
            pm1 = f"{min(1, p * m_this):.2g}"
            pm2 = f"{min(1, p * n_all):.2g}" if n_all else "--"
        else:
            dp, pm1, pm2 = f"{100*(pick['acc']-bv):+.1f}, (no per-input record)", "--", "--"
        name = f"{pick['cell']} {T - pick['k']} + {pick['pol']} {pick['k']}"
        hh = [h for h in hhyb if h["T"] == T and h["cell"] == pick["cell"] and h["pol"] == pick["pol"]
              and h["k"] == pick["k"] and h["draw"] == pick["draw"]]
        fdev = max(druns["fp32"][0][:2 * T])
        fhel = max(hruns["fp32"][0][:2 * T]) if hruns.get("fp32") and hruns["fp32"] else None
        if hh and bn in hruns and bs <= len(hruns[bn][0]):
            h = hh[0]
            hc = hruns[bn][1].get(bs)
            if hc is not None and len(hc) == len(h["corr"]):
                n10, n01, p = MF.mcnemar(h["corr"], hc)
                hv = hc.mean()
                held = (f"{100*h['acc']:.1f} | {bn} {bs}: {100*hv:.1f} | {100*(h['acc']-hv):+.1f} ({n10}/{n01}) | {p:.2g}")
            else:
                held = f"{100*h['acc']:.1f} | {bn} {bs}: {100*hruns[bn][0][bs-1]:.1f} (no per-input) | -- | --"
        else:
            held = "not run | not run | -- | --"
        # paired test vs fp32 at its best depth <= 2T (held-out)
        vsf = "--"
        if hh and hruns.get("fp32"):
            fcur = hruns["fp32"][0][:2 * T]
            sbest = int(np.argmax(fcur)) + 1
            fa = hruns["fp32"][1].get(sbest)
            if fa is not None and len(fa) == len(hh[0]["corr"]):
                n10, n01, p = MF.mcnemar(hh[0]["corr"], fa)
                vsf = f"fp32@{sbest}: {100*fa.mean():.1f}, {100*(hh[0]['acc']-fa.mean()):+.1f} ({n10}/{n01}), p={p:.2g}"
        print(f"| {T} | {name} | {100*pick['acc']:.1f} | {bn} {bs} | {100*bv:.1f} | {dp} | {pm1} | {pm2} | {held} | "
              f"{100*fdev:.1f} | {'--' if fhel is None else f'{100*fhel:.1f}'} | {vsf} |")
    if SHORT:
        print("\ncomparator curves too short for the matched traffic (lower bounds only; run longer): " +
              "; ".join(sorted({f"{n} needs {nd} steps, has {h}" for n, nd, h in SHORT})))
    if per_task:
        print("\nschedules compared on dev, by task and checkpoint: " +
              "; ".join(f"{t}{' ' + c.split('/')[-2] if c else ''}: {n}" for (t, c), n in sorted(per_task.items())))


if __name__ == "__main__":
    main()
