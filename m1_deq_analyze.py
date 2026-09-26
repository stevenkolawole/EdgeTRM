#!/usr/bin/env python3
"""M1 on MDEQ (Small or XL, from the record meta): matched-traffic frontier on dev, frozen picks evaluated on held-out images.

Cost: weight-bit traffic of the fixed-point block, (t_c x bits(cheap) + k x bits(precise)) / 32 in fp32-iteration
units (injection and head, computed once per model, are ignored); memory = stored bits per weight (nested n8c4 with
INT8 polish = one INT8 copy). Single formats: fp32, w8c and every cheap map, Broyden from z0 = 0, any recorded budget.
Frozen rule (written before the held-out read): for each total-iteration budget B = t_c + k in --budgets, the pick is
the hybrid with the highest dev accuracy among t_c + k == B; its comparator is the single-format run with the highest
dev accuracy at <= the pick's traffic and <= its memory. Held-out: the same (cheap, precise, t_c, k) and the same
comparator (format, iterations); exact McNemar on held-out images.

    python m1_deq_analyze.py results/breadth/babel/m1/deq_small_dev.json results/breadth/babel/m1/deq_small_held.json
"""
import json
import sys

import numpy as np

import m1_frontier as MF

BITS = {"fp32": 32.0, "w8c": 8.0, "w4c": 4.0, "w4a": 4.0, "w4g32": 4.5, "n8c4": 4.0}   # + scales (<= 0.1 b/w at 3x3 convs)
MEM = {"fp32": 32.0, "w8c": 8.0, "w4c": 4.0, "w4a": 4.0, "w4g32": 4.5, "n8c4": 8.0}


def arr(s):
    return MF.arr(s)


def main():
    dev = json.load(open(sys.argv[1]))
    held = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else None
    budgets = [8, 12, 16, 20, 26]
    curve = dev["meta"]["curve"]
    singles = {s: v for s, v in dev["curve"].items()}

    def best_single(D, cost, mem):
        out = (0.0, None, 0)
        for s, v in D["curve"].items():
            if MEM[s] > mem + 1e-9:
                continue
            for k, a in zip(curve, v["acc"]):
                if k * BITS[s] / 32 <= cost + 1e-9 and a > out[0]:
                    out = (a, s, k)
        return out

    def mem_of(c, p):
        return MEM["w8c"] if (c == "n8c4" and p == "w8c") else MEM[c] + MEM[p]

    m = dev["meta"]
    name = {"mdeq_small": "MDEQ-Small", "mdeq_xl": "MDEQ-XL"}.get(m.get("model"), m.get("model", "MDEQ"))
    print(f"## {name} dev (images {m.get('lo', 0)}-{m.get('hi', 2500) - 1}): single formats from z0 = 0 (top-1 %)\n")
    print("| format | " + " | ".join(str(k) for k in curve) + " |")
    print("|---|" + "---|" * len(curve))
    for s, v in singles.items():
        print(f"| {s} | " + " | ".join(f"{100*a:.1f}" for a in v["acc"]) + " |")
    rows = []
    for key, d in dev["hybrid"].items():
        c, p = key.split("|")
        for tk, v in d.items():
            t, k = map(int, tk.split("+"))
            cost = (t * BITS[c] + k * BITS[p]) / 32
            rows.append(dict(c=c, p=p, t=t, k=k, B=t + k, acc=v["acc"], corr=v["correct"], cost=cost, mem=mem_of(c, p)))
    print("\n## frozen picks per total-iteration budget B (dev), and held-out evaluation\n")
    print("| B | pick (dev) | dev acc | traffic (fp32 its) | memory | comparator (dev) | dev acc | dev delta, p | "
          "held-out pick | held-out comparator | held-out delta (discordant), p | held-out fp32 best <= 60 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for B in budgets:
        cand = [r for r in rows if r["B"] == B]
        if not cand:
            continue
        pk = max(cand, key=lambda r: r["acc"])
        bv, bn, bk = best_single(dev, pk["cost"], pk["mem"])
        n10, n01, p = MF.mcnemar(arr(pk["corr"]), arr(dev["curve"][bn]["correct"][str(bk)]))
        line = (f"| {B} | {pk['c']} {pk['t']} + {pk['p']} {pk['k']} | {100*pk['acc']:.1f} | {pk['cost']:.2f} | {pk['mem']:.0f} | "
                f"{bn} {bk} | {100*bv:.1f} | {100*(pk['acc']-bv):+.1f}, {p:.2g} ({n10}/{n01}) | ")
        if held is not None:
            hp = held["hybrid"][f"{pk['c']}|{pk['p']}"][f"{pk['t']}+{pk['k']}"]
            hc = held["curve"][bn]["correct"][str(bk)]
            hn10, hn01, hpv = MF.mcnemar(arr(hp["correct"]), arr(hc))
            hf = max(held["curve"]["fp32"]["acc"]) if "fp32" in held["curve"] else float("nan")
            line += (f"{100*hp['acc']:.1f} | {bn} {bk}: {100*arr(hc).mean():.1f} | {100*(hp['acc']-arr(hc).mean()):+.1f} "
                     f"({hn10}/{hn01}), {hpv:.2g} | {100*hf:.1f} |")
        else:
            line += "-- | -- | -- | -- |"
        print(line)
    # Rule 2 (committed before any held-out read, after rule 1 proved uninformative: every rule-1 pick sits at the
    # converged ceiling): for each traffic cap C (fp32-iteration units), the pick is the hybrid with the highest dev
    # accuracy at traffic <= C (ties: lower traffic); comparator = best single format at <= the pick's traffic and
    # <= its memory; held-out: the same pick and comparator.
    print("\n## rule 2: frozen picks per traffic cap C (fp32-iteration units; C = 1 is 4 INT8 iterations)\n")
    print("| C | pick (dev) | dev acc | traffic | memory | comparator (dev) | dev acc | dev delta, p | "
          "held-out pick | held-out comparator | held-out delta (discordant), p |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for C in [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]:
        cand = [r for r in rows if r["cost"] <= C + 1e-9]
        if not cand:
            continue
        pk = max(cand, key=lambda r: (r["acc"], -r["cost"]))
        bv, bn, bk = best_single(dev, pk["cost"], pk["mem"])
        if bn is None:
            continue
        n10, n01, p = MF.mcnemar(arr(pk["corr"]), arr(dev["curve"][bn]["correct"][str(bk)]))
        line = (f"| {C} | {pk['c']} {pk['t']} + {pk['p']} {pk['k']} | {100*pk['acc']:.1f} | {pk['cost']:.2f} | {pk['mem']:.0f} | "
                f"{bn} {bk} | {100*bv:.1f} | {100*(pk['acc']-bv):+.1f}, {p:.2g} ({n10}/{n01}) | ")
        if held is not None and str(bk) in held["curve"][bn]["correct"]:
            hp = held["hybrid"][f"{pk['c']}|{pk['p']}"][f"{pk['t']}+{pk['k']}"]
            hc = arr(held["curve"][bn]["correct"][str(bk)])
            hn10, hn01, hpv = MF.mcnemar(arr(hp["correct"]), hc)
            line += f"{100*hp['acc']:.1f} | {bn} {bk}: {100*hc.mean():.1f} | {100*(hp['acc']-hc.mean()):+.1f} ({hn10}/{hn01}), {hpv:.2g} |"
        else:
            line += "-- | -- | -- |"
        print(line)
    # dev frontier: every hybrid vs best single at <= cost and memory
    print("\n## dev: every hybrid against the best single format at <= traffic and <= memory (top 12 by delta)\n")
    print("| hybrid | acc | traffic | memory | best single (its) | acc | delta |")
    print("|---|---|---|---|---|---|---|")
    scored = []
    for r in rows:
        bv, bn, bk = best_single(dev, r["cost"], r["mem"])
        scored.append((r["acc"] - bv, r, bn, bk, bv))
    for d, r, bn, bk, bv in sorted(scored, key=lambda x: -x[0])[:12]:
        print(f"| {r['c']} {r['t']} + {r['p']} {r['k']} | {100*r['acc']:.1f} | {r['cost']:.2f} | {r['mem']:.0f} | {bn} {bk} | "
              f"{100*bv:.1f} | {100*d:+.1f} |")


if __name__ == "__main__":
    main()
