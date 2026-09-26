#!/usr/bin/env python3
"""M1 looped-LM read-out (Babel Ouro re-test, frozen rule in M1.md): pick = best dev switch; comparator = best single
format at <= the pick's weight traffic (bits x loops) and <= its memory; held-out: same pick and comparator, McNemar.
    python m1_lm_readout.py results/breadth/babel/m1lm/m1o26 results/breadth/babel/m1lm/m1o14"""
import json
import re
import sys
from math import comb
from pathlib import Path

BITS = {"bf16": 16.0, "w8c": 8.0, "w4g128": 4.125, "w4c": 4.0}
MEM = {"bf16": 16.0, "w8c": 8.0, "w4g128": 4.125, "w4c": 4.0}


def cells(d):
    out = {}
    for sub in sorted(Path(d).glob("*_core")):
        m = re.fullmatch(r"ouro\d+_s(\d+)_(.+)_core", sub.name)
        loops, spec = int(m.group(1)), m.group(2)
        recs = sorted((json.loads(l) for l in open(sub / "greedy.jsonl")), key=lambda r: r["idx"])
        corr = [r["correct"] for r in recs]
        sw = re.fullmatch(r"(\w+)~P(\d+)@(\w+)", spec)
        if sw:
            base, pl, pol = sw.group(1), [int(c) for c in sw.group(2)], sw.group(3)
            cost = (loops - len(pl)) * BITS[base] + len(pl) * BITS[pol]
            mem = MEM[base] + MEM[pol]
        else:
            cost, mem = loops * BITS[spec], MEM[spec]
        out[sub.name] = dict(loops=loops, spec=spec, switch=bool(sw), cost=cost, mem=mem, corr=corr,
                             acc=sum(corr) / len(corr), n=len(corr))
    return out


def mcnemar(a, b):
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if y and not x)
    n = n10 + n01
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(n10, n01) + 1)) / 2 ** n)
    return n10, n01, p


for root in sys.argv[1:]:
    dev, held = cells(Path(root) / "dev"), cells(Path(root) / "held")
    print(f"\n## {Path(root).name}: dev n={next(iter(dev.values()))['n']}, held-out n={next(iter(held.values()))['n']}\n")
    print("| cell | traffic | memory | dev | held-out |\n|---|---|---|---|---|")
    for k, v in sorted(dev.items(), key=lambda kv: kv[1]["cost"]):
        print(f"| {k} | {v['cost']:.2f} | {v['mem']:.1f} | {100*v['acc']:.1f} | {100*held[k]['acc']:.1f} |")
    best = max(v["acc"] for v in dev.values() if v["switch"])
    for pk, pv in [(k, v) for k, v in dev.items() if v["switch"] and v["acc"] == best]:
        comp = max((v for k, v in dev.items() if not v["switch"] and v["cost"] <= pv["cost"] + 1e-9 and v["mem"] <= pv["mem"] + 1e-9),
                   key=lambda v: v["acc"])
        ck = [k for k, v in dev.items() if v is comp][0]
        d10, d01, dp = mcnemar(pv["corr"], comp["corr"])
        h10, h01, hp = mcnemar(held[pk]["corr"], held[ck]["corr"])
        print(f"\npick (dev) {pk} {100*pv['acc']:.1f} (traffic {pv['cost']:.2f}, memory {pv['mem']:.1f}); comparator {ck} {100*comp['acc']:.1f}: "
              f"dev {100*(pv['acc']-comp['acc']):+.1f} ({d10}/{d01}, p={dp:.2g}); held-out {100*held[pk]['acc']:.1f} vs "
              f"{100*held[ck]['acc']:.1f}: {100*(held[pk]['acc']-held[ck]['acc']):+.1f} ({h10}/{h01}, p={hp:.2g})")
