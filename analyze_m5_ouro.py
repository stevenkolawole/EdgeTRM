#!/usr/bin/env python3
"""Read-out of ouro_exit.py summaries: overthinking across loops and the learned exit gate's recovery.

Per (model, weight spec), from per-problem greedy GSM8K correctness at L = 1, 2, 3 loops and at 4 loops (tau 1.0):
  V_rel     share of the 4-loop errors that were solved with fewer loops (visit then lose across loops)
  oracle    accuracy of choosing the best loop count per problem (right at any L)
  gate      the trained exit gate at threshold tau (per token, 4 loops computed): accuracy, mean exit loop
  recovery  (acc_gate - acc_4) / (oracle - acc_4)

    python analyze_m5_ouro.py results/m5/ouro/ouro14_summary.json [...]
"""
import json
import sys

import numpy as np


def load(paths):
    """merge serial-driver summaries (<model>_summary.json) and per-cell files (<model>_<spec>_<cfg>.summary.json)."""
    by_model = {}
    for p in paths:
        model = p.split("/")[-1].split("_")[0]
        by_model.setdefault(model, {}).update(json.load(open(p)))
    return by_model


def main(paths):
    out = {}
    for model, d in load(paths).items():
        specs = sorted({k.split("|")[0] for k in d})
        for spec in specs:
            g = lambda k: d.get(f"{spec}|{k}")  # noqa: E731
            full = g("tau1.0")
            Ls = [g(f"L{i}") for i in (1, 2, 3)]
            if full is None or any(x is None for x in Ls):
                print(f"{model} {spec}: incomplete ({sorted(k for k in d if k.startswith(spec + '|'))})")
                continue
            C = np.array([x["correct"] for x in Ls] + [full["correct"]], bool)   # [4, n]
            fin = C[-1]
            anyc = C.any(0)
            V_abs = (anyc & ~fin).mean() * 100
            V_rel = (anyc & ~fin).sum() / max((~fin).sum(), 1) * 100
            orc = anyc.mean() * 100
            row = {"n": int(C.shape[1]), "acc_by_loops": [float(c.mean() * 100) for c in C], "V_abs": V_abs,
                   "V_rel": V_rel, "oracle": orc, "gate": {}}
            for tau in ("0.7", "0.5"):
                x = g(f"tau{tau}")
                if x is None:
                    continue
                a = x["acc"] * 100
                gap = orc - fin.mean() * 100
                row["gate"][tau] = {"acc": a, "exit_loop": x["mean_exit_loop"],
                                    "recovery": (a - fin.mean() * 100) / gap if gap >= 2 else None,
                                    "degenerate": x["degenerate"] * 100}
            out[f"{model}|{spec}"] = row
            gs = " | ".join(f"gate tau {t}: {v['acc']:.1f} at {v['exit_loop']:.2f} loops, recovery "
                            + (f"{v['recovery']:+.2f}" if v["recovery"] is not None else "-") for t, v in row["gate"].items())
            print(f"{model} {spec:6s} n={row['n']}: acc at 1/2/3/4 loops " + " / ".join(f"{a:.1f}" for a in row["acc_by_loops"])
                  + f" | V_rel {V_rel:.1f}% (V_abs {V_abs:.1f}) | oracle {orc:.1f} | " + gs)
    return out


if __name__ == "__main__":
    res = main(sys.argv[1:])
    json.dump(res, open("results/breadth/m5_ouro.json", "w"), indent=1, default=float)
