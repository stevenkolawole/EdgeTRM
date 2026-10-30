#!/usr/bin/env python3
"""Label-free step-1 / step-2 signals of every TensorRT gate chain (TRT_GATE_AND_LAUNCH.md, section 5).

Per chain (records of trt_gate_chain.py: <task>.json, <task>_<engine>_preds.npz, <task>_host_fp32_preds.npz):
  phi_H, phi_L    carry cosine with the host fp32 carry (from the record)
  pred_agree      fraction of cells where the chain's argmax equals the host fp32 argmax (all cells, no labels)
  q_agree         fraction of puzzles where the sign of q_halt equals the host's
at steps 1 and 2, against the step-8 verdict (PASS / WARN / FAIL as trt_gate_chain.score). AUROC (tools/e3_gate_phi.auroc,
PASS ranked high) for PASS vs FAIL and PASS vs not-PASS, per GPU and pooled.

    python trt_gate_signals.py --dirs l40s=results/trt_gate a100=results/trt_gate_a100 --out results/trt_gate_signals.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tools"))
from e3_gate_phi import auroc  # noqa: E402

TASKS = ("sudoku_mlp", "sudoku_attn", "maze", "hrm_sudoku")
SIGNALS = ("phi_H", "phi_L", "pred_agree", "q_agree")


def cls(r):
    return "PASS" if r >= 0.95 else ("FAIL" if r < 0.5 else "WARN")


def rows(gpu, d):
    d = pathlib.Path(d)
    out = []
    for t in TASKS:
        p = d / f"{t}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        host = np.load(d / f"{t}_host_fp32_preds.npz")
        he8 = rec["host_fp32"]["exact"][-1]
        for prec, c in rec["chains"].items():
            ch = np.load(d / f"{t}_{prec}_preds.npz")
            r = dict(gpu=gpu, task=t, prec=prec, exact8=c["exact"][-1], host_exact8=he8,
                     verdict=cls(c["exact"][-1] / max(he8, 1e-9)))
            for s in (1, 2):
                r[f"phi_H_s{s}"], r[f"phi_L_s{s}"] = c["phi_H"][s - 1], c["phi_L"][s - 1]
                r[f"pred_agree_s{s}"] = float((ch["preds"][s - 1] == host["preds"][s - 1]).mean())
                r[f"q_agree_s{s}"] = float(((ch["q"][s - 1] > 0) == (host["q"][s - 1] > 0)).mean())
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="gpu=dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allr = []
    for g in a.dirs:
        gpu, d = g.split("=", 1)
        allr += rows(gpu, HERE / d if not pathlib.Path(d).is_absolute() else d)
    au = {}
    for scope in sorted({r["gpu"] for r in allr}) + ["pooled"]:
        rs = [r for r in allr if scope == "pooled" or r["gpu"] == scope]
        for s in (1, 2):
            for sig in SIGNALS:
                k = f"{sig}_s{s}"
                P = [r[k] for r in rs if r["verdict"] == "PASS"]
                au[f"{scope} | {k}"] = dict(
                    pass_vs_fail=auroc(P, [r[k] for r in rs if r["verdict"] == "FAIL"]),
                    pass_vs_notpass=auroc(P, [r[k] for r in rs if r["verdict"] != "PASS"]),
                    n_pass=len(P), n_warn=sum(r["verdict"] == "WARN" for r in rs), n_fail=sum(r["verdict"] == "FAIL" for r in rs))
    pathlib.Path(a.out).write_text(json.dumps(dict(rows=allr, auroc=au), indent=1))
    for r in allr:
        print(f"{r['gpu']:5s} {r['task']:12s} {r['prec']:9s} {r['verdict']:4s} " + " ".join(
            f"{sig} {r[f'{sig}_s1']:.3f}/{r[f'{sig}_s2']:.3f}" for sig in SIGNALS))
    for k, v in au.items():
        print(f"{k:28s} PASSvsFAIL {v['pass_vs_fail']:.3f} PASSvsNOT {v['pass_vs_notpass']:.3f} (P{v['n_pass']} W{v['n_warn']} F{v['n_fail']})")


if __name__ == "__main__":
    main()
