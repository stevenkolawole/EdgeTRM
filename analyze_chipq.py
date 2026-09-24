#!/usr/bin/env python3
"""Read-out for E0 (chip-realisable quantizers) and the carry-storage precision sweep
(chipq_sweep.py records pulled to results/devices/chipq/).

    python analyze_chipq.py            -> results/devices/chipq_tables.md (+ .json)

Tables: (1) per task and (H, n): exact accuracy (and cell, fidelity_H) for the chip formats, weight
granularity x activation precision; (2) carry precision x weight format; every cell names n.
"""
import collections
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "results" / "devices" / "chipq"
W_ROWS = ["fp32", "w8t", "w8c", "w4t", "w4c", "w4g128", "w4g32", "w4l128", "w4l64", "w4l32"]
A_COLS = ["", "+a16", "+a8"]
CARRY = ["", "+cs_fp16", "+cs_bf16", "+cs_i8t", "+cs_i8c", "+cs_i8td", "+cs_i8tok", "+cs_i4t", "+cs_i4c",
         "+cs_i4td", "+cs_i4tok"]


def load(prefix):
    recs = collections.defaultdict(dict)
    ns = {}
    for f in sorted(glob.glob(str(SRC / f"{prefix}_*.json"))):
        d = json.loads(Path(f).read_text())
        for r in d["records"]:
            key = (r["task"], r["H_cycles"], r["nsup"])
            recs[key][r["quant"]["spec"]] = r
            ns[key] = r["n"]
    return recs, ns


def fmt(r, what="pexact"):
    if r is None:
        return "-"
    return f"{r[what]*100:.1f}" if what != "fidelity_H" else f"{r[what]:.3f}"


def main():
    out, js = [], {}
    recs, ns = load("chipq")
    out.append("# E0: chip-realisable quantizers (exact accuracy %, harness `chipq_sweep.py`)\n")
    out.append("Rows: weight format (t per-tensor, c per-channel, g<k> float block scales, l<k> LPBQ-grid block "
               "scales); columns: activations (none / static per-tensor INT16 / INT8 at linear inputs). "
               "Cells: exact % (fidelity_H).\n")
    for key in sorted(recs):
        task, H, n = key
        out.append(f"\n## {task}, H={H}, n={n} (n_puzzles={ns[key]})\n")
        out.append("| weights | " + " | ".join(c or "w-only" for c in A_COLS) + " |")
        out.append("|---|" + "---|" * len(A_COLS))
        for w in W_ROWS:
            cells = []
            for a in A_COLS:
                r = recs[key].get(w + a) if w != "fp32" or a == "" else None
                cells.append("-" if r is None else f"{fmt(r)} ({fmt(r, 'fidelity_H')})")
            out.append(f"| {w} | " + " | ".join(cells) + " |")
        js[f"{task}|H{H}|n{n}"] = {s: {k: r[k] for k in ("pexact", "cell", "fidelity_H", "fidelity_L")}
                                   for s, r in recs[key].items()}
    crecs, cns = load("carry")
    out.append("\n# Carry-storage precision (carry quantized after every supervision step)\n")
    for key in sorted(crecs):
        task, H, n = key
        weights = sorted({s.split("+")[0] for s in crecs[key]})
        out.append(f"\n## {task}, H={H}, n={n} (n_puzzles={cns[key]})\n")
        out.append("| carry | " + " | ".join(weights) + " |")
        out.append("|---|" + "---|" * len(weights))
        for c in CARRY:
            cells = []
            for w in weights:
                r = crecs[key].get(w + c)
                cells.append("-" if r is None else f"{fmt(r)} ({fmt(r, 'fidelity_H')})")
            out.append(f"| {c or 'fp32 carry'} | " + " | ".join(cells) + " |")
        extra = [s for s in crecs[key] if "+a" in s]
        for s in extra:
            r = crecs[key][s]
            out.append(f"| {s} | {fmt(r)} ({fmt(r, 'fidelity_H')}) |")
        js[f"carry|{task}|H{H}|n{n}"] = {s: {k: r[k] for k in ("pexact", "cell", "fidelity_H", "fidelity_L")}
                                         for s, r in crecs[key].items()}
    (HERE / "results" / "devices" / "chipq_tables.md").write_text("\n".join(out) + "\n")
    (HERE / "results" / "devices" / "chipq_tables.json").write_text(json.dumps(js, indent=1))
    print("\n".join(out[:60]))


if __name__ == "__main__":
    main()
