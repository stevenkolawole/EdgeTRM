#!/usr/bin/env python3
"""Label-free check on the AGX forced-INT8 faults: per step, cell agreement of each TensorRT engine's answer with the
host fp32 reference (check.ref.cell_agreement_per_step in the gate records), and the gap between the worst passing
engine of a model (fp32, fp16, calibrated int8, DLA fp16) and its forced-INT8 engine.
usage: agx_int8_check.py <dir with agx_<task>_trt_<prec>_gate.json>"""
import json, sys
from pathlib import Path

D = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
PASS = ["fp32", "fp16", "int8", "dla0fp16"]
for task in ["sudoku_mlp", "sudoku_attn", "maze", "hrm_sudoku"]:
    rows = {}
    for prec in PASS + ["int8strict"]:
        f = D / f"agx_{task}_trt_{prec}_gate.json"
        if f.exists():
            c = json.load(open(f))["check"]
            rows[prec] = (c["ref"]["cell_agreement_per_step"], c["exact_count_per_step"][-1], c["ref"]["ref_exact_count_per_step"][-1])
    ok = [rows[p][0] for p in PASS if p in rows]
    bad, ex, ref = rows["int8strict"]
    print(f"{task}: forced-INT8 exact at step 16 {ex} (host {ref}); passing engines {len(ok)}")
    print("  step  pass_min  int8strict  gap")
    for s in (1, 2, 3, len(bad)):
        lo = min(a[s - 1] for a in ok)
        print(f"  {s:>4}  {lo:8.3f}  {bad[s - 1]:10.3f}  {lo - bad[s - 1]:+.3f}")
