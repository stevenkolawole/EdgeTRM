#!/usr/bin/env python3
"""X2 pre-registered width prediction from the sensitivity alone (no accuracy labels).

    sigma_1/2 (predicted) = DELTA_STAR / S(0.05)

DELTA_STAR = 0.4033 is fixed here (commit it before any predicted width is measured): the one-parameter width law
fitted on the five Babel TRM/HRM models whose sigma_1/2 AND S were both measured on Babel L40S (TRM Sudoku MLP, TRM
Sudoku attention, TRM Maze, HRM Sudoku, HRM Maze). Their leave-one-out error band: median x1.46, worst x1.92.
S(0.05) = median over the 3 noise draws of (median over inputs of delta_T) / 0.05, from sens_fit.py on Babel.

    python x2_predict.py SENS_JSON [--log predictions.jsonl]

Appends {model, S, predicted sigma_1/2, time, sens file mtime} to the log BEFORE the model's width sweep runs
(x2_model.sh calls it between sens_fit and breadth_sweep).
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

DELTA_STAR = 0.4033
SIGMA0 = 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sens")
    ap.add_argument("--log", default=str(Path.home() / "breadth_results" / "x2" / "sens" / "predictions.jsonl"))
    a = ap.parse_args()
    j = json.load(open(a.sens))
    d = [r["delta_med"] for r in j["records"] if r.get("sigma") == SIGMA0 and "@" not in r["spec"]]
    if len(d) < 3:
        raise SystemExit(f"need 3 draws at sigma {SIGMA0}, have {len(d)}")
    S = float(np.median(d)) / SIGMA0
    rec = {"model": j["variant"], "S": S, "sigma_half_pred": DELTA_STAR / S, "delta_star": DELTA_STAR,
           "fp32_acc_n256": j.get("fp32_acc"), "sens_file": a.sens,
           "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.uname().nodename}
    with open(a.log, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"PREDICTION {rec['model']}: S {S:.3f} -> sigma_1/2 {rec['sigma_half_pred']:.3f} ({rec['time_utc']})", flush=True)


if __name__ == "__main__":
    main()
