#!/usr/bin/env python3
"""Readout of the HRM phone chains (reports/edgetrm_exp/AIHUB_HRM_GATE.md): per chain and step exact, cell, phi_H, phi_L,
label-free agreement with host fp32 predictions; the gate verdict at step 2 against the step-8 outcome.
    python hrm_gate_readout.py <records dir = aihub_hrm_records>
"""
import glob
import json
import os
import sys

import numpy as np

R = sys.argv[1]
INF = os.path.join(R, "art", "hrm_sudoku", "infer")
host = np.load(os.path.join(R, "art", "hrm_sudoku", "host_fp32_n8.npz"))   # preds (8,100,81); carries of steps 1-2 only
data = np.load(os.path.join(R, "art", "hrm_sudoku", "hrm_data.npz"))
lab, hp = data["lab"], host["preds"].astype(np.int64)
host_ex = [(hp[s] == lab).all(-1).mean() for s in range(8)]
CUT_H, CUT_L = 0.8206705565659004, 0.8779769526762154
rows = []
print("host fp32 exact", [round(float(x), 2) for x in host_ex])
for f in sorted(glob.glob(os.path.join(INF, "dev_*_main.json"))):
    tag = os.path.basename(f)[4:-10]
    m = json.load(open(f))
    phi = json.load(open(f[:-5] + ".phi.json"))
    P = np.load(f[:-5] + ".npz")["preds"].astype(np.int64)
    n = len(P)
    ex = [float((P[s] == lab).all(-1).mean()) for s in range(n)]
    ce = [float((P[s] == lab).mean()) for s in range(n)]
    ag = [float((P[s] == hp[s]).mean()) for s in range(n)]
    agp = [float((P[s] == hp[s]).all(-1).mean()) for s in range(n)]
    print(f"\n{tag}  compile {m.get('compile_job')} qairt {m.get('qairt_sdk_version')} guard {(m.get('guard') or {}).get('verdict')}")
    print("step  exact  cell   phi_H  phi_L  phiHmin agree_cell agree_puzzle")
    for s in range(n):
        p = phi[str(s + 1)]
        print(f"{s + 1:>3}  {ex[s]:.2f}  {ce[s]:.4f}  {p['phi_H']:.4f} {p['phi_L']:.4f} {p['phi_H_min']:.3f}  {ag[s]:.4f}  {agp[s]:.2f}")
    if n >= 8:
        rows.append((tag, phi["2"]["phi_H"], phi["2"]["phi_L"], ag[1], agp[1], ex[7], ex[7] / host_ex[7]))
print("\nGATE (step 2) vs step-8 outcome (fail = exact_8/host_exact_8 < 0.5, not-pass < 0.95)")
print("chain  phi_H2  gate_H  phi_L2  gate_L  agree_cell2  agree_puzzle2  exact8  ratio8  outcome")
for t, ph, pl, a, ap, e8, r8 in rows:
    out = "FAIL" if r8 < 0.5 else ("PASS" if r8 >= 0.95 else "WARN")
    print(f"{t}  {ph:.4f}  {'PASS' if ph >= CUT_H else 'FAIL'}  {pl:.4f}  {'PASS' if pl >= CUT_L else 'FAIL'}  {a:.4f}  {ap:.2f}  {e8:.2f}  {r8:.2f}  {out}")
