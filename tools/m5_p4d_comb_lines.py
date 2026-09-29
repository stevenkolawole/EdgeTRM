#!/usr/bin/env python3
"""E_COMBINED (MLSys, Sep 29): the four moves on one resident 8-bit copy, run together. p4d backlog lines, owner M5.

Question: does per-step perturbation on the cheap loops (dither drawn from the INT8 copy, M2 arm 6) combined with the
controller (exit on the frozen head rule, then j <= 4 finishing steps from the same INT8 copy) beat the controller alone
and dither alone at the same honest cost (finishing step = 8 / b_eff cheap steps)?
Predictions are in reports/edgetrm_exp/E_COMBINED.md (written before any line ran).

Records (exit_polish.py, --K 64 --J 4: finishing after EVERY cheap step 1..64): per cell and row set
  <cheap>__w8c.npz                         round-to-nearest cheap trajectory (the controller-alone arm; also the fixed-depth arm)
  <cheap>__w8c__dither_mw8c_s{0,1,2}.npz   dither from the INT8 copy, three re-draw sequences (M2 arm 6 seeds)
Cells: TRM Sudoku-MLP w4c, w3g32; Sudoku-attention w3c, w4t; HRM Sudoku w3c. Rows 0-999 (dev) and 1000-1999 (held out).
Output p4d ~/m5/comb/polish_o{0,1000}/<task>/; repo copy results/m5/comb/p4d/. Read-out: python m5_comb.py.

    python tools/m5_p4d_comb_lines.py [--smoke]     # prints "<prio> <mem_GB> <owner> <tag> <command>" lines
"""
import sys

H = "/home/ubuntu"
PRE = f"cd {H}/edgetrm_src && HOME={H} HF_HOME={H}/hf_cache HF_HUB_OFFLINE=1 {H}/venvs/kvcache/bin/python"
CELLS = {"sudoku_mlp": ["w4c", "w3g32"], "sudoku_attn": ["w3c", "w4t"], "hrm_sudoku": ["w3c"]}
S = {"sudoku_mlp": "mlp", "sudoku_attn": "attn", "hrm_sudoku": "hrm"}
smoke = "--smoke" in sys.argv
OUT = f"{H}/m5/comb_smoke" if smoke else f"{H}/m5/comb"
N = "--n 16" if smoke else ""
P = 5
L = []
for off in (0, 1000):
    for f, cl in CELLS.items():
        for c in cl:
            t = f"{S[f]}_{c}_o{off}" + ("_smoke" if smoke else "")
            base = f"{PRE} exit_polish.py --task {f} --cheap {c} --polish w8c --K 64 --J 4 {N} --offset {off} --outdir {OUT}/polish_o{off}/{f}"
            if smoke:
                L.append(f"1 8 M5 comb_{t} {base} ; {base} --dither w8c --seeds 2")
                continue
            L.append(f"{P} 8 M5 comb_rtn_{t} {base}")
            for k in range(3):
                # one seed per line so the cards pack; --seeds k+1 skips the seeds whose files exist (the earlier lines)
                L.append(f"{P} 8 M5 comb_dit{k}_{t} {base} --dither w8c --seeds {k + 1}")
print("\n".join(L))
