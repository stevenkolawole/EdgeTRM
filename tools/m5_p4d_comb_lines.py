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
Maze (added 08:10 UTC, --maze): TRM Maze w2a, w4t; HRM Maze w2a, w2g32; rows 0-499 (dev) and 500-999 (held out), n = 500,
K = 32 cheap steps (the mazes settle by 16; a K = 16 maze polish record took 850-2030 s, so K = 32 is 30-70 min each).
Output p4d ~/m5/comb/polish_o{0,1000}/<task>/; repo copy results/m5/comb/p4d/. Read-out: python m5_comb.py.

    python tools/m5_p4d_comb_lines.py [--smoke]     # prints "<prio> <mem_GB> <owner> <tag> <command>" lines
"""
import sys

H = "/home/ubuntu"
PRE = f"cd {H}/edgetrm_src && HOME={H} HF_HOME={H}/hf_cache HF_HUB_OFFLINE=1 {H}/venvs/kvcache/bin/python"
CELLS = {"sudoku_mlp": ["w4c", "w3g32"], "sudoku_attn": ["w3c", "w4t"], "hrm_sudoku": ["w3c"]}
S = {"sudoku_mlp": "mlp", "sudoku_attn": "attn", "hrm_sudoku": "hrm"}
smoke = "--smoke" in sys.argv
if "--maze" in sys.argv:
    CELLS = {"maze_attn": ["w2a", "w4t"], "hrm_maze": ["w2a", "w2g32"]}
    S = {"maze_attn": "mza", "hrm_maze": "hmz"}
OFFS, KK, NN = ((0, 500), 32, "--n 500") if "--maze" in sys.argv else ((0, 1000), 64, "")
OUT = f"{H}/m5/comb_smoke" if smoke else f"{H}/m5/comb"
N = "--n 16" if smoke else NN
P = 5
L = []
for off in OFFS:
    for f, cl in CELLS.items():
        for c in cl:
            t = f"{S[f]}_{c}_o{off}" + ("_smoke" if smoke else "")
            base = f"{PRE} exit_polish.py --task {f} --cheap {c} --polish w8c --K {KK} --J 4 {N} --offset {off} --outdir {OUT}/polish_o{off}/{f}"
            if smoke:
                L.append(f"1 8 M5 comb_{t} {base} ; {base} --dither w8c --seeds 2")
                continue
            L.append(f"{P} 8 M5 comb_rtn_{t} {base}")
            for k in range(3):
                # one seed per line so the cards pack. (The first Sudoku launch used --seeds k+1, relying on the
                # exists-skip; lines launched together then computed the same seed twice and one os.replace failed.)
                L.append(f"{P} 8 M5 comb_dit{k}_{t} {base} --dither w8c --seeds {k}:{k + 1}")
if "--rerun" in sys.argv:            # low-priority repeats of the exposed lines (tag _r1): a finished record is skipped in seconds
    L = [l.replace(" 5 8 M5 comb_", " 9 8 M5 comb_", 1).replace(f" {l.split()[3]} ", f" {l.split()[3]}_r1 ", 1) for l in L]
print("\n".join(L))
