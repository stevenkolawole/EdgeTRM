#!/usr/bin/env python3
"""M5 x GPTQ (ICLR comparison, Sep 26): p4d backlog lines, prio 5, owner M5. RTN vs GPTQ (gq<fmt>, m5_gptq.py), each at
fixed depth and under the controller with w8c finishing, TRM Sudoku-MLP / Sudoku-attn / HRM Sudoku, rows 0-999 (dev) and
1000-1999 (held out). Rollouts K8 (existing RTN K16 records are truncated to 8 in the read-out, m5_gptq_ho.py)."""
H = "/home/ubuntu"
OUT = f"{H}/m5/ctl"
PRE = f"cd {H}/edgetrm_src && HOME={H} HF_HOME={H}/hf_cache HF_HUB_OFFLINE=1 {H}/venvs/kvcache/bin/python"
S = {"sudoku_mlp": "mlp", "sudoku_attn": "attn", "hrm_sudoku": "hrm"}
KDET = {"sudoku_mlp": 192, "sudoku_attn": 192, "hrm_sudoku": 128}
GQ = {"sudoku_mlp": ["gqw3c", "gqw3g32", "gqw2g32", "gqw4t"], "sudoku_attn": ["gqw3c", "gqw3g32", "gqw2g32"],
      "hrm_sudoku": ["gqw3c", "gqw3g32", "gqw2g32"]}
RTN_NEW = {"sudoku_mlp": ["w3c", "w2g32"], "sudoku_attn": ["w3g32", "w2g32"], "hrm_sudoku": ["w3c", "w3g32"]}
L = []
for f in GQ:
    L.append(f"5 6 M5 m5g_build_{S[f]} {PRE} m5_gptq.py --task {f} --specs {','.join(GQ[f])}")
for off in (1000, 0):
    for f in GQ:
        for c in GQ[f] + RTN_NEW[f]:
            t = f"{S[f]}_{c}_o{off}"
            L.append(f"5 8 M5 m5g_det_{t} {PRE} exit_sweep.py --task {f} --offset {off} --K {KDET[f]} --specs {c} --outdir {OUT}/det_o{off}/{f}")
            L.append(f"5 10 M5 m5g_m2_{t} {PRE} m2_bestofk.py --task {f} --specs {c} --noise ptrm --sigmas 0.3 --K 8 --D 16 --polish 2 "
                     f"--offset {off} --outdir {OUT}/m2cells --index {OUT}/index_g_{t}.jsonl")
            L.append(f"5 8 M5 m5g_pol_{t} {PRE} exit_polish.py --task {f} --cheap {c} --polish w8c --K 16 --J 4 --offset {off} "
                     f"--outdir {OUT}/polish_o{off}/{f}")
print("\n".join(L))
