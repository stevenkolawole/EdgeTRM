#!/usr/bin/env python3
"""M5 unified controller exhibit on p5 (self-contained: every record, fp32 included, made on p5).

Released Sudoku models (TRM MLP, TRM attention, HRM): rows 0-999 (select) and 1000-1999 (held out).
Trained seeds (attn512 / mlp512, s0-2): the same two row sets. Mazes (1000 test rows only): rows 0-499 select,
500-999 held out. Deep-thinking mazes: rows 0-499 / 500-999 of one 1000-row run.
Per cell: exit_sweep (deterministic trajectory, halting logits), m2_bestofk (PTRM z_L-noise rollouts stopped by the head,
2 fp32 polish steps from each rollout end), exit_polish (fp32 polish from the deterministic state after every step t <= 16).
Prints p5 autofill backlog lines "<prio> <mem_GB> <owner> <tag> <command>".

    python tools/m5_p5_ctl_lines.py > m5_ctl_lines.txt
"""
R = "/opt/dlami/nvme/edgetrm"
OUT = f"{R}/results/m5ctl"
PRE = f"cd {R}/EdgeTRM && HOME={R} HF_HOME={R}/hf_cache HF_HUB_OFFLINE=1 {R}/venv/bin/python"
CK = f"{R}/edgetrm_ckpt/variants"
CELLS = {"sudoku_mlp": "fp32,noise0.15@s0,noise0.15@s2,noise0.1@s0,w3a,w3c,w3g32,w4c,w4t",
         "sudoku_attn": "fp32,noise0.5@s0,noise0.5@s1,noise0.5@s2,noise0.7@s0,noise0.7@s1,noise0.7@s2,w3a,w3c,w3g32,w4c,w4t",
         "hrm_sudoku": "fp32,noise1.0@s0,noise1.0@s1,noise1.0@s2,w2a,w2g32,w3c,w3g32,w4c,w4t"}
KDET = {"sudoku_mlp": 768, "sudoku_attn": 768, "hrm_sudoku": 512}
MZ = {"maze_attn": "fp32,noise1.0@s0,noise1.5@s0,w2a,w4t", "hrm_maze": "fp32,noise1.0@s0,noise1.5@s0,w2a,w2g32"}
KMZ = {"maze_attn": 48, "hrm_maze": 32}
SHORT = {"sudoku_mlp": "mlp", "sudoku_attn": "attn", "hrm_sudoku": "hrm", "maze_attn": "mza", "hrm_maze": "hmz"}
lines = []


def add(mem, tag, cmd):
    lines.append(f"10 {mem} M5 m5c_{tag} {cmd}")


def groups(lst, k):
    n = (len(lst) + k - 1) // k
    return [lst[i:i + n] for i in range(0, len(lst), n)]


for f, cells in CELLS.items():
    cl, s = cells.split(","), SHORT[f]
    cheap = ",".join(c for c in cl if c != "fp32")
    for off in (0, 1000):
        for g, grp in enumerate(groups(cl, 3)):
            add(8, f"{s}_det{g}_o{off}", f"{PRE} exit_sweep.py --task {f} --offset {off} --K {KDET[f]} --specs {','.join(grp)} "
                                         f"--outdir {OUT}/det_o{off}/{f}")
            add(16, f"{s}_m2{g}_o{off}", f"{PRE} m2_bestofk.py --task {f} --specs {','.join(grp)} --noise ptrm --sigmas 0.3 --K 32 --D 16 "
                                         f"--polish 2 --offset {off} --outdir {OUT}/m2cells --index {OUT}/index_{s}_m2{g}_o{off}.jsonl")
        add(8, f"{s}_pol_o{off}", f"{PRE} exit_polish.py --task {f} --cheap {cheap} --polish fp32 --K 16 --J 4 --offset {off} "
                                  f"--outdir {OUT}/polish_o{off}/{f}")

for arch, task in (("attn512", "sudoku_attn"), ("mlp512", "sudoku_mlp")):
    for sd in range(3):
        ck = f"{CK}/trm_{task}_w512_s{sd}/step_65100"
        for off in (0, 1000):
            c1 = f"{PRE} exit_sweep.py --task {task} --ckpt {ck} --offset {off} --K 192 --specs fp32,w4c,w3c,w3g32 --outdir {OUT}/det_o{off}/var/{arch}_s{sd}"
            c2 = (f"{PRE.split(' && ')[1]} exit_polish.py --task {task} --ckpt {ck} --cheap w4c,w3c,w3g32 --polish fp32 --K 16 --J 4 "
                  f"--offset {off} --outdir {OUT}/polish_o{off}/var/{arch}_s{sd}")
            c3 = (f"{PRE.split(' && ')[1]} m2_bestofk.py --task {task} --ckpt {ck} --tag _{arch}s{sd} --specs fp32,w4c,w3c,w3g32 --noise ptrm "
                  f"--sigmas 0.3 --K 16 --D 16 --polish 2 --offset {off} --outdir {OUT}/m2cells --index {OUT}/index_var_{arch}s{sd}_o{off}.jsonl")
            add(16, f"var_{arch}s{sd}_o{off}", f"{c1} ; {c2} ; {c3}")

for f, cells in MZ.items():
    cl, s = cells.split(","), SHORT[f]
    cheap = ",".join(c for c in cl if c != "fp32")
    add(12, f"{s}_det", f"{PRE} exit_sweep.py --task {f} --K {KMZ[f]} --specs {cells} --outdir {OUT}/det_o0/{f}")
    for off in (0, 500):
        for g, grp in enumerate(groups(cl, 2)):
            add(24, f"{s}_m2{g}_o{off}", f"{PRE} m2_bestofk.py --task {f} --specs {','.join(grp)} --noise ptrm --sigmas 1 --K 16 --D 16 "
                                         f"--polish 2 --n 500 --offset {off} --outdir {OUT}/m2cells --index {OUT}/index_{s}_m2{g}_o{off}.jsonl")
        add(12, f"{s}_pol_o{off}", f"{PRE} exit_polish.py --task {f} --cheap {cheap} --polish fp32 --K 16 --J 4 --n 500 --offset {off} "
                                   f"--outdir {OUT}/polish_o{off}/{f}")

DT = {13: (200, "fp32,w8c,w4c,w4t,w3a,noise0.2@s0,noise0.2@s1,noise0.2@s2"),
      25: (400, "fp32,w8c,w4c,w4t,w3a,noise0.2@s0,noise0.2@s1,noise0.2@s2"),
      33: (500, "fp32,w8c,w4g32,w4c,w4t,w3a,w8c+a8,noise0.2@s0,noise0.2@s1,noise0.2@s2,noise0.3@s0,noise0.3@s1,noise0.3@s2")}
for size, (T, specs) in DT.items():
    add(8, f"dt{size}", f"{PRE} dt_exit.py --size {size} --T {T} --specs {specs} --outdir {OUT}/dt_maze{size}")

print("\n".join(lines))
