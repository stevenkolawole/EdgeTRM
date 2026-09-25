#!/usr/bin/env python3
"""M5 on p4d (Sep 25 13:00 UTC; p5 returned to Steven at 14:00, nothing of M5 finished there): the unified controller
(R18) and the overthinking phi/V exhibit (R19), each self-contained on p4d (A100 40 GB), fp32 references included.

Scope cut to fit the remaining day (stated in M5.md R18): released Sudoku deterministic trajectories to 64 default-H steps
(K = 192 / 128 H=1 steps, as the trained seeds), PTRM rollouts K16 with 2 fp32 polish steps everywhere, a reduced cell list
per model (every weight family and at least two draws per noise level kept), trained seeds fp32 / w4c / w3c.
The DT mazes' controller read-out uses the m5phi DT records (same p4d runs, rows 0-499 / 500-999).
Prints p4d autofill backlog lines "<prio> <mem_GB> <owner> <tag> <command>"; small lines (one or two cells each) so they pack.

    python tools/m5_p4d_lines.py > m5_p4d_lines.txt
"""
H = "/home/ubuntu"
OUT = f"{H}/m5"
PRE = f"cd {H}/edgetrm_src && HOME={H} HF_HOME={H}/hf_cache HF_HUB_OFFLINE=1 {H}/venvs/kvcache/bin/python"
CK = f"{H}/edgetrm_ckpt/variants"
P = 5
lines = []


def add(mem, tag, cmd):
    lines.append(f"{P} {mem} M5 {tag} {cmd}")


# ---------------------------------------------------------------- R18 controller
CELLS = {"sudoku_mlp": ["fp32", "noise0.15@s0", "noise0.15@s2", "w3g32", "w4c", "w4t"],
         "sudoku_attn": ["fp32", "noise0.7@s0", "noise0.7@s1", "noise0.7@s2", "w3c", "w4c"],
         "hrm_sudoku": ["fp32", "noise1.0@s0", "noise1.0@s1", "w2g32", "w4c", "w4t"]}
KDET = {"sudoku_mlp": 192, "sudoku_attn": 192, "hrm_sudoku": 128}
S = {"sudoku_mlp": "mlp", "sudoku_attn": "attn", "hrm_sudoku": "hrm", "maze_attn": "mza", "hrm_maze": "hmz"}
for f, cl in CELLS.items():
    s = S[f]
    for off in (0, 1000):
        add(8, f"m5d_{s}_det_o{off}", f"{PRE} exit_sweep.py --task {f} --offset {off} --K {KDET[f]} --specs {','.join(cl)} "
                                      f"--outdir {OUT}/ctl/det_o{off}/{f}")
        for i in range(0, len(cl), 2):
            add(10, f"m5d_{s}_m2{i // 2}_o{off}", f"{PRE} m2_bestofk.py --task {f} --specs {','.join(cl[i:i + 2])} --noise ptrm --sigmas 0.3 "
                                                  f"--K 16 --D 16 --polish 2 --offset {off} --outdir {OUT}/ctl/m2cells "
                                                  f"--index {OUT}/ctl/index_{s}_m2{i // 2}_o{off}.jsonl")
        add(8, f"m5d_{s}_pol_o{off}", f"{PRE} exit_polish.py --task {f} --cheap {','.join(c for c in cl if c != 'fp32')} --polish fp32 "
                                      f"--K 16 --J 4 --offset {off} --outdir {OUT}/ctl/polish_o{off}/{f}")
for arch, task in (("attn512", "sudoku_attn"), ("mlp512", "sudoku_mlp")):
    for sd in range(3):
        ck = f"{CK}/trm_{task}_w512_s{sd}/step_65100"
        for off in (0, 1000):
            add(10, f"m5d_var_{arch}s{sd}_o{off}",
                f"{PRE} exit_sweep.py --task {task} --ckpt {ck} --offset {off} --K 192 --specs fp32,w4c,w3c --outdir {OUT}/ctl/det_o{off}/var/{arch}_s{sd} ; "
                f"{PRE.split(' && ')[1]} exit_polish.py --task {task} --ckpt {ck} --cheap w4c,w3c --polish fp32 --K 16 --J 4 --offset {off} "
                f"--outdir {OUT}/ctl/polish_o{off}/var/{arch}_s{sd} ; "
                f"{PRE.split(' && ')[1]} m2_bestofk.py --task {task} --ckpt {ck} --tag _{arch}s{sd} --specs fp32,w4c,w3c --noise ptrm --sigmas 0.3 "
                f"--K 16 --D 16 --polish 2 --offset {off} --outdir {OUT}/ctl/m2cells --index {OUT}/ctl/index_var_{arch}s{sd}_o{off}.jsonl")
MZ = {"maze_attn": ["fp32", "noise1.0@s0", "noise1.5@s0", "w2a", "w4t"], "hrm_maze": ["fp32", "noise1.0@s0", "noise1.5@s0", "w2a", "w2g32"]}
KMZ = {"maze_attn": 48, "hrm_maze": 32}
for f, cl in MZ.items():
    s = S[f]
    add(12, f"m5d_{s}_det", f"{PRE} exit_sweep.py --task {f} --K {KMZ[f]} --specs {','.join(cl)} --outdir {OUT}/ctl/det_o0/{f}")
    for off in (0, 500):
        for i in range(0, len(cl), 2):
            add(16, f"m5d_{s}_m2{i // 2}_o{off}", f"{PRE} m2_bestofk.py --task {f} --specs {','.join(cl[i:i + 2])} --noise ptrm --sigmas 1 --K 16 "
                                                  f"--D 16 --polish 2 --n 500 --offset {off} --outdir {OUT}/ctl/m2cells "
                                                  f"--index {OUT}/ctl/index_{s}_m2{i // 2}_o{off}.jsonl")
        add(12, f"m5d_{s}_pol_o{off}", f"{PRE} exit_polish.py --task {f} --cheap {','.join(c for c in cl if c != 'fp32')} --polish fp32 "
                                       f"--K 16 --J 4 --n 500 --offset {off} --outdir {OUT}/ctl/polish_o{off}/{f}")

# ---------------------------------------------------------------- R19 overthinking (phi and V from one trajectory)
W = {"sudoku_mlp": "fp32,w8c,w8c+a8,w4c,w4c+a8,w4g32,w4t,w3g32,w3c,w3a",
     "sudoku_attn": "fp32,w8c,w4c,w4c+a8,w4g32,w4t,w3g32,w3c,w3t,w3a",
     "hrm_sudoku": "fp32,w8c,w4c,w4t,w3g32,w3c,w3t,w3a,w2g32,w2a",
     "maze_attn": "fp32,w8c,w4c,w4a,w4t,w4g32,w3c,w3a,w3t,w3g32,w2c,w2a,w2t,w2g32",
     "hrm_maze": "fp32,w8c,w4c,w4a,w4t,w4g32,w3c,w3a,w3t,w3g32,w2c,w2a,w2t,w2g32"}
NZ = {"sudoku_mlp": (("noise", (0.05, 0.1, 0.15, 0.2, 0.3)), ("anoise", (0.1, 0.3))),
      "sudoku_attn": (("noise", (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)), ("anoise", (0.3, 0.5))),
      "hrm_sudoku": (("noise", (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)), ("anoise", (0.3, 0.5, 0.7))),
      "maze_attn": (("noise", (0.5, 1.0, 1.5, 2.0)), ("anoise", (0.5, 0.7, 1.0))),
      "hrm_maze": (("noise", (0.5, 1.0, 1.5, 2.0)), ("anoise", (0.5, 0.7, 1.0)))}
KP = {"sudoku_mlp": 48, "sudoku_attn": 48, "hrm_sudoku": 32, "maze_attn": 48, "hrm_maze": 32}


def es(f, specs, extra="", od=None):
    return f"{PRE} exit_sweep.py --task {f} --K {KP[f]} --phi 4,8,16 {extra} --specs {specs} --outdir {od or OUT + '/phi/' + f}"


for f in W:
    w = W[f].split(",")
    z = [f"{k}{v:g}@s{s}" for k, vs in NZ[f] for v in vs for s in range(3)]
    step = 3 if "maze" in f else 6
    chunks = [w[i:i + step] for i in range(0, len(w), step)] + [z[i:i + step] for i in range(0, len(z), step)]
    for i, ch in enumerate(chunks):
        add(12 if "maze" in f else 8, f"m5p_{f}_{i}", es(f, ",".join(ch)))
for arch, task in (("attn512", "sudoku_attn"), ("mlp512", "sudoku_mlp")):
    nzs = "noise0.3@s0,noise0.5@s0,noise0.7@s0,anoise0.3@s0" if arch.startswith("attn") else "noise0.1@s0,noise0.15@s0,noise0.2@s0,anoise0.3@s0"
    for sd in range(3):
        add(8, f"m5p_var_{arch}_s{sd}", es(task, "fp32,w8c,w4c,w4t,w3g32,w3c," + nzs, f"--ckpt {CK}/trm_{task}_w512_s{sd}/step_65100",
                                           f"{OUT}/phi/var/{arch}_s{sd}"))
for size, T, ph, specs in ((13, 200, "50,100,200", "fp32,w8c,w4c,w4t,w3a,noise0.2@s0,noise0.2@s1,noise0.2@s2"),
                           (25, 400, "100,200,400", "fp32,w8c,w4c,w4t,w3a,noise0.2@s0,noise0.2@s1,noise0.2@s2"),
                           (33, 500, "100,300,500", "fp32,w8c,w4g32,w4c,w4t,w3a,w8c+a8,noise0.2@s0,noise0.2@s1,noise0.2@s2,noise0.3@s0,noise0.3@s1,noise0.3@s2")):
    add(8, f"m5p_dt{size}", f"{PRE} dt_exit.py --size {size} --T {T} --phi {ph} --specs {specs} --outdir {OUT}/phi/dt_maze{size}")

print("\n".join(lines))
