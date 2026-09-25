#!/usr/bin/env python3
"""M5 Babel backlog files (Sep 25): controller held-out records (released Sudoku rows 1000-1999, trained seeds dev+held-out,
mazes rows 0-499 / 500-999) and overthinking phi+V-from-one-trajectory runs. Writes <prio>_M5_<tag>.sbatch in the cwd."""
import os
H = "/home/skolawol"
R = f"{H}/breadth_results"
EXC = "babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28"


def job(fname, name, part, lines, comment, hours=12, mem=64):
    pre = "#SBATCH --requeue\n#SBATCH --qos=preempt_qos\n" if part == "preempt" else ""
    body = f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude={EXC}
#SBATCH --cpus-per-task=6
#SBATCH --mem={mem}G
#SBATCH --time={hours}:00:00
{pre}#SBATCH --output={R}/logs/%x_%j.out
# M5 ({comment}). Lines run concurrently on the one GPU; every script is resumable (finished records skipped).
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
python -c "import torch; assert torch.cuda.is_available(), \\"NO CUDA on this node\\"; print(torch.cuda.get_device_name(0))" || exit 1
cd ~/edgetrm_src || exit 1
echo "$(date -u) start on $(hostname) job=$SLURM_JOB_ID restart=${{SLURM_RESTART_COUNT:-0}}"
"""
    for i, l in enumerate(lines, 1):
        body += f'( {l} ) > >(sed -u "s/^/[{i}] /") 2>&1 &\n'
    body += 'wait\necho "$(date -u) all done"\n'
    open(fname, "w").write(body)


def chain(*cmds):
    return " ; ".join(cmds)


# ---------------- A: controller, released Sudoku models, held-out rows 1000-1999 (general)
CELLS = {"sudoku_mlp": "fp32,noise0.15@s0,noise0.15@s2,noise0.1@s0,w3a,w3c,w3g32,w4c,w4t",
         "sudoku_attn": "fp32,noise0.5@s0,noise0.5@s1,noise0.5@s2,noise0.7@s0,noise0.7@s1,noise0.7@s2,w3a,w3c,w3g32,w4c,w4t",
         "hrm_sudoku": "fp32,noise1.0@s0,noise1.0@s1,noise1.0@s2,w2a,w2g32,w3c,w3g32,w4c,w4t"}
KDET = {"sudoku_mlp": 768, "sudoku_attn": 768, "hrm_sudoku": 512}
ab = {"sudoku_mlp": "a1", "sudoku_attn": "a2", "hrm_sudoku": "a3"}
for f, cells in CELLS.items():
    cl = cells.split(",")
    half = (len(cl) + 1) // 2
    cheap = ",".join(c for c in cl if c != "fp32")
    det = lambda specs, f=f: f"python exit_sweep.py --task {f} --offset 1000 --K {KDET[f]} --specs {specs} --outdir {R}/m5ctl/det_o1000/{f}"
    m2 = lambda specs, f=f: (f"python m2_bestofk.py --task {f} --specs {specs} --noise ptrm --sigmas 0.3 --K 32 --D 16 --polish 2 "
                             f"--offset 1000 --outdir {R}/m5ctl/m2cells --index {R}/m5ctl/index_ho_{f}.jsonl")
    pol = chain(f"python exit_polish.py --task {f} --cheap {cheap} --polish fp32 --K 16 --J 4 --offset 1000 --outdir {R}/m5ctl/polish_o1000/{f}",
                f"python exit_polish.py --task {f} --cheap {cheap} --polish fp32 --K 16 --J 4 --outdir {R}/m5/polish/{f}")
    job(f"10_M5_{ab[f]}_ctlho_{f}_det.sbatch", f"M5_{ab[f]}_ctlho_{f}_det", "general",
        [det(",".join(cl[:half])), det(",".join(cl[half:])), pol],
        f"controller held-out rows 1000-1999, {f}: deterministic exit records + fp32 polish records")
    job(f"10_M5_{ab[f]}_ctlho_{f}_m2.sbatch", f"M5_{ab[f]}_ctlho_{f}_m2", "general",
        [m2(",".join(cl[:half])), m2(",".join(cl[half:]))],
        f"controller held-out rows 1000-1999, {f}: PTRM rollouts K32 with 2 fp32 polish steps")

# ---------------- B: controller, trained seeds, dev rows 0-999 and held-out 1000-1999 (general)
CK = f"{H}/edgetrm_ckpt/variants"
for arch, task, bb in (("attn512", "sudoku_attn", "b1"), ("mlp512", "sudoku_mlp", "b2")):
    lines = []
    for s in range(3):
        ck = f"{CK}/trm_{task}_w512_s{s}/step_65100"
        tag = f"{arch}s{s}"
        cmds = [f"python exit_sweep.py --task {task} --ckpt {ck} --offset 1000 --K 192 --specs fp32,w4c,w3c,w3g32 "
                f"--outdir {R}/m5ctl/det_o1000/var/{arch}_s{s}"]
        for off in (0, 1000):
            cmds.append(f"python exit_polish.py --task {task} --ckpt {ck} --cheap w4c,w3c,w3g32 --polish fp32 --K 16 --J 4 "
                        f"--offset {off} --outdir {R}/m5ctl/polish_o{off}/var/{arch}_s{s}")
        for off in (0, 1000):
            cmds.append(f"python m2_bestofk.py --task {task} --ckpt {ck} --tag _{tag} --specs fp32,w4c,w3c,w3g32 --noise ptrm --sigmas 0.3 "
                        f"--K 16 --D 16 --polish 2 --offset {off} --outdir {R}/m5ctl/m2cells --index {R}/m5ctl/index_var_{tag}.jsonl")
        lines.append(chain(*cmds))
    job(f"10_M5_{bb}_ctl_var_{arch}.sbatch", f"M5_{bb}_ctl_var_{arch}", "general", lines,
        f"controller on trained seeds {arch} s0-2: held-out exit records, dev+held-out polish and PTRM rollouts K16 P2")

# ---------------- C: controller, mazes: select on rows 0-499, score on rows 500-999 (preempt)
MZ = {"maze_attn": "fp32,noise1.0@s0,noise1.5@s0,w2a,w4t", "hrm_maze": "fp32,noise1.0@s0,noise1.5@s0,w2a,w2g32"}
for f, cells in MZ.items():
    cheap = ",".join(c for c in cells.split(",") if c != "fp32")
    m2 = lambda specs, off, f=f: (f"python m2_bestofk.py --task {f} --specs {specs} --noise ptrm --sigmas 1 --K 16 --D 16 --polish 2 --n 500 "
                                  f"--offset {off} --outdir {R}/m5ctl/m2cells --index {R}/m5ctl/index_mz_{f}.jsonl")
    cl = cells.split(",")
    lines = [chain(m2(",".join(cl[:3]), 500), m2(",".join(cl[:3]), 0)),
             chain(m2(",".join(cl[3:]), 500), m2(",".join(cl[3:]), 0)),
             chain(f"python exit_polish.py --task {f} --cheap {cheap} --polish fp32 --K 16 --J 4 --n 500 --offset 500 --outdir {R}/m5ctl/polish_o500/{f}",
                   f"python exit_polish.py --task {f} --cheap {cheap} --polish fp32 --K 16 --J 4 --n 500 --outdir {R}/m5/polish/{f}")]
    job(f"10_M5_c_ctl_{f}.sbatch", f"M5_c_ctl_{f}", "preempt", lines,
        f"controller {f}: rows 0-499 select, rows 500-999 held out (the maze test set has 1000 rows): PTRM rollouts K16 P2, fp32 polish")

# ---------------- D: phi and V from one trajectory (overthinking audit item 70) (preempt)
PHI = "--phi 4,8,16"
W = {"sudoku_mlp": "fp32,w8c,w8c+a8,w4c,w4c+a8,w4g32,w4t,w3g32,w3c,w3a",
     "sudoku_attn": "fp32,w8c,w4c,w4c+a8,w4g32,w4t,w3g32,w3c,w3t,w3a",
     "hrm_sudoku": "fp32,w8c,w4c,w4t,w3g32,w3c,w3t,w3a,w2g32,w2a",
     "maze_attn": "fp32,w8c,w8a,w8t,w4c,w4a,w4t,w4g32,w4g128,w3c,w3a,w3t,w3g32,w2c,w2a,w2t,w2g32",
     "hrm_maze": "fp32,w8c,w4c,w4a,w4t,w4g32,w3c,w3a,w3t,w3g32,w2c,w2a,w2t,w2g32"}
NZ = {"sudoku_mlp": (("noise", (0.05, 0.1, 0.15, 0.2, 0.3)), ("anoise", (0.1, 0.3))),
      "sudoku_attn": (("noise", (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)), ("anoise", (0.3, 0.5))),
      "hrm_sudoku": (("noise", (0.1, 0.2, 0.3, 0.5, 0.7, 1.0)), ("anoise", (0.3, 0.5, 0.7))),
      "maze_attn": (("noise", (0.5, 1.0, 1.5, 2.0)), ("anoise", (0.5, 0.7, 1.0))),
      "hrm_maze": (("noise", (0.5, 1.0, 1.5, 2.0)), ("anoise", (0.5, 0.7, 1.0)))}
KP = {"sudoku_mlp": 48, "sudoku_attn": 48, "hrm_sudoku": 32, "maze_attn": 48, "hrm_maze": 32}


def nz(f):
    return [f"{k}{v:g}@s{s}" for k, vs in NZ[f] for v in vs for s in range(3)]


def es(f, specs, extra="", od=None):
    return f"python exit_sweep.py --task {f} --K {KP[f]} {PHI} {extra} --specs {specs} --outdir {od or R + '/m5phi/' + f}"


job("10_M5_d1_phi_sudoku.sbatch", "M5_d1_phi_sudoku", "preempt",
    [chain(es(f, W[f]), es(f, ",".join(nz(f)))) for f in ("sudoku_mlp", "sudoku_attn", "hrm_sudoku")],
    "overthinking: V and phi from one trajectory, released Sudoku models, K=16 default-H steps, n=1000")
lines = []
for arch, task in (("attn512", "sudoku_attn"), ("mlp512", "sudoku_mlp")):
    nzs = "noise0.3@s0,noise0.5@s0,noise0.7@s0,anoise0.3@s0" if arch.startswith("attn") else "noise0.1@s0,noise0.15@s0,noise0.2@s0,anoise0.3@s0"
    cmds = []
    for s in range(3):
        ck = f"{CK}/trm_{task}_w512_s{s}/step_65100"
        cmds.append(es(task, "fp32,w8c,w4c,w4t,w3g32,w3c," + nzs, f"--ckpt {ck}", f"{R}/m5phi/var/{arch}_s{s}"))
    lines.append(chain(*cmds))
dtl = [f"python dt_exit.py --size 13 --T 200 --phi 50,100,200 --specs fp32,w8c,w4c,w4t,w3a,noise0.2@s0,noise0.2@s1,noise0.2@s2 --outdir {R}/m5phi/dt_maze13",
       f"python dt_exit.py --size 25 --T 400 --phi 100,200,400 --specs fp32,w8c,w4c,w4t,w3a,noise0.2@s0,noise0.2@s1,noise0.2@s2 --outdir {R}/m5phi/dt_maze25",
       f"python dt_exit.py --size 33 --T 500 --phi 100,300,500 --specs fp32,w8c,w4g32,w4c,w4t,w3a,w8c+a8,noise0.2@s0,noise0.2@s1,noise0.2@s2,"
       f"noise0.3@s0,noise0.3@s1,noise0.3@s2 --outdir {R}/m5phi/dt_maze33"]
job("10_M5_d2_phi_var_dt.sbatch", "M5_d2_phi_var_dt", "preempt", lines + [chain(dtl[0], dtl[1]), dtl[2]],
    "overthinking: V and phi from one trajectory, trained seeds (attn512/mlp512 s0-2) and deep-thinking mazes")
for f in ("maze_attn", "hrm_maze"):
    w = W[f].split(",")
    z = nz(f)
    parts = [w[:len(w) // 2], w[len(w) // 2:], z[:len(z) // 2], z[len(z) // 2:]]
    for j, (p1, p2) in enumerate(((parts[0], parts[2]), (parts[1], parts[3]))):
        job(f"10_M5_d3_phi_{f}_{j}.sbatch", f"M5_d3_phi_{f}_{j}", "preempt",
            [es(f, ",".join(p1)), es(f, ",".join(p2))], f"overthinking: V and phi from one trajectory, {f} part {j}")
