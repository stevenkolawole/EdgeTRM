#!/bin/bash
# X5 Babel preempt program (Sep 26): writes 20_X5_<tag>.sbatch into ~/autofill/babel_backlog/preempt/ (run on Babel).
D=~/autofill/babel_backlog/preempt; V=$HOME/edgetrm_ckpt/variants
hdr() {  # name time
cat <<EOF
#!/bin/bash
#SBATCH --job-name=$1
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=$2
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/home/skolawol/q7/logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export Q7_ROOT=\$HOME/q7 Q7_DATA=\$HOME/q7/data/sudoku-extreme-1k-aug-1000 Q7_PY=\$(which python) Q7_SRC=\$HOME/edgetrm_src/TinyRecursiveModels HF_HOME=\$HOME/hf_cache
export WANDB_DIR_OVERRIDE=/tmp/\$USER/q7_wandb PYTHONPATH=\$HOME/q7/stubs
python -c "import torch,sys; ok=torch.cuda.is_available(); print('[preflight]', ok, torch.cuda.get_device_name(0) if ok else 'NO CUDA'); sys.exit(0 if ok else 3)" || exit 3
EOF
}
# (1) new mixer/width seeds: train (no inline eval, keep only the latest checkpoint) -> X2 S + blind width prediction ->
#     S_mix -> width sweep (q7_eval, full grid) -- in this order, so every prediction is written before its width
var() {  # run seed time archargs
  f=$D/20_X5_$1.sbatch
  { hdr $1 $3; cat <<EOF
# X5 width-law model $1 (seed $2): 65,104-step Lambda recipe with the fixed trainer; resumable (resume.pt); keeps only the
# latest step_N; then X2's S and blind width prediction (x2_sonly.sh), S_mix, and only then the width sweep (q7_eval).
C=\$HOME/q7/ckpt/$1; CK=\$C/step_65100; O=\$HOME/breadth_results/x2/sens
bash \$HOME/q7/q7_train2.sh $1 $2 $4 +q7_inline_eval=False +q7_keep_last_ckpt=True || exit 1
[ -f \$C/DONE ] && [ -f \$CK ] || { echo "no final checkpoint"; exit 1; }
[ -f \$O/sens_var_$1.json ] || bash \$HOME/x2_jobs/x2_sonly.sh $1 \$CK
[ -f \$O/smix_$1.json ] || bash \$HOME/x2_jobs/x2_smix.sh $1 sudoku_mlp 256 \$CK
cd \$HOME/edgetrm_src && python q7_eval.py --last --full --rundir \$C
echo "\$(date -u) [x5] $1 finished pipeline"
EOF
  } > $f
}
A256="arch.mlp_t=False arch.pos_encodings=rope arch.hidden_size=256 arch.num_heads=4"
M256="arch.hidden_size=256 arch.num_heads=4"
A768="arch.mlp_t=False arch.pos_encodings=rope arch.hidden_size=768 arch.num_heads=12"
M768="arch.hidden_size=768 arch.num_heads=12"
for s in 3 4 5; do var var_attn256_s$s $s 1-00:00:00 "$A256"; var var_mlp256_s$s $s 1-00:00:00 "$M256"; done
for s in 0 1 2; do var var_attn768_s$s $s 2-00:00:00 "$A768"; done
for s in 1 2 3; do var var_mlp768_s$s $s 2-00:00:00 "$M768"; done
# fine-tunes from the Lambda w256 finals (12k steps, one checkpoint at the end, full q7_eval at the end)
ft() {  # tag run seed extra
  f=$D/20_X5_$1.sbatch
  { hdr $2 12:00:00; cat <<EOF
# X5 M6 fine-tune $2 (Babel exhibit; matched controls bab_ft_fp32_s*): 12k steps from the Lambda w256 final, one checkpoint.
bash \$HOME/q7/q7_train2.sh $2 $3 epochs=9216 eval_interval=9216 lr_warmup_steps=500 +load_checkpoint=$V/trm_sudoku_mlp_w256_s$3/step_65100 $4
EOF
  } > $f
}
# (2) SAM seed 2 and its control (s0/s1 queued earlier as 20_X5_sam_*)
ft sam_samft05_s2 bab_samft05_s2 2 "+q7_reg=sam +q7_rho=0.05 +q7_start_step=0"
ft sam_samft1_s2 bab_samft1_s2 2 "+q7_reg=sam +q7_rho=0.1 +q7_start_step=0"
ft sam_ftfp32_s2 bab_ft_fp32_s2 2 ""
# (3) contraction-margin penalty: hinge on rho(J) of one ACT step at the late carry, margins 0.2 and 0.4
for s in 0 1 2; do for m in 0.2 0.4; do t=$(echo $m | tr -d .); ft ctr_m${t}_s$s bab_ctr_m${t}_s$s $s "+q7_reg=contract +q7_margin=$m +q7_lambda=1.0 +q7_sens_sub=32 +q7_sens_min_steps=8 +q7_pi_iters=8 +q7_jac_every=1 +q7_ramp_steps=1000 +q7_start_step=0"; done; done
ls $D | grep X5 | wc -l
