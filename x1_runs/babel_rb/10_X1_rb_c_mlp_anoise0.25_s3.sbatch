#!/bin/bash
#SBATCH --job-name=X1_mlp_anoise0.25_s3
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/home/skolawol/breadth_results/x1rb/logs/%x_%j.out
# X1 robustness block, self-contained Babel exhibit (activation vs weight noise, 1800 puzzles, K=512).
# Resumable: difficulty_sweep.py skips finished specs and resumes chunks (--checkpoint).
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA on this node'; print(torch.cuda.get_device_name(0))" || { echo "NO CUDA, abort"; exit 1; }
R=$HOME/breadth_results/x1rb; mkdir -p $R/logs
cd ~/edgetrm_src || exit 1
echo "$(date -u) start mlp_anoise0.25_s3 on $(hostname)"
python difficulty_sweep.py --K 512 --per_bin 300 --batch 300 --checkpoint --task sudoku_mlp --quants anoise0.25@s3 --ref_wait --ref_cache $R/ref_sudoku_mlp.pt --out $R/rb_sudoku_mlp_anoise0.25_s3.json
echo "$(date -u) done mlp_anoise0.25_s3 rc=$?"
