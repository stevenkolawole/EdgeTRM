#!/bin/bash
#SBATCH --job-name=X1_onehot
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skolawol/breadth_results/x1rb/logs/%x_%j.out
# X1 pre-registered one-hot probe (X1.md, 2026-09-26T02:55Z). Resumable per spec.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA on this node'; print(torch.cuda.get_device_name(0))" || { echo "NO CUDA, abort"; exit 1; }
R=$HOME/breadth_results/x1rb; mkdir -p $R/logs
cd ~/edgetrm_src || exit 1
echo "$(date -u) start onehot on $(hostname)"
python onehot_probe.py --task sudoku_attn --n 128 --K 512 --quants fp32,w3c,w3a,w3g32,w4t,noise0.5@s1,noise0.7@s0 --out $R/onehot_sudoku_attn.json > $R/logs/onehot_attn.log 2>&1 &
python onehot_probe.py --task sudoku_mlp --n 128 --K 512 --quants fp32,w4c,w4a,w4g32 --out $R/onehot_sudoku_mlp.json > $R/logs/onehot_mlp.log 2>&1 &
wait
echo "$(date -u) done onehot"
