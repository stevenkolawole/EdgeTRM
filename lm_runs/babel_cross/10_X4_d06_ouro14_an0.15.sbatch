#!/bin/bash
#SBATCH --job-name=x4x_ouro14_an0.15
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/home/skolawol/breadth_results/lm/logs/%x_%j.out
# X4 within-model crossing exhibit on Babel L40S (re-homed from p5): ouro14, 4 loops, an0.15, GSM8K n=500,
# batch 48 for every cell of this model (bf16 reference included). Resumable (looped_decode skips finished batches).
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=/data/hf_cache/skolawol PYTHONPATH=$HOME/pylibs/tf4554 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True X4_OURO_AUTOPATCH=1
python -c "import torch, transformers; assert torch.cuda.is_available(), 'NO CUDA on this node'; print(torch.cuda.get_device_name(0), transformers.__version__)" || exit 1
mkdir -p /data/user_data/skolawol/lm_ref_cross $HOME/breadth_results/lm/cross_babel
cd $HOME/edgetrm_src/lm || exit 1
python looped_decode.py --model ouro14 --steps 4 --quants an0.15 --decodes greedy --n 500 --batch 48 --outdir $HOME/breadth_results/lm/cross_babel --refdir /data/user_data/skolawol/lm_ref_cross
