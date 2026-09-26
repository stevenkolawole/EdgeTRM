#!/bin/bash
#SBATCH --job-name=x4ret_c3_ouro14_3
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/home/skolawol/breadth_results/lm/logs/%x_%j.out
# X4 return test (prereg/X4_return_test_prereg.md, commit 8a14e1f): ouro14, 4 loops, hand-off k=2, cells w8c,w4g32;
# GSM8K n=100, batch 16, Babel L40S, transformers 4.55.4 overlay. Resumable per cell.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=/data/hf_cache/skolawol PYTHONPATH=$HOME/pylibs/tf4554 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True X4_OURO_AUTOPATCH=1
python -c "import torch, transformers; assert torch.cuda.is_available(), 'NO CUDA on this node'; print(torch.cuda.get_device_name(0), transformers.__version__)" || exit 1
cd $HOME/edgetrm_src/lm || exit 1
python ret_test.py --model ouro14 --steps 4 --k 2 --quants w8c,w4g32 --n 100 --batch 16 --outdir $HOME/breadth_results/lm/ret
