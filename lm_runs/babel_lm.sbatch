#!/bin/bash
# EdgeTRM X4: looped-LM decode/quant jobs on Babel.  JOBFILE line 1 = looped_decode.py arguments.
# Resumable (looped_decode.py skips finished batches), so it is safe on the preempt partition
# with --requeue.  Results go to $HOME/breadth_results/lm (visible from the login node); model
# weights and bf16 references live under /data (compute nodes only).
#SBATCH --job-name=x4lm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=babel-o9-24,babel-o5-16
#SBATCH --requeue
#SBATCH --output=/home/skolawol/breadth_results/lm/logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=/data/hf_cache/skolawol
export PYTHONPATH=$HOME/pylibs/tf4571
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch, transformers; assert torch.cuda.is_available(), 'NO CUDA on this node'; print(torch.cuda.get_device_name(0), transformers.__version__)" || exit 1
mkdir -p /data/user_data/skolawol/lm_ref $HF_HOME
cd ~/edgetrm_src/lm || exit 1
ARGS=$(sed -n 1p "$JOBFILE")
echo "$(date -u) host $(hostname) restart ${SLURM_RESTART_COUNT:-0}: $ARGS"
python looped_decode.py $ARGS --refdir /data/user_data/skolawol/lm_ref
rc=$?
echo "$(date -u) done rc=$rc"
exit $rc
