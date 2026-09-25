#!/bin/bash
# Q7 (X5) training run on Babel preempt: requeued on preemption, resumes from resume.pt (pretrain_q7.py).
#   sbatch --job-name=q7_<run> ~/q7/q7_babel.sbatch RUN SEED [hydra overrides]
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --open-mode=append
#SBATCH --output=/home/skolawol/q7/logs/%x_%j.out
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export Q7_ROOT=${Q7_ROOT:-$HOME/q7} Q7_DATA=$HOME/q7/data/sudoku-extreme-1k-aug-1000 Q7_PY=$(which python) \
       Q7_SRC=$HOME/edgetrm_src/TinyRecursiveModels HF_HOME=$HOME/hf_cache
export WANDB_DIR_OVERRIDE=/tmp/$USER/q7_wandb
export PYTHONPATH=$HOME/q7/stubs${PYTHONPATH:+:$PYTHONPATH}   # argdantic stub (imported, unused, by puzzle_dataset.py)
echo "$(date -u) job $SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader)"
bash $HOME/q7/q7_train2.sh "$@"
