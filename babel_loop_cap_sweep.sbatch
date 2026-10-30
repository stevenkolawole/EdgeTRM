#!/bin/bash
#SBATCH --job-name=loop_cap_sweep
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/skolawol/capsweep/logs/%x_%j.out
# Loop-cap sweep of one model on one L40S (loop_cap_sweep.py); usage: sbatch --export=ALL,TASK=<task> this.sbatch
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
export E1_WORK=/data/user_data/$USER/e1/work E1_RES=$HOME/e1/results E1_ONNX=$HOME/e1/onnx E1_X3ART=$HOME/e1/x3
PYL=/data/user_data/$USER/e1/pylibs
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA on this node'; print(torch.cuda.get_device_name(0))" || { echo "PREFLIGHT FAILED: no CUDA"; exit 1; }
nvidia-smi --query-gpu=name,uuid,driver_version,power.limit,clocks.max.sm --format=csv
export PYTHONPATH=$PYL/nvml:$HOME/e1/code
python -c "import pynvml; pynvml.nvmlInit(); print('nvml ok')" || { echo "pynvml import FAILED"; exit 1; }
cd $HOME/e1/code || exit 1
echo "$(date -u) start capsweep $TASK on $(hostname) job=$SLURM_JOB_ID"
python $HOME/capsweep/code/loop_cap_sweep.py --task $TASK --out $HOME/capsweep/results/$TASK.json
rc=$?
echo "$(date -u) done rc=$rc"
exit $rc
