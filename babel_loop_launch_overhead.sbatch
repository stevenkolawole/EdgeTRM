#!/bin/bash
#SBATCH --job-name=loop_launch_overhead
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skolawol/trt_gate/logs/%x_%j.out
# k-step unrolled engine vs k launches (carry on device / via host), loop_launch_overhead.py.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print(torch.cuda.get_device_name(0))" || { echo "PREFLIGHT FAILED: no CUDA"; exit 1; }
D=/data/user_data/$USER
PYL=$D/e1/pylibs
[ -e $PYL/trt/tensorrt/__init__.py ] || { echo "TensorRT missing under $PYL"; exit 1; }
export PYTHONPATH=$PYL/trt
export TG_WORK=$D/trt_gate/work_launch
R=$HOME/trt_gate/results/launch_overhead; mkdir -p $R
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv,noheader
echo "$(date -u) start on $(hostname)"
cd $HOME/trt_gate/code || exit 1
for t in sudoku_mlp maze; do python loop_launch_overhead.py --task $t --out $R/$t.json || echo "RUN FAILED $t"; done
echo "$(date -u) done" && touch $R/DONE
