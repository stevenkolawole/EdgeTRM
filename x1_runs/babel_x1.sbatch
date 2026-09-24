#!/bin/bash
#SBATCH --job-name=x1diff
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/home/skolawol/breadth_results/logs/%x_%j.out
# X1 (Sudoku difficulty axis). JOBFILE: one line per run, "<script> <args>"; the lines run
# concurrently on this job's one GPU. Resumable: difficulty_sweep.py skips specs already in
# its --out, so a preempted-and-requeued job continues where it stopped.
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=$HOME/hf_cache
echo "$(date -u) start on $(hostname) job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0}"
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA on this node'; print(torch.cuda.get_device_name(0))" || exit 1
cd ~/edgetrm_src || exit 1
i=0
while IFS= read -r line; do
  [ -z "$line" ] && continue
  i=$((i+1))
  echo "$(date -u) [$i] python $line"
  python $line > >(sed -u "s/^/[$i] /") 2>&1 &
done < "$JOBFILE"
wait
echo "$(date -u) all done"
