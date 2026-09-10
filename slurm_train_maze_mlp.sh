#!/bin/bash
#SBATCH --job-name=trm_mlp_maze
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=/home/skolawol/trm_mlp_maze.out
# The reverse ablation reviewers 9tYk and SSXB asked for: an MLP-mixing TRM
# on Maze-Hard, so the architecture claim rests on two tasks. Same recipe as
# the README's single-L40S Maze run with arch.mlp_t=True and
# pos_encodings=none, which is the MLP-mixing configuration the Sudoku
# checkpoint uses. Checkpoints land in $HOME (visible from the login node).
set -u
cd "$HOME/workspace/EdgeTRM/TinyRecursiveModels"
export WANDB_MODE=offline WANDB_DIR=$HOME/wandb_trm PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export HF_HOME=/data/hf_cache/skolawol
mkdir -p "$WANDB_DIR"
python3 -c "import torch,sys; ok=torch.cuda.is_available(); print('[preflight] cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" || exit 1

# Optimizer: build the CUDA extension if a compiler is available, else use
# the pure-torch fallback dropped next to pretrain.py.
python3 -c "import adam_atan2" 2>/dev/null || pip install --user -q adam-atan2 2>/dev/null || cp "$HOME/adam_atan2_fallback.py" ./adam_atan2.py
python3 -c "import adam_atan2; print('[opt]', adam_atan2.__file__)"

# Data: the README's Maze-Hard build (1k mazes, 8 augments) from the HF dataset.
if [ ! -d data/maze-30x30-hard-1k ]; then
  python3 dataset/build_maze_dataset.py || { echo "[data] build failed"; exit 1; }
fi
ls data/maze-30x30-hard-1k

run_name="pretrain_mlp_t_maze30x30_1gpu"
python3 pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 eval_interval=5000 checkpoint_every_eval=True \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True
echo "[train] finished rc=$? $(date -u)"
ls -la checkpoints/*/${run_name} 2>/dev/null
