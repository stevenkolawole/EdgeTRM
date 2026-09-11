#!/bin/bash
#SBATCH --job-name=maze_ema
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=0-06:00:00
#SBATCH --output=/home/skolawol/maze_ema_rebuild.out
#SBATCH --exclude=babel-o9-24,babel-o5-16
# Rebuild the EMA weights of the H100-trained MLP-mixing Maze TRM. The H100
# run reached 95.5% exact accuracy with its EMA copy (TRM evaluates the EMA),
# but only the raw step_130200 weights were saved before the box was
# terminated, and the raw weights under a constant 1e-4 learning rate score
# 0% exact. Continue training from step_130200 at the same learning rate with
# EMA registered from the loaded weights, for ~4.7k steps (EMA horizon 1k),
# then evaluate: pretrain.py evaluates the EMA copy and saves it in resume.pt
# ("ema_state"). Extract that state dict afterwards.
set -u
cd "$HOME/workspace/EdgeTRM/TinyRecursiveModels"
export WANDB_MODE=offline WANDB_DIR=$HOME/wandb_trm PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export HF_HOME=/data/hf_cache/skolawol
mkdir -p "$WANDB_DIR"
python3 -c "import torch,sys; ok=torch.cuda.is_available(); print('[preflight] cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" || exit 1
python3 -c "import adam_atan2" 2>/dev/null || cp "$HOME/adam_atan2_fallback.py" ./adam_atan2.py
python3 -c "import adam_atan2; print('[opt]', adam_atan2.__file__)"
# The 09-10 fallback run died with a FileNotFoundError in a DataLoader
# worker: verify the train split is complete, rebuild otherwise.
if [ ! -f data/maze-30x30-hard-1k/train/all__inputs.npy ] || [ ! -f data/maze-30x30-hard-1k/train/dataset.json ]; then
  echo "[data] train split incomplete, rebuilding"; rm -rf data/maze-30x30-hard-1k
  python3 dataset/build_maze_dataset.py || { echo "[data] build failed"; exit 1; }
fi
ls data/maze-30x30-hard-1k data/maze-30x30-hard-1k/train

run_name="pretrain_mlp_t_maze30x30_ema_rebuild"
python3 pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  +load_checkpoint="$HOME/workspace/EdgeTRM/maze_mlp_h100/step_130200" \
  epochs=600 eval_interval=600 checkpoint_every_eval=True +checkpoint_interval=100 \
  lr=1e-4 lr_warmup_steps=0 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True
echo "[train] finished rc=$? $(date -u)"
ls -la checkpoints/*/${run_name} 2>/dev/null
