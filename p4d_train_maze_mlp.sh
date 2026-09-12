#!/bin/bash
# Second attempt at the reverse ablation on p4d (free): MLP-mixing TRM on
# Maze-Hard with the README's single-GPU recipe (batch 128), EMA on,
# resume.pt (with ema_state) every 5,000 optimizer steps so the EMA copy can
# be evaluated while training runs. The batch-384 H100 run did not converge
# (halting head never fired, 0% exact) and its EMA copy was lost.
#   CUDA_VISIBLE_DEVICES=7 nohup bash p4d_train_maze_mlp.sh > ~/p4d_train_maze_mlp.log 2>&1 &
set -u
cd "$HOME/edgetrm_src/TinyRecursiveModels"
export WANDB_MODE=offline WANDB_DIR=$HOME/wandb_trm PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export HF_HOME=$HOME/hf_cache
PY=$HOME/venvs/kvcache/bin/python
mkdir -p "$WANDB_DIR"
$PY -c "import torch,sys; ok=torch.cuda.is_available(); print('[preflight] cuda', ok, torch.cuda.get_device_name(0) if ok else ''); sys.exit(0 if ok else 1)" || exit 1
$PY -c "import adam_atan2" 2>/dev/null || cp "$HOME/adam_atan2_fallback.py" ./adam_atan2.py
$PY -c "import adam_atan2; print('[opt]', adam_atan2.__file__)"
if [ ! -f data/maze-30x30-hard-1k/train/all__inputs.npy ]; then
  rm -rf data/maze-30x30-hard-1k
  $PY dataset/build_maze_dataset.py || { echo "[data] build failed"; exit 1; }
fi
ls data/maze-30x30-hard-1k data/maze-30x30-hard-1k/train
run_name="pretrain_mlp_t_maze30x30_p4d_b128"
$PY pretrain.py \
  arch=trm \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=25000 eval_interval=2500 checkpoint_every_eval=True +checkpoint_interval=5000 \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=128 \
  arch.mlp_t=True arch.pos_encodings=none \
  arch.L_layers=2 \
  arch.H_cycles=3 arch.L_cycles=4 \
  +run_name=${run_name} ema=True
echo "[train] finished rc=$? $(date -u)"
ls -la checkpoints/*/${run_name} 2>/dev/null
