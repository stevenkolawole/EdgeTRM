#!/bin/bash
# Q7 (X5): one TRM-MLP Sudoku w256 training run, resumable. Recipe = the Lambda recipe that produced
# checkpoints_trained/trm_sudoku_mlp_w256_s* (reconstructed from their all_config.yaml): global batch 768,
# lr 1e-4 (warmup 2000, constant), wd 1.0, puzzle-emb lr 1e-4 wd 1.0, 50k epochs, eval/ckpt every 5k epochs
# (6,510 steps), EMA 0.999, H3 L6 L_layers 2, hidden 256, 4 heads, mlp_t, no pos enc, bf16, AdamW fallback.
# Differences: pretrain_q7.py (+ regularizer hooks, in-run eval on 3 test batches instead of the full
# 422k test set, inline q7_eval after every checkpoint), and the 1k training subset (seeded rebuild).
#   usage: q7_train.sh RUN SEED [extra hydra overrides, "+q7_*=..."]
#   env:   Q7_ROOT (ckpt/, logs/, wandb/), Q7_DATA (dataset dir), Q7_PY (python), Q7_SRC (TinyRecursiveModels)
#          Q7_EPOCHS / Q7_EVAL_INTERVAL (smoke tests only)
set -u
RUN=$1; SEED=$2; shift 2
CK=$Q7_ROOT/ckpt/$RUN
if [ -f "$CK/DONE" ]; then echo "$(date -u) $RUN already DONE"; exit 0; fi
mkdir -p "$CK" "${WANDB_DIR_OVERRIDE:-$Q7_ROOT/wandb}"
$Q7_PY -c "import torch,sys; ok=torch.cuda.is_available(); print('[preflight]', 'cuda', ok, torch.cuda.get_device_name(0) if ok else 'NO CUDA', torch.__version__); sys.exit(0 if ok else 1)" || { echo "[preflight] no CUDA; abort"; exit 3; }
cd "$Q7_SRC" || exit 4
export WANDB_MODE=offline WANDB_DIR=${WANDB_DIR_OVERRIDE:-$Q7_ROOT/wandb} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
echo "$(date -u) [q7_train] $RUN seed=$SEED host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-} extra: $*"
# recipe keys may be replaced from the command line (epochs=, eval_interval=, lr_warmup_steps=) without
# duplicating them for Hydra
EPOCHS=${Q7_EPOCHS:-50000}; EVAL_INTERVAL=${Q7_EVAL_INTERVAL:-5000}; WARMUP=2000; EXTRA=()
for a in "$@"; do
  case "$a" in
    epochs=*) EPOCHS=${a#epochs=} ;;
    eval_interval=*) EVAL_INTERVAL=${a#eval_interval=} ;;
    lr_warmup_steps=*) WARMUP=${a#lr_warmup_steps=} ;;
    *) EXTRA+=("$a") ;;
  esac
done
$Q7_PY pretrain_q7.py arch=trm data_paths="[$Q7_DATA]" evaluators="[]" \
  epochs=$EPOCHS eval_interval=$EVAL_INTERVAL lr_warmup_steps=$WARMUP checkpoint_every_eval=True \
  lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 global_batch_size=768 \
  arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
  arch.hidden_size=256 arch.num_heads=4 ema=True seed=$SEED \
  +run_name=$RUN +checkpoint_path=$CK "${EXTRA[@]}"
rc=$?
echo "$(date -u) [q7_train] $RUN exited rc=$rc DONE=$([ -f "$CK/DONE" ] && echo yes || echo no)"
exit $rc
