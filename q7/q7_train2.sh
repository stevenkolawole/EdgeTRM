#!/bin/bash
# Q7/M6/variants (X5): one TRM training run, resumable (pretrain_q7.py, fixed resume). Generalised copy of
# q7_train.sh (which must not be edited while runs use it): every recipe key has a default (the Lambda
# Sudoku-MLP w256 recipe) and any "key=value" argument whose key is in the table REPLACES it (no duplicate
# Hydra keys); other arguments ("+q7_*=...") pass through.
#   usage: q7_train2.sh RUN SEED [key=value ...] [+extra=...]
#   env:   Q7_ROOT, Q7_DATA (default dataset), Q7_PY, Q7_SRC; WANDB_DIR_OVERRIDE
set -u
RUN=$1; SEED=$2; shift 2
CK=$Q7_ROOT/ckpt/$RUN
if [ -f "$CK/DONE" ]; then echo "$(date -u) $RUN already DONE"; exit 0; fi
mkdir -p "$CK" "${WANDB_DIR_OVERRIDE:-$Q7_ROOT/wandb}"
$Q7_PY -c "import torch,sys; ok=torch.cuda.is_available(); print('[preflight]', 'cuda', ok, torch.cuda.get_device_name(0) if ok else 'NO CUDA', torch.__version__); sys.exit(0 if ok else 1)" || { echo "[preflight] no CUDA; abort"; exit 3; }
cd "$Q7_SRC" || exit 4
export WANDB_MODE=offline WANDB_DIR=${WANDB_DIR_OVERRIDE:-$Q7_ROOT/wandb} PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HUB_OFFLINE=1
declare -A R=(
  [data_paths]="[$Q7_DATA]" [epochs]=50000 [eval_interval]=5000 [lr_warmup_steps]=2000
  [lr]=1e-4 [puzzle_emb_lr]=1e-4 [weight_decay]=1.0 [puzzle_emb_weight_decay]=1.0 [global_batch_size]=768
  [arch.mlp_t]=True [arch.pos_encodings]=none [arch.L_layers]=2 [arch.H_cycles]=3 [arch.L_cycles]=6
  [arch.hidden_size]=256 [arch.num_heads]=4 [ema]=True
)
ORDER="data_paths epochs eval_interval lr_warmup_steps lr puzzle_emb_lr weight_decay puzzle_emb_weight_decay global_batch_size arch.mlp_t arch.pos_encodings arch.L_layers arch.H_cycles arch.L_cycles arch.hidden_size arch.num_heads ema"
EXTRA=()
for a in "$@"; do
  k=${a%%=*}
  if [ -n "${R[$k]+x}" ]; then R[$k]=${a#*=}; else EXTRA+=("$a"); fi
done
ARGS=(); for k in $ORDER; do ARGS+=("$k=${R[$k]}"); done
echo "$(date -u) [q7_train2] $RUN seed=$SEED host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-} recipe: ${ARGS[*]} extra: ${EXTRA[*]}"
$Q7_PY pretrain_q7.py arch=trm evaluators="[]" checkpoint_every_eval=True "${ARGS[@]}" seed=$SEED \
  +run_name=$RUN +checkpoint_path=$CK "${EXTRA[@]}"
rc=$?
echo "$(date -u) [q7_train] $RUN exited rc=$rc DONE=$([ -f "$CK/DONE" ] && echo yes || echo no)"
exit $rc
