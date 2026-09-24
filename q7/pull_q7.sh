#!/bin/bash
# Q7 (X5): pull evaluation records (no weights) from p4d and Babel into EdgeTRM/results/q7/{p4d,babel}/<run>/
set -u
D=$(cd "$(dirname "$0")/.." && pwd)/results/q7
mkdir -p "$D/p4d" "$D/babel"
INC=(--include='*/' --include='eval_step_*.json' --include='metrics.jsonl' --include='all_config.yaml' --include='DONE' --exclude='*')
rsync -a "${INC[@]}" sk-p4d:q7/ckpt/ "$D/p4d/"
rsync -a sk-p4d:q7/logs/ "$D/p4d/_logs/" --include='*.log' --exclude='*' 2>/dev/null
rsync -a "${INC[@]}" babel:q7/ckpt/ "$D/babel/" 2>/dev/null
rsync -a babel:q7/logs/ "$D/babel/_logs/" --include='*.out' --exclude='*' 2>/dev/null
rsync -a sk-p4d:q7/lambda_eval/ "$D/lambda_finals/" --include='*.json' --exclude='*' 2>/dev/null
echo "pulled into $D: $(find "$D" -name 'eval_step_*.json' | wc -l) eval files"
