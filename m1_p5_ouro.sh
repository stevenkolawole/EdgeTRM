#!/bin/bash
# M1 on p5 (card 4): Ouro-2.6B depth-for-precision grid, dev (GSM8K test[0:500]) and held-out (test[500:1000]),
# with its own bf16 / INT8 / w4c references on this platform (H200, transformers 4.55.4 overlay).
# Resumable: looped_decode.py skips finished batches; a job whose summary exists is skipped by the harness.
# Usage: bash m1_p5_ouro.sh [PARALLEL]      (default 7 concurrent processes on the card)
set -u
R=/opt/dlami/nvme/edgetrm
OUT=$R/results/m1
mkdir -p $OUT/logs
PY=$R/venv/bin/python
export CUDA_VISIBLE_DEVICES=4 HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/pylibs/tf4554
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -c "import torch, transformers; assert torch.cuda.is_available(), 'NO CUDA'; n=torch.cuda.get_device_name(0); assert 'H200' in n, n; print(n, transformers.__version__)" || { echo "PREFLIGHT FAILED"; exit 1; }
cd $R/EdgeTRM || exit 1
# name | steps | quants  (loop indices of the INT8 loops after ~P; traffic per loop: bf16 16, INT8 8, w4c 4 bits/weight)
CFG=$(cat <<'EOF'
bf16_s4|4|bf16
bf16_s23|2,3|bf16
w8_s4|4|w8c
w8_s23|2,3|w8c
w8_s56|5,6|w8c
w4_s4|4|w4c
w4_s56|5,6|w4c
sw_P3|4|w4c~P3@w8c
sw_P0|4|w4c~P0@w8c
sw_P23|4|w4c~P23@w8c
sw_P03|4|w4c~P03@w8c
sw_P01_P123|4|w4c~P01@w8c,w4c~P123@w8c
deep6|6|w4c~P45@w8c,w4c~P5@w8c
deep8|8|w4c,w4c~P67@w8c
EOF
)
JOBS=$OUT/jobs.txt
: > $JOBS
for SET in dev held; do
  OFF=$([ $SET = dev ] && echo 0 || echo 500)
  while IFS='|' read -r name steps quants; do
    [ -z "$name" ] && continue
    echo "M1_GSM_OFFSET=$OFF $PY m1_looped.py --model ouro26 --steps $steps --quants $quants --decodes greedy --n 500 --batch 24 --outdir $OUT/ouro26_$SET --refdir $OUT/ref_${SET}_$name > $OUT/logs/${SET}_$name.log 2>&1; echo \"\$(date -u) ${SET}_$name rc=\$?\" >> $OUT/logs/done.log" >> $JOBS
  done <<< "$CFG"
done
echo "$(date -u) launching $(wc -l < $JOBS) jobs, parallel ${1:-7}" >> $OUT/logs/done.log
tr '\n' '\0' < $JOBS | xargs -0 -P ${1:-7} -I{} bash -c '{}'
echo "$(date -u) all jobs finished" >> $OUT/logs/done.log
