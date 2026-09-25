#!/bin/bash
# M1 p5 autofill cell: MDEQ (locuslab MDEQ-Vision, X6's deq_sweep harness) cheap-then-precise Broyden, one image half.
# Usage: bash m1_p5_deq.sh <small|xl> <dev|held>      dev = images 0-2499, held-out = 2500-4999 (X6's 5-per-class cache)
set -u
MODEL=$1; SET=$2
R=/opt/dlami/nvme/edgetrm; X=$R/m1_x6
OUT=$R/results/m1/deq; mkdir -p $OUT/logs
PY=$R/venv/bin/python
export PYTHONPATH=$X/pylib DEQ_ROOT=$X/deq X6_CKPT=$X/ckpt X6_CACHE=$OUT/cache HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; n=torch.cuda.get_device_name(0); assert 'H200' in n, n; print(n)" || { echo "PREFLIGHT FAILED"; exit 1; }
LO=$([ "$SET" = dev ] && echo 0 || echo 2500); HI=$([ "$SET" = dev ] && echo 2500 || echo 5000)
cd $R/EdgeTRM || exit 1
$PY m1_deq.py --model $MODEL --cheap w4a,w4g32,w4c,n8c4 --polish fp32,w8c \
    --curve 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,26,32,40 --tc 4,5,6,7,8,10,12,16,26 --kp 1,2,3,4,5,6,8,12,16 \
    --lo $LO --hi $HI --out $OUT/p5_deq_${MODEL}_${SET}.json
rc=$?
echo "$(date -u) deq_${MODEL}_${SET} rc=$rc" >> $OUT/logs/done.log
exit $rc
