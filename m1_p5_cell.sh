#!/bin/bash
# M1 p5 autofill cell: one Ouro-2.6B depth-for-precision config on one GSM8K question set.
# Usage: bash m1_p5_cell.sh <dev|held> <name> <steps> <quants> [batch=24] [n=500] [outdir-suffix]
# The autofill daemon sets CUDA_VISIBLE_DEVICES. Resumable (looped_decode.py skips finished batches).
# dev = GSM8K test[0:500] at batch 24, held-out = test[500:1000] at batch 64 (coordinator, Sep 25 11:00: every cell of
# one exhibit shares one batch size; held-out is its own comparison); greedy, 512 new tokens.
set -u
SET=$1; NAME=$2; STEPS=$3; QUANTS=$4; BATCH=${5:-24}; N=${6:-500}; SFX=${7:-}
R=/opt/dlami/nvme/edgetrm
OUT=$R/results/m1
mkdir -p $OUT/logs
PY=$R/venv/bin/python
export HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/pylibs/tf4554 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; n=torch.cuda.get_device_name(0); assert 'H200' in n, n; print(n)" || { echo "PREFLIGHT FAILED"; exit 1; }
OFF=$([ "$SET" = dev ] && echo 0 || echo 500)
cd $R/EdgeTRM || exit 1
M1_GSM_OFFSET=$OFF $PY m1_looped.py --model ouro26 --steps $STEPS --quants "$QUANTS" --decodes greedy --n $N --batch $BATCH \
    --outdir $OUT/ouro26_$SET$SFX --refdir $OUT/ref_${SET}${SFX}_$NAME
rc=$?
echo "$(date -u) ${SET}_$NAME rc=$rc" >> $OUT/logs/done.log
exit $rc
