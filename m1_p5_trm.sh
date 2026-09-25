#!/bin/bash
# M1 p5 autofill cell: one TRM Sudoku-attention (released) run of polish.py.
# Usage: bash m1_p5_trm.sh <offset 0|1000> <tag> <cells> <fp32_ext> [extra polish.py args...]
# T = 640; hybrids at T in {64,128,256,512} are extra schedules of the same run; per-step records for full curves.
set -u
OFF=$1; TAG=$2; CELLS=$3; FEXT=$4; shift 4
R=/opt/dlami/nvme/edgetrm
OUT=$R/results/m1/trm; mkdir -p $OUT/logs
PY=$R/venv/bin/python
export HOME=$R HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1          # breadth_sweep reads checkpoints from $HOME/edgetrm_ckpt
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; n=torch.cuda.get_device_name(0); assert 'H200' in n, n; print(n)" || { echo "PREFLIGHT FAILED"; exit 1; }
cd $R/EdgeTRM || exit 1
X="c56.P8;c48.P16;c120.P8;c112.P16;c96.P32;c248.P8;c240.P16;c224.P32;c192.P64;c504.P8;c496.P16;c480.P32;c448.P64"
$PY polish.py --task sudoku_attn --n 1000 --offset $OFF --T 640 --ks 64 --cells $CELLS --polish w8c --only_extra \
    --extra_sched "$X" --fp32_ext $FEXT --per_step --diag none --inner none "$@" --out $OUT/p5_attn_T640_${TAG}_o${OFF}.json
rc=$?
echo "$(date -u) ${TAG}_o$OFF rc=$rc" >> $OUT/logs/done.log
exit $rc
