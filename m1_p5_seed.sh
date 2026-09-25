#!/bin/bash
# M1 p5 autofill cell: frozen seed test on an attention-512 seed (X5 var_attn512_s<N>), puzzles 0-999.
# Schedule w3c 224 + INT8-c 32 (T=256, frozen on the released model); comparator curves to 320 steps for
# {fp32 (to 512), w8c, w3c, n8c3, w3g32, w4g32, w4a, w4c, n8a4}; one exact McNemar per seed (Holm over s3-s5).
set -u
S=$1
R=/opt/dlami/nvme/edgetrm
OUT=$R/results/m1/trm; mkdir -p $OUT/logs
PY=$R/venv/bin/python
export HOME=$R HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1
$PY -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; n=torch.cuda.get_device_name(0); assert 'H200' in n, n; print(n)" || { echo "PREFLIGHT FAILED"; exit 1; }
cd $R/EdgeTRM || exit 1
$PY polish.py --task sudoku_attn --ckpt $R/edgetrm_ckpt/variants/var_attn512_s$S/step_65100 --n 1000 --T 320 \
    --cells w3c,w8c,n8c3,w3g32,w4g32,w4a,w4c,n8a4 --polish w8c --only_extra --extra_sched c224.P32 --ks 1 \
    --fp32_ext 512 --per_step --diag none --inner none --out $OUT/p5_seed${S}_T320_seedtest.json
rc=$?
echo "$(date -u) seed${S}_test rc=$rc" >> $OUT/logs/done.log
exit $rc
