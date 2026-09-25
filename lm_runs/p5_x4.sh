#!/bin/bash
# EdgeTRM X4 on p5 (H200), card 5.  Waits for /opt/dlami/nvme/edgetrm/READY, preflights CUDA, then runs every
# line of p5_x4_jobs.txt as its own background process on card 5 (looped_decode.py is resumable; rerunning
# this script skips finished batches).  HOME is pointed at the work root so that looped_decode's
# ~/hf_cache (GSM8K) and ~/hf_models (huginn, olmo_rec32) resolve to the p5 copies.
R=/opt/dlami/nvme/edgetrm
until [ -e $R/READY ]; do sleep 60; done
OUT=$R/results/X4; mkdir -p $OUT/logs
cd $R/EdgeTRM || exit 1
export HOME=$R HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=5 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$R/pylibs/tf4554
$R/venv/bin/python -c "import torch, transformers; assert torch.cuda.is_available(), 'NO CUDA'; print(torch.cuda.get_device_name(0), transformers.__version__)" > $OUT/logs/preflight.log 2>&1 || { echo PREFLIGHT FAILED; exit 1; }
i=0
while read -r line; do
  [ -z "$line" ] && continue; case "$line" in \#*) continue;; esac
  i=$((i+1)); tag=$(echo "$line" | awk '{print $2}')_$i
  setsid nohup $R/venv/bin/python looped_decode.py $line >> $OUT/logs/$tag.log 2>&1 < /dev/null &
  echo "$(date -u) pid $! $tag: $line" >> $OUT/logs/launch.log
  sleep 20
done < $R/EdgeTRM/lm_runs/p5_x4_jobs.txt
