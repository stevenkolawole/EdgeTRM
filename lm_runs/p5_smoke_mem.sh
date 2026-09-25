#!/bin/bash
# Peak-memory smoke for p5 batch sizes: one full batch of w4t (runs to the 512-token cap) per model on card 7.
R=/opt/dlami/nvme/edgetrm; O=$R/results/X4/smoke_mem; mkdir -p $O
cd $R/EdgeTRM
export HOME=$R HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$R/pylibs/tf4554
for spec in "olmo_rec32 32 64" "ouro26 4 64" "huginn 32 32" "ouro14 4 128"; do
  set -- $spec
  $R/venv/bin/python looped_decode.py --model $1 --steps $2 --quants w4t --decodes greedy --n $3 --batch $3 --outdir $O/b$3 > $O/$1_b$3.log 2>&1
  echo "$1 batch $3 rc=$? $(grep -h peak_mem $O/$1_b$3.log)" >> $O/summary.txt
done
