#!/bin/bash
# Push (Eq. 2) for the model-format pairs the law-selects-format table lacks: one model per card, n=256, T=16.
export PYTHONPATH=/home/ubuntu/m5w8c/pylib HOME=/home/ubuntu/m5w8c/home HF_HOME=/home/ubuntu/m5w8c/hf HF_HUB_OFFLINE=1 M5_GPTQ_DIR=/home/ubuntu/m5w8c/gptq
cd /home/ubuntu/m5w8c/src; P=/opt/dlami/nvme/kvcache/venv/bin/python; O=/home/ubuntu/m5w8c/push_missing; mkdir -p $O
W=/home/ubuntu/m5w8c/seed_ckpt
run(){ CUDA_VISIBLE_DEVICES=$1 $P push_vs_weight_distance.py --task $2 --specs $3 --n 256 --T 16 ${5:+--ckpt $5} --out $O/$4.json > $O/$4.log 2>&1; echo "rc=$? $4" >> $O/DONE; }
c=3; for m in attn mlp; do for s in 0 1 2; do
  [ $c -gt 7 ] && c=0; run $c sudoku_$m w3c,w3g32,w4c,w4g32,w6c,w8c ${m}512_s$s $W/trm_sudoku_${m}_w512_s$s/step_65100 & c=$((c+1)); done; done
wait
