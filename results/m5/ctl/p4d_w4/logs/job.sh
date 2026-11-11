#!/bin/bash
# usage: job.sh TAG script.py args...   (card 7 only; log per tag; DONE marker with rc)
TAG=$1; shift
export PYTHONPATH=/home/ubuntu/m5w8c/pylib CUDA_VISIBLE_DEVICES=7 HOME=/home/ubuntu/m5w8c/home HF_HOME=/home/ubuntu/m5w8c/hf HF_HUB_OFFLINE=1
cd /home/ubuntu/m5w8c/src
/opt/dlami/nvme/kvcache/venv/bin/python "$@" > /home/ubuntu/m5w8c/logs/w4_$TAG.log 2>&1
echo "rc=$? $TAG" >> /home/ubuntu/m5w8c/logs/W4DONE
