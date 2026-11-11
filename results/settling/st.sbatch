#!/bin/bash
#SBATCH --job-name=settling
#SBATCH --partition=general
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/skolawol/settling/logs/%x_%j.out
# usage: sbatch po.sbatch TASK CELL TAG   (writes $HOME/settling/out/TAG.json; marker TAG.done)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA on this node'; print(torch.cuda.get_device_name(0), torch.__version__)" || { echo "NOCUDA" > $HOME/settling/out/$3.failed; exit 1; }
cd ~/precision_order/src || exit 1
export HOME_CKPT=$HOME
python polish.py --task $1 --cells $2 --polish w8c --n 1000 --offset 0 --ks 8 --fp32_ext 264 --only_extra --extra_sched "c16;c24;P8.c16;c16.P8;c32;c40;P8.c32;c32.P8;c64;c72;P8.c64;c64.P8;c128;c136;P8.c128;c128.P8;c256;c264;P8.c256;c256.P8" --out $HOME/settling/out/$3.json && echo rc=0 > $HOME/settling/out/$3.done || echo rc=$? > $HOME/settling/out/$3.failed
