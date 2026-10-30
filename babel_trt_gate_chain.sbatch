#!/bin/bash
#SBATCH --job-name=trt_gate_chain
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --constraint=L40S
#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/skolawol/trt_gate/logs/%x_%j.out
# TensorRT engines of E1 under the label-free gate (trt_gate_chain.py). Records -> $HOME/trt_gate/results/trt_gate.
set -u
source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache
export HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
python -c "import torch; assert torch.cuda.is_available(), 'NO CUDA'; print(torch.cuda.get_device_name(0))" || { echo "PREFLIGHT FAILED: no CUDA"; exit 1; }
D=/data/user_data/$USER
PYL=$D/e1/pylibs
[ -e $PYL/trt/tensorrt/__init__.py ] || { echo "TensorRT missing under $PYL"; exit 1; }
ORT=$D/trt_gate/pylibs/ort
[ -e $ORT/onnxruntime/__init__.py ] || pip install -q --no-deps --target $ORT onnxruntime==1.22.0 || { echo "ORT install failed"; exit 1; }
export PYTHONPATH=$PYL/trt:$ORT
export TG_WORK=$D/trt_gate/work TG_RES=$HOME/trt_gate/results/trt_gate
mkdir -p $TG_RES
python -c "import tensorrt, onnxruntime, numpy; print('trt', tensorrt.__version__, 'ort', onnxruntime.__version__, 'numpy', numpy.__version__)"
nvidia-smi --query-gpu=name,driver_version,clocks.sm,clocks.max.sm,power.limit --format=csv,noheader
echo "$(date -u) start on $(hostname)"
for t in sudoku_mlp sudoku_attn maze hrm_sudoku; do
  H=1; [ $t = hrm_sudoku ] && H=2
  mkdir -p $TG_WORK/$t
  cp -n $D/e1/work/$t/calib_dyn_H$H.cache $TG_WORK/$t/ 2>/dev/null || echo "$t: no E1 calibration cache"
  miss=""
  for p in fp16 int8mm int8proj; do
    e=dyn_H${H}_b1_$p
    cp -n $D/e1/work/$t/$e.engine $TG_WORK/$t/ 2>/dev/null
    [ -e $TG_WORK/$t/$e.engine ] && echo "$t $e reused from E1 $(md5sum < $TG_WORK/$t/$e.engine | cut -c1-12)" || miss="$miss,$e"
  done
  if [ -n "$miss" ]; then
    echo "$t: building ${miss#,} (E1 engines missing)"
    E1_WORK=$TG_WORK E1_RES=$TG_WORK/e1res E1_ONNX=$HOME/e1/onnx E1_X3ART=$HOME/e1/x3 \
      python $HOME/e1/code/e1_energy.py --stage build --task $t --only ${miss#,} || echo "BUILD FAILED $t"
  fi
done
cd $HOME/trt_gate/code || exit 1
for t in ${TG_TASKS:-sudoku_mlp sudoku_attn maze hrm_sudoku}; do python trt_gate_chain.py run --task $t || echo "RUN FAILED $t"; done
python trt_gate_chain.py score --out $TG_RES/trt_gate.json
echo "$(date -u) done" && touch $TG_RES/DONE
