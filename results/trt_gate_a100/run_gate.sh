#!/bin/bash
# G1 on the A100: build E1 batch-1 engines from E1 calibration caches, then trt_gate_chain.py run per task.
set -u
R=/opt/dlami/nvme/trt_gate
export HOME=$R/home HF_HOME=$R/home/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0
PY=$R/venv/bin/python
$PY -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))" || { echo PREFLIGHT FAILED; touch $R/FAILED; exit 1; }
nvidia-smi -i 0 --query-gpu=name,uuid,driver_version,clocks.max.sm,power.limit --format=csv,noheader
export TG_WORK=$R/work TG_RES=$HOME/trt_gate/results/trt_gate_a100 TG_E1CODE=$HOME/e1/code TG_X3=$HOME/e1/x3 TG_ONNX=$HOME/e1/onnx
mkdir -p $TG_RES
for t in sudoku_mlp sudoku_attn maze hrm_sudoku; do
  H=1; [ $t = hrm_sudoku ] && H=2
  mkdir -p $TG_WORK/$t; cp -n $HOME/calib/$t/calib_dyn_H$H.cache $TG_WORK/$t/
  (cd $HOME/e1/code && E1_WORK=$TG_WORK E1_RES=$TG_WORK/e1res E1_ONNX=$HOME/e1/onnx E1_X3ART=$HOME/e1/x3 \
    $PY e1_energy.py --stage build --task $t --only dyn_H${H}_b1_fp16,dyn_H${H}_b1_int8mm,dyn_H${H}_b1_int8proj) || echo "BUILD FAILED $t"
done
cd $HOME/trt_gate/code
for t in sudoku_mlp sudoku_attn maze hrm_sudoku; do $PY trt_gate_chain.py run --task $t || echo "RUN FAILED $t"; done
$PY trt_gate_chain.py score --out $TG_RES/trt_gate.json
cp $TG_WORK/*/e1res/*/engines.json $TG_RES/ 2>/dev/null
for t in sudoku_mlp sudoku_attn maze hrm_sudoku; do cp $TG_WORK/$t/../e1res/$t/engines.json $TG_RES/${t}_engines.json 2>/dev/null; done
touch $R/DONE
