# AGX INT8 chain (MAXN, jetson_clocks): calibrated INT8+FP16 engines (build_int8_engine.py), check passes vs the
# host fp32 reference, energy rows for every engine whose step-16 count is at least half the fp16 engine's, then
# the INT8+FP32-fallback variant (built from the same calibration cache) with its check pass. Waits for the energy
# chain (ENERGY_DONE, no edge_loop.py running) and writes INT8_DONE on exit, whatever happens.
#   cd ~/edge && nohup bash run_int8.sh >> results/int8_all.log 2>&1 &
cd ~/edge
trap 'echo INT8_DONE' EXIT
until grep -q ENERGY_DONE results/energy_all.log 2>/dev/null && ! pgrep -f "edge_loo[p].py" >/dev/null; do sleep 300; done
echo "start $(date)"; sudo jetson_clocks
D=/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon4; E=/sys/bus/i2c/drivers/ina3221/1-0041/hwmon/hwmon5
PF="$D/in1_input*$D/curr1_input,$D/in2_input*$D/curr2_input,$D/in3_input*$D/curr3_input,$E/in1_input*$E/curr1_input,$E/in2_input*$E/curr2_input,$E/in3_input*$E/curr3_input,sum"
RN="GPU,CPU,SOC,CV,VDDRQ,SYS5V,TOTAL"
TASKS="hrm_sudoku sudoku_mlp sudoku_attn maze"
graph() { ls assets/$1/stepc_H[23]_op13_fold.onnx | head -1; }
plan() { g=$(basename $(graph $1) .onnx); echo engines/$1/${g}_$2.plan; }
build() { # build <task> <variant>
  t=$1; v=$2; o=$(plan $t $v); [ -f $o ] && { echo "$o exists"; return; }
  python3 build_int8_engine.py --onnx $(graph $t) --calib assets/$t/calib_carries.npz --variant $v \
    --cache engines/$t/$(basename $(graph $t) .onnx)_int8.calib.cache --out $o > $o.stdout 2>&1
  echo "build $t $v rc=$? $(tail -1 $o.stdout)"
}
check() { # check <task> <variant>
  t=$1; v=$2; o=results/agx_${t}_trt_${v}_gate.json; [ -f $o ] && { echo "$o exists"; return; }
  python3 edge_loop.py --backend trt --task $t --model $(plan $t $v) --init assets/$t/init.npz \
    --puzzles $(ls assets/$t/puzzles_*.npz | head -1) --ref assets/$t/host_fp32.npz --board agx --precision $v \
    --nwin 0 --no_power --note "gate: MAXN, jetson_clocks on, folded op13 graph, TRT 8.5 Volta, $v entropy-calibrated on 128 real carries" \
    --out $o > results/agx_${t}_trt_${v}_gate.log 2>&1
  echo "check $t $v rc=$?"
}
sane() { # sane <task>: int8 step-16 count >= half of fp16's
  python3 -c "
import json,sys
c=lambda p: json.load(open(p))['check']['exact_count_per_step'][-1]
a,b=c('results/agx_$1_trt_int8_gate.json'),c('results/agx_$1_trt_fp16_gate.json')
print('sane $1 int8 %d fp16 %d' % (a,b), file=sys.stderr); sys.exit(0 if 2*a>=b else 1)"
}
run() { # as run_energy_trim.sh: run <task> <engine-suffix> <precision-label> <mode> <depth> <win_s> <tag> [extra...]
  t=$1; e=$2; p=$3; m=$4; d=$5; w=$6; tag=$7; shift 7
  [ -f results/agx_${t}_${p}_${m}${d}_${tag%% *}.json ] && { echo "$t $p $m $d ${tag%% *} exists"; return; }
  P=$(ls assets/$t/puzzles_*.npz | head -1); M=$(plan $t $e)
  python3 edge_loop.py --backend trt --task $t --model $M --init assets/$t/init.npz --puzzles $P --board agx \
    --precision $p --mode $m --depth $d --no_check --nidle 1 --nwin 5 --win_s $w --power_dir / --power_files "$PF" \
    --rail_names "$RN" --total_rail 6 --note "$tag" "$@" \
    --out results/agx_${t}_${p}_${m}${d}_${tag%% *}.json > results/agx_${t}_${p}_${m}${d}_${tag%% *}.log 2>&1
  echo "$t $p $m $d ${tag%% *} rc=$?"
}
for t in $TASKS; do build $t int8; done
for t in $TASKS; do check $t int8; done
for t in sudoku_mlp sudoku_attn hrm_sudoku; do sane $t || { echo "skip energy $t (collapsed)"; continue; }
  for m in exit fixed; do run $t int8 int8 $m 16 60 "maxn nvpmodel 0 MAXN, jetson_clocks on, INT8+FP16 calibrated"; done
  [ $t = hrm_sudoku ] && continue
  run $t int8 int8 exit 16 60 "maxnfin2fp16 nvpmodel 0 MAXN, INT8 exit + 2 fp16 finishing steps" \
    --finish engines/$t/stepc_H3_op13_fold_fp16.plan --finish_steps 2 --finish_precision fp16
done
if sane maze; then for m in exit fixed; do run maze int8 int8 $m 8 120 "maxn nvpmodel 0 MAXN, jetson_clocks on, INT8+FP16 calibrated"; done
else echo "skip energy maze (collapsed)"; fi
for t in $TASKS; do build $t int8f32; check $t int8f32; done
echo "end $(date)"
