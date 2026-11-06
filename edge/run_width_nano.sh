# J per step against model width on a Nano: Sudoku-MLP at widths 256/768/512, fixed depth 16, fp32, MAXN.
cd ~/edge
until ! pgrep -f "edge_loo[p]" >/dev/null; do sleep 20; done
D=/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0; sudo chmod a+r $D/in_power*_input $D/in_voltage*_input $D/in_current*_input
cat $D/in_power0_input >/dev/null || { echo POWER_UNREADABLE; exit 1; }
cool() { n=0; until [ $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) -le 65000 ] || [ $n -ge 30 ]; do sleep 10; n=$((n+1)); done; }
sudo nvpmodel -m 0 >/dev/null 2>&1; sudo jetson_clocks; H=$(hostname)
for w in 256 768 512; do t=sudoku_mlp_w$w
  mkdir -p engines/$t; /usr/src/tensorrt/bin/trtexec --onnx=assets/$t/stepc_H3_op13_fold.onnx --saveEngine=engines/$t/stepc_H3_op13_fold_fp32.plan > results/${H}_${t}_build.log 2>&1; echo "$(date +%T) build $t rc=$?"
  cool; python3 edge_loop_new.py --backend trt --task sudoku_mlp --model engines/$t/stepc_H3_op13_fold_fp32.plan --init assets/$t/init.npz --puzzles assets/$t/puzzles_0_100.npz --board $H --precision fp32 --mode fixed --depth 16 --no_check --nidle 1 --nwin 3 --win_s 60 --note "width $w fixed16 maxn" --out results/${H}_${t}_fp32_fixed16_maxn.json > results/${H}_${t}_fp32_fixed16_maxn.log 2>&1
  echo "$(date +%T) $t fixed16 rc=$?"; done
echo WIDTH_DONE
