# Exit vs fixed depth on the full puzzle set: Sudoku-MLP and Sudoku-attn, fixed 16, best fixed depth, head exit, answer-stable exit.
# Usage: bash run_fullset_nano.sh <nvpmodel mode: 0 = 10 W, 1 = 5 W> <window seconds>
cd ~/edge; MODE=$1; W=$2; H=$(hostname); TAG=$([ "$MODE" = 0 ] && echo maxn || echo 5w)
until ! pgrep -f "edge_loo[p]" >/dev/null; do sleep 30; done
D=/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0; sudo chmod a+r $D/in_power*_input $D/in_voltage*_input $D/in_current*_input
cat $D/in_power0_input >/dev/null || { echo POWER_UNREADABLE; exit 1; }
cool() { n=0; until [ $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) -le 65000 ] || [ $n -ge 30 ]; do sleep 10; n=$((n+1)); done; }
sudo nvpmodel -m $MODE >/dev/null 2>&1; sudo jetson_clocks
for row in "sudoku_mlp:fixed:16:" "sudoku_mlp:fixed:13:" "sudoku_mlp:exit:16:" "sudoku_mlp:exit:16:stable1" "sudoku_attn:fixed:16:" "sudoku_attn:fixed:12:" "sudoku_attn:exit:16:" "sudoku_attn:exit:16:stable1"; do
  IFS=: read t m d r <<< "$row"; R=""; [ -n "$r" ] && R="--rule $r"; cool
  O=results/${H}_${t}_fp32_${m}${d}${r:+_$r}_${TAG}_fullset
  python3 edge_loop_new.py --backend trt --task $t --model engines/$t/stepc_H3_op13_fold_fp32.plan --init assets/$t/init.npz --puzzles assets/$t/puzzles_0_100.npz --board $H --precision fp32 --mode $m --depth $d $R --no_check --nidle 1 --nwin 2 --win_s $W --note "$TAG full-set windows ($W s)" --out $O.json > $O.log 2>&1
  echo "$(date +%T) $t $m$d $r rc=$?"; done
echo FULLSET_DONE
