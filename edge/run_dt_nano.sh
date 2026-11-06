# DeepThinking mazes on a Jetson Nano: build, check against the host chain, then energy rows (MAXN, jetson_clocks).
cd ~/edge
until ! pgrep -f "edge_loo[p]" >/dev/null && ! pgrep -f "run_headfre[e].sh" >/dev/null; do sleep 30; done
D=/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0; sudo chmod a+r $D/in_power*_input $D/in_voltage*_input $D/in_current*_input
cat $D/in_power0_input >/dev/null || { echo POWER_UNREADABLE; exit 1; }
cool() { n=0; until [ $(cat /sys/devices/virtual/thermal/thermal_zone1/temp) -le 65000 ] || [ $n -ge 30 ]; do sleep 10; n=$((n+1)); done; }
sudo nvpmodel -m 0 >/dev/null 2>&1; sudo jetson_clocks
H=$(hostname)
sh build_engines.sh dt_mazes ~/edge/assets/dt_mazes ~/edge/engines/dt_mazes > results/${H}_dt_build.log 2>&1; echo "$(date +%T) build rc=$?"
P="--backend trt --proj engines/dt_mazes/dt_proj_fold_fp32.plan --step engines/dt_mazes/dt_step_fold_fp32.plan --precision fp32 --mazes assets/dt_mazes/mazes33_0_50.npz --board $H"
python3 dt_loop.py $P --ref assets/dt_mazes/host_fp32.npz --check_only --out results/${H}_dt_check.json > results/${H}_dt_check.log 2>&1; echo "$(date +%T) check rc=$?"
for row in "exit --rule stable1 --depth 300:stable1" "fixed --depth 300:fixed300" "fixed --depth 100:fixed100" "fixed --depth 30:fixed30" "exit --rule resid0.0794 --depth 300:resid"; do
  a=${row%%:*}; t=${row##*:}; cool
  python3 dt_loop.py $P --mode $a --no_check --nidle 1 --nwin 3 --win_s 60 --out results/${H}_dt_${t}_maxn.json > results/${H}_dt_${t}_maxn.log 2>&1
  echo "$(date +%T) $t rc=$?"; done
echo DT_DONE
