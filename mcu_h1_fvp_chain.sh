#!/bin/bash
# Build the four H=1 INT8 FVP images one after another, then run the four FVPs in parallel.
set -u
S=$HOME/mcu_h1/src/mcu_h1_fvp.sh
for t in sudoku_mlp sudoku_attn; do for n in u55_256 u65_256; do
  bash $S build $t int8 $n 1; tail -n 2 $HOME/mcu_h1/logs/fvp_${t}_h1_int8_${n}.build.log
done; done
for t in sudoku_mlp sudoku_attn; do for n in u55_256 u65_256; do
  bash $S run $t int8 $n 1 &
  echo "[chain] run $t $n pid $!"
done; done
wait
echo "[chain] all done $(date -u)"
