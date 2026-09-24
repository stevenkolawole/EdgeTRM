#!/bin/bash
# H=3 (real-carry calibration) INT8 Sudoku graphs on the U55-256 FVP, for a same-toolchain H=3 vs H=1 comparison.
set -u
S=$HOME/mcu_h1/src/mcu_h1_fvp.sh
for t in sudoku_mlp sudoku_attn; do bash $S build $t int8 u55_256 3; tail -n 2 $HOME/mcu_h1/logs/fvp_${t}_h3_int8_u55_256.build.log; done
for t in sudoku_mlp sudoku_attn; do bash $S run $t int8 u55_256 3 & echo "[chain] run $t pid $!"; done
wait
echo "[chain] all done $(date -u)"
