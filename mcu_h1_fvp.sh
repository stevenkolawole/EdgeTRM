#!/bin/bash
# Corstone-300 FVP run of one Vela-compiled inner step (modelled on
# ~/corstone/corstone_chain.sh, which produced the H=3 Sudoku-MLP number).
# The eval-kit tree already carries the Sep 10 patches (patch_ld.py: tensor
# arena in DDR; patch_resolver.py: EMBEDDING_LOOKUP registered; patch_runner.py:
# token-id inputs zero-filled), so this only configures, builds and runs.
#
#   bash mcu_h1_fvp.sh <build|run> <task> <variant> <u55_256|u65_256> [H]
#
# Builds share the kit source tree (the TFLM library build runs inside it), so
# run the builds one after another and only the FVP runs in parallel.
#
# Input graph: ~/mcu_h1/<task>/H<H>/fvp/<variant>_<npu>/*_vela.tflite (mcu_h1.py --stage fvpprep).
# Build dir:   ~/corstone/ml-embedded-evaluation-kit/build_<task>_h<H>_<variant>_<npu>
# Logs:        ~/mcu_h1/logs/fvp_<task>_h<H>_<variant>_<npu>.{build,run}.log
set -u
MODE=$1; TASK=$2; VAR=$3; NPU=$4; H=${5:-1}
KIT=$HOME/corstone/ml-embedded-evaluation-kit
GDIR=$HOME/mcu_h1/$TASK/H$H/fvp/${VAR}_${NPU}
MODEL=$(ls "$GDIR"/*_vela.tflite | head -n1)
BDIR=build_${TASK}_h${H}_${VAR}_${NPU}
LOGB=$HOME/mcu_h1/logs/fvp_${TASK}_h${H}_${VAR}_${NPU}.build.log
LOGR=$HOME/mcu_h1/logs/fvp_${TASK}_h${H}_${VAR}_${NPU}.run.log
export PATH=$HOME/corstone/gcc/bin:$PATH
case $NPU in
  u55_256) ID=U55; CFG=H256; MEM=Shared_Sram; EXTRA=""; FVPBIN=FVP_Corstone_SSE-300_Ethos-U55 ;;
  u65_256) ID=U65; CFG=Y256; MEM=Dedicated_Sram; EXTRA="-DETHOS_U_NPU_CACHE_SIZE=393216"; FVPBIN=FVP_Corstone_SSE-300_Ethos-U65 ;;
  *) echo "unknown npu $NPU"; exit 2 ;;
esac
if [ "$MODE" = build ]; then
{
  echo "[build] start $(date -u) task=$TASK H=$H variant=$VAR npu=$NPU model=$MODEL"
  cd "$KIT" && rm -rf "$BDIR" && mkdir -p "$BDIR" && cd "$BDIR" || exit 3
  cmake .. -DUSE_CASE_BUILD=inference_runner -DTARGET_PLATFORM=mps3 -DTARGET_SUBSYSTEM=sse-300 \
    -DETHOS_U_NPU_ENABLED=ON -DETHOS_U_NPU_ID=$ID -DETHOS_U_NPU_CONFIG_ID=$CFG -DETHOS_U_NPU_MEMORY_MODE=$MEM $EXTRA \
    -DCMAKE_TOOLCHAIN_FILE=scripts/cmake/toolchains/bare-metal-gcc.cmake \
    -Dinference_runner_MODEL_PATH="$MODEL" -Dinference_runner_ACTIVATION_BUF_SZ=0x900000 2>&1 | tail -n 12
  nice -n 5 make -j 12 2>&1 | tail -n 4
  ls -la bin/*.axf
  echo "[build] done $(date -u)"
} > "$LOGB" 2>&1
exit 0
fi
AXF=$KIT/$BDIR/bin/mlek_inference_runner.axf
[ -f "$AXF" ] || { echo "[fvp] no axf, see $LOGB" > "$LOGR"; exit 4; }
FVP=$(ls $HOME/corstone/FVP/models/*/$FVPBIN | head -n1)
{
  echo "[fvp] $FVP axf=$AXF macs=256 $(date -u)"
  nice -n 5 timeout 10800 "$FVP" -a "$AXF" \
    -C mps3_board.visualisation.disable-visualisation=1 \
    -C mps3_board.telnetterminal0.start_telnet=0 \
    -C mps3_board.uart0.out_file=- \
    -C mps3_board.uart0.unbuffered_output=1 \
    -C mps3_board.uart0.shutdown_on_eot=1 \
    -C cpu0.CFGITCMSZ=15 -C cpu0.CFGDTCMSZ=15 \
    -C ethosu.num_macs=256 \
    --stat
  echo "[fvp] rc=$? $(date -u)"
} > "$LOGR" 2>&1
