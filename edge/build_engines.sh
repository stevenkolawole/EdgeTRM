#!/bin/sh
# Build TensorRT step engines on a Jetson from the X3 stepc ONNX exports (carry in/out, logits, q_halt).
#
#   sh build_engines.sh <task> [onnx_dir] [engine_dir]
#   sh build_engines.sh sudoku_mlp ~/edge/assets/sudoku_mlp ~/edge/engines/sudoku_mlp
#
# Per H in {1, 3} present in onnx_dir: stepc_H<H>_fp32.plan (no TF32; Maxwell has none anyway) and
# stepc_H<H>_fp16.plan (--fp16: TensorRT picks FP16 kernels, I/O stays fp32). No INT8: JetPack 4.6 / Maxwell
# has no INT8 kernels (BRINGUP.md A.5); INT8 rows are the Pi's onnxruntime QDQ graphs.
# k-unrolled engines: for every stepc_k<k>_H<H>.onnx in onnx_dir (an export that applies k steps per call; none
# exists yet, e1_energy/aihub_h1 export single steps) the same two engines are built as stepc_k<k>_H<H>_<prec>.plan
# and run with edge_loop.py --steps_per_call <k>.
# Build logs (layer list, timing) land next to each plan as <name>.log; one build takes minutes on a Nano, so
# keep the desktop off and add swap on the 2 GB kit (BRINGUP.md B). Batch is fixed at 1 (the exports are static).
set -e
TASK=${1:?task (sudoku_mlp | sudoku_attn | maze)}
SRC=${2:-$HOME/edge/assets/$TASK}
DST=${3:-$HOME/edge/engines/$TASK}
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
WS=${WS:-1024}                       # workspace MiB (TRT 8.2: --workspace; TRT >= 8.4 also takes --memPoolSize)
mkdir -p "$DST"
[ -x "$TRTEXEC" ] || { echo "trtexec not found at $TRTEXEC (set TRTEXEC=...)"; exit 1; }
"$TRTEXEC" --help 2>/dev/null | grep -q -- '--memPoolSize' && WSFLAG="--memPoolSize=workspace:${WS}M" || WSFLAG="--workspace=$WS"

build() {  # build <onnx> <plan> [extra flags]
    onnx=$1; plan=$2; shift 2
    if [ -f "$plan" ]; then echo "exists: $plan"; return; fi
    echo "== $plan"
    # --noTF32 is accepted by TRT 8.x; --buildOnly skips the inference timing pass; --verbose keeps the layer log
    "$TRTEXEC" --onnx="$onnx" --saveEngine="$plan" $WSFLAG --noTF32 --buildOnly --verbose "$@" \
        > "${plan%.plan}.log" 2>&1 || { echo "BUILD FAILED, see ${plan%.plan}.log (last lines):"; tail -5 "${plan%.plan}.log"; return 1; }
    ls -l "$plan"
}

# prefer the opset-13 restamps (convert_opset.py) when present: TRT 8.2 parses opset <= 13
if ls "$SRC"/stepc_H*_op13.onnx >/dev/null 2>&1; then LIST="$SRC/stepc_H*_op13.onnx $SRC/stepc_k*_H*_op13.onnx"; else LIST="$SRC/stepc_H*.onnx $SRC/stepc_k*_H*.onnx"; fi
for onnx in $LIST; do
    [ -f "$onnx" ] || continue
    base=$(basename "$onnx" .onnx)
    build "$onnx" "$DST/${base}_fp32.plan"
    build "$onnx" "$DST/${base}_fp16.plan" --fp16
done

# Timing of the engine alone (no host loop, no carry feedback), useful as the launch-overhead floor for the
# unrolled-k rows; the number of the paper comes from edge_loop.py, not from here.
#   $TRTEXEC --loadEngine=$DST/stepc_H1_fp16.plan --iterations=200 --avgRuns=50 --warmUp=2000
#
# If the TRT 8.2 parser rejects the opset-17 export (it should not: the graphs use opset<=13 ops), downgrade on the
# host:  python3 -c "import onnx; from onnx import version_converter as v; m=onnx.load('stepc_H1.onnx'); \
#        onnx.save(v.convert_version(m, 13), 'stepc_H1_op13.onnx')"
echo "done: $(ls "$DST"/*.plan 2>/dev/null | wc -l) engines in $DST"
