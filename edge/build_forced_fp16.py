#!/usr/bin/env python3
"""Build a TensorRT engine that runs every floating-point layer in fp16. trtexec --fp16 only allows fp16, and on the
Jetson Nano (TensorRT 8.2) the builder timed fp32 kernels faster for every layer, so its "fp16" engines came out
byte-identical to the fp32 ones. Here each layer whose inputs and outputs are all float gets precision HALF under
the builder's precision-constraint flag; graph inputs and outputs stay fp32. Python 3.6, TensorRT 8.x bindings.

    python3 build_forced_fp16.py assets/sudoku_mlp/stepc_H1_op13_fold.onnx engines/sudoku_mlp/stepc_H1_op13_fold_fp16forced.plan
"""
import sys

import tensorrt as trt

onnx_path, plan_path = sys.argv[1], sys.argv[2]
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open(onnx_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        sys.exit(1)

config = builder.create_builder_config()
config.max_workspace_size = 1 << 30
config.set_flag(trt.BuilderFlag.FP16)
flag = getattr(trt.BuilderFlag, "OBEY_PRECISION_CONSTRAINTS", None) or trt.BuilderFlag.STRICT_TYPES
config.set_flag(flag)

forced = 0
for i in range(network.num_layers):
    layer = network.get_layer(i)
    tensors = [layer.get_input(j) for j in range(layer.num_inputs)] + \
              [layer.get_output(j) for j in range(layer.num_outputs)]
    if all(t is not None and t.dtype == trt.float32 for t in tensors) and not any(t.is_shape_tensor for t in tensors):
        layer.precision = trt.float16
        forced += 1
print("layers %d, forced to fp16 %d, constraint flag %s" % (network.num_layers, forced, flag))

engine = builder.build_serialized_network(network, config)
if engine is None:
    sys.exit("build failed")
with open(plan_path, "wb") as f:
    f.write(engine)
print("wrote", plan_path)
