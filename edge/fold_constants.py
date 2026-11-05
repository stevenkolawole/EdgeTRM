#!/usr/bin/env python3
"""Fold the shape arithmetic of the opset-13 step exports into constants so TensorRT 8.2's ONNX parser accepts them
(it rejects the Pad on puzzle_embedding, whose pads come from a Reshape of a shape tensor). Uses onnxruntime's basic
graph optimizations (constant folding only; standard ONNX ops, no ORT-specific fusions) and checks the folded graph
against the original on random inputs. Writes <name>_fold.onnx next to each input.

    python fold_constants.py assets/*/stepc_H*_op13.onnx
"""
import sys

import numpy as np
import onnxruntime as ort


def session(path, out=None):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    if out:
        so.optimized_model_filepath = out
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def feeds(sess, rng):
    f = {}
    for i in sess.get_inputs():
        if i.type == "tensor(int32)":
            f[i.name] = rng.integers(0, 6, size=i.shape).astype(np.int32)
        else:
            f[i.name] = rng.standard_normal(i.shape).astype(np.float32)
    return f


for src in sys.argv[1:]:
    dst = src.replace(".onnx", "_fold.onnx")
    session(src, out=dst)
    plain = ort.SessionOptions()
    plain.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    a = ort.InferenceSession(src, plain, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(dst, plain, providers=["CPUExecutionProvider"])
    f = feeds(a, np.random.default_rng(0))
    worst = max(float(np.max(np.abs(x - y))) for x, y in zip(a.run(None, f), b.run(None, f)))
    print("%s -> %s  max|diff| %.3g" % (src, dst, worst))
