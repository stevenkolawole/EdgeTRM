#!/usr/bin/env python3
"""INT8 QDQ step graphs for the onnxruntime CPU boards (Raspberry Pi): onnxruntime's static quantizer, calibrated on
the real carries of make_calib_carries.py (calibration puzzles disjoint from the test puzzles). int8 symmetric
per-channel weights, uint8 activations (MinMax), QDQ format, so the graph stays a plain ONNX file the Pi's
onnxruntime runs with its int8 kernels. Writes <step graph stem>_int8qdq.onnx next to the input.

    python quantize_int8_qdq.py assets/sudoku_mlp/stepc_H3_op13_fold.onnx assets/sudoku_mlp/calib_carries.npz
"""
import sys

import numpy as np
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

INPUTS = ["z_H", "z_L", "inputs", "puzzle_embedding"]


class Carries(CalibrationDataReader):
    def __init__(self, path):
        d = np.load(path)
        self.rows = [{k: d[k][i:i + 1] for k in INPUTS} for i in range(d["z_H"].shape[0])]
        self.i = 0

    def get_next(self):
        if self.i >= len(self.rows):
            return None
        self.i += 1
        return self.rows[self.i - 1]


src, calib = sys.argv[1], sys.argv[2]
dst = src.replace(".onnx", "_int8qdq.onnx")
quantize_static(src, dst, Carries(calib), quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8, per_channel=True)
print("wrote", dst)
