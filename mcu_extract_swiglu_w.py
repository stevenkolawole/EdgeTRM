#!/usr/bin/env python3
"""Pull one block's SwiGLU weights (gate_up [3072,512], down [512,1536]) as integer values from the
INT8 and W4A8-per-channel sudoku_attn H=1 graphs, for the TOSA INT4 check with real weights (mcu_tosa_int4.py --real_w).

    ~/venvs/mcu/bin/python mcu_extract_swiglu_w.py ~/mcu_h1/tosa_int4/real_w.npz
"""
import sys, numpy as np
from ai_edge_litert.tools import flatbuffer_utils as fu
out = sys.argv[1]
res = {}
for tag, g in (("int8", "/home/ubuntu/mcu_h1/sudoku_attn/H1/inner_step_int8.tflite"),
               ("int4", "/home/ubuntu/mcu_h1/sudoku_attn/H1/inner_step_w4a8_c.tflite")):
    m = fu.read_model(g); sg = m.subgraphs[0]
    got = {}
    for op in sg.operators:
        if fu.opcode_to_name(m, op.opcodeIndex) != "FULLY_CONNECTED":
            continue
        t = sg.tensors[op.inputs[1]]
        shp = tuple(t.shape)
        key = {(3072, 512): "up", (512, 1536): "down"}.get(shp)
        if key is None or key in got:
            continue
        raw = np.frombuffer(bytes(m.buffers[t.buffer].data), dtype=np.uint8)
        if t.type == 17:  # INT4, packed two per byte, low nibble first
            lo = (raw & 0x0F).astype(np.int8); hi = (raw >> 4).astype(np.int8)
            v = np.empty(raw.size * 2, np.int8); v[0::2] = lo; v[1::2] = hi
            v = np.where(v > 7, v - 16, v).astype(np.int8)[: int(np.prod(shp))]
        else:
            v = raw.view(np.int8)
        got[key] = v.reshape(shp)
        print(tag, key, shp, "type", t.type, "range", int(v.min()), int(v.max()))
    res[f"{tag}_up"] = got["up"]; res[f"{tag}_down"] = got["down"]
np.savez(out, **res)
