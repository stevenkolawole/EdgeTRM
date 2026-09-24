"""TFLite builtin REFERENCE kernels (the ones TFLite Micro shares) vs the default optimized
kernels, op by op, on one saved input; prints the first ops whose outputs differ by more than THR
integer steps. This is how the TFLM collapse was traced to the INT8 REDUCE SUM of the RMS norm.

    ~/venvs/mcu/bin/python mcu_ref_kernel_diff.py ~/mcu_h1/sudoku_mlp/H1/inner_step_int8.tflite \
        ~/mcu_h1/sudoku_mlp/H1/variants/int8.raw.npz 14
"""
import sys, numpy as np
from ai_edge_litert import interpreter as I
from ai_edge_litert.tools import flatbuffer_utils as fu
graph, raw = sys.argv[1:3]; THR = float(sys.argv[3]) if len(sys.argv) > 3 else 2
d = np.load(raw)
def run(kind):
    it = I.Interpreter(model_path=graph, experimental_preserve_all_tensors=True, num_threads=1,
                       experimental_op_resolver_type=kind)
    it.allocate_tensors()
    for j, det in enumerate(it.get_input_details()):
        it.set_tensor(det["index"], d[f"in{j}"])
    it.invoke()
    return it
a = run(I.OpResolverType.BUILTIN_REF)
b = run(I.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
m = fu.read_model(graph); sg = m.subgraphs[0]
shown = 0
for k, op in enumerate(sg.operators):
    name = fu.opcode_to_name(m, op.opcodeIndex)
    for o in op.outputs:
        x, y = a.get_tensor(o), b.get_tensor(o)
        diff = float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max())
        if diff > THR:
            ins = [(int(i), sg.tensors[i].type, list(sg.tensors[i].shape)) for i in op.inputs if i >= 0]
            q = sg.tensors[o].quantization
            print(k, name, "out", o, list(x.shape), x.dtype, "maxdiff", diff, "ref", x.ravel()[:6], "opt", y.ravel()[:6], "inputs", ins,
                  "oq", (list(q.scale)[:2], list(q.zeroPoint)[:2]) if q is not None and q.scale is not None else None)
            for i in op.inputs:
                if i >= 0:
                    qi = sg.tensors[i].quantization
                    print("    in", i, a.get_tensor(i).ravel()[:4], b.get_tensor(i).ravel()[:4],
                          (list(qi.scale)[:2], list(qi.zeroPoint)[:2]) if qi is not None and qi.scale is not None else None)
            shown += 1
    if shown >= 4:
        break
print("final equal frac", (a.get_tensor(a.get_output_details()[0]["index"]) == b.get_tensor(b.get_output_details()[0]["index"])).mean())
