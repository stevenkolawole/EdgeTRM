#!/usr/bin/env python3
"""Host-side: restamp the X3 step exports (opset 17, torch.onnx) to an older opset for the boards' runtimes.
onnxruntime 1.10 (the last Python 3.6 wheel) refuses opset > 15 outright, and TensorRT 8.2's parser is of the same
generation; the graphs themselves only use ops that exist unchanged at opset 13, so the version converter is a
no-op on the arithmetic. Writes <name>_op<N>.onnx next to each input and checks the converted graph with the
host's onnxruntime on random inputs of the declared shapes (max abs difference printed; 0 expected).

    python3 convert_opset.py --opset 13 assets/*/stepc_H*.onnx
Needs onnx (and onnxruntime for the check); run on the host, not the board.
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--no_check", action="store_true")
    a = ap.parse_args()
    import onnx
    from onnx import version_converter
    for src in a.files:
        dst = os.path.splitext(src)[0] + "_op%d.onnx" % a.opset
        m = onnx.load(src)
        cur = [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")][0]
        try:
            c = version_converter.convert_version(m, a.opset)
            how = "version_converter"
        except RuntimeError as ex:               # e.g. "No Adapter From Version $16 for Identity": the converter
            c = onnx.ModelProto()                # lacks trivial adapters; restamp and let the checker validate
            c.CopyFrom(m)                        # every node against the opset-%d schemas
            for o in c.opset_import:
                if o.domain in ("", "ai.onnx"):
                    o.version = a.opset
            for n in c.graph.node:               # Reshape.allowzero is opset 14+; its default 0 is the opset-13
                if n.op_type == "Reshape":       # behaviour, so dropping it changes nothing (1 would)
                    for at in list(n.attribute):
                        if at.name == "allowzero":
                            assert at.i == 0, "Reshape allowzero=1 has no opset-13 equivalent: " + n.name
                            n.attribute.remove(at)
            how = "restamp (%s)" % str(ex).strip().splitlines()[-1][:60]
        onnx.checker.check_model(c, full_check=True)
        onnx.save(c, dst)
        ops = sorted(set(n.op_type for n in c.graph.node))
        msg = "%s: opset %d -> %d by %s, %d nodes -> %d, %.1f MB, ops %s" % (
            src, cur, a.opset, how, len(m.graph.node), len(c.graph.node), os.path.getsize(dst) / 1e6, ops)
        if not a.no_check:
            import onnxruntime as ort
            rng = np.random.RandomState(0)
            feed = {}
            for i in m.graph.input:
                shp = [d.dim_value for d in i.type.tensor_type.shape.dim]
                if i.type.tensor_type.elem_type == onnx.TensorProto.INT32:
                    feed[i.name] = rng.randint(1, 6, size=shp).astype(np.int32)
                else:
                    feed[i.name] = rng.randn(*shp).astype(np.float32)
            outs = [o.name for o in m.graph.output]
            s0 = ort.InferenceSession(src, providers=["CPUExecutionProvider"]).run(outs, feed)
            s1 = ort.InferenceSession(dst, providers=["CPUExecutionProvider"]).run(outs, feed)
            msg += ", max abs diff vs source %s" % ["%.1e" % float(np.abs(x - y).max()) for x, y in zip(s0, s1)]
        print(msg, flush=True)


if __name__ == "__main__":
    main()
