#!/usr/bin/env python3
"""Write an AI Hub-compilable QDQ variant for an M3 activation allocation: start from the w8a16 graph and give
every activation tensor NOT in the keep-16 set the w8a8 graph's own 8-bit encoding (same AIMET calibration),
exactly as wa_collapse.build_variant does per group, but for an explicit tensor list.

    python m3_variant.py --task sudoku_mlp --act results/m3/act_sudoku_mlp.json --key greedy@20 \
        --art ~/aihub_artifacts/h1 --name m3_greedy20
The variant is saved next to the w8a16 model (its model.data stays shared) and a JSON with the tensor list.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--act", required=True, help="m3_act output JSON (orders)")
    ap.add_argument("--key", required=True, help="<score>@<k>: keep the top-k tensors of that order at 16 bit")
    ap.add_argument("--art", default=str(Path.home() / "aihub_artifacts" / "h1"))
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    J = json.load(open(a.act))
    sc, k = a.key.split("@")
    keep = set(J["orders"][sc][:int(k)])
    d = Path(a.art) / a.task / "infer"
    m = onnx.load(str(d / "qdq_w8a16" / "model.onnx"), load_external_data=False)
    g8 = onnx.load(str(d / "qdq_w8a8" / "model.onnx"), load_external_data=False).graph
    g = m.graph
    inits = {i.name for i in g.initializer}
    vals8 = {i.name: i for i in g8.initializer}
    vals8.update({n.output[0]: n.attribute[0].t for n in g8.node if n.op_type == "Constant"})
    q8 = {n.input[0]: n for n in g8.node if n.op_type == "QuantizeLinear" and n.input[0] not in {i.name for i in g8.initializer}}
    cons = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    new, n8 = [], 0
    for q in [n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] not in inits]:
        t = q.input[0]
        if t in keep:
            continue
        e = q8[t]
        s = numpy_helper.to_array(vals8[e.input[1]]).astype(np.float32)
        zt = vals8[e.input[2]]
        z = numpy_helper.to_array(zt)
        sn, zn = q.name + "_m3s", q.name + "_m3z"
        new.append(numpy_helper.from_array(s, sn))
        new.append(helper.make_tensor(zn, zt.data_type, list(z.shape), z.reshape(-1).tolist()))
        for node in [q] + [c for c in cons.get(q.output[0], []) if c.op_type == "DequantizeLinear"]:
            node.input[1], node.input[2] = sn, zn
        n8 += 1
    g.initializer.extend(new)
    out = d / "qdq_w8a16" / f"variant_{a.name}.onnx"
    onnx.save_model(m, str(out))
    info = {"task": a.task, "key": a.key, "source": a.act, "n16": len(keep), "n8": n8, "keep16": sorted(keep),
            "path": str(out)}
    Path(str(out).replace(".onnx", ".json")).write_text(json.dumps(info, indent=1))
    print(json.dumps({k_: v for k_, v in info.items() if k_ != "keep16"}))


if __name__ == "__main__":
    main()
