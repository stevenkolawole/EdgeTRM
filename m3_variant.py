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
    # HTP legality (X3, Sep 25; wa_collapse.build_variant propagate_movement): (1) a data-movement op's output takes
    # its input's encoding; (2) an embedding Gather's table takes the w8a8 table encoding when its output is 8 bit.
    import collections as _c
    prod = {o: n for n in g.node for o in n.output}
    cons = _c.defaultdict(list)
    for n in g.node:
        for x in n.input:
            cons[x].append(n)
    wq8 = {n.input[0]: n for n in g8.node if n.op_type == "QuantizeLinear" and n.input[0] in {i.name for i in g8.initializer}}
    n_prop = n_table = 0
    tbl = []
    for n in g.node:
        if n.op_type not in ("Gather", "Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze", "Flatten"):
            continue
        q_out = next((c for c in cons.get(n.output[0], []) if c.op_type == "QuantizeLinear"), None)
        dq_in = prod.get(n.input[0])
        if q_out is None or dq_in is None or dq_in.op_type != "DequantizeLinear":
            continue
        q_in = prod.get(dq_in.input[0])
        if q_in is None or q_in.op_type != "QuantizeLinear":
            continue
        if q_in.input[0] in inits:                      # embedding table: follows its output (rule 2)
            if n.op_type == "Gather" and q_out.input[0] not in keep and q_in.input[0] in wq8:
                e = wq8[q_in.input[0]]
                sn, zn = q_in.name + "_m3ts", q_in.name + "_m3tz"
                zt = vals8[e.input[2]]
                tbl.append(numpy_helper.from_array(numpy_helper.to_array(vals8[e.input[1]]).astype(np.float32), sn))
                tbl.append(helper.make_tensor(zn, zt.data_type, list(numpy_helper.to_array(zt).shape),
                                              numpy_helper.to_array(zt).reshape(-1).tolist()))
                for node in [q_in, dq_in]:
                    node.input[1], node.input[2] = sn, zn
                n_table += 1
            continue
        if (q_out.input[1], q_out.input[2]) == (dq_in.input[1], dq_in.input[2]):
            continue
        for node in [q_out] + [c for c in cons[q_out.output[0]] if c.op_type == "DequantizeLinear"]:
            node.input[1], node.input[2] = dq_in.input[1], dq_in.input[2]
            kept = [x for x in node.attribute if x.name != "axis"]
            del node.attribute[:]
            node.attribute.extend(kept + [x for x in dq_in.attribute if x.name == "axis"])
        n_prop += 1
    g.initializer.extend(tbl)
    out = d / "qdq_w8a16" / f"variant_{a.name}.onnx"
    onnx.save_model(m, str(out))
    info = {"task": a.task, "key": a.key, "source": a.act, "n16": len(keep), "n8": n8, "keep16": sorted(keep),
            "device_legal": {"movement_outputs_propagated": n_prop, "gather_tables_8bit": n_table},
            "path": str(out)}
    Path(str(out).replace(".onnx", ".json")).write_text(json.dumps(info, indent=1))
    print(json.dumps({k_: v for k_, v in info.items() if k_ != "keep16"}))


if __name__ == "__main__":
    main()
