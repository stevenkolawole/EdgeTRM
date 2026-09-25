"""Do the activation encodings we declared in the AI Hub w8a8 QDQ graph survive into the compiled DLC?
For every activation Q node of the QDQ ONNX: is its tensor name in the DLC flatbuffer, and does its exact
float32 scale appear in the DLC (anywhere; and within 4 KB after the name)? Grouped as in wa_collapse."""
import json, struct, sys, collections, re
import numpy as np, onnx
from onnx import numpy_helper
sys.path.insert(0, "/home/ubuntu/x3")
import wa_collapse as W
qdq = sys.argv[1]; dlc_model = sys.argv[2]
buf = open(dlc_model, "rb").read()
g = onnx.load(qdq, load_external_data=False).graph
vals = W.const_map(g); qn = W.act_quant_nodes(g); grp = W.classify(g)
rows = []
for t, q in qn.items():
    s = float(numpy_helper.to_array(vals[q.input[1]])) if numpy_helper.to_array(vals[q.input[1]]).ndim == 0 else None
    zp = numpy_helper.to_array(vals[q.input[2]])
    name_pos = [m.start() for m in re.finditer(re.escape(t.encode()), buf)]
    sb = struct.pack("<f", s) if s is not None else None
    anywhere = (sb in buf) if sb else None
    near = any(sb in buf[p:p + 4096] for p in name_pos) if sb and name_pos else False
    # also the "_updated" / "_qdq" renamings the converter uses
    alt = [m.start() for m in re.finditer(re.escape((t + "_updated").encode()), buf)] + \
          [m.start() for m in re.finditer(re.escape((t + "_qdq").encode()), buf)]
    near_alt = any(sb in buf[p:p + 4096] for p in alt) if sb and alt else False
    rows.append(dict(tensor=t, group=grp[t], scale=s, zp=int(np.asarray(zp).reshape(-1)[0]),
                     name_in_dlc=bool(name_pos or alt), scale_anywhere=anywhere, scale_near_name=bool(near or near_alt)))
by = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    c = by[r["group"]]; c["n"] += 1; c["name"] += r["name_in_dlc"]; c["scale_any"] += bool(r["scale_anywhere"]); c["scale_near"] += r["scale_near_name"]
print(f"{'group':14s} {'n':>4s} {'name in DLC':>11s} {'scale in DLC':>12s} {'scale near name':>15s}")
for k, c in sorted(by.items()):
    print(f"{k:14s} {c['n']:4d} {c['name']:11d} {c['scale_any']:12d} {c['scale_near']:15d}")
json.dump(rows, open(sys.argv[3], "w"), indent=1)
