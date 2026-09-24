import sys, time, json, numpy as np, torch, onnx
sys.path.insert(0, "/home/ubuntu/x3"); sys.path.insert(0, "/home/ubuntu/m3")
torch.set_num_threads(16)
import aihub_h1 as A, aihub_infer as I, wa_collapse as W
from qdq_torch import QDQRunner
task = sys.argv[1]; variants = sys.argv[2].split(",")
d = I.idir(task)
g16 = onnx.load(str(d / "qdq_w8a16" / "model.onnx"), load_external_data=False).graph
grp = W.classify(g16)
json.dump(grp, open(f"/home/ubuntu/m3/groups_{task}.json", "w"))
r = QDQRunner(d / "qdq_w8a16" / "model.onnx", d / "qdq_w8a8" / "model.onnx", device="cpu")
inp, lab = A.load_data(task, list(range(0, 100))); inp = inp.astype(np.int32)
init = I.initial_state(task)
for v in variants:
    spec = W.VARIANTS[v]
    eight = spec["eight"]
    if spec.get("sixteen_default", True):
        names = [t for t, gname in grp.items() if gname in eight]
    else:
        names = [t for t, gname in grp.items() if gname not in eight]
    r.set_act8(names)
    t0 = time.time()
    res = r.rollout(inp, init, 16, batch=100)
    ex = (res["preds"][-1] == lab).all(-1).mean(); cell = (res["preds"][-1] == lab).mean()
    ref = d / "ablate" / f"eval_{v}_0_100.json"
    rr = json.load(open(ref)) if ref.exists() else None
    print(f"{v}: torch exact {ex:.2f} cell {cell:.4f} ({time.time()-t0:.0f}s, 8-bit tensors {len(names)}) | ORT "
          f"{rr['exact'] if rr else '-'} {round(rr['cell'],4) if rr else '-'}", flush=True)
