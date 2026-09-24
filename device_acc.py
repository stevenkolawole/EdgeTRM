#!/usr/bin/env python3
"""Device-faithful accuracy table: exact accuracy per supervision step of the very QDQ graphs AI Hub
compiles (host simulation, ONNX Runtime; files written by aihub_infer.py --stage host/hostbig/hostarc),
plus on-device chains when present. Output: results/devices/device_faithful_acc.json, rows
{task, prec, n_steps_run, step, exact, cell, count, source}. Run on p4d (reads ~/aihub_artifacts/h1)."""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A
import aihub_infer as I

rows = []
def add(task, prec, preds, lab, source, mask=None):
    for s in range(preds.shape[0]):
        if mask is None:
            ex = (preds[s] == lab).all(-1).mean(); cell = (preds[s] == lab).mean()
        else:
            c = (preds[s] == lab) & mask
            ex = (c.sum(-1) == mask.sum(-1)).mean(); cell = c.sum() / mask.sum()
        rows.append({"task": task, "prec": prec, "step": s + 1, "exact": float(ex), "cell": float(cell),
                     "count": int(lab.shape[0]), "source": source})

for task in ("sudoku_mlp", "sudoku_attn", "maze"):
    d = A.ART / task / "infer"
    _, lab = I.test_set(task)
    for f in sorted(d.glob("host_*.npz")):
        prec = f.stem[5:]
        if "_n" in prec:
            continue
        add(task, prec, np.load(f)["preds"], lab, f"host sim of compiled QDQ graph, test 0-{len(lab)-1}")
    for f in sorted(d.glob("hostbig_*.npz")):
        z = np.load(f); prec = f.stem.split("_")[1]
        add(task, prec, z["preds"], z["lab"], f"host sim of compiled QDQ graph, {f.stem.split('_',2)[2]} (start_count)")
    for f in sorted(d.glob("dev_*.json")):
        meta = json.loads(f.read_text())
        if not meta.get("done") or not (d / (f.stem + ".npz")).exists():
            continue
        z = np.load(d / (f.stem + ".npz")); prec = f.stem.split("_")[1]
        add(task, prec, z["preds"], lab, f"ON DEVICE {f.stem[len('dev_'+prec)+1:]}")
d = A.ART / "arc" / "infer"
for f in sorted(d.glob("host_*_100.npz")):
    z = np.load(f); prec = f.stem.split("_")[1]
    add("arc", prec, z["preds"], z["lab"], "host sim of compiled QDQ graph, 100 canonical rows, token exact", mask=z["lab"] != 0)
out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "device_faithful_acc.json")
out.write_text(json.dumps(rows, indent=1))
print(len(rows), "rows ->", out)
