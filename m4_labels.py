#!/usr/bin/env python3
"""M4: correctness labels of every device-faithful cell, from the written records only.

Backends and sources (all on p4d):
  qnn_qdq_host  X3's host simulation of the exact AI Hub QDQ graphs:
                ~/aihub_artifacts/h1/<task>/infer/host_<prec>.npz (Sudoku 100 puzzles x 16 steps, maze 50 x 8)
                host_<prec>_100.npz (ARC, 100 canonical rows x 8 steps, token-exact over non-pad cells)
                ablate/eval_<variant>_<start>_<count>.json (activation-group ablations)
                hostbig_<prec>_1000_1000.npz (Sudoku-MLP puzzles 1000-1999)
  tflite_auto / tflite_ref   M4's TFLite interpreter chains: ~/m4/tflite/<task>/label_<v>_<rt>_<s>_<n>.json
  qnn_device    AI Hub on-device chains (X3): ~/aihub_artifacts/h1/<task>/infer/dev_<prec>_<device>.npz
Every label is paired with fp32 on the same backend, inputs and step count; retained = exact / exact_fp32.

    ~/venvs/kvcache/bin/python m4_labels.py  -> ~/m4/labels.json
"""
import json
import pathlib
import sys

import numpy as np

HOME = pathlib.Path.home()
ART = HOME / "aihub_artifacts" / "h1"
M4 = HOME / "m4"
for extra in (pathlib.Path(__file__).resolve().parent, HOME / "x3"):
    if (extra / "aihub_h1.py").exists():
        sys.path.append(str(extra))
import aihub_h1 as A  # noqa: E402

NSTEPS = {"sudoku_mlp": 16, "sudoku_attn": 16, "maze": 8, "arc": 8}
_LAB = {}


def labels(task, start, n):
    key = (task, start, n)
    if key not in _LAB:
        _LAB[key] = A.load_data(task, list(range(start, start + n)))[1]
    return _LAB[key]


def score(preds, lab, arc=False):
    if arc:
        mask = lab != 0
        corr = (preds == lab) & mask
        ex = corr.sum(-1) == mask.sum(-1)
        return ex, corr.sum(-1) / mask.sum(-1)
    return (preds == lab).all(-1), (preds == lab).mean(-1)


def rec(backend, task, cell, ex, cl, ref_ex, ref_cl, src, start, family, extra=None):
    ex, cl = np.asarray(ex, float), np.asarray(cl, float)
    r = dict(backend=backend, task=task, cell=cell, family=family, n=int(ex.size), start=start,
             exact=float(ex.mean()), cellacc=float(cl.mean()), exact_fp32=float(np.mean(ref_ex)),
             cellacc_fp32=float(np.mean(ref_cl)), source=str(src))
    r["retained"] = r["exact"] / r["exact_fp32"] if r["exact_fp32"] > 0 else None
    # fraction of fp32-solved inputs still solved (X2's measure), when per-input records exist
    if ex.size == np.asarray(ref_ex).size:
        m = np.asarray(ref_ex, bool)
        r["kept_of_fp32_solved"] = float(ex[m].mean()) if m.any() else None
    r.update(extra or {})
    return r


def family_of(cell):
    if cell.startswith("abl_A"):
        return "act_group_8bit"          # one activation group moved to 8 bit in a 16-bit graph
    if cell.startswith("abl_B"):
        return "act_group_16bit"         # one activation group kept at 16 bit in an 8-bit graph
    if cell in ("w8a8", "w4a8", "int8", "w4a8_c", "w4a8_t", "w4a8_allops"):
        return "act8_all"
    if cell in ("w8a16", "w4a16", "w4a16_c"):
        return "act16_all"
    if cell == "w8a8_fc":
        return "act8_fc_only"
    return "other"


def main():
    out = []
    # ---------------- QNN QDQ host simulation (X3)
    for task in ("sudoku_mlp", "sudoku_attn", "maze", "arc"):
        d = ART / task / "infer"
        arc = task == "arc"
        if arc:
            f0 = d / "host_fp32_100.npz"
            if not f0.exists():
                continue
            z0 = np.load(f0)
            lab = z0["lab"]
            rex, rcl = score(z0["preds"][-1], lab, True)
            for prec in ("w8a8", "w4a8", "w8a16", "w4a16"):
                f = d / f"host_{prec}_100.npz"
                if not f.exists():
                    f = M4 / "arc" / f"host_{prec}_100.npz"      # M4's runs of the same protocol
                if f.exists():
                    z = np.load(f)
                    ex, cl = score(z["preds"][-1], lab, True)
                    out.append(rec("qnn_qdq_host", task, prec, ex, cl, rex, rcl, f, 0, family_of(prec)))
            continue
        f0 = d / "host_fp32.npz"
        if not f0.exists():
            continue
        z0 = np.load(f0)
        n = z0["preds"].shape[1]
        lab = labels(task, 0, n)
        rex, rcl = score(z0["preds"][-1], lab)
        for prec in ("w8a8", "w4a8", "w8a16", "w4a16"):
            f = d / f"host_{prec}.npz"
            if f.exists():
                z = np.load(f)
                ex, cl = score(z["preds"][-1], lab)
                out.append(rec("qnn_qdq_host", task, prec, ex, cl, rex, rcl, f, 0, family_of(prec)))
        # ablations (100 puzzles 0-99; fp32 reference = host_fp32 on the same puzzles)
        for f in sorted((d / "ablate").glob("eval_*_0_100.json")) if (d / "ablate").exists() else []:
            e = json.loads(f.read_text())
            ex = np.asarray(e["correct"], float)
            cl = np.full(ex.shape, e["cell"])        # per-puzzle cell accuracy not stored; mean only
            name = "abl_" + e["variant"]
            out.append(rec("qnn_qdq_host", task, name, ex, cl, rex, rcl, f, 0, family_of(name)))
        # 1000-puzzle slices
        fb = d / "hostbig_fp32_1000_1000.npz"
        if fb.exists():
            zb = np.load(fb)
            labb = zb["lab"]
            bex, bcl = score(zb["preds"][-1], labb)
            for prec in ("w8a8", "w4a8", "w8a16", "w4a16"):
                f = d / f"hostbig_{prec}_1000_1000.npz"
                if f.exists():
                    z = np.load(f)
                    ex, cl = score(z["preds"][-1], labb)
                    out.append(rec("qnn_qdq_host", task, prec, ex, cl, bex, bcl, f, 1000, family_of(prec),
                                   {"slice": "1000-1999"}))
            for f in sorted((d / "ablate").glob("eval_*_1000_1000.json")):
                e = json.loads(f.read_text())
                ex = np.asarray(e["correct"], float)
                name = "abl_" + e["variant"]
                out.append(rec("qnn_qdq_host", task, name, ex, np.full(ex.shape, e["cell"]), bex, bcl, f, 1000,
                               family_of(name), {"slice": "1000-1999"}))
        # on-device chains
        for f in sorted(d.glob("dev_*.npz")):
            z = np.load(f)
            meta = json.loads(f.with_suffix(".json").read_text()) if f.with_suffix(".json").exists() else {}
            k = z["preds"].shape[0]
            if not meta.get("done"):              # never read a chain still in flight
                continue
            prec, dev = f.stem[4:].split("_", 1)
            ex, cl = score(z["preds"][-1], lab)
            kex, kcl = score(z0["preds"][k - 1], lab)          # host fp32 at the same step count
            out.append(rec("qnn_device", task, prec, ex, cl, kex, kcl, f, 0, family_of(prec),
                           {"device": dev.replace("_", " "), "steps": int(k)}))
    # ---------------- M4's own QDQ label runs (~/m4/onnx_labels): reference QDQ semantics (ORT basic) for
    # every cell, and ORT-all labels of the vendor-emulated (fused RMSNorm) variants
    od = M4 / "onnx_labels"
    if od.exists():
        recs = {}
        for f in sorted(od.glob("*.json")):
            e = json.loads(f.read_text())
            recs[(e["task"], e["cell"], e["level"])] = (f, e)
        for (task, cell, level), (f, e) in sorted(recs.items()):
            if cell == "fp32self":
                continue
            ref = recs.get((task, "fp32self", level))
            if ref is None and level == "all":
                # X3's ORT-all fp32 host run on the same inputs
                src = ART / task / "infer" / ("host_fp32_100.npz" if task == "arc" else "host_fp32.npz")
                if not src.exists():
                    continue
                z0 = np.load(src)
                lab0 = z0["lab"] if task == "arc" else labels(task, 0, z0["preds"].shape[1])
                rex, rcl = score(z0["preds"][-1], lab0, task == "arc")
            elif ref is None:
                continue
            else:
                rex = np.asarray(ref[1]["correct"], float)
                rcl = np.full(rex.shape, ref[1]["cell_acc"])
            ex = np.asarray(e["correct"], float)
            backend = "qnn_qdq_ref" if level == "basic" else "qnn_qdq_host"
            if backend == "qnn_qdq_host" and "@" not in cell:
                continue                                    # X3's own ORT-all labels are used for these
            out.append(rec(backend, task, cell, ex, np.full(ex.shape, e["cell_acc"]), rex, rcl, f, 0,
                           family_of(cell.split("@")[0]) if "@" not in cell else "vendor_emulation"))
    # ---------------- M3's reference-QDQ-semantics runs of X3's ablation rows (torch QDQ executor,
    # `EdgeTRM/qdq_torch.py`; Sudoku-MLP puzzles 0-99, 16 steps): used where M4 has no ORT-basic label
    import re
    have = {(r["backend"], r["task"], r["cell"]) for r in out}
    z0 = np.load(ART / "sudoku_mlp" / "infer" / "host_fp32.npz")
    lab0 = labels("sudoku_mlp", 0, z0["preds"].shape[1])
    rex, rcl = score(z0["preds"][-1], lab0)
    for logf in (HOME / "m3" / "val_mlp.log", HOME / "m3" / "val_mlp_B.log"):
        if not logf.exists():
            continue
        for line in logf.read_text().splitlines():
            mm = re.match(r"(\S+): torch exact ([0-9.]+) cell ([0-9.]+)", line)
            if not mm:
                continue
            cell = "abl_" + mm.group(1)
            if ("qnn_qdq_ref", "sudoku_mlp", cell) in have:
                continue
            ex_m, cl_m = float(mm.group(2)), float(mm.group(3))
            r = dict(backend="qnn_qdq_ref", task="sudoku_mlp", cell=cell, family=family_of(cell), n=100, start=0,
                     exact=ex_m, cellacc=cl_m, exact_fp32=float(rex.mean()), cellacc_fp32=float(rcl.mean()),
                     retained=ex_m / float(rex.mean()), source=f"{logf} (M3 torch QDQ executor)")
            out.append(r)
            have.add(("qnn_qdq_ref", "sudoku_mlp", cell))
    # ---------------- TFLite chains (M4)
    for task in ("sudoku_mlp", "sudoku_attn", "maze"):
        td = M4 / "tflite" / task
        for rt in ("auto", "ref"):
            refs = sorted(td.glob(f"label_fp32_{rt}_*.json")) or sorted(td.glob("label_fp32_auto_*.json"))
            for f in sorted(td.glob(f"label_*_{rt}_*.json")):
                e = json.loads(f.read_text())
                if e["variant"] == "fp32":
                    continue
                if not e.get("ok"):
                    # a runtime that cannot execute the graph solves nothing (e.g. the reference kernels'
                    # RSQRT on a non-positive REDUCE_SUM result); other errors are not labels
                    if "failed to invoke" in e.get("error", "") and refs:
                        r0 = json.loads(refs[0].read_text())
                        out.append(dict(backend=f"tflite_{rt}", task=task, cell=e["variant"],
                                        family=family_of(e["variant"]), n=e["n"], start=e["start"], exact=0.0,
                                        cellacc=0.0, exact_fp32=r0["exact"], cellacc_fp32=r0["cell"],
                                        retained=0.0, source=str(f), runtime_error=e["error"][:200]))
                    continue
                rf = [r for r in refs if json.loads(r.read_text()).get("n") == e["n"]
                      and json.loads(r.read_text()).get("start") == e["start"]]
                if not rf:
                    continue
                r0 = json.loads(rf[0].read_text())
                z = np.load(f.with_suffix(".npz"))
                z0 = np.load(rf[0].with_suffix(".npz"))
                ex, cl = score(z["preds"][-1], z["lab"])
                rex, rcl = score(z0["preds"][-1], z0["lab"])
                out.append(rec(f"tflite_{rt}", task, e["variant"], ex, cl, rex, rcl, f, e["start"],
                               family_of(e["variant"]), {"fp32_source": str(rf[0])}))
    (M4 / "labels.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(f"{r['backend']:13s} {r['task']:12s} {r['cell']:28s} n={r['n']:4d} exact {r['exact']:.3f} "
              f"fp32 {r['exact_fp32']:.3f} retained {r['retained'] if r['retained'] is None else round(r['retained'], 3)}")


if __name__ == "__main__":
    main()
