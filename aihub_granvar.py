#!/usr/bin/env python3
"""Q4b: which weight-quantization granularities compile and run on which chips.

AI Hub's quantize job has no granularity option; its QDQ output (AIMET) always
carries per-output-channel symmetric weights (INT4 or INT8, `axis=1` on the
[in, out] MatMul weight) and per-tensor asymmetric activations. To test the
other granularities on the same compiler, this script downloads that QDQ graph
and rewrites only the weight Quantize/Dequantize pairs:

  w4a8_t, w8a8_t   per-tensor symmetric (one scale per weight matrix)
  w4a8_g32/_g128   block-wise symmetric along the input (reduction) dimension,
                   ONNX opset-21 `block_size` on axis 0 (scale [ceil(in/g), out]),
                   the same grouping the accuracy harness calls w4g32 / w4g128
  w4a16_g32        the same with the 16-bit-activation AI Hub graph

Activations, embeddings and every non-weight Q/DQ are left exactly as AI Hub
produced them, so a row differs from the per-channel row only in the weight
scales. Each variant is checked on the host with onnxruntime against the fp32
graph on real calibration carries (logit agreement), uploaded, and recorded as
a `granvar` entry in ~/aihub_artifacts/h1/<task>/jobs.json; aihub_h1.py
--stage compile/profile then treats it like any other precision.

    python aihub_granvar.py --tasks sudoku_mlp,maze --variants w4a8_t,w8a8_t,w4a8_g32,w4a8_g128
"""
import argparse
import json
import pathlib
import shutil
import sys
import time
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402

WORK = A.ART / "granvar"


def download_qdq(task, graph, H, prec):
    import qai_hub as hub
    rec = A.load_rec(task)
    q = A.find(rec, kind="quantize", graph=graph, H=H, prec=prec)
    assert q, f"no quantize job for {task} {graph} H{H} {prec}"
    job = hub.get_job(q[-1]["job_id"])
    assert job.get_status().code == "SUCCESS", job.get_status()
    d = WORK / task / f"{graph}_H{H}_{prec}"
    if not (d / "src").exists():
        d.mkdir(parents=True, exist_ok=True)
        z = job.get_target_model().download(str(d / "qdq"))
        with zipfile.ZipFile(z) as f:
            f.extractall(d / "unz")
        inner = next((d / "unz").iterdir())
        inner.rename(d / "src")
        shutil.rmtree(d / "unz")
        pathlib.Path(z).unlink()
    return d / "src", q[-1]["job_id"]


def _const_value(g, name):
    for i in g.initializer:
        if i.name == name:
            return i
    for n in g.node:
        if n.op_type == "Constant" and n.output[0] == name:
            return n.attribute[0].t
    return None


def rewrite(src_dir, out_dir, gran, bits):
    """gran: 't' per-tensor or int block size. Rewrites weight Q/DQ pairs whose zero point is a
    signed integer of `bits` bits (the AI Hub weight encodings); returns counts."""
    m = onnx.load(str(src_dir / "model.onnx"), load_external_data=True)
    g = m.graph
    inits = {i.name: i for i in g.initializer}
    consumers = {}
    for n in g.node:
        for x in n.input:
            consumers.setdefault(x, []).append(n)
    zp_type = TensorProto.INT4 if bits == 4 else TensorProto.INT8
    qmax = 2 ** (bits - 1) - 1
    new_inits, n_done, shapes = [], 0, {}
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] not in inits:
            continue
        zp = _const_value(g, n.input[2]) if len(n.input) > 2 else None
        if zp is None or zp.data_type != zp_type:
            continue
        W = numpy_helper.to_array(inits[n.input[0]]).astype(np.float32)
        if W.ndim != 2:
            continue
        din, dout = W.shape
        if gran == "t":
            s = np.array(max(float(np.abs(W).max()), 1e-12) / qmax, dtype=np.float32)
            z_dims, attrs = [], {}
        elif isinstance(gran, str) and gran.startswith("l"):
            # LPBQ grid (AIMET lpbq_utils): per-channel float scale = max block scale / 2^4,
            # per-block integer multiplier round(block scale / channel scale) clipped to [1, 16].
            b = int(gran[1:])
            if din % b:
                # QNN block expansion needs the block to divide the input dim (Sudoku-MLP's
                # 97-token mixer); such layers keep the AI Hub per-channel encoding.
                shapes[f"kept_per_channel_{din}x{dout}"] = shapes.get(f"kept_per_channel_{din}x{dout}", 0) + 1
                continue
            nb = din // b
            sb = np.maximum(np.abs(W).reshape(nb, b, dout).max(1), 1e-12) / qmax      # [nb, out]
            pcfs = sb.max(0, keepdims=True) / 16.0
            kb = np.clip(np.round(sb / pcfs), 1, 16)
            s = (kb * pcfs).astype(np.float32)
            z_dims, attrs = [nb, dout], {"axis": 0, "block_size": b}
        else:
            b = int(gran)
            nb = -(-din // b)
            pad = nb * b - din
            Wp = np.pad(np.abs(W), ((0, pad), (0, 0))).reshape(nb, b, dout)
            s = np.maximum(Wp.max(1), 1e-12).astype(np.float32) / qmax
            z_dims, attrs = [nb, dout], {"axis": 0, "block_size": b}
        sname, zname = n.name + "_x3s", n.name + "_x3z"
        new_inits.append(numpy_helper.from_array(s, sname))
        nz = int(np.prod(z_dims)) if z_dims else 1
        new_inits.append(helper.make_tensor(zname, zp_type, z_dims, [0] * nz))
        dqs = [c for c in consumers.get(n.output[0], []) if c.op_type == "DequantizeLinear"]
        for node in [n] + dqs:
            node.input[1], node.input[2] = sname, zname
            keep = [a for a in node.attribute if a.name not in ("axis", "block_size")]
            del node.attribute[:]
            node.attribute.extend(keep + [helper.make_attribute(k, v) for k, v in attrs.items()])
        n_done += 1
        shapes[(din, dout)] = shapes.get((din, dout), 0) + 1
    g.initializer.extend(new_inits)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx.save_model(m, str(out_dir / "model.onnx"), save_as_external_data=True, all_tensors_to_one_file=True,
                    location="model.data", size_threshold=1024)
    return n_done, {(f"{k[0]}x{k[1]}" if isinstance(k, tuple) else k): v for k, v in shapes.items()}


def host_check(task, graph, H, qdq_dir, n=8):
    """Logit agreement (argmax) of the QDQ graph with the fp32 graph on real calibration carries."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 16
    fp = ort.InferenceSession(str(A.ART / task / A.onnx_name(graph, H)), so, providers=["CPUExecutionProvider"])
    q = ort.InferenceSession(str(qdq_dir / "model.onnx"), so, providers=["CPUExecutionProvider"])
    cal = A.calib_dict(task, H, graph)
    agree, cos = [], []
    for i in range(min(n, len(cal["z_H"]))):
        feed = {k: v[i] for k, v in cal.items()}
        a = fp.run(["logits"], feed)[0]
        b = q.run(["logits"], feed)[0]
        agree.append(float((a.argmax(-1) == b.argmax(-1)).mean()))
        cos.append(float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
    return float(np.mean(agree)), float(np.mean(cos))


VARIANTS = {  # name -> (AI Hub base precision, granularity, weight bits)
    "w4a8_c": ("w4a8", None, 4), "w8a8_c": ("w8a8", None, 8),
    "w4a8_t": ("w4a8", "t", 4), "w8a8_t": ("w8a8", "t", 8),
    "w4a8_g32": ("w4a8", 32, 4), "w4a8_g128": ("w4a8", 128, 4), "w4a16_g32": ("w4a16", 32, 4),
    "w8a8_g32": ("w8a8", 32, 8),
    "w4a8_l64": ("w4a8", "l64", 4), "w4a16_l64": ("w4a16", "l64", 4), "w4a8_l32": ("w4a8", "l32", 4),
}


def main():
    import qai_hub as hub
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="sudoku_mlp,maze")
    ap.add_argument("--graph", default="step")
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--variants", default="w4a8_c,w4a8_t,w8a8_t,w4a8_g32,w4a8_g128")
    ap.add_argument("--no_upload", action="store_true")
    a = ap.parse_args()
    for task in a.tasks.split(","):
        for var in a.variants.split(","):
            base, gran, bits = VARIANTS[var]
            rec = A.load_rec(task)
            if A.find(rec, kind="granvar", graph=a.graph, H=a.H, prec=var):
                print(f"[{task}] {var} already recorded", flush=True)
                continue
            src, qid = download_qdq(task, a.graph, a.H, base)
            t0 = time.time()
            if gran is None:   # the AI Hub graph itself, checked on the host for reference
                out, n_done, shapes = src, 0, {}
            else:
                out = WORK / task / f"{a.graph}_H{a.H}_{var}.onnx_dir"
                if out.exists():
                    shutil.rmtree(out)
                n_done, shapes = rewrite(src, out, gran, bits)
            agree, cos = host_check(task, a.graph, a.H, out)
            print(f"[{task}] {var}: rewrote {n_done} weight Q/DQ {shapes}; host argmax agreement {agree:.4f} "
                  f"logit cos {cos:.4f} ({time.time()-t0:.0f}s)", flush=True)
            entry = {"kind": "granvar", "graph": a.graph, "H": a.H, "prec": var, "base_quantize_job": qid,
                     "gran": gran, "bits": bits, "n_rewritten": n_done, "shapes": shapes,
                     "host_argmax_agree": agree, "host_logit_cos": cos, "path": str(out),
                     "made": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            if gran is not None and not a.no_upload:
                mdl = hub.upload_model(str(out), name=f"x3_{task}_{a.graph}_H{a.H}_{var}")
                entry["model_id"] = mdl.model_id
                print(f"[{task}] {var} uploaded -> {mdl.model_id}", flush=True)
            if gran is None:
                entry["kind"] = "hostcheck"
            A.add_job(task, entry)
            if gran is not None and not a.no_upload:
                shutil.rmtree(out, ignore_errors=True)   # the uploaded copy is the record; p4d disk is shared
        # the downloaded AI Hub graphs are re-downloadable from their quantize jobs
        shutil.rmtree(WORK / task, ignore_errors=True)


if __name__ == "__main__":
    main()
