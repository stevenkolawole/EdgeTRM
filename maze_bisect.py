#!/usr/bin/env python3
"""Bisect the Maze w8a8 on-device failure (S24 and 8 Elite run the compiled w8a8 graph wrongly from step 1
while the fp16 graph matches the host). Each variant is the AI Hub Maze stepc graph with all activations at
the w8a8 encodings except the named groups at w8a16 (wa_collapse M* variants; both HTP legality rules:
Gather table follows its output, data-movement ops keep their input encoding). For each variant: upload and
compile (qnn_dlc) on acct2, run ONE step on 20 mazes from the initial carry on the S24, and compare the
device's z_H / logits with the host run of the SAME graph (ONNX Runtime, reference QDQ = ORT_DISABLE_ALL,
and ORT-fused) and with host fp32. A variant whose device z_H cosine to its host graph exceeds 0.99 names
the group whose 8-bit encoding the HTP mis-executes.

    python maze_bisect.py submit   (idempotent)      python maze_bisect.py check
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_infer as I  # noqa: E402
from aihub_acct import write_io_enc  # noqa: E402
from aihub_sched import named_outputs  # noqa: E402

# BISECT_TASK=sudoku_attn runs the sequence-length control: the same RoPE-8-bit graph pattern (same
# architecture: 2 layers, 512 hidden, 8 heads, RoPE) at seq 97 instead of 916 (audit item 111).
TASK = os.environ.get("BISECT_TASK", "maze")
N = 20
# BISECT_DEV=e8 runs the inference on the Snapdragon 8 Elite QRD, reusing the S24 record's DLCs (qnn_dlc is
# device-agnostic); results go to bisect_e8.json
DEV, OS_ = {"s24": ("Samsung Galaxy S24", "14"), "e8": ("Snapdragon 8 Elite QRD", "15")}[os.environ.get("BISECT_DEV", "s24")]
VARIANTS = {"maze": ["M0_all8", "Ma_rope16", "Mb_concat16", "Mc_scores16", "Md_notlinear16",
                     "Me_ropeconst16", "Mf_ropemul16", "Mg_rope_scores16", "Mh_notlinear_notscores16"],
            "sudoku_attn": ["M0_all8", "Ma_rope16", "Mc_scores16"],
            "maze_s97": ["M0_all8", "Ma_rope16", "Mc_scores16"]}[TASK]
D = I.idir(TASK)
REC = D / ("bisect.json" if DEV.startswith("Samsung") else "bisect_e8.json")
if not REC.exists() and (D / "bisect.json").exists():
    REC.write_text(json.dumps({v: {k: e[k] for k in ("model_id", "compile") if k in e}
                               for v, e in json.loads((D / "bisect.json").read_text()).items()}, indent=1))


def load():
    return json.loads(REC.read_text()) if REC.exists() else {}


def save(r):
    REC.write_text(json.dumps(r, indent=1))


def feeds(v, tshapes):
    inp, _ = I.test_set(TASK)
    zH0, zL0, emb = I.initial_state(TASK)
    enc = json.loads((D / f"io_enc_bisect_{v}.json").read_text())["inputs"]
    f = {"z_H": [zH0] * N, "z_L": [zL0] * N, "inputs": [inp[i:i + 1].astype(np.int32) for i in range(N)],
         "puzzle_embedding": [emb] * N}
    for k in list(f):
        if tshapes[k][1] in ("uint8", "uint16", "int8", "int16"):
            assert enc[k]["dtype"] == tshapes[k][1], (k, enc[k], tshapes[k])
            f[k] = [I.quantize_input(x, enc[k]) for x in f[k]]
    return {k: f[k] for k in tshapes}


def submit():
    import qai_hub as hub
    C = hub.Client(profile="acct2")
    r = load()
    for v in VARIANTS:
        e = r.setdefault(v, {})
        up = D / f"upload_bisect_{v}"
        if not (up / "model.onnx").exists():
            up.mkdir(exist_ok=True)
            os.link(D / "qdq_w8a16" / "model.data", up / "model.data")
            os.link(D / "qdq_w8a16" / f"variant_{v}.onnx", up / "model.onnx")
            write_io_enc(up / "model.onnx", D / f"io_enc_bisect_{v}.json")
        if "model_id" not in e:
            e["model_id"] = C.upload_model(str(up), name=f"x3_acct2_{TASK}_bisect_{v}").model_id
            save(r)
        if "compile" not in e:
            e["compile"] = C.submit_compile_job(model=C.get_model(e["model_id"]), device=hub.Device(DEV, OS_),
                                                options="--target_runtime qnn_dlc --quantize_io",
                                                name=f"x3_acct2_{TASK}_bisect_{v}").job_id
            save(r)
            print(v, "compile", e["compile"], flush=True)
    while True:
        pending = 0
        for v in VARIANTS:
            e = r[v]
            if "inference" in e or e.get("compile_failed"):
                continue
            cj = C.get_job(e["compile"])
            st = cj.get_status()
            if st.code == "FAILED":
                e["compile_failed"] = (st.message or "")[:400]
                save(r)
                print(v, "COMPILE FAILED", e["compile_failed"][:160], flush=True)
                continue
            if st.code != "SUCCESS":
                pending += 1
                continue
            ij = C.submit_inference_job(model=cj.get_target_model(), device=hub.Device(DEV, OS_),
                                        inputs=feeds(v, cj.get_target_shapes()), name=f"x3_acct2_{TASK}_bisect_{v}_s0_{os.environ.get('BISECT_DEV', 's24')}")
            e["inference"] = ij.job_id
            save(r)
            print(v, "inference", ij.job_id, flush=True)
        if not pending:
            break
        time.sleep(60)


def host_outputs(path, opt):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 16
    if opt == "ref":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp, _ = I.test_set(TASK)
    zH0, zL0, emb = I.initial_state(TASK)
    o = [sess.run(I.OUTS, {"z_H": zH0, "z_L": zL0, "inputs": inp[i:i + 1].astype(np.int32), "puzzle_embedding": emb})
         for i in range(N)]
    return np.concatenate([x[0].reshape(1, -1) for x in o]), np.concatenate([x[2] for x in o]).argmax(-1)


def cos(a, b):
    return float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)).mean())


def check():
    import qai_hub as hub
    C = hub.Client(profile="acct2")
    r = load()
    fz, fl = host_outputs(I.A.ART / TASK / "stepc_H1.onnx", "ref") if hasattr(I, "A") else (None, None)
    for v in VARIANTS:
        e = r.get(v, {})
        if "result" in e or "inference" not in e:
            continue
        j = C.get_job(e["inference"])
        st = j.get_status().code
        if st != "SUCCESS":
            if st == "FAILED":
                e["result"] = {"inference_failed": (j.get_status().message or "")[:400]}
                save(r)
                print(v, "INFERENCE FAILED", flush=True)
            continue
        o = named_outputs(j.download_output_data())
        dz = np.concatenate([x.reshape(1, -1).astype(np.float32) for x in o["z_H_out"][:N]])
        dl = np.concatenate([x.astype(np.float32) for x in o["logits"][:N]]).argmax(-1)
        path = D / "qdq_w8a16" / f"variant_{v}.onnx"
        rz, rl = host_outputs(path, "ref")
        oz, ol = host_outputs(path, "fused")
        res = {"device_vs_host_ref_zH_cos": cos(dz, rz), "device_vs_host_ref_argmax": float((dl == rl).mean()),
               "device_vs_host_fused_zH_cos": cos(dz, oz), "host_ref_vs_host_fused_zH_cos": cos(rz, oz),
               "host_ref_vs_host_fused_argmax": float((rl == ol).mean())}
        if fz is not None:
            res.update(device_vs_fp32_zH_cos=cos(dz, fz), host_ref_vs_fp32_zH_cos=cos(rz, fz))
        e["result"] = res
        save(r)
        print(v, json.dumps({k: round(x, 4) for k, x in res.items()}), flush=True)


if __name__ == "__main__":
    {"submit": submit, "check": check}[sys.argv[1]]()
