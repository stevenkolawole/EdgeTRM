#!/usr/bin/env python3
"""Is the v79 long-sequence integer-attention fault specific to the looped Maze model, or general?
(reports/edgetrm_exp/E3_V79_GENERIC.md; follow-up to E3_V79_LENGTH.md.)

A plain, non-looped encoder (google/bert_uncased_L-4_H-512_A-8: 4 layers, d=512, 8 heads, the Maze dims; learned
absolute positions interpolated 512 -> 1024) is exported at fixed sequence lengths, quantized w8a8 by the AI Hub
quantize job, compiled qnn_dlc on the pinned QAIRT, and run on the Galaxy S25 (hexagon v79) and S23 (v73) on the
same 32 inputs. The reference is the same QDQ graph in onnxruntime with ORT_DISABLE_ALL (each Q/DQ pair executed
as written, the semantics the devices follow). Readout per (length, device): mean cosine of the (L, 512) output,
argmax-over-channels agreement per token, and the per-position cosine curve (does disagreement grow with position?).

Runs on the Babel login node (API work only; no GPU, no sbatch) from ~/v79gen with the kvcache env:
    python aihub_v79_generic.py build            # export ONNX per length, write calibration + test tokens
    python aihub_v79_generic.py submit           # quantize -> compile -> inference, all lengths and devices
    python aihub_v79_generic.py collect          # wait, download outputs, host reference, metrics, records
Tokens are never read or printed here: the qai_hub client loads ~/.qai_hub/client.ini itself.
"""
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("V79GEN_ROOT", Path.home() / "v79gen"))
MODEL = "google/bert_uncased_L-4_H-512_A-8"
LENGTHS = [97, 256, 512, 916, 1024]
MAXPOS = 1024
N_CALIB = 200
N_TEST = 32
QAIRT = "2.45"
DEVICES = {"S25": ("Samsung Galaxy S25", "15"), "S23": ("Samsung Galaxy S23", "13")}  # v79, v73
CONTROL = ("fp16", 916, "S25")  # float graph at 916 on the S25
STATE = ROOT / "state.json"


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"jobs": []}


def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def find(st, **k):
    return [j for j in st["jobs"] if all(j.get(a) == b for a, b in k.items())]


# ------------------------------------------------------------------ build
def english_tokens(tok, n_needed):
    """Token ids from English text already on the machine (license / README files of the conda env), so the
    calibration and test inputs are real text and attention is peaked rather than uniform."""
    import site
    files = []
    for sp in site.getsitepackages():
        files += glob.glob(f"{sp}/**/LICENSE*", recursive=True) + glob.glob(f"{sp}/**/README*", recursive=True)
    files = sorted(set(files))
    ids = []
    for f in files:
        try:
            txt = Path(f).read_text(errors="ignore")
        except Exception:
            continue
        if len(txt) < 500:
            continue
        ids += tok(txt, add_special_tokens=False)["input_ids"]
        if len(ids) >= n_needed:
            break
    if len(ids) < n_needed:
        raise SystemExit(f"only {len(ids)} text tokens found, need {n_needed}")
    return np.array(ids, dtype=np.int64), len(files)


def build():
    import torch
    from transformers import BertModel, BertTokenizer

    ROOT.mkdir(parents=True, exist_ok=True)
    tok = BertTokenizer.from_pretrained(MODEL)
    m = BertModel.from_pretrained(MODEL, attn_implementation="eager", add_pooling_layer=False).eval()
    # extend the learned position table 512 -> 1024 by linear interpolation along the position axis
    pe = m.embeddings.position_embeddings.weight.detach()  # (512, 512)
    pe2 = torch.nn.functional.interpolate(pe.T[None], size=MAXPOS, mode="linear", align_corners=True)[0].T.contiguous()
    m.embeddings.position_embeddings = torch.nn.Embedding(MAXPOS, pe.shape[1])
    m.embeddings.position_embeddings.weight.data = pe2
    m.embeddings.register_buffer("position_ids", torch.arange(MAXPOS)[None], persistent=False)
    m.embeddings.register_buffer("token_type_ids", torch.zeros(1, MAXPOS, dtype=torch.long), persistent=False)
    m.config.max_position_embeddings = MAXPOS

    class Wrap(torch.nn.Module):
        def __init__(s, b):
            super().__init__()
            s.b = b

        def forward(s, input_ids):
            return s.b(input_ids=input_ids).last_hidden_state

    w = Wrap(m).eval()
    Lmax = max(LENGTHS)
    ids, nfiles = english_tokens(tok, (N_CALIB + N_TEST) * Lmax + Lmax)
    rng = np.random.default_rng(0)
    starts = rng.choice(len(ids) - Lmax - 1, size=N_CALIB + N_TEST, replace=False)
    # fixed windows: input k at length L is the first L tokens of window k, with [CLS] first and [SEP] last
    def window(k, L):
        s = ids[starts[k]: starts[k] + L - 2]
        return np.concatenate([[tok.cls_token_id], s, [tok.sep_token_id]]).astype(np.int32)[None]

    meta = {"model": MODEL, "maxpos": MAXPOS, "n_calib": N_CALIB, "n_test": N_TEST, "text_files": nfiles,
            "n_text_tokens": int(len(ids)), "lengths": {}}
    for L in LENGTHS:
        calib = np.stack([window(k, L) for k in range(N_CALIB)])  # (N, 1, L)
        test = np.stack([window(N_CALIB + k, L) for k in range(N_TEST)])
        np.savez(ROOT / f"inputs_L{L}.npz", calib=calib, test=test)
        onnx_path = ROOT / f"bert_small_L{L}.onnx"
        x = torch.from_numpy(test[0])
        with torch.no_grad():
            ref = w(x).numpy()
        torch.onnx.export(w, (x,), str(onnx_path), input_names=["input_ids"], output_names=["last_hidden_state"],
                          opset_version=17, dynamo=False, do_constant_folding=True)
        import onnxruntime as ort
        s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        y = s.run(None, {"input_ids": test[0]})[0]
        err = float(np.abs(y - ref).max())
        meta["lengths"][str(L)] = {"onnx": onnx_path.name, "onnx_mb": round(onnx_path.stat().st_size / 2**20, 2),
                                  "torch_vs_ort_maxabs": err, "out_shape": list(y.shape)}
        print(f"L={L}: onnx {onnx_path.name} {meta['lengths'][str(L)]['onnx_mb']} MB, torch-vs-ort max|d| {err:.2e}",
              flush=True)
    (ROOT / "build_meta.json").write_text(json.dumps(meta, indent=1))
    print("build done", flush=True)


# ------------------------------------------------------------------ submit
def submit():
    import qai_hub as hub
    st = load_state()
    fw = [f for f in hub.get_frameworks() if "qairt" in f.name.lower() or "qnn" in f.name.lower()]
    print("frameworks:", [(f.name, f.api_version, f.full_version) for f in fw][:6], flush=True)
    # 1. quantize jobs (w8a8), one per length
    for L in LENGTHS:
        if find(st, kind="quantize", L=L):
            continue
        d = np.load(ROOT / f"inputs_L{L}.npz")
        calib = {"input_ids": [d["calib"][k] for k in range(N_CALIB)]}
        qj = hub.submit_quantize_job(model=str(ROOT / f"bert_small_L{L}.onnx"), calibration_data=calib,
                                     weights_dtype=hub.QuantizeDtype.INT8, activations_dtype=hub.QuantizeDtype.INT8,
                                     name=f"v79gen_quant_w8a8_L{L}")
        st["jobs"].append({"kind": "quantize", "L": L, "prec": "w8a8", "job_id": qj.job_id})
        save_state(st)
        print(f"quantize L={L} -> {qj.job_id}", flush=True)
    # 2. compiles: qnn_dlc is device-agnostic, one per (prec, L); pinned QAIRT
    opts = f"--target_runtime qnn_dlc --qairt_version {QAIRT}"
    cells = [("w8a8", L) for L in LENGTHS] + [(CONTROL[0], CONTROL[1])]
    for prec, L in cells:
        if find(st, kind="compile", L=L, prec=prec):
            continue
        if prec == "w8a8":
            q = find(st, kind="quantize", L=L)[-1]
            qj = hub.get_job(q["job_id"])
            print(f"waiting quantize L={L} {q['job_id']}", flush=True)
            qj.wait()
            if not qj.get_status().success:
                st["jobs"].append({"kind": "compile", "L": L, "prec": prec, "job_id": None,
                                   "error": f"quantize failed: {qj.get_status().message}"})
                save_state(st)
                print(f"quantize L={L} FAILED: {qj.get_status().message}", flush=True)
                continue
            model = qj.get_target_model()
        else:
            model = str(ROOT / f"bert_small_L{L}.onnx")
        dev = hub.Device(*DEVICES[CONTROL[2]])  # the compile's device only names the SoC family for a DLC
        cj = hub.submit_compile_job(model=model, device=dev, options=opts, name=f"v79gen_compile_{prec}_L{L}")
        st["jobs"].append({"kind": "compile", "L": L, "prec": prec, "options": opts, "job_id": cj.job_id})
        save_state(st)
        print(f"compile {prec} L={L} -> {cj.job_id}", flush=True)
    # 3. inference jobs: every w8a8 (L, device) + the control
    runs = [("w8a8", L, dk) for L in LENGTHS for dk in DEVICES] + [CONTROL]
    for prec, L, dk in runs:
        if find(st, kind="inference", L=L, prec=prec, device=dk):
            continue
        c = find(st, kind="compile", L=L, prec=prec)
        if not c or not c[-1]["job_id"]:
            continue
        cj = hub.get_job(c[-1]["job_id"])
        cj.wait()
        if not cj.get_status().success:
            st["jobs"].append({"kind": "inference", "L": L, "prec": prec, "device": dk, "job_id": None,
                               "error": f"compile failed: {cj.get_status().message}"})
            save_state(st)
            print(f"compile {prec} L={L} FAILED: {cj.get_status().message}", flush=True)
            continue
        d = np.load(ROOT / f"inputs_L{L}.npz")
        feeds = {"input_ids": [d["test"][k] for k in range(N_TEST)]}
        ij = hub.submit_inference_job(model=cj.get_target_model(), device=hub.Device(*DEVICES[dk]), inputs=feeds,
                                      name=f"v79gen_infer_{prec}_L{L}_{dk}")
        st["jobs"].append({"kind": "inference", "L": L, "prec": prec, "device": dk, "job_id": ij.job_id,
                           "from_compile": c[-1]["job_id"]})
        save_state(st)
        print(f"inference {prec} L={L} {dk} -> {ij.job_id}", flush=True)
    print("submit done", flush=True)


# ------------------------------------------------------------------ collect
def host_run(onnx_path, test, disable_opt):
    import onnxruntime as ort
    so = ort.SessionOptions()
    if disable_opt:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 8
    s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    return np.stack([s.run(None, {"input_ids": test[k]})[0][0] for k in range(test.shape[0])])  # (N, L, 512)


def qdq_model_path(qj, L):
    """The hub downloads a QDQ ONNX as a zip (model + external data); download() returns the real path."""
    import zipfile
    d = ROOT / f"qdq_L{L}"
    found = sorted(d.rglob("*.onnx")) if d.exists() else []
    if found:
        return found[0]
    d.mkdir(exist_ok=True)
    p = Path(qj.get_target_model().download(str(d / f"bert_small_L{L}_w8a8_qdq")))
    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as z:
            z.extractall(d)
        p.unlink()
    found = sorted(d.rglob("*.onnx"))
    if not found:
        raise SystemExit(f"no .onnx in {d}: {list(d.rglob('*'))[:5]}")
    return found[0]


def metrics(dev, ref):
    """dev, ref: (N, L, D). cosine per input over the flattened output; per position cosine (mean over inputs);
    argmax over channels agreement per token."""
    def cos(a, b, axis):
        num = (a * b).sum(axis)
        return num / (np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis) + 1e-12)
    per_input = cos(dev.reshape(dev.shape[0], -1), ref.reshape(ref.shape[0], -1), 1)
    per_pos = cos(dev, ref, 2).mean(0)  # (L,)
    agree_pos = (dev.argmax(-1) == ref.argmax(-1)).mean(0)  # (L,)
    L = dev.shape[1]
    blocks = []
    for b0 in range(0, L, 64):
        b1 = min(L, b0 + 64)
        blocks.append({"pos": [b0, b1 - 1], "cos": float(per_pos[b0:b1].mean()),
                       "argmax_agree": float(agree_pos[b0:b1].mean()),
                       "maxabs": float(np.abs(dev[:, b0:b1] - ref[:, b0:b1]).max())})
    head = per_pos[: L // 2].mean()
    tail = per_pos[L - min(128, L // 4):].mean()
    return {"cos_mean": float(per_input.mean()), "cos_min": float(per_input.min()),
            "argmax_agree": float(agree_pos.mean()), "maxabs": float(np.abs(dev - ref).max()),
            "cos_first_half": float(head), "cos_last_128": float(tail),
            "worst_block_cos": float(per_pos.reshape(-1)[:L].min()),
            "blocks": blocks}


def collect():
    import qai_hub as hub
    st = load_state()
    recdir = ROOT / "records"
    recdir.mkdir(exist_ok=True)
    summary = {"model": MODEL, "qairt": QAIRT, "n_test": N_TEST, "cells": {}}
    hosts = {}
    for L in LENGTHS:
        d = np.load(ROOT / f"inputs_L{L}.npz")
        test = d["test"]
        fp32 = host_run(ROOT / f"bert_small_L{L}.onnx", test, False)
        hosts[("fp32", L)] = fp32
        q = find(st, kind="quantize", L=L)
        if q:
            qj = hub.get_job(q[-1]["job_id"])
            qj.wait()
            if qj.get_status().success:
                qdq = qdq_model_path(qj, L)
                ref = host_run(qdq, test, True)
                hosts[("w8a8", L)] = ref
                summary["cells"][f"host_qdq_vs_fp32_L{L}"] = metrics(ref, fp32) | {"note": "host QDQ (ORT_DISABLE_ALL) vs host fp32: quantization noise bound"}
                print(f"L={L} host QDQ vs fp32: cos {summary['cells'][f'host_qdq_vs_fp32_L{L}']['cos_mean']:.4f} "
                      f"argmax {summary['cells'][f'host_qdq_vs_fp32_L{L}']['argmax_agree']:.3f}", flush=True)
    for j in find(st, kind="inference"):
        tag = f"{j['prec']}_L{j['L']}_{j['device']}"
        rec = dict(j)
        if not j["job_id"]:
            summary["cells"][tag] = {"error": j.get("error")}
            (recdir / f"{tag}.json").write_text(json.dumps(rec, indent=1))
            continue
        ij = hub.get_job(j["job_id"])
        ij.wait()
        stt = ij.get_status()
        rec["status"] = {"success": stt.success, "message": stt.message, "code": str(stt.code)}
        cj = hub.get_job(j["from_compile"]) if j.get("from_compile") else None
        if cj is not None:
            try:
                rec["compile_message"] = cj.get_status().message
            except Exception:
                pass
        if not stt.success:
            summary["cells"][tag] = {"error": stt.message, "job_id": j["job_id"]}
            (recdir / f"{tag}.json").write_text(json.dumps(rec, indent=1))
            print(f"{tag}: FAILED {stt.message}", flush=True)
            continue
        out = ij.download_output_data()
        key = [k for k in out][0]
        dev = np.stack([np.asarray(a)[0] for a in out[key]]).astype(np.float32)
        np.savez(recdir / f"{tag}.npz", dev=dev)
        ref = hosts[("w8a8", j["L"])] if j["prec"] == "w8a8" else hosts[("fp32", j["L"])]
        m = metrics(dev, ref)
        m["ref"] = "host QDQ ORT_DISABLE_ALL" if j["prec"] == "w8a8" else "host fp32 ORT"
        m["job_id"] = j["job_id"]
        m["from_compile"] = j.get("from_compile")
        m["device"] = DEVICES[j["device"]][0]
        m["hexagon"] = "v79" if j["device"] == "S25" else "v73"
        rec["metrics"] = m
        summary["cells"][tag] = m
        (recdir / f"{tag}.json").write_text(json.dumps(rec, indent=1))
        print(f"{tag}: cos {m['cos_mean']:.4f} (min {m['cos_min']:.4f}) argmax {m['argmax_agree']:.3f} "
              f"first-half cos {m['cos_first_half']:.4f} last-128 cos {m['cos_last_128']:.4f} maxabs {m['maxabs']:.3f}",
              flush=True)
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("collect done", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "submit", "collect"])
    a = ap.parse_args()
    {"build": build, "submit": submit, "collect": collect}[a.cmd]()


if __name__ == "__main__":
    main()
