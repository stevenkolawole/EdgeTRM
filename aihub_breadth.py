#!/usr/bin/env python3
"""Device breadth on Qualcomm AI Hub: one recursion step of every task on one
device per chipset the account can reach, at INT8 and INT4 (and FP32 on the
phones), reusing the quantize jobs already recorded in
~/aihub_artifacts/<task>/job_ids.json. Every job id is written to
~/aihub_artifacts/breadth/<task>_jobs.json so any number is traceable.

    ~/venvs/kvcache/bin/python aihub_breadth.py --stage submit [--tasks maze,sudoku_mlp,sudoku_attn]
    ~/venvs/kvcache/bin/python aihub_breadth.py --stage status

Qualcomm devices compile to a QNN context binary with quantized I/O (the
NPU path the paper reports); Google Tensor and Samsung Exynos phones have no
QNN, so they compile to TFLite (CPU/GPU delegate) and are reported as such.
"""
import argparse
import json
import pathlib
import time

import qai_hub as hub

ART = pathlib.Path.home() / "aihub_artifacts"
OUT = ART / "breadth"
QIO = "--target_runtime qnn_context_binary --quantize_io"
QNN = "--target_runtime qnn_context_binary"
TFL = "--target_runtime tflite"

# one device per chipset family; (device name, os, qualcomm?)
DEVICES = [
    ("Google Pixel 3", "10", True),            # Snapdragon 845
    ("Google Pixel 4", "11", True),            # 855
    ("Google Pixel 5", "12", True),            # 765G
    ("Samsung Galaxy Tab S7", "11", True),     # 865+
    ("Samsung Galaxy S21", "12", True),        # 888
    ("Samsung Galaxy A73 5G", "12", True),     # 778G
    ("Samsung Galaxy S22 5G", "13", True),     # 8 Gen 1
    ("Samsung Galaxy S23", "13", True),        # 8 Gen 2
    ("Samsung Galaxy S24", "14", True),        # 8 Gen 3
    ("Samsung Galaxy S25", "15", True),        # 8 Elite for Galaxy
    ("Samsung Galaxy S26", "16", True),        # 8 Elite Gen 5 for Galaxy
    ("Snapdragon 8 Elite QRD", "15", True),
    ("Snapdragon 7 Gen 4 QRD", "15", True),
    ("Snapdragon X Elite CRD", "11", True),    # laptop
    ("Snapdragon X Plus 8-Core CRD", "11", True),
    ("Snapdragon X2 Elite CRD", "11", True),
    ("SA8295P ADP", "14", True),               # automotive
    ("SA8775P ADP", "14", True),
    ("SA7255P ADP", "14", True),
    ("QCS8550 (Proxy)", "12", True),           # IoT
    ("Dragonwing RB3 Gen 2 Vision Kit", "1.6", True),   # QCS6490
    ("Dragonwing IQ-9075 EVK", "1.9", True),   # QCS9075
    ("Dragonwing Q-6690 MTP", "15", True),     # QCM6690
    ("Arduino VENTUNO Q", "24.04", True),      # QCS8275
    ("Google Pixel 6", "13", False),           # Tensor G1
    ("Google Pixel 7", "14", False),           # Tensor G2
    ("Google Pixel 8", "14", False),           # Tensor G3
    ("Google Pixel 9", "15", False),           # Tensor G4
    ("Google Pixel 10", "16", False),          # Tensor G5
    ("Samsung Galaxy Note 20 (Intl)", "11", False),  # Exynos 990
    ("Samsung Galaxy A53 5G", "12", False),    # Exynos 1280
    ("Samsung Galaxy A14 5G", "13", False),    # Exynos 1330
]
TASK_ART = {"maze": ART / "maze", "sudoku_mlp": ART / "sudoku_mlp", "sudoku_attn": ART / "sudoku_attn",
            "arc": ART}


def dev(name, os_):
    return hub.Device(name, os_)


def quantized_models(task):
    rec = json.loads((TASK_ART[task] / "job_ids.json").read_text())
    out = {}
    for j in rec["jobs"]:
        if j["kind"] == "quantize":
            job = hub.get_job(j["job_id"])
            if job.get_status().code == "SUCCESS":
                out[j["precision"]] = job.get_target_model()
    return out


def onnx_path(task):
    p = TASK_ART[task] / ("trm_inner_step.onnx" if task == "arc" else "inner_step.onnx")
    return str(p)


def stage_submit(tasks, devices, do_fp32):
    OUT.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        rec_path = OUT / f"{task}_jobs.json"
        rec = json.loads(rec_path.read_text()) if rec_path.exists() else {"task": task, "jobs": []}
        have = {(j["precision"], j["device"], j["kind"]) for j in rec["jobs"]}
        qms = quantized_models(task)
        print(f"[{task}] quantized models: {list(qms)}", flush=True)

        def save():
            rec_path.write_text(json.dumps(rec, indent=2))

        for name, os_, is_q in devices:
            variants = [(p, qms[p]) for p in ("int8", "int4") if p in qms]
            if do_fp32:
                variants.append(("fp32", onnx_path(task)))
            for prec, model in variants:
                if (prec, name, "compile") in have:
                    continue
                if is_q:
                    opts = QNN if prec == "fp32" else QIO
                    runtime = "qnn_context_binary" + ("" if prec == "fp32" else "_qio")
                else:
                    opts, runtime = TFL, "tflite"
                try:
                    cj = hub.submit_compile_job(model=model, device=dev(name, os_), options=opts,
                                                name=f"breadth_{task}_{prec}_{name}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[{task}] compile {prec} {name}: submit failed: {str(exc)[:120]}", flush=True)
                    rec["jobs"].append({"kind": "compile", "precision": prec, "runtime": runtime, "device": name,
                                        "os": os_, "job_id": None, "error": str(exc)[:200]})
                    save()
                    continue
                rec["jobs"].append({"kind": "compile", "precision": prec, "runtime": runtime, "device": name,
                                    "os": os_, "job_id": cj.job_id})
                print(f"[{task}] compile {prec:5s} {name:32s} -> {cj.job_id}", flush=True)
                save()
        print(f"[{task}] submitted; run --stage profile once compiles finish", flush=True)


def stage_profile(tasks):
    for task in tasks:
        rec_path = OUT / f"{task}_jobs.json"
        rec = json.loads(rec_path.read_text())
        done = {(j["precision"], j["device"]) for j in rec["jobs"] if j["kind"] == "profile"}
        for j in list(rec["jobs"]):
            if j["kind"] != "compile" or not j.get("job_id") or (j["precision"], j["device"]) in done:
                continue
            job = hub.get_job(j["job_id"])
            st = job.get_status()
            if st.code != "SUCCESS":
                if st.code == "FAILED":
                    j["error"] = (st.message or "")[:200]
                continue
            try:
                pj = hub.submit_profile_job(model=job.get_target_model(), device=dev(j["device"], j["os"]),
                                            name=f"breadth_profile_{task}_{j['precision']}_{j['device']}")
            except Exception as exc:  # noqa: BLE001
                print(f"[{task}] profile {j['precision']} {j['device']}: {str(exc)[:120]}", flush=True)
                continue
            rec["jobs"].append({"kind": "profile", "precision": j["precision"], "runtime": j["runtime"],
                                "device": j["device"], "os": j["os"], "job_id": pj.job_id,
                                "from_compile": j["job_id"]})
            print(f"[{task}] profile {j['precision']:5s} {j['device']:32s} -> {pj.job_id}", flush=True)
            rec_path.write_text(json.dumps(rec, indent=2))
        rec_path.write_text(json.dumps(rec, indent=2))


def stage_status(tasks):
    rows = []
    for task in tasks:
        rec_path = OUT / f"{task}_jobs.json"
        if not rec_path.exists():
            continue
        rec = json.loads(rec_path.read_text())
        for j in rec["jobs"]:
            if not j.get("job_id"):
                continue
            job = hub.get_job(j["job_id"])
            st = job.get_status()
            row = {"task": task, **{k: j[k] for k in ("kind", "precision", "runtime", "device", "job_id")},
                   "status": st.code}
            if j["kind"] == "profile" and st.code == "SUCCESS":
                p = job.download_profile()
                s = p["execution_summary"]
                units = {}
                for l in p.get("execution_detail", []):
                    units[l.get("compute_unit", "?")] = units.get(l.get("compute_unit", "?"), 0) + 1
                row.update({"ms": s["estimated_inference_time"] / 1000,
                            "load_peak_mb": s["first_load_peak_memory"] / 2 ** 20,
                            "infer_peak_mb": s["inference_memory_peak_range"][1] / 2 ** 20,
                            "peak_mb": s["estimated_inference_peak_memory"] / 2 ** 20, "units": units})
            elif st.code == "FAILED":
                row["error"] = (st.message or "")[:160]
            rows.append(row)
            print(f"{task:11s} {j['kind']:7s} {j['precision']:5s} {j['device'][:30]:30s} {st.code:9s} "
                  f"{row.get('ms', ''):>8} {row.get('peak_mb', '')}", flush=True)
    (OUT / "status.json").write_text(json.dumps(rows, indent=2))
    print("wrote", OUT / "status.json", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["submit", "profile", "status"], required=True)
    ap.add_argument("--tasks", default="maze,sudoku_mlp,sudoku_attn")
    ap.add_argument("--fp32", action="store_true", help="also compile/profile FP32 (QNN on Qualcomm)")
    a = ap.parse_args()
    tasks = [t for t in a.tasks.split(",") if t]
    if a.stage == "submit":
        stage_submit(tasks, DEVICES, a.fp32)
    elif a.stage == "profile":
        stage_profile(tasks)
    else:
        stage_status(tasks)


if __name__ == "__main__":
    main()
