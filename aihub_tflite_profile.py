#!/usr/bin/env python3
"""Profile our own TFLite flatbuffers (the MCU-path INT8 and W4A8 graphs from ~/mcu/<task>/) on the
devices where AI Hub's ONNX->TFLite conversion failed: the non-Qualcomm phones (Tensor, Exynos) and
the integer-only Hexagon parts. A profile job takes a .tflite directly, so the converter is bypassed.
Job ids go to ~/aihub_artifacts/breadth/tflite_jobs.json in the same record shape as the other tasks,
so `aihub_breadth.py --stage status --tasks ...,tflite` reads them.

    ~/venvs/kvcache/bin/python aihub_tflite_profile.py [--precisions int8,w4a8] [--tasks maze,sudoku_mlp,sudoku_attn]
"""
import argparse
import json
import pathlib

import qai_hub as hub

ART = pathlib.Path.home() / "aihub_artifacts"
MCU = pathlib.Path.home() / "mcu"
OUT = ART / "breadth" / "tflite_jobs.json"
DEVICES = [  # TFLite-only targets and the integer-only Hexagon parts (int8/fp32 QNN compiles failed there)
    ("Google Pixel 6", "13"), ("Google Pixel 7", "14"), ("Google Pixel 8", "14"), ("Google Pixel 9", "15"),
    ("Google Pixel 10", "16"), ("Samsung Galaxy Note 20 (Intl)", "11"), ("Samsung Galaxy A53 5G", "12"),
    ("Samsung Galaxy A14 5G", "13"),
    ("Samsung Galaxy S21", "12"), ("Samsung Galaxy A73 5G", "12"), ("Snapdragon 7 Gen 4 QRD", "15"),
    ("Dragonwing RB3 Gen 2 Vision Kit", "1.6"), ("Dragonwing Q-6690 MTP", "15"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="maze,sudoku_mlp,sudoku_attn")
    ap.add_argument("--precisions", default="int8,w4a8")
    a = ap.parse_args()
    rec = {"jobs": []}
    if OUT.exists():
        rec = json.loads(OUT.read_text())
    done = {(j["task"], j["precision"], j["device"]) for j in rec["jobs"]}
    uploaded = {}
    for task in a.tasks.split(","):
        for prec in a.precisions.split(","):
            path = MCU / task / f"inner_step_{prec}.tflite"
            if not path.exists():
                print(f"[{task}] no {path.name}", flush=True)
                continue
            for dev, os_ in DEVICES:
                if (task, prec, dev) in done:
                    continue
                if (task, prec) not in uploaded:
                    uploaded[(task, prec)] = hub.upload_model(str(path), name=f"breadth_tflite_{task}_{prec}")
                try:
                    pj = hub.submit_profile_job(model=uploaded[(task, prec)], device=hub.Device(dev, os_),
                                                name=f"breadth_tflite_profile_{task}_{prec}_{dev}")
                except Exception as exc:  # noqa: BLE001
                    print(f"[{task}] {prec} {dev}: {str(exc)[:120]}", flush=True)
                    continue
                rec["jobs"].append({"kind": "profile", "task": task, "precision": prec + "_tflite", "runtime": "tflite_direct",
                                    "device": dev, "os": os_, "job_id": pj.job_id, "model": path.name})
                print(f"[{task}] {prec:5s} {dev:32s} -> {pj.job_id}", flush=True)
                OUT.write_text(json.dumps(rec, indent=2))
    OUT.write_text(json.dumps(rec, indent=2))
    print("recorded", len(rec["jobs"]), "jobs in", OUT)


if __name__ == "__main__":
    main()
