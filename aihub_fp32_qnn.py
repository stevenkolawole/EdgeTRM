#!/usr/bin/env python3
"""Compile and profile the FP32 step on QNN, so the INT4 comparison is same-target.

INT4 on the S24 came back at 58.70 ms via QNN against 144.22 ms for FP32 --
but the FP32 number was measured on the TF-Lite target. A 2.5x gap across two
different runtimes is suggestive, not decisive. This puts FP32 on the same QNN
context-binary path so the only thing that differs between the rows is
precision. Profile jobs are submitted automatically once each compile lands.
"""
import json
import pathlib
import sys
import time

import qai_hub as hub

ART = pathlib.Path.home() / "aihub_artifacts"
DEVICES = ["Samsung Galaxy S24", "Samsung Galaxy S22 5G", "Dragonwing RB3 Gen 2 Vision Kit"]


def main():
    rec = json.loads((ART / "job_ids.json").read_text())
    onnx = ART / "trm_inner_step.onnx"
    compiles = []
    for d in DEVICES:
        cj = hub.submit_compile_job(model=str(onnx), device=hub.Device(d),
                                    options="--target_runtime qnn_context_binary",
                                    name=f"trm_step_fp32_qnn_{d}")
        rec["jobs"].append({"kind": "compile", "precision": "fp32",
                            "runtime": "qnn_context_binary", "device": d,
                            "job_id": cj.job_id})
        compiles.append((d, cj))
        print(f"[fp32-qnn] compile {d:32s} -> {cj.job_id}")
    (ART / "job_ids.json").write_text(json.dumps(rec, indent=2))

    # Wait for compiles and chain profiles; a compile takes a few minutes.
    for d, cj in compiles:
        cj.wait()
        if cj.get_status().code != "SUCCESS":
            print(f"[fp32-qnn] compile FAILED {d}: {cj.get_status().message[:120]}")
            continue
        pj = hub.submit_profile_job(model=cj.get_target_model(), device=hub.Device(d),
                                    name=f"trm_step_profile_fp32_qnn_{d}")
        rec["jobs"].append({"kind": "profile", "precision": "fp32",
                            "runtime": "qnn_context_binary", "device": d,
                            "job_id": pj.job_id, "from_compile": cj.job_id})
        print(f"[fp32-qnn] profile {d:32s} -> {pj.job_id}")
        (ART / "job_ids.json").write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
