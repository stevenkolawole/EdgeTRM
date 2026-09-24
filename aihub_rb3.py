#!/usr/bin/env python3
"""Compile and profile on the Dragonwing RB3 with quantized I/O.

QCS6490 refuses any floating-point graph input on the QNN path, and the
streamed puzzle embedding is fp32. `--quantize_io` folds the input/output
quantization into the graph so the device sees only fixed-point tensors. This
is the last device missing from the same-target table; FP32 on the RB3 is
already known to run on the GPU, so the interesting rows are INT8 and INT4.
"""
import json
import pathlib
import sys

import qai_hub as hub

ART = pathlib.Path.home() / "aihub_artifacts"
DEV = "Dragonwing RB3 Gen 2 Vision Kit"
OPTS = "--target_runtime qnn_context_binary --quantize_io"


def main():
    rec = json.loads((ART / "job_ids.json").read_text())
    quant = {j["precision"]: j["job_id"] for j in rec["jobs"] if j["kind"] == "quantize"}
    dev = hub.Device(DEV)
    jobs = []
    for prec, qid in sorted(quant.items()):
        model = hub.get_job(qid).get_target_model()
        cj = hub.submit_compile_job(model=model, device=dev, options=OPTS,
                                    name=f"trm_step_{prec}_qnn_qio_{DEV}")
        rec["jobs"].append({"kind": "compile", "precision": prec,
                            "runtime": "qnn_context_binary_qio", "device": DEV,
                            "job_id": cj.job_id})
        jobs.append((prec, cj))
        print(f"[rb3] compile {prec} -> {cj.job_id}")
    (ART / "job_ids.json").write_text(json.dumps(rec, indent=2))
    for prec, cj in jobs:
        cj.wait()
        st = cj.get_status()
        if st.code != "SUCCESS":
            print(f"[rb3] compile {prec} FAILED: {(st.message or '')[:150]}")
            continue
        pj = hub.submit_profile_job(model=cj.get_target_model(), device=dev,
                                    name=f"trm_step_profile_{prec}_qnn_qio_{DEV}")
        rec["jobs"].append({"kind": "profile", "precision": prec,
                            "runtime": "qnn_context_binary_qio", "device": DEV,
                            "job_id": pj.job_id, "from_compile": cj.job_id})
        print(f"[rb3] profile {prec} -> {pj.job_id}")
        (ART / "job_ids.json").write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
