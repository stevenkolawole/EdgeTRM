#!/usr/bin/env python3
"""Recompile the quantized TRM step against QNN instead of TF-Lite.

AI Hub defaults to TF-Lite, and both quantized variants failed there for
target-specific reasons rather than anything about the model:

  INT8: 'tfl.strided_slice' op quantization parameters violate the same scale
        constraint -- TF-Lite requires slice input and output to share a scale.
  INT4: QuantizeLinear ... cannot have output dtype 'int4' -- TF-Lite has no
        INT4 tensor type at all.

That second one matters for the paper directly. If the on-device INT4 numbers
were produced through a TF-Lite target, INT4 could not have executed as INT4,
which is exactly consistent with the observed INT4 step latency matching FP32
to three significant figures on all three chips.

The Hexagon NPU path is QNN, and the rebuttal states the original runs used
Qualcomm AI Hub's QNN execution provider. So this recompiles the same quantized
models against QNN and records what the compiler says. Either it succeeds, and
we get INT4 latency traceable to a job id, or it fails, and the paper drops its
INT4 latency rows rather than carrying a projection.
"""
import json
import pathlib
import sys

import qai_hub as hub

ART = pathlib.Path.home() / "aihub_artifacts"
DEVICES = [
    "Samsung Galaxy S24",
    "Samsung Galaxy S22 5G",
    "Dragonwing RB3 Gen 2 Vision Kit",
]
# qnn_context_binary is device-specific and gives the most faithful HTP result;
# qnn_dlc is the portable fallback if a device rejects the context binary.
RUNTIMES = ["qnn_context_binary", "qnn_dlc"]


def main():
    rec = json.loads((ART / "job_ids.json").read_text())
    quant_jobs = {j["precision"]: j["job_id"]
                  for j in rec["jobs"] if j["kind"] == "quantize"}
    if not quant_jobs:
        print("no quantize jobs recorded; run aihub_export.py --stage quantize first")
        return 1

    for prec, qid in sorted(quant_jobs.items()):
        qmodel = hub.get_job(qid).get_target_model()
        for runtime in RUNTIMES:
            ok_any = False
            for dname in DEVICES:
                cj = hub.submit_compile_job(
                    model=qmodel, device=hub.Device(dname),
                    options=f"--target_runtime {runtime}",
                    name=f"trm_step_{prec}_{runtime}_{dname}",
                )
                rec["jobs"].append({"kind": "compile", "precision": prec,
                                    "runtime": runtime, "device": dname,
                                    "job_id": cj.job_id})
                print(f"[qnn] {prec:4s} {runtime:20s} {dname:32s} -> {cj.job_id}")
                ok_any = True
            if ok_any:
                # Try the portable fallback only if the context binary route
                # turns out to fail; checked by aihub_status.py, not here.
                break

    (ART / "job_ids.json").write_text(json.dumps(rec, indent=2))
    print(f"[qnn] wrote {ART/'job_ids.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
