#!/usr/bin/env python3
"""Recompile the phones with quantized I/O, to explain the 6 MB vs 112 MB gap.

On the Galaxy S24 and S22 5G the QNN profiles report ~110 MB peak memory at
every precision, FP32 through INT4. On the Dragonwing RB3 the INT4 profile with
`--quantize_io` reports 6 MB. The RB3's HTP refuses any floating-point tensor,
so its graph was forced fully fixed-point; the phones' compiler was allowed to
keep FP16 activations and I/O buffers, which at seq=916 dominate the footprint.

If the phones also drop to single digits with `--quantize_io`, the ~110 MB was
a compile choice, not a property of the model, and the paper's 8 MB framing is
rescued on all three devices with measured numbers. If they stay at ~110 MB,
the 8 MB claim holds only on a fully fixed-point pipeline -- which is exactly
the MCU class the paper targets -- and the paper must say so.
"""
import json
import pathlib
import sys

import qai_hub as hub

ART = pathlib.Path.home() / "aihub_artifacts"
DEVICES = ["Samsung Galaxy S24", "Samsung Galaxy S22 5G"]
OPTS = "--target_runtime qnn_context_binary --quantize_io"


def main():
    rec = json.loads((ART / "job_ids.json").read_text())
    quant = {j["precision"]: j["job_id"] for j in rec["jobs"] if j["kind"] == "quantize"}
    jobs = []
    for prec, qid in sorted(quant.items()):
        model = hub.get_job(qid).get_target_model()
        for d in DEVICES:
            cj = hub.submit_compile_job(model=model, device=hub.Device(d), options=OPTS,
                                        name=f"trm_step_{prec}_qnn_qio_{d}")
            rec["jobs"].append({"kind": "compile", "precision": prec,
                                "runtime": "qnn_context_binary_qio", "device": d,
                                "job_id": cj.job_id})
            jobs.append((prec, d, cj))
            print(f"[qio] compile {prec} {d} -> {cj.job_id}", flush=True)
    (ART / "job_ids.json").write_text(json.dumps(rec, indent=2))
    for prec, d, cj in jobs:
        cj.wait()
        st = cj.get_status()
        if st.code != "SUCCESS":
            print(f"[qio] compile {prec} {d} FAILED: {(st.message or '')[:150]}", flush=True)
            continue
        pj = hub.submit_profile_job(model=cj.get_target_model(), device=hub.Device(d),
                                    name=f"trm_step_profile_{prec}_qnn_qio_{d}")
        rec["jobs"].append({"kind": "profile", "precision": prec,
                            "runtime": "qnn_context_binary_qio", "device": d,
                            "job_id": pj.job_id, "from_compile": cj.job_id})
        print(f"[qio] profile {prec} {d} -> {pj.job_id}", flush=True)
        (ART / "job_ids.json").write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
