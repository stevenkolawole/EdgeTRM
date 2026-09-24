#!/usr/bin/env python3
"""Profile every successfully compiled TRM variant on real devices.

This is the measurement the paper needs and does not have. The published INT4
step latency matched FP32 to three significant figures on all three chips, which
is what a compiler folding INT4 to FP16 looks like, and there were no job ids to
check it against.

We now know part of the answer already: INT4 cannot compile to TF-Lite at all
(no INT4 tensor type), and both quantized variants compile fine against QNN on
the two phones. If the QNN INT4 profile also matches FP32, the fold hypothesis
survives the change of target and the paper should drop its INT4 latency rows.
If INT4 is meaningfully faster, the original numbers were a target artifact and
we can report real ones with provenance.

Every job id is recorded so any number that reaches the paper can be traced.
"""
import json
import pathlib
import sys

import qai_hub as hub

ART = pathlib.Path.home() / "aihub_artifacts"


def main():
    rec = json.loads((ART / "job_ids.json").read_text())
    already = {(j["device"], j["precision"], j.get("runtime", "tflite"))
               for j in rec["jobs"] if j["kind"] == "profile"}

    submitted = 0
    for j in list(rec["jobs"]):
        if j["kind"] != "compile":
            continue
        key = (j["device"], j["precision"], j.get("runtime", "tflite"))
        if key in already:
            continue
        job = hub.get_job(j["job_id"])
        if job.get_status().code != "SUCCESS":
            continue
        try:
            model = job.get_target_model()
        except Exception as exc:  # noqa: BLE001
            print(f"[profile] skip {key}: {exc}")
            continue
        pj = hub.submit_profile_job(
            model=model, device=hub.Device(j["device"]),
            name=f'trm_step_profile_{j["precision"]}_{j["device"]}',
        )
        rec["jobs"].append({"kind": "profile", "precision": j["precision"],
                            "runtime": j.get("runtime", "tflite"),
                            "device": j["device"], "job_id": pj.job_id,
                            "from_compile": j["job_id"]})
        print(f'[profile] {j["precision"]:5s} {j["device"][:28]:30s} -> {pj.job_id}')
        submitted += 1

    (ART / "job_ids.json").write_text(json.dumps(rec, indent=2))
    print(f"[profile] submitted {submitted}; wrote {ART/'job_ids.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
