#!/usr/bin/env python3
"""Poll the AI Hub jobs recorded by aihub_export.py and dump their reports.

The compile report is the point, not the latency. The paper's INT4 step latency
matched FP32 to three significant figures on all three chips, which is what a
compiler folding INT4 to FP16 looks like. The report says which precision the
ops actually landed in, so it settles whether the INT4 rows measure INT4.
"""
import json
import pathlib
import sys

import qai_hub as hub

ART = pathlib.Path.home() / "aihub_artifacts"


def main():
    rec = json.loads((ART / "job_ids.json").read_text())
    out = []
    for j in rec["jobs"]:
        job = hub.get_job(j["job_id"])
        st = job.get_status()
        row = {**j, "state": st.code, "message": getattr(st, "message", "") or ""}
        if st.code == "SUCCESS" and j["kind"] == "compile":
            try:
                tm = job.get_target_model()
                row["target_model"] = str(getattr(tm, "name", tm))
            except Exception as e:  # noqa: BLE001
                row["target_model_error"] = str(e)[:200]
        if st.code == "SUCCESS" and j["kind"] == "profile":
            # The number the paper needs, traceable to this job id. Inference
            # time is reported in microseconds; peak memory in bytes.
            try:
                prof = job.download_profile()
                es = prof.get("execution_summary", {})
                row["inference_ms"] = es.get("estimated_inference_time", 0) / 1000.0
                row["peak_mem_mb"] = es.get("estimated_inference_peak_memory", 0) / 1e6
                # Which compute unit ran the layers: NPU vs GPU vs CPU split is
                # exactly the FP16-fold question.
                units = {}
                for layer in prof.get("execution_detail", []):
                    u = layer.get("compute_unit", "?")
                    units[u] = units.get(u, 0) + 1
                row["compute_units"] = units
                row["message"] = (f'{row["inference_ms"]:.2f} ms, '
                                  f'{row["peak_mem_mb"]:.0f} MB, units={units}')
            except Exception as e:  # noqa: BLE001
                row["profile_error"] = str(e)[:200]
        out.append(row)
        dev = row["device"][:26]
        rt = row.get("runtime", "tflite")
        msg = row["message"].replace("\n", " ")[:120]
        print(f'{dev:28s} {row["kind"]:8s} {row["precision"]:5s} {rt:20s} '
              f'{row["state"]:9s} {msg}')
    (ART / "job_status.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {ART/'job_status.json'}")
    return 0 if all(r["state"] in ("SUCCESS", "RUNNING", "PENDING") for r in out) else 1


if __name__ == "__main__":
    sys.exit(main())
