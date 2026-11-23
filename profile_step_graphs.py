"""Profile one loop iteration of DeepThinking and MDEQ-Small on every AI Hub device (per-step latency).

Same device list and compile targets as the TRM step table (aihub_h1.DEVICES / compile_opts):
  fp32   QNN (qnn_dlc, fp16 on the HTP) on Snapdragon devices with fp16 HTP; TFLite on Tensor/Exynos/older
  w8a16  one AI Hub quantize job per graph (real calibration states), then qnn_dlc --quantize_io on every
         Snapdragon device, integer-only HTPs included
Every job id is recorded in <root>/jobs.json, so the loop can be killed and restarted.

    python profile_step_graphs.py --root ~/aihub_steps advance   (submit what is ready; loop with --loop)
    python profile_step_graphs.py --root ~/aihub_steps table     (write <root>/step_latency.json)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

GRAPHS = ["dt_maze33_step", "mdeq_small_step"]
DEVICES = [  # (name, os, has QNN, fp16 HTP), as aihub_h1.DEVICES
    ("Samsung Galaxy S21", "12", True, False), ("Samsung Galaxy A73 5G", "12", True, False),
    ("Dragonwing RB3 Gen 2 Vision Kit", "1.6", True, False), ("SA8295P ADP", "14", True, True),
    ("Samsung Galaxy S22 5G", "12", True, True), ("Samsung Galaxy S23", "13", True, True),
    ("Snapdragon 7 Gen 4 QRD", "15", True, False), ("Dragonwing Q-6690 MTP", "15", True, False),
    ("QCS8550 (Proxy)", "12", True, True), ("Dragonwing IQ-9075 EVK", "1.9", True, True),
    ("SA8775P ADP", "14", True, True), ("Snapdragon X Elite CRD", "11", True, True),
    ("Snapdragon X Plus 8-Core CRD", "11", True, True), ("Samsung Galaxy S24", "14", True, True),
    ("SA7255P ADP", "14", True, True), ("Arduino VENTUNO Q", "24.04", True, True),
    ("Samsung Galaxy S25", "15", True, True), ("Snapdragon 8 Elite QRD", "15", True, True),
    ("Samsung Galaxy S26", "16", True, True), ("Snapdragon 8 Elite Gen 5 QRD", "16", True, True),
    ("Snapdragon X2 Elite CRD", "11", True, True),
    ("Google Pixel 3", "10", False, False), ("Google Pixel 3a", "10", False, False),
    ("Google Pixel 4", "11", False, False), ("Google Pixel 5", "12", False, False),
    ("Samsung Galaxy Tab S7", "11", False, False), ("Xiaomi Redmi Note 10 5G", "11", False, False),
    ("Google Pixel 6", "13", False, False), ("Google Pixel 7", "14", False, False),
    ("Google Pixel 8", "14", False, False), ("Google Pixel 9", "15", False, False),
    ("Google Pixel 10", "16", False, False), ("Samsung Galaxy Note 20 (Intl)", "11", False, False),
    ("Samsung Galaxy A53 5G", "12", False, False), ("Samsung Galaxy A14 5G", "13", False, False),
]


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load(root):
    p = root / "jobs.json"
    return json.loads(p.read_text()) if p.exists() else {"uploads": {}, "jobs": []}


def save(root, rec):
    (root / "jobs.json").write_text(json.dumps(rec, indent=1))


def find(rec, **kw):
    return [j for j in rec["jobs"] if all(j.get(k) == v for k, v in kw.items())]


def plan(prec):
    for d in DEVICES:
        name, os_, qnn, fp16 = d
        if prec == "fp32" and not qnn:
            yield d, "--target_runtime tflite", "tflite"
        elif prec == "fp32" and fp16:
            yield d, "--target_runtime qnn_dlc", "qnn_dlc"
        elif prec == "w8a16" and qnn:
            yield d, "--target_runtime qnn_dlc --quantize_io", "qnn_dlc_qio"


def advance(root):
    import qai_hub as hub
    rec = load(root)
    status = {}

    def st(jid):
        if jid not in status:
            status[jid] = hub.get_job(jid).get_status()
        return status[jid]

    for g in GRAPHS:
        if g not in rec["uploads"]:
            rec["uploads"][g] = hub.upload_model(str(root / "graphs" / f"{g}.onnx"), name=f"step_{g}_fp32").model_id
            save(root, rec)
        if not find(rec, kind="quantize", graph=g):
            cal = np.load(root / "graphs" / f"{g}_cal.npz")
            calib = {k: [cal[k][i] for i in range(cal[k].shape[0])] for k in cal.files}
            qj = hub.submit_quantize_job(model=hub.get_model(rec["uploads"][g]), calibration_data=calib,
                                         weights_dtype=hub.QuantizeDtype.INT8,
                                         activations_dtype=hub.QuantizeDtype.INT16, name=f"step_{g}_w8a16")
            rec["jobs"].append({"kind": "quantize", "graph": g, "prec": "w8a16", "job_id": qj.job_id, "at": now()})
            save(root, rec)
        sources = {"fp32": hub.get_model(rec["uploads"][g])}
        q = find(rec, kind="quantize", graph=g)[-1]
        if st(q["job_id"]).code == "SUCCESS":
            sources["w8a16"] = hub.get_job(q["job_id"]).get_target_model()
        for prec, model in sources.items():
            for (name, os_, _, _), opts, runtime in plan(prec):
                if find(rec, kind="compile", graph=g, prec=prec, device=name):
                    continue
                try:
                    cj = hub.submit_compile_job(model=model, device=hub.Device(name, os_), options=opts,
                                                name=f"step_{g}_{prec}_{name}")
                    jid, err = cj.job_id, None
                except Exception as exc:  # noqa: BLE001
                    jid, err = None, str(exc)[:300]
                rec["jobs"].append({"kind": "compile", "graph": g, "prec": prec, "device": name, "os": os_,
                                    "runtime": runtime, "job_id": jid, "submit_error": err, "at": now()})
                save(root, rec)
    for c in find(rec, kind="compile"):
        if not c["job_id"] or find(rec, kind="profile", from_compile=c["job_id"]):
            continue
        s = st(c["job_id"])
        if s.code != "SUCCESS":
            c["status"] = s.code
            continue
        pj = hub.submit_profile_job(model=hub.get_job(c["job_id"]).get_target_model(),
                                    device=hub.Device(c["device"], c["os"]),
                                    name=f"step_profile_{c['graph']}_{c['prec']}_{c['device']}")
        rec["jobs"].append({"kind": "profile", "graph": c["graph"], "prec": c["prec"], "device": c["device"],
                            "runtime": c["runtime"], "job_id": pj.job_id, "from_compile": c["job_id"], "at": now()})
        save(root, rec)
    for p in find(rec, kind="profile"):
        if p.get("status") in ("SUCCESS", "FAILED"):
            continue
        s = st(p["job_id"])
        p["status"] = s.code
        if s.code == "FAILED":
            p["error"] = (s.message or "")[:300]
        if s.code == "SUCCESS":
            prof = hub.get_job(p["job_id"]).download_profile()
            ex = prof["execution_summary"]
            times = ex.get("all_inference_times") or []
            p["ms_min"] = ex["estimated_inference_time"] / 1000.0
            p["ms_median"] = float(np.median(times)) / 1000.0 if times else None
            p["peak_mb"] = ex.get("estimated_inference_peak_memory", 0) / 2 ** 20
            units = {}
            for layer in prof.get("execution_detail", []):
                units[layer.get("compute_unit")] = units.get(layer.get("compute_unit"), 0) + 1
            p["units"] = units
        save(root, rec)
    todo = [j for j in rec["jobs"] if j["kind"] == "profile" and j.get("status") not in ("SUCCESS", "FAILED")]
    pending_c = [c for c in find(rec, kind="compile") if c["job_id"] and c.get("status") not in ("SUCCESS", "FAILED")
                 and not find(rec, kind="profile", from_compile=c["job_id"])]
    pending_q = [q for q in find(rec, kind="quantize") if st(q["job_id"]).code not in ("SUCCESS", "FAILED")]
    n_ok = len([j for j in rec["jobs"] if j["kind"] == "profile" and j.get("status") == "SUCCESS"])
    print(f"{now()} profiles done {n_ok}, profiles pending {len(todo)}, compiles pending {len(pending_c)}, "
          f"quantize pending {len(pending_q)}", flush=True)
    return bool(todo or pending_c or pending_q)


def table(root):
    rec = load(root)
    rows = [{k: j.get(k) for k in ("graph", "prec", "device", "runtime", "ms_min", "ms_median", "peak_mb", "units",
                                     "job_id")}
            for j in rec["jobs"] if j["kind"] == "profile" and j.get("status") == "SUCCESS"]
    fails = [{k: j.get(k) for k in ("kind", "graph", "prec", "device", "status", "error", "submit_error", "job_id")}
             for j in rec["jobs"] if j.get("status") == "FAILED" or j.get("submit_error")]
    (root / "step_latency.json").write_text(json.dumps({"rows": rows, "failed": fails}, indent=1))
    for r in sorted(rows, key=lambda r: (r["graph"], r["prec"], r["ms_min"])):
        print(f"{r['graph']:16s} {r['prec']:6s} {r['device'][:28]:28s} {r['runtime']:11s} {r['ms_min']:9.2f} ms")
    print(f"{len(rows)} profiles, {len(fails)} failures")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["advance", "table"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--loop", type=int, default=0, help="seconds between passes; 0 = one pass")
    a = ap.parse_args()
    root = Path(a.root).expanduser()
    if a.stage == "table":
        return table(root)
    while True:
        try:
            more = advance(root)
        except Exception as exc:  # noqa: BLE001  (API hiccup: retry next pass)
            print(f"{now()} pass error: {str(exc)[:200]}", flush=True)
            more = True
        if not a.loop or not more:
            break
        time.sleep(a.loop)
    print(f"{now()} DONE" if not a.loop or not more else "", flush=True)


if __name__ == "__main__":
    main()
