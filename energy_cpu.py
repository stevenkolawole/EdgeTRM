#!/usr/bin/env python3
"""Measured energy per recursion step on a laptop SoC CPU (AMD Ryzen 7 PRO 7840U, Zen 4, 8 cores),
from the SoC's own package-power telemetry, with idle subtraction.

Sensor: amdgpu hwmon `power1_input` labelled PPT (package power tracking, microwatts), reported by
the SMU from the voltage-regulator telemetry for the whole APU socket (CPU cores + iGPU + NPU + SoC).
It is readable without root, unlike RAPL `energy_uj`. It is a package-level reading, so the number
includes whatever else the machine was doing; the protocol below brackets every active window with
idle windows and reports both the total and the idle-subtracted (incremental) energy.

Protocol (MLPerf-Tiny/Power style, adapted): per model, `reps` repetitions of
[idle `idle_s`] [active `active_s` of back-to-back single-sample inferences] [idle `idle_s`];
PPT sampled every 20 ms in a thread; first 1 s of each active window dropped (ramp).
  P_active, P_idle (mean of the two bracketing idle windows), t_step (active time / inferences)
  E_total = P_active x t_step,  E_incr = (P_active - P_idle) x t_step
Reported per repetition and as median over repetitions.

    python energy_cpu.py --models sudoku_mlp/step_H1.onnx,... --threads 8 --out energy.json
"""
import argparse
import json
import pathlib
import platform
import statistics
import threading
import time

import numpy as np
import onnxruntime as ort

PPT = None
for h in pathlib.Path("/sys/class/hwmon").iterdir():
    try:
        if (h / "name").read_text().strip() == "amdgpu" and (h / "power1_label").read_text().strip() == "PPT":
            PPT = h / "power1_input"
    except OSError:
        pass


class Sampler(threading.Thread):
    def __init__(self, period=0.02):
        super().__init__(daemon=True)
        self.period, self.samples, self.stop_flag = period, [], False

    def run(self):
        while not self.stop_flag:
            try:
                self.samples.append((time.perf_counter(), int(PPT.read_text()) / 1e6))
            except OSError:
                pass
            time.sleep(self.period)


def mean_power(samples, t0, t1):
    v = [p for t, p in samples if t0 <= t < t1]
    return float(np.mean(v)) if v else float("nan"), len(v)


def feeds_for(sess, calib=None):
    feed = {}
    rng = np.random.default_rng(0)
    for i in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in i.shape]
        if calib is not None and i.name in calib:
            feed[i.name] = calib[i.name][0]
        elif "int" in i.type:
            feed[i.name] = rng.integers(1, 6, shape).astype(np.int32)
        else:
            feed[i.name] = rng.normal(0, 1, shape).astype(np.float32)
    return feed


def measure(path, threads, reps, idle_s, active_s, calib):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    feed = feeds_for(sess, calib)
    outs = [o.name for o in sess.get_outputs()]
    for _ in range(3):
        sess.run(outs, feed)
    smp = Sampler()
    smp.start()
    rows = []
    time.sleep(idle_s)
    for r in range(reps):
        i0 = time.perf_counter() - idle_s
        a0 = time.perf_counter()
        n = 0
        while time.perf_counter() - a0 < active_s:
            sess.run(outs, feed)
            n += 1
        a1 = time.perf_counter()
        time.sleep(idle_s)
        i1 = time.perf_counter()
        p_act, n_act = mean_power(smp.samples, a0 + 1.0, a1)
        p_idle_pre, _ = mean_power(smp.samples, i0, a0)
        p_idle_post, _ = mean_power(smp.samples, a1 + 1.0, i1)
        p_idle = float(np.nanmean([p_idle_pre, p_idle_post]))
        t_step = (a1 - a0) / n
        rows.append({"rep": r, "inferences": n, "t_step_ms": t_step * 1e3, "p_active_w": p_act,
                     "p_idle_pre_w": p_idle_pre, "p_idle_post_w": p_idle_post, "p_idle_w": p_idle,
                     "e_total_mj": p_act * t_step * 1e3, "e_incr_mj": (p_act - p_idle) * t_step * 1e3,
                     "power_samples_active": n_act})
        print(f"  rep {r}: {n} inf, {t_step*1e3:.2f} ms/step, P_act {p_act:.2f} W, P_idle {p_idle:.2f} W "
              f"-> E_total {p_act*t_step*1e3:.2f} mJ, E_incr {(p_act-p_idle)*t_step*1e3:.2f} mJ", flush=True)
    smp.stop_flag = True
    med = {k: statistics.median([r[k] for r in rows]) for k in ("t_step_ms", "p_active_w", "p_idle_w",
                                                                 "e_total_mj", "e_incr_mj")}
    return {"model": path, "threads": threads, "reps": rows, "median": med,
            "spread_e_incr_mj": [min(r["e_incr_mj"] for r in rows), max(r["e_incr_mj"] for r in rows)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--calib", default="", help="comma list of calib npz aligned with --models (optional)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--idle_s", type=float, default=15)
    ap.add_argument("--active_s", type=float, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert PPT is not None, "no amdgpu PPT sensor"
    res = {"host": platform.node(), "cpu": "AMD Ryzen 7 PRO 7840U", "sensor": str(PPT), "ort": ort.__version__,
           "protocol": {"reps": a.reps, "idle_s": a.idle_s, "active_s": a.active_s, "sample_period_s": 0.02},
           "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    for f in ("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", "/sys/firmware/acpi/platform_profile",
              "/sys/class/power_supply/AC/online"):
        try:
            res[f.split("/")[-1]] = pathlib.Path(f).read_text().strip()
        except OSError:
            pass
    cals = a.calib.split(",") if a.calib else []
    for i, m in enumerate(a.models.split(",")):
        calib = None
        if i < len(cals) and cals[i]:
            d = np.load(cals[i])
            calib = {k: d[k] for k in d.files}
        print(f"[energy] {m}", flush=True)
        res["runs"].append(measure(m, a.threads, a.reps, a.idle_s, a.active_s, calib))
        pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
