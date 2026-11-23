"""Accuracy reachable within a deadline on each phone, for one set of weights.

Joins two measured records:
  - per-step exact accuracy of the deployable H=1 carry step, simulated on the host from the
    compiled QDQ graph (results/devices/device_faithful_acc.json, test rows 0-99);
  - per-step latency of the same H=1 step graph on each AI Hub device
    (results/devices/latency/e5_latency_table.json, hub estimated inference time).
Steps reachable = floor(deadline / ms per step), capped at the longest measured step.
Latency is kernel time per step: per-step dispatch and carry copy are not included.

Also reports, for every on-device chain with a host curve of the same graph and precision,
the mean absolute gap between device and host exact accuracy across steps.

DeepThinking and MDEQ rows use the fp32 step-graph latencies of results/devices/step_latency (profile_step_graphs.py)
with the host fp32 iteration curves; on Snapdragon NPUs that graph runs in fp16, whose accuracy is not measured on device.

Usage: python3 deadline_accuracy.py  (writes results/devices/deadline_accuracy.json)
"""
import json
from collections import defaultdict
from pathlib import Path

DEV = Path(__file__).resolve().parent / "results" / "devices"
DEADLINES_MS = (50, 100, 250, 500, 1000)
SERIES = [("sudoku_mlp", "fp32"), ("sudoku_mlp", "w8a16"), ("sudoku_attn", "w8a16"),
          ("maze", "fp32"), ("maze", "w8a16")]


def iteration_curves():
    """Host fp32 accuracy against iteration count for the DeepThinking and MDEQ step graphs.

    DeepThinking 33x33 mazes: n = 1000 at 10-100 iterations (results/breadth/dt_mazes.json) and the Jetson Nano's
    fixed-300 run (n = 50, edge/results/nano-a). MDEQ-Small ImageNet top-1 from z = 0: the fp32 row of
    results/breadth/m1_mdeq.md (images 0-2499). Accuracy at k iterations is read as the value at the largest
    measured count <= k.
    """
    root = DEV.parent
    dt = {}
    for r in json.loads((root / "breadth" / "dt_mazes.json").read_text())["records"]:
        if str(r["size"]) == "33" and r["quant"].get("spec") == "fp32":
            dt[r["iters"]] = r["acc"]
    nano = json.loads((root.parent / "edge" / "results" / "nano-a" / "nano-a_dt_fixed300_maxn.json").read_text())
    dt[300] = nano["windows"]["trt_fp32_b1_fixed300"]["median_clean"]["acc_window"]
    lines = (root / "breadth" / "m1_mdeq.md").read_text().splitlines()
    head = next(l for l in lines if l.startswith("| format |"))
    row = next(l for l in lines if l.startswith("| fp32 |"))
    iters = [int(c) for c in head.strip("|").split("|")[1:]]
    top1 = [float(c) / 100 for c in row.strip("|").split("|")[1:]]
    return {"dt_maze33_step": dict(sorted(dt.items())), "mdeq_small_step": dict(zip(iters, top1))}


def board_rows():
    """Per-step time of the same weights on the boards and the simulated microcontroller NPU, per series.

    Sudoku-MLP boards ran the H=3 full step; ms per H=1 step = full step / 3 (the H=3/H=1 ratio is 3.00 on the Ethos-U55
    FVP and 2.8-3.0 on 40 phone pairs). Ethos-U rows are INT8 H=1 graphs: FVP NPU cycles for U55-256, Vela estimates
    for U65 and U85, at 500 MHz; TFLite Micro numerics for this graph are broken, so these rows give time only.
    """
    edge = DEV.parent.parent / "edge" / "results"

    def rec(path, window=None):
        d = json.loads((edge / path).read_text())
        return d["windows"][window]["median_clean"]["ms_per_step"] if window else d["record"]["step_ms"]["median"]

    mcu = {}
    for line in (DEV / "mcu_h1.md").read_text().splitlines():
        c = [x.strip() for x in line.strip("|").split("|")]
        if len(c) == 11 and c[0] == "sudoku_mlp" and c[1] == "1":
            mcu[c[2]] = float(c[10]) if c[10] != "-" else float(c[6])
    sudoku = [("Jetson AGX Xavier (TensorRT fp16)", "board", rec("agx/agx_sudoku_mlp_fp16_fixed16_maxn.json") / 3),
              ("Jetson Nano (TensorRT fp32)", "board", rec("nano-a/nano-a_sudoku_mlp_fp32_fixed16_maxn_fullset.json") / 3),
              ("Raspberry Pi 4 (ONNX Runtime fp32)", "board", rec("pi-a/pi-a_sudoku_mlp_H3_fp32_fixed16_nopower.json") / 3)]
    sudoku += [(f"Arm Ethos-{k} (simulated, INT8, time only)", "mcu", v) for k, v in mcu.items()]
    dt = [("Jetson Nano (TensorRT fp32)", "board", rec("nano-a/nano-a_dt_fixed300_maxn.json")),
          ("Jetson AGX Xavier (PyTorch fp32)", "board",
           json.loads((edge / "agx_family" / "dt" / "energy.json").read_text())["windows"]["fixed_T100"]["median"]["ms_per_step"])]
    return {"sudoku_mlp/fp32": sudoku, "dt_maze33_step/fp32": dt}


def reach_at(curve, ms):
    reach = {}
    for d in DEADLINES_MS:
        k = int(d // ms)
        done = [n for n in curve if n <= k]
        reach[d] = {"steps": k, "exact": curve[max(done)] if done else 0.0}
    return reach


def main():
    acc = json.loads((DEV / "device_faithful_acc.json").read_text())
    host, dev = defaultdict(dict), defaultdict(dict)
    for r in acc:
        if r["source"].startswith("host"):
            host[(r["task"], r["prec"])].setdefault(r["step"], r["exact"])
        elif r["source"].startswith("ON DEVICE"):
            dev[(r["source"][len("ON DEVICE "):], r["task"], r["prec"])][r["step"]] = r["exact"]

    table = json.loads((DEV / "latency" / "e5_latency_table.json").read_text())["table1"]
    out = {"deadlines_ms": DEADLINES_MS, "series": {}, "device_vs_host": []}
    for task, prec in SERIES:
        curve = host[(task, prec)]
        last = max(curve)
        rows, seen = [], set()
        for r in sorted((r for r in table[f"{task}/{prec}"] if r.get("ms")), key=lambda r: r["ms"]):
            if r["device"] in seen:
                continue
            seen.add(r["device"])
            reach = {}
            for d in DEADLINES_MS:
                k = min(int(d // r["ms"]), last)
                reach[d] = {"steps": k, "exact": curve.get(k, 0.0)}
            rows.append({"device": r["device"], "chip": r["chip"], "runtime": r["runtime"],
                         "ms_per_step": r["ms"], "reach": reach})
        out["series"][f"{task}/{prec}"] = {"host_curve": curve, "devices": rows}

    lat = json.loads((DEV / "step_latency" / "step_latency.json").read_text())["rows"]
    for graph, curve in iteration_curves().items():
        rows = []
        for r in sorted((r for r in lat if r["graph"] == graph and r["prec"] == "fp32"), key=lambda r: r["ms_min"]):
            reach = {}
            for d in DEADLINES_MS:
                k = int(d // r["ms_min"])
                done = [n for n in curve if n <= k]
                reach[d] = {"steps": k, "exact": curve[max(done)] if done else 0.0}
            rows.append({"device": r["device"], "runtime": r["runtime"], "ms_per_step": r["ms_min"], "reach": reach})
        out["series"][f"{graph}/fp32"] = {"host_curve": curve, "devices": rows}

    for key, rows in board_rows().items():
        curve = out["series"][key]["host_curve"]
        for name, tier, ms in rows:
            out["series"][key]["devices"].append({"device": name, "tier": tier, "ms_per_step": round(ms, 2),
                                                  "reach": reach_at(curve, ms)})

    for (name, task, prec), c in sorted(dev.items()):
        h = host.get((task, prec))
        common = [s for s in c if h and s in h]
        if not common:
            continue
        gap = sum(abs(c[s] - h[s]) for s in common) / len(common)
        out["device_vs_host"].append({"device": name, "task": task, "prec": prec,
                                      "steps": len(common), "mean_abs_gap": round(gap, 4),
                                      "device_last": c[max(common)], "host_last": h[max(common)]})

    path = DEV / "deadline_accuracy.json"
    path.write_text(json.dumps(out, indent=1))
    gaps = [g["mean_abs_gap"] for g in out["device_vs_host"]]
    print(f"wrote {path}: {sum(len(s['devices']) for s in out['series'].values())} device rows, "
          f"{len(gaps)} device-host pairs, {sum(g <= 0.03 for g in gaps)} within 0.03")


if __name__ == "__main__":
    main()
