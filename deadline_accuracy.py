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

Usage: python3 deadline_accuracy.py  (writes results/devices/deadline_accuracy.json)
"""
import json
from collections import defaultdict
from pathlib import Path

DEV = Path(__file__).resolve().parent / "results" / "devices"
DEADLINES_MS = (50, 100, 250, 500, 1000)
SERIES = [("sudoku_mlp", "fp32"), ("sudoku_mlp", "w8a16"), ("sudoku_attn", "w8a16"),
          ("maze", "fp32"), ("maze", "w8a16")]


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
