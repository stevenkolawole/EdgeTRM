#!/usr/bin/env python3
"""MCU-tier breadth: compile the INT8 (and W4A8) TFLite inner steps already
exported under ~/mcu/<task>/ with Arm Vela for every Ethos-U accelerator
configuration and the system configurations Arm ships, and collect SRAM,
flash and cycle estimates into one JSON.

    ~/venvs/mcu/bin/python vela_breadth.py --out ~/breadth/results/vela_breadth.json
"""
import argparse
import csv
import json
import pathlib
import subprocess
import sys

MCU = pathlib.Path.home() / "mcu"
TASKS = ["maze", "sudoku_mlp", "sudoku_attn"]
GRAPHS = ["inner_step_int8.tflite", "inner_step_w4a8.tflite"]
# (accelerator, system config, memory mode)
CONFIGS = [
    ("ethos-u55-32", "Ethos_U55_Deep_Embedded", "Sram_Only"),
    ("ethos-u55-64", "Ethos_U55_Deep_Embedded", "Shared_Sram"),
    ("ethos-u55-128", "Ethos_U55_High_End_Embedded", "Shared_Sram"),
    ("ethos-u55-256", "Ethos_U55_High_End_Embedded", "Shared_Sram"),
    ("ethos-u55-256", "Ethos_U55_High_End_Embedded", "Dedicated_Sram"),
    ("ethos-u65-256", "Ethos_U65_Embedded", "Shared_Sram"),
    ("ethos-u65-256", "Ethos_U65_Mid_End", "Dedicated_Sram"),
    ("ethos-u65-512", "Ethos_U65_High_End", "Dedicated_Sram"),
    ("ethos-u85-128", "Ethos_U85_SYS_Flash_Low", "Shared_Sram"),
    ("ethos-u85-256", "Ethos_U85_SYS_Flash_High", "Shared_Sram"),
    ("ethos-u85-512", "Ethos_U85_SYS_DRAM_Low", "Dedicated_Sram"),
    ("ethos-u85-1024", "Ethos_U85_SYS_DRAM_Mid", "Dedicated_Sram"),
    ("ethos-u85-2048", "Ethos_U85_SYS_DRAM_High", "Dedicated_Sram"),
]


def run_vela(graph, accel, sysconf, mem, outdir):
    vela = pathlib.Path(sys.executable).parent / "vela"
    cmd = [str(vela), str(graph), "--accelerator-config", accel, "--system-config", sysconf,
           "--memory-mode", mem, "--output-dir", str(outdir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec = {"rc": r.returncode}
    if r.returncode != 0:
        rec["error"] = (r.stderr or r.stdout)[-600:]
        return rec
    # vela writes <graph>_summary_<sysconf>.csv with the numbers
    for p in outdir.glob("*_summary_*.csv"):
        rows = list(csv.DictReader(open(p)))
        if rows:
            row = rows[-1]
            for k, v in row.items():
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    results = []
    for task in TASKS:
        for g in GRAPHS:
            graph = MCU / task / g
            if not graph.exists():
                print(f"skip {graph}")
                continue
            for accel, sysconf, mem in CONFIGS:
                outdir = MCU / task / "vela_breadth" / f"{g[:-7]}_{accel}_{sysconf}_{mem}"
                outdir.mkdir(parents=True, exist_ok=True)
                rec = run_vela(graph, accel, sysconf, mem, outdir)
                rec.update({"task": task, "graph": g, "accelerator": accel, "system_config": sysconf,
                            "memory_mode": mem})
                results.append(rec)
                keys = [k for k in rec if "sram" in k.lower() or "flash" in k.lower() or "cycles" in k.lower()
                        or "inference" in k.lower()]
                print(f"{task:12s} {g:26s} {accel:14s} {sysconf:28s} {mem:15s} rc={rec['rc']} "
                      + " ".join(f"{k}={rec[k]}" for k in keys[:6]), flush=True)
                json.dump(results, open(a.out, "w"), indent=1)
    print("wrote", a.out, len(results))


if __name__ == "__main__":
    main()
