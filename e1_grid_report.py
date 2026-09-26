#!/usr/bin/env python3
"""E1 grid read-out: one table per model (one L40S each) from <root>/<task>/grid.json.
Energy per solved (total and idle-subtracted) and latency with 95% t-intervals over the 5 windows; ranking checks
between time and energy for the comparisons named in E1.md G1-G6.

    python e1_grid_report.py --root results/energy/babel_l40s > results/energy/babel_l40s/e1_grid_tables.md
"""
import argparse
import json
import pathlib
import statistics

TASKS = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze"]
NICE = {"sudoku_mlp": "TRM Sudoku-MLP", "sudoku_attn": "TRM Sudoku-attn", "hrm_sudoku": "HRM Sudoku",
        "maze": "TRM Maze-Hard"}
TOOLS = ["eager_fp32", "eager_fp16", "ort_fp16", "trt_fp16", "trt_int8proj"]


def f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def ci(v, d=3):
    return "" if not v else f" [{v[0]:.{d}f}, {v[1]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    print("# E1 grid (Babel L40S, one exclusive card per model; DATACENTER-GPU energy, not edge)\n")
    for task in TASKS:
        p = root / task / "grid.json"
        if not p.exists():
            print(f"## {NICE[task]}: no grid record\n")
            continue
        g = json.loads(p.read_text())
        m = g["meta"]
        idle = [w["P_W"] for w in g["idle"] if w["clean"]]
        print(f"## {NICE[task]} (H={m['H']}, rule {m['rule']}, n={m['n']}; {m['gpu']} on {m['host']}, job {m['slurm_job']}, "
              f"driver {m['driver']}, torch {m['torch']}, TensorRT {m.get('tensorrt')}, ORT {m.get('onnxruntime')})")
        print(f"Idle median {statistics.median(idle):.1f} W ({len(idle)} windows). Source `{task}/grid.json`.\n")
        print("| config | exact % | mean steps | J/solved [95% CI] | J/solved dyn [95% CI] | latency ms/puzzle [95% CI] | ms/solved (1/throughput) | W | SM MHz |")
        print("|---|---|---|---|---|---|---|---|---|")
        C = g["configs"]
        rows = {}
        for name, c in C.items():
            if c.get("skipped"):
                print(f"| {name} | skipped: {c['reason'][:90]} |||||||")
                continue
            s = c["summary"]
            rows[name] = s
            extra = ""
            if "polish_mean_polish_steps" in s:
                extra = f" (INT8 {s['polish_mean_cheap_steps']:.2f} + FP16 {s['polish_mean_polish_steps']:.2f})"
            print(f"| {name} | {100*s['acc']:.1f} | {s['mean_steps']:.2f}{extra} | {f(s['J_solved_median'])}{ci(s['J_solved_ci95'])} | "
                  f"{f(s['J_solved_dyn_median'])}{ci(s['J_solved_dyn_ci95'])} | {f(s['lat_ms_median'], 2)}{ci(s['lat_ms_ci95'], 2)} | "
                  f"{f(s['ms_per_solved_median'], 3)} | {f(s['P_W_median'], 1)} | {f(s['sm_MHz_median'], 0)} |")
        # ranking checks: A vs B, by per-solved energy and per-solved time (1/throughput) and per-puzzle latency
        print("\nA/B ratios (energy per solved, time per solved = 1/throughput, per-puzzle latency). 'split' = energy and "
              "time per solved order A and B differently.\n")
        print("| A | B | energy | dyn energy | time/solved | latency | order |")
        print("|---|---|---|---|---|---|---|")
        pairs = []
        for B in (1, 8, 64, 256):
            for mode in ("fixed", "exit"):
                pairs += [(f"trt_int8proj_b{B}_{mode}", f"trt_fp16_b{B}_{mode}"),
                          (f"eager_fp16_b{B}_{mode}", f"trt_fp16_b{B}_{mode}"),
                          (f"ort_fp16_b{B}_{mode}", f"trt_fp16_b{B}_{mode}"),
                          (f"eager_fp16_b{B}_{mode}", f"eager_fp32_b{B}_{mode}")]
            for tool in TOOLS:
                pairs.append((f"{tool}_b{B}_exit", f"{tool}_b{B}_fixed"))
        for tool in TOOLS:
            for mode in ("fixed", "exit"):
                for B in (8, 64, 256):
                    pairs.append((f"{tool}_b{B}_{mode}", f"{tool}_b1_{mode}"))
        pairs += [("trt_int8proj_b1_polish", "trt_fp16_b1_exit"), ("trt_int8proj_b1_polish", "trt_int8proj_b1_exit")]
        nsplit = ncmp = 0
        for A, Bn in pairs:
            if A not in rows or Bn not in rows:
                continue
            ra, rb = rows[A], rows[Bn]
            if not (ra["J_solved_median"] and rb["J_solved_median"]):
                continue
            e = ra["J_solved_median"] / rb["J_solved_median"]
            ed = ra["J_solved_dyn_median"] / rb["J_solved_dyn_median"]
            tm = ra["ms_per_solved_median"] / rb["ms_per_solved_median"]
            lt = ra["lat_ms_median"] / rb["lat_ms_median"]
            split = (e < 1) != (tm < 1)
            ncmp += 1
            nsplit += split
            tag = "SPLIT" if split else ("split (dyn only)" if (ed < 1) != (tm < 1) else "agree")
            print(f"| {A} | {Bn} | {e:.3f} | {ed:.3f} | {tm:.3f} | {lt:.3f} | {tag} |")
        print(f"\n{nsplit} of {ncmp} comparisons split energy from time per solved.\n")


if __name__ == "__main__":
    main()
