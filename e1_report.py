#!/usr/bin/env python3
"""E1 read-out: tables from ~/e1/results/<task>/{energy,engines,record,prep}.json (or a local copy given by
--root). A100-SXM4-40GB datacenter-GPU energy only; one platform per table.

Energy per solved = median(energy per step over the 5 clean windows) x mean steps per puzzle / accuracy, where
mean steps and accuracy come from the same run's first pass over all n puzzles (energy per step does not depend
on which puzzles a 60 s window happened to cover; energy per window per solved does). 'dyn' = idle power of the
same card in the same session subtracted.

    python e1_report.py --root results/energy/p4d > results/energy/p4d/e1_tables.md
"""
import argparse
import json
import pathlib
import statistics

TASKS = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze"]
NICE = {"sudoku_mlp": "TRM Sudoku-MLP", "sudoku_attn": "TRM Sudoku-attn", "hrm_sudoku": "HRM Sudoku",
        "maze": "TRM Maze-Hard"}


def load(root, task, name):
    p = root / task / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def derive(w):
    m = w["median_clean"]
    fp = w["first_pass"]
    steps, acc = fp["mean_steps"], fp["exact"]
    d = dict(P=m["P_W"], ms_step=m["ms_per_step"], mJ_step=m["mJ_per_step"], mJ_step_dyn=m["mJ_per_step_dyn"],
             acc=acc, steps=steps, n=fp["puzzles_covered"], n_clean=w["n_clean"],
             sm=statistics.median([x.get("sm_MHz_mean", float("nan")) for x in w["windows"]]),
             temp=max(x.get("temp_C_end", 0) for x in w["windows"]),
             util=statistics.median([x.get("util_gpu_mean", float("nan")) for x in w["windows"]]))
    B = w["B"]
    d["J_puzzle"] = d["mJ_step"] * steps / 1e3
    d["J_solved"] = d["J_puzzle"] / acc if acc else float("inf")
    d["J_solved_dyn"] = d["mJ_step_dyn"] * steps / 1e3 / acc if acc else float("inf")
    d["ms_solved"] = d["ms_step"] * steps / acc if acc else float("inf")
    # time until an answer is emitted (per-answer latency): batch-1 = steps x step time; batched = whole batch
    d["ms_answer"] = d["ms_step"] * steps * (B if B > 1 else 1)
    cv = [x["mJ_per_step"] for x in w["windows"] if x["clean"]]
    d["cv_mJ_step"] = (statistics.pstdev(cv) / statistics.mean(cv)) if len(cv) > 1 else float("nan")
    return d


def energy_table(root, task):
    e = load(root, task, "energy")
    if not e:
        return f"\n### {NICE[task]}: no energy record\n", {}
    idle = [w["P_W"] for w in e["idle"] if w["clean"]]
    meta = e["meta"]
    out = [f"\n### {NICE[task]} (H={meta['H']}, {meta['nsup']} steps max, exit rule {meta['rule']}, n={meta['n_puzzles']}; "
           f"{meta['gpu']}, driver {meta['driver']}, torch {meta['torch']}, TensorRT {meta.get('tensorrt')})",
           f"Idle (process alive, no work): median {statistics.median(idle):.1f} W over {len(idle)} windows "
           f"(range {min(idle):.1f}-{max(idle):.1f}). Source: `{task}/energy.json`.", "",
           "| config | windows | W | ms/step | mJ/step | mJ/step dyn | exact | mean steps | J/solved | J/solved dyn | ms/solved | SM MHz | util % | CV mJ/step |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = {}
    for name, w in e["windows"].items():
        if not w.get("median_clean"):
            continue
        d = derive(w)
        rows[name] = d
        out.append(f"| {name} | {d['n_clean']} | {d['P']:.1f} | {d['ms_step']:.3f} | {d['mJ_step']:.1f} | "
                   f"{d['mJ_step_dyn']:.1f} | {100*d['acc']:.1f} | {d['steps']:.2f} | {d['J_solved']:.3f} | "
                   f"{d['J_solved_dyn']:.3f} | {d['ms_solved']:.1f} | {d['sm']:.0f} | {d['util']:.0f} | {100*d['cv_mJ_step']:.1f}% |")
    return "\n".join(out) + "\n", rows


def rank_checks(task, rows):
    """Pairs whose order by time and by energy is compared (per solved)."""
    B = {"maze": 32}.get(task, 256)
    pairs = [("INT8 GEMMs vs FP16, TRT, batch 1 fixed", "trt_int8mm_b1_fixed", "trt_fp16_b1_fixed"),
             ("INT8 GEMMs vs FP16, TRT, batch 1 exit", "trt_int8mm_b1_exit", "trt_fp16_b1_exit"),
             (f"INT8 GEMMs vs FP16, TRT, batch {B}", f"trt_int8mm_b{B}_fixed", f"trt_fp16_b{B}_fixed"),
             ("TRT's own INT8 choice vs FP16, batch 1", "trt_int8_b1_fixed", "trt_fp16_b1_fixed"),
             (f"TRT's own INT8 choice vs FP16, batch {B}", f"trt_int8_b{B}_fixed", f"trt_fp16_b{B}_fixed"),
             ("exit vs fixed, TRT FP16, batch 1", "trt_fp16_b1_exit", "trt_fp16_b1_fixed"),
             ("exit vs fixed, eager FP16, batch 1", "eager_fp16_b1_exit", "eager_fp16_b1_fixed"),
             ("exit vs fixed, eager FP32, batch 1", "eager_fp32_b1_exit", "eager_fp32_b1_fixed"),
             ("exit vs fixed, CUDA graph FP16, batch 1", "graph_fp16_b1_exit", "graph_fp16_b1_fixed"),
             ("framework tax: eager FP16 vs TRT FP16, batch 1", "eager_fp16_b1_fixed", "trt_fp16_b1_fixed"),
             ("framework tax: eager FP32 vs TRT FP32, batch 1", "eager_fp32_b1_fixed", "trt_fp32_b1_fixed"),
             ("dispatch only: eager FP16 vs CUDA graph FP16, batch 1", "eager_fp16_b1_fixed", "graph_fp16_b1_fixed"),
             ("kernels only: CUDA graph FP16 vs TRT FP16, batch 1", "graph_fp16_b1_fixed", "trt_fp16_b1_fixed"),
             ("FP16 vs FP32, eager, batch 1", "eager_fp16_b1_fixed", "eager_fp32_b1_fixed"),
             ("FP16 vs FP32, TRT, batch 1", "trt_fp16_b1_fixed", "trt_fp32_b1_fixed"),
             (f"FP16 vs FP32, eager, batch {B}", f"eager_fp16_b{B}_fixed", f"eager_fp32_b{B}_fixed"),
             (f"batch {B} vs batch 1, TRT FP16", f"trt_fp16_b{B}_fixed", "trt_fp16_b1_fixed"),
             (f"batch {B} vs batch 1, eager FP16", f"eager_fp16_b{B}_fixed", "eager_fp16_b1_fixed"),
             (f"batch {B} fixed vs batch 1 exit, TRT FP16", f"trt_fp16_b{B}_fixed", "trt_fp16_b1_exit")]
    out = ["", f"Ratios A/B per solved puzzle ({NICE[task]}); 'agree' = time and energy order A and B the same way.",
           "", "| comparison | time A/B | energy A/B | dyn energy A/B | exact A, B | order |", "|---|---|---|---|---|---|"]
    for label, a, b in pairs:
        if a not in rows or b not in rows:
            continue
        A, Bb = rows[a], rows[b]
        rt = A["ms_solved"] / Bb["ms_solved"]
        re = A["J_solved"] / Bb["J_solved"]
        rd = A["J_solved_dyn"] / Bb["J_solved_dyn"]
        agree = (rt < 1) == (re < 1)
        agree_d = (rt < 1) == (rd < 1)
        tag = "agree" if agree and agree_d else ("DISAGREE (total)" if not agree else "DISAGREE (dyn only)")
        out.append(f"| {label} | {rt:.3f} | {re:.3f} | {rd:.3f} | {100*A['acc']:.1f}, {100*Bb['acc']:.1f} | {tag} |")
    return "\n".join(out) + "\n"


def engines_table(root, task):
    e = load(root, task, "engines")
    if not e:
        return ""
    out = ["", f"Engines ({NICE[task]}; `{task}/engines.json`): layers by (input | output) precision from the TensorRT "
           "engine inspector.", "", "| engine | build s | opt level | MB | INT8-output layers | layers | failed |",
           "|---|---|---|---|---|---|---|"]
    for k, v in e.items():
        if v.get("BUILD_FAILED"):
            out.append(f"| {k} | {v['build_s']} | {v.get('builder_optimization_level', 3)} | - | - | - | "
                       f"{v['errors'][-1][:120] if v.get('errors') else 'yes'} |")
            continue
        i8 = sum(c for kk, c in v["by_io_dtype"].items() if "Int8" in kk.split("out:")[1])
        out.append(f"| {k} | {v['build_s']} | {v.get('builder_optimization_level', 3)} | {v['size_MB']} | {i8} | "
                   f"{v['n_layers']} | |")
    return "\n".join(out) + "\n"


def record_table(root, task):
    r = load(root, task, "record")
    if not r:
        return ""
    out = ["", f"Correctness ({NICE[task]}; `{task}/record.json`): 16-step carry chain, n puzzles, vs the eager fp32 "
           "reference at the same H (batched).", "",
           "| config | n | exact @8 | exact @16 | first 100 @8 / @16 | cell agree w/ ref @16 | puzzle agree @16 | z_H cos @1 / @8 / @16 (min @16) | exit exact / steps | agree w/ X3 ORT fp32 @8 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for k, s in r.items():
        cs = s["zH_cos_per_step_meanK"]
        x3 = s.get("x3_ort_host_fp32")
        x3s = f"{x3['cell_agreement_per_step'][min(7, x3['steps']-1)]:.4f} (n={x3['n']})" if x3 else ""
        f100 = s["first100_exact_per_step"]
        out.append(f"| {k} | {s['n']} | {100*s['exact_per_step'][7]:.1f} | {100*s['exact_per_step'][-1]:.1f} | "
                   f"{f100[7]} / {f100[-1]} | {s['ref_cell_agreement_per_step'][-1]:.4f} | {s['ref_puzzle_agreement_final']:.3f} | "
                   f"{cs[0]:.4f} / {cs[7]:.4f} / {cs[-1]:.4f} ({s['zH_cos_min_final']:.3f}) | "
                   f"{100*s['exit_exact']:.1f} / {s['exit_mean_steps']:.2f} | {x3s} |")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(pathlib.Path.home() / "e1" / "results"))
    ap.add_argument("--tasks", default=",".join(TASKS))
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    print("# E1 tables (generated by e1_report.py; NVIDIA A100-SXM4-40GB on AWS p4d = DATACENTER-GPU energy, not edge)")
    for task in a.tasks.split(","):
        txt, rows = energy_table(root, task)
        print(txt)
        if rows:
            print(rank_checks(task, rows))
        print(engines_table(root, task))
        print(record_table(root, task))


if __name__ == "__main__":
    main()
