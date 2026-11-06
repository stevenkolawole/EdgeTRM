#!/usr/bin/env python3
"""Compare the check pass of one or more edge_loop.py records against a reference record (the laptop ORT fp32 run
of the same graph on the same puzzles): exact-answer counts per step, per-puzzle agreement of the exact flag at the
last step, the halting logit's largest difference, and per-step latency. Prints one line per record.

    python3 compare_checks.py --ref host/laptop_sudoku_mlp_H3_ort_fp32.json nano-b_sudoku_mlp_H3_trt_fp32_gate.json ...
"""
import argparse
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("records", nargs="+")
a = ap.parse_args()

ref = json.load(open(a.ref))["check"]
rp = {p["i"]: p for p in ref["per_puzzle"]}
print("ref %s: exact per step %s, exit %.2f @ %.2f steps, step ms %.1f" % (
    a.ref, ref["exact_count_per_step"], ref["exit_exact"], ref["exit_mean_steps"], ref["step_ms"]["median"]))
for path in a.records:
    d = json.load(open(path))
    c, m = d["check"], d["meta"]
    counts_equal = c["exact_count_per_step"] == ref["exact_count_per_step"]
    last_agree = qdiff = 0.0
    n = 0
    for p in c["per_puzzle"]:
        r = rp.get(p["i"])
        if r is None:
            continue
        n += 1
        last_agree += p["exact_per_step"][-1] == r["exact_per_step"][-1]
        qdiff = max(qdiff, max(abs(x - y) for x, y in zip(p["q_halt"], r["q_halt"])))
    print("%s [%s %s %s]: counts %s equal=%s | last-step agreement %.2f (n=%d) | q_halt maxdiff %.3f | exit %.2f @ %.2f | step ms %.1f" % (
        path, m.get("board"), m.get("backend"), m.get("precision"), c["exact_count_per_step"], counts_equal,
        last_agree / max(n, 1), n, qdiff, c["exit_exact"], c["exit_mean_steps"], c["step_ms"]["median"]))
