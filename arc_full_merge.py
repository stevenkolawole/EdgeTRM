#!/usr/bin/env python3
"""Merge the puzzle-sharded official-protocol ARC-AGI-1 runs (arc_breadth.py --full --shard K/N)
into one row per (H, nsup, quantizer): pass@1 / pass@2 / cell are n_puzzles-weighted means over
the shards (each shard holds whole puzzles, so the vote is exact); token metrics and fidelity are
the same canonical-row numbers in every shard and are taken from shard 0.

    python arc_full_merge.py ~/breadth/results/arc_full --out ~/breadth/results/arc_attn_full.json
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    groups = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.dir, "*_s*of*.json"))):
        m = re.match(r"(.+)_s(\d+)of(\d+)\.json$", os.path.basename(f))
        if not m:
            continue
        try:
            recs = json.load(open(f))["records"]
        except (json.JSONDecodeError, ValueError):
            continue
        for r in recs:
            if "arc_pass1" not in r:
                continue
            groups[(m.group(1), r["H_cycles"], r["nsup"], r["quant"]["spec"], int(m.group(3)))].append((int(m.group(2)), r))
    rows = []
    for (tag, H, nsup, spec, nsh), items in sorted(groups.items()):
        n = sum(r["n_puzzles"] for _, r in items)
        row = {"tag": tag, "H_cycles": H, "nsup": nsup, "quant": items[0][1]["quant"], "protocol": "full_aug_vote",
               "shards_done": len(items), "shards": nsh, "n_puzzles": n,
               "arc_pass1": sum(r["arc_pass1"] * r["n_puzzles"] for _, r in items) / n,
               "arc_pass2": sum(r["arc_pass2"] * r["n_puzzles"] for _, r in items) / n,
               "cell": sum(r["cell"] * r["n_puzzles"] for _, r in items) / n,
               "pexact_token": items[0][1].get("pexact_token"), "cell_token": items[0][1].get("cell_token"),
               "fidelity_H": items[0][1].get("fidelity_H")}
        rows.append(row)
        flag = "" if len(items) == nsh else f"  (partial: {len(items)}/{nsh} shards)"
        print(f"{tag:14s} H={H} nsup={nsup:2d} {spec:8s} pass@1 {row['arc_pass1']*100:6.2f} pass@2 {row['arc_pass2']*100:6.2f} "
              f"cell {row['cell']*100:6.2f} fidH {row['fidelity_H']:.3f} n={n}{flag}")
    if a.out:
        json.dump({"task": "arc_attn", "protocol": "full_aug_vote", "records": rows}, open(a.out, "w"), indent=1)
        print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()
