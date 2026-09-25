#!/usr/bin/env python3
"""Audit numbers for the X3 device rows (AUDIT_SEP25.md), reproducible from the written records.

1. Paired exact McNemar tests (two-sided binomial on the discordant puzzles) for every device comparison
   at step 8 on the same puzzles: each device run vs the device fp32 run, vs the host simulation of the
   same graph, and across devices. Host and device runs are deterministic (replicates are bit-identical),
   so there is no run-to-run noise band; the uncertainty is over puzzles.
2. B2ab / B2abu tensor counts from variants.json (16-bit = 555 - 8-bit).
3. DLC encoding coverage from dlc_encodings_w8a8.json (tensors kept by name, of which with our scale
   adjacent to the name).

Run on p4d (reads ~/aihub_artifacts/h1):  python device_stats.py > device_stats.json
"""
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402
import aihub_infer as I  # noqa: E402

STEP = 8


def mcnemar(a, b):
    """a, b: bool arrays of per-puzzle correctness. Returns counts and exact two-sided p."""
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else 1.0
    return {"a_correct": int(a.sum()), "b_correct": int(b.sum()), "a_only": n10, "b_only": n01,
            "n": int(len(a)), "p_exact": p}


def correct(task, kind, name, lab):
    d = I.idir(task)
    f = d / (f"dev_{name}.npz" if kind == "dev" else f"host_{name}.npz")
    if not f.exists():
        return None
    if kind == "dev":
        m = json.loads((d / f"dev_{name}.json").read_text())
        if not m.get("done") or m.get("failed"):
            return None
    p = np.load(f)["preds"]
    if p.shape[0] < STEP:
        return None
    return (p[STEP - 1] == lab).all(-1)


PAIRS = []
for task in ("sudoku_mlp", "sudoku_attn", "maze"):
    for dev in ("Samsung_Galaxy_S24_acct3", "Snapdragon_8_Elite_QRD_acct3"):
        for prec in ("w8a8", "w8a16", "w8a8sqvar16u"):
            PAIRS.append((task, ("dev", f"{prec}_{dev}"), ("dev", f"fp32_{dev}")))
            host = "w8a8" if prec == "w8a8" else ("w8a16" if prec == "w8a16" else None)
            if host:
                PAIRS.append((task, ("dev", f"{prec}_{dev}"), ("host", host)))
        PAIRS.append((task, ("dev", f"fp32_{dev}"), ("host", "fp32")))
    for prec in ("w8a8", "w8a16", "fp32"):
        PAIRS.append((task, ("dev", f"{prec}_Samsung_Galaxy_S24_acct3"), ("dev", f"{prec}_Snapdragon_8_Elite_QRD_acct3")))
PAIRS.append(("sudoku_mlp", ("dev", "w8a8rms16u_Samsung_Galaxy_S24_acct2"), ("dev", "fp32_Samsung_Galaxy_S24_acct3")))
PAIRS.append(("sudoku_mlp", ("dev", "w8a8_Samsung_Galaxy_S24"), ("host", "w8a8")))
PAIRS.append(("maze", ("dev", "w8a8_Samsung_Galaxy_S24"), ("host", "w8a8")))


def main():
    out = {"step": STEP, "mcnemar": []}
    labs = {t: I.test_set(t)[1] for t in ("sudoku_mlp", "sudoku_attn", "maze")}
    for task, (ka, na), (kb, nb) in PAIRS:
        a, b = correct(task, ka, na, labs[task]), correct(task, kb, nb, labs[task])
        if a is None or b is None:
            continue
        r = mcnemar(a, b)
        r.update(task=task, a=f"{ka}:{na}", b=f"{kb}:{nb}")
        out["mcnemar"].append(r)
    reg = json.loads((I.idir("sudoku_mlp") / "ablate" / "variants.json").read_text())
    out["tensor_counts"] = {k: {"8bit": v["n_tensors_8bit"], "16bit": 555 - v["n_tensors_8bit"]}
                            for k, v in reg.items() if k.startswith(("B2ab", "B2_", "B2b2", "B2t", "B2u", "B2abt", "B2abu"))}
    dl = pathlib.Path.home() / "x3" / "dlc" / "enc_w8a8.json"
    if dl.exists():
        rows = json.loads(dl.read_text())
        kept = [r for r in rows if r["name_in_dlc"]]
        out["dlc"] = {"activation_tensors": len(rows), "kept_by_name": len(kept),
                      "kept_with_our_scale_adjacent": sum(r["scale_near_name"] for r in kept),
                      "dropped": len(rows) - len(kept)}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
