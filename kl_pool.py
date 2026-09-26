#!/usr/bin/env python3
"""Does teacher-forced KL(bf16 || compressed) give ONE collapse threshold across looped LMs where latent fidelity
does not?  Uses only saved records: tf2.json (looped_decode v2) holds per-cell KL on the bf16 top-64 support + a
tail bin, fidelity and NLL.  One platform per model (the platform with the most KL cells), batch-matched to the bf16
reference, one row per model x quantizer x loops (largest n).  Collapse = greedy retained < 50%.
    python kl_pool.py          # the platform with the most KL cells per model
    python kl_pool.py p4d      # p4d only (audit batch 8)
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_looped as A  # noqa: E402
from ret_analyze import auroc, best_cut  # noqa: E402

ROOTS = A.POOL_ROOTS + ["results/looped/p5/cross"]
PLATFORM = sys.argv[1] if len(sys.argv) > 1 else None   # e.g. "p4d": restrict every model to one platform


def main():
    rows, _ = A.pool_rows(ROOTS, 50.0, single_platform=False, dedupe=False)
    rows = [r for r in rows if r.get("kl") is not None and r["platform"] != "lambda"]
    per = defaultdict(lambda: defaultdict(set))
    for r in rows:
        per[r["model"]][r["platform"]].add((r["quant"], r["loops"]))
    choice = {m: (PLATFORM if PLATFORM in d else None) if PLATFORM else max(d, key=lambda p: len(d[p]))
              for m, d in per.items()}
    rows = [r for r in rows if r["platform"] == choice[r["model"]]]
    best = {}
    for r in rows:
        k = (r["model"], r["quant"], r["loops"])
        if k not in best or min(r["n_gen"], r["n_tf"]) > min(best[k]["n_gen"], best[k]["n_tf"]):
            best[k] = r
    rows = sorted(best.values(), key=lambda r: (r["model"], r["kl"]))
    print("## KL(bf16 || compressed) vs latent fidelity as a pooled collapse threshold (saved tf2 records)\n")
    print("platform per model: " + ", ".join(f"{m} {p}" for m, p in sorted(choice.items())) + "\n")
    print("| model | loops | quant | n | retained | collapse | KL | fidelity | ΔNLL % |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['loops']} | {r['quant']} | {min(r['n_gen'], r['n_tf'])} | {r['ret']:.0f}% | "
              f"{'yes' if r['collapse'] else ''} | {r['kl']:.3f} | {r['fid']:.3f} | "
              f"{100 * (r['nll'] - r['nll_bf16']) / r['nll_bf16']:+.0f}% |")
    lab = np.array([r["collapse"] for r in rows])
    models = sorted(set(r["model"] for r in rows))
    print()
    for name, sc in (("KL", np.array([r["kl"] for r in rows])), ("1-φ", 1 - np.array([r["fid"] for r in rows])),
                     ("ΔNLL %", np.array([(r["nll"] - r["nll_bf16"]) / r["nll_bf16"] for r in rows]))):
        au = auroc(sc[lab], sc[~lab])
        ok, per_m = 0, {}
        for m in models:
            te = np.array([r["model"] == m for r in rows])
            tr = ~te
            t = best_cut(sc[tr], lab[tr]) if lab[tr].any() and not lab[tr].all() else np.median(sc[tr])
            c = (sc[te] > t) == lab[te]
            per_m[m] = f"{c.mean():.2f}"
            ok += int(c.sum())
        colmin = sc[lab].min() if lab.any() else float("nan")
        heldmax = sc[~lab].max() if (~lab).any() else float("nan")
        print(f"- {name}: pooled AUROC {au:.3f}; collapsed min {colmin:.3f} vs held max {heldmax:.3f}; "
              f"leave-one-model-out accuracy {ok / len(rows):.3f} ({', '.join(f'{m} {v}' for m, v in per_m.items())})")
    print(f"\n{len(rows)} cells ({int(lab.sum())} collapsed). Hidden states of compressed models were not saved "
          "(only bf16 prompt latents in ref files), so φ without the top-k massive dimensions / standardized φ cannot "
          "be recomputed without a rerun.")


if __name__ == "__main__":
    main()
