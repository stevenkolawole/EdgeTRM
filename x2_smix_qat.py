#!/usr/bin/env python3
"""S_mix for LSQ-QAT / naive-QAT / fine-tune vs the fp32 MLP w256 models (Babel L40S only): S, S_mix, S_mlp,
sigma_1/2, implied delta* = sigma_1/2 x S (and x S_mix), and the S / S_mix laws' predictions (fit set of x2_smix.json)."""
import json
from pathlib import Path

import numpy as np

import x2_smix_analyze as X

HERE = Path(__file__).resolve().parent
D = X.D
law = json.load(open(HERE / "results/breadth/x2_smix.json"))
fit = [r for r in law["rows"] if not r["held_out"] and r["sigma_half"] and r["fp32"] >= 0.3]
logd = {k: float(np.mean([np.log(r["sigma_half"] * r[k]) for r in fit])) for k in ("S", "S_mix")}
rows = []
for m, stem, sf in ([(f"trm_sudoku_mlp_w256_s{i}", f"trm_sudoku_mlp_w256_s{i}_babel", f"sens_var_mlp_w256_s{i}.json") for i in range(3)]
                    + [(m, m, f"sens_var_{m}.json") for m in ("ft_fp32_s0", "qat_naive_s0", "qat_lsq_s0", "qat_lsq_s1", "qat_lsq_s2")]):
    (w, a0) = X.width(stem)
    S = X.s_of(D / sf, "noise0.05")
    Sm = X.s_of(D / f"smix_{m}.json", "noise0.05@mixer")
    Sl = X.s_of(D / f"smix_{m}.json", "noise0.05@mlp")
    sh = w[0] if w else None
    r = {"model": m, "fp32": a0, "sigma_half": sh, "bound": w[1] if w else None, "S": S, "S_mix": Sm, "S_mlp": Sl,
         "delta_S": sh * S if sh else None, "delta_Smix": sh * Sm if sh else None,
         "ratio_S": float(np.exp(logd["S"]) / S / sh) if sh else None, "ratio_Smix": float(np.exp(logd["S_mix"]) / Sm / sh) if sh else None}
    rows.append(r)
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    print(f"| {m} | {f(a0)} | {f(sh) if sh else r['bound']} | {f(S,2)} | {f(Sm,2)} | {f(Sl,2)} | {f(r['delta_S'])} | {f(r['delta_Smix'])} | {f(r['ratio_S'],2)} | {f(r['ratio_Smix'],2)} |")
print("fit delta* S", np.exp(logd["S"]), "S_mix", np.exp(logd["S_mix"]))
json.dump({"platform": "babel", "fit_delta_star": {k: float(np.exp(v)) for k, v in logd.items()}, "rows": rows},
          open(HERE / "results/breadth/x2_smix_qat.json", "w"), indent=1)
