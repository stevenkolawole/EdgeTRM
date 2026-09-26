#!/usr/bin/env python3
"""P3 bug check (diagnostic; X2.md 02:47 UTC entry): ties / saturation of the halting-logit argmax over SR draws, and
the head pick re-scored against the picked draw's correctness and phi vs the median draw (sign tests)."""
import glob
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import sys
PLAT = sys.argv[1] if len(sys.argv) > 1 else "babel"
RES = HERE / f"results/breadth/{PLAT}/x2_ret"
SAT = 7.6          # logit above which sigmoid rounds to 1.0 in fp16


def binom_p(k, n):
    if n == 0:
        return None
    pm = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(q for q in pm if q <= pm[k] * (1 + 1e-9))))


out = {}
for task in ("sudoku_mlp", "sudoku_attn", "hrm_sudoku"):
    recs = []
    short = {"sudoku_mlp": "mlp", "sudoku_attn": "attn", "hrm_sudoku": "hrm"}[task]
    for f in sorted(glob.glob(str(RES / f"ret_sr_{task}_*.json")) + glob.glob(str(RES / f"ret_sr_{short}_[ab].json"))):
        recs += json.load(open(f))["records"]
    recs.sort(key=lambda r: int(r["cell"].split("~")[1]))
    q = np.array([r["per"]["qhalt"] for r in recs], float)          # (K, n)
    c = np.array([r["per"]["corr_q"] for r in recs], bool)
    phi = np.array([r["per"]["phi"] for r in recs], float)
    tc = np.array([r["per"]["t_c"] for r in recs], float)
    K, n = q.shape
    srt = np.sort(q, 0)
    tie = (srt[-1] - srt[-2]) <= 1e-3
    sat_all = (q > SAT).all(0)
    sat_top2 = srt[-2] > SAT
    pick = q.argmax(0)
    ar = np.arange(n)
    res = {"draws": K, "n": n, "frac_tied_top2": float(tie.mean()), "frac_all_draws_saturated": float(sat_all.mean()),
           "frac_top2_saturated": float(sat_top2.mean()), "frac_pick_index0": float((pick == 0).mean()),
           "qhalt_median": float(np.median(q)), "qhalt_p10_p90": [float(np.percentile(q, 10)), float(np.percentile(q, 90))]}
    for name, v in (("correct", c.astype(float)), ("phi", phi), ("t_c", tc)):
        vp, vm = v[pick, ar], np.median(v, 0)
        up, dn = int((vp > vm).sum()), int((vp < vm).sum())
        res[f"{name}_picked_gt_median"] = up
        res[f"{name}_picked_lt_median"] = dn
        res[f"{name}_sign_p"] = binom_p(up, up + dn)
        res[f"{name}_mean_picked_vs_all"] = [float(vp.mean()), float(v.mean())]
    # among untied, unsaturated examples only
    ok = ~tie & ~sat_top2
    vp, vm = c[pick, ar].astype(float), np.median(c.astype(float), 0)
    res["n_clean"] = int(ok.sum())
    res["correct_picked_clean"] = float(vp[ok].mean()) if ok.any() else None
    res["correct_meandraw_clean"] = float(c.astype(float).mean(0)[ok].mean()) if ok.any() else None
    res["acc_rtn"] = float(c[0].mean())
    res["acc_pick"] = float(c[pick, ar].mean())
    res["acc_oracle"] = float(c.any(0).mean())
    # does t_c discriminate draws at all? fraction of examples whose t_c is identical across all draws
    res["frac_tc_constant_across_draws"] = float((tc.max(0) == tc.min(0)).mean())
    res["frac_tc_censored_all_draws"] = float((tc >= 16).all(0).mean())
    out[task] = res
    print(task, json.dumps(res))
(HERE / ("results/breadth/x2_p3_check.json" if PLAT == "babel" else f"results/breadth/x2_p3_check_{PLAT}.json")).write_text(json.dumps(out, indent=1))
