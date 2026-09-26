#!/usr/bin/env python3
"""X2 recovery depth / returning-map read-out: K1, K2 scored by the rules written before any run
(results/breadth/x2_retk_predictions.md), plus the POST HOC residual comparison. One platform per call.

    python x2_retk_analyze.py --platform babel|p4d  -> results/breadth/x2_retk_readout_<platform>.md, x2_retk_<platform>.json
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

from x2_return_analyze import family, ctype, auroc, spearman, fmt

HERE = Path(__file__).resolve().parent


def load(res):
    rows = []
    for f in sorted(glob.glob(str(res / "retk_*.json"))):
        b = json.load(open(f))
        for r in b["records"]:
            p = r["per"]
            cf = np.array(p["corr_f"], bool)
            g = np.array(p["gain"], float)
            rq = np.array(p["r_q"], float)
            shift = np.where(g < 1, rq / np.maximum(1 - g, 1e-9), np.inf)
            rows.append({"file": Path(f).name, "model": b["model"], "family": family(b["model"], b["family"]), "cell": r["cell"],
                         "ctype": ctype(r["cell"]), "acc_f": r["acc_f"], "acc_q": r["acc_q"], "retained": r["retained"],
                         "collapse": r["retained"] < 0.5, "maps": r["maps"], "ks": b["ks"], "n_fc": int(cf.sum()),
                         "r_q": float(np.median(rq[cf])) if cf.any() else float(np.median(rq)),
                         "res_q": float(np.median(np.array(p["res_q"])[cf])) if cf.any() else None,
                         "res_f": float(np.median(np.array(p["res_f"])[cf])) if cf.any() else None,
                         "gain": float(np.median(g[cf])) if cf.any() else float(np.median(g)),
                         "shift": float(np.median(shift[cf])) if cf.any() else float(np.median(shift))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="babel", choices=["babel", "p4d"])
    a = ap.parse_args()
    res = HERE / f"results/breadth/{a.platform}/x2_retk"
    rows = [r for r in load(res) if r["acc_f"] >= 0.10 and r["n_fc"] > 0]
    main_ = [r for r in rows if r["ctype"] in ("weight-quant", "act-quant", "weight-noise")]
    md = [f"# X2 recovery depth / returning map: read-out ({a.platform}; `python x2_retk_analyze.py --platform {a.platform}`)\n",
          f"Records: {len({r['file'] for r in rows})} files, {len(rows)} cells (fp32 acc >= 10%). Rules: `x2_retk_predictions.md`.\n"]
    js = {"platform": a.platform}
    ks = rows[0]["ks"] if rows else []

    # K1
    col = [r for r in main_ if r["collapse"]]
    rec = lambda r, mp, k: r["maps"].get(mp, {}).get("rec_fc", {}).get(str(k))
    acc = lambda r, mp, k: r["maps"].get(mp, {}).get("acc", {}).get(str(k))
    k4 = [rec(r, "fp32", 4) for r in col]
    med4 = float(np.median(k4)) if k4 else None
    frac4 = float(np.mean([v >= 0.8 for v in k4])) if k4 else None
    k1v = "supported" if (med4 is not None and med4 >= 0.8 and frac4 >= 0.8) else ("partly" if med4 is not None and (med4 >= 0.8 or frac4 >= 0.5) else "falsified")
    js["K1"] = {"n_collapsed": len(col), "median_rec_k4": med4, "frac_cells_ge_0.8_k4": frac4, "verdict": k1v,
                "median_rec_by_k": {k: float(np.median([rec(r, "fp32", k) for r in col])) for k in ks} if col else {}}
    md.append("## K1: collapsed cells recover with <= 4 exact steps in >= 80% of fp32-correct examples\n")
    md.append(f"Collapsed cells: {len(col)}. At k = 4 (fp32 map): median recovery {fmt(med4)}, cells >= 0.8: {fmt(frac4)} -> **{k1v.upper()}**\n")
    md.append("Median recovery over collapsed cells by k (fp32 map): " + ", ".join(f"k={k}: {fmt(v)}" for k, v in js["K1"]["median_rec_by_k"].items()) + "\n")
    md.append("| family | model | cell | fp32 acc | compressed acc | recovery (fp32-correct) k=1/2/4/8/16 | accuracy after k exact steps k=1/2/4/8/16 | w8c recovery k=1/2/4/8/16 |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(col, key=lambda r: (r["family"], r["model"], r["cell"])):
        md.append(f"| {r['family']} | {r['model']} | {r['cell']} | {fmt(r['acc_f'])} | {fmt(r['acc_q'])} | "
                  + " / ".join(fmt(rec(r, 'fp32', k), 2) for k in ks) + " | " + " / ".join(fmt(acc(r, 'fp32', k), 2) for k in ks) + " | "
                  + " / ".join(fmt(rec(r, 'w8c', k), 2) for k in ks) + " |")
    md.append("")

    # K2
    out = []
    for r in main_ + [x for x in rows if x["ctype"] == "act-noise(fresh)"]:
        if "w8c" not in r["maps"] or "fp32" not in r["maps"]:
            continue
        d4 = rec(r, "w8c", 4) - rec(r, "fp32", 4)
        d16 = rec(r, "w8c", 16) - rec(r, "fp32", 16) if 16 in ks else None
        w4 = (rec(r, "w4c", 16) - rec(r, "fp32", 16)) if "w4c" in r["maps"] and 16 in ks else None
        out.append({"family": r["family"], "model": r["model"], "cell": r["cell"], "d4": d4, "d16": d16, "d16_w4c": w4,
                    "ok": abs(d4) <= 0.02 and (d16 is None or abs(d16) <= 0.02), "collapse": r["collapse"]})
    frac = float(np.mean([o["ok"] for o in out])) if out else None
    k2v = None if frac is None else ("supported" if frac >= 0.9 else ("partly" if frac >= 0.7 else "falsified"))
    js["K2"] = {"n": len(out), "frac_within_2pts": frac, "verdict": k2v,
                "median_abs_d4": float(np.median([abs(o["d4"]) for o in out])) if out else None,
                "worst": sorted(out, key=lambda o: -abs(o["d4"]))[:8]}
    md.append("## K2: w8c recovery within 2 points of fp32 recovery (k = 4 and 16)\n")
    md.append(f"Cells: {len(out)}; within 2 points at both k: {fmt(frac)} -> **{str(k2v).upper()}**; median |w8c - fp32| at k=4: "
              f"{fmt(js['K2']['median_abs_d4'])}. w4c - fp32 at k=16, median over cells: "
              f"{fmt(float(np.median([o['d16_w4c'] for o in out if o['d16_w4c'] is not None])) if out else None)}.\n")
    md.append("| family | model | cell | collapsed | w8c - fp32 (k=4) | w8c - fp32 (k=16) | w4c - fp32 (k=16) |\n|---|---|---|---|---|---|---|")
    for o in sorted(out, key=lambda o: -abs(o["d4"]))[:25]:
        md.append(f"| {o['family']} | {o['model']} | {o['cell']} | {o['collapse']} | {fmt(o['d4'])} | {fmt(o['d16'])} | {fmt(o['d16_w4c'])} |")
    md.append("")

    # POST HOC
    y = np.array([r["collapse"] for r in main_])
    ph = {"auroc_r_q": auroc([r["r_q"] for r in main_], y), "auroc_res_q": auroc([r["res_q"] for r in main_], y),
          "auroc_shift": auroc([min(r["shift"], 1e6) for r in main_], y), "auroc_gain": auroc([r["gain"] for r in main_], y)}
    fams = sorted({r["family"] for r in main_})
    ph["by_family"] = {}
    for f in fams:
        s = np.array([r["family"] == f for r in main_])
        sub = [r for r in main_ if r["family"] == f]
        ph["by_family"][f] = {"n": int(s.sum()), "n_collapse": int(y[s].sum()), "auroc_r_q": auroc([r["r_q"] for r in sub], y[s]),
                              "auroc_shift": auroc([min(r["shift"], 1e6) for r in sub], y[s]),
                              "r_q_collapsed_min": float(min([r["r_q"] for r in sub if r["collapse"]], default=float("nan"))),
                              "r_q_held_max": float(max([r["r_q"] for r in sub if not r["collapse"]], default=float("nan")))}
    # S x sigma vs measured r_q on weight-noise cells (S from the width-law table, same platform: Babel only)
    sxs = []
    if a.platform == "babel":
        sm = {w["model"]: w for w in json.load(open(HERE / "results/breadth/x2_smix.json"))["rows"]}
        for r in main_:
            if r["ctype"] == "weight-noise" and r["model"] in sm:
                sig = float(r["cell"][5:].split("#")[0])
                sxs.append({"model": r["model"], "cell": r["cell"], "S_sigma": sm[r["model"]]["S"] * sig, "r_q": r["r_q"],
                            "collapse": r["collapse"]})
        if sxs:
            ph["S_sigma_vs_r_q_spearman"] = spearman([x["S_sigma"] for x in sxs], [x["r_q"] for x in sxs])
            ph["S_sigma_over_r_q_median"] = float(np.median([x["S_sigma"] / x["r_q"] for x in sxs]))
            ph["auroc_S_sigma"] = auroc([x["S_sigma"] for x in sxs], [x["collapse"] for x in sxs])
            ph["auroc_r_q_same_cells"] = auroc([x["r_q"] for x in sxs], [x["collapse"] for x in sxs])
    js["post_hoc"] = ph
    md.append("## POST HOC (labelled): what separates collapsed from held cells\n")
    md.append(f"Pooled AUROC for collapse: r_q = ||F_q(z*) - F(z*)||/||z*|| {fmt(ph['auroc_r_q'])}; ||F_q(z*) - z*||/||z*|| "
              f"{fmt(ph['auroc_res_q'])}; linear-response shift r_q/(1-g) {fmt(ph['auroc_shift'])}; gain g {fmt(ph['auroc_gain'])}.\n")
    if sxs:
        md.append(f"Weight-noise cells with a width-law S ({len(sxs)}): Spearman(S x sigma, r_q) {fmt(ph['S_sigma_vs_r_q_spearman'])}, "
                  f"median S x sigma / r_q {fmt(ph['S_sigma_over_r_q_median'])}; AUROC S x sigma {fmt(ph['auroc_S_sigma'])} vs r_q "
                  f"{fmt(ph['auroc_r_q_same_cells'])} on the same cells.\n")
    md.append("| family | cells | collapsed | AUROC r_q | AUROC shift | min r_q collapsed | max r_q held |\n|---|---|---|---|---|---|---|")
    for f, v in ph["by_family"].items():
        md.append(f"| {f} | {v['n']} | {v['n_collapse']} | {fmt(v['auroc_r_q'],2)} | {fmt(v['auroc_shift'],2)} | {fmt(v['r_q_collapsed_min'])} | {fmt(v['r_q_held_max'])} |")
    md.append("\n| family | model | cell | retained | r_q | g | shift | res_f (exact residual at z*) |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(main_, key=lambda r: (r["family"], r["model"], r["r_q"])):
        md.append(f"| {r['family']} | {r['model']} | {r['cell']} | {fmt(r['retained'])} | {fmt(r['r_q'],4)} | {fmt(r['gain'])} | "
                  f"{fmt(r['shift'],4) if math.isfinite(r['shift']) else 'inf'} | {fmt(r['res_f'],5)} |")
    (HERE / f"results/breadth/x2_retk_readout_{a.platform}.md").write_text("\n".join(md) + "\n")
    (HERE / f"results/breadth/x2_retk_{a.platform}.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md[:30]))


if __name__ == "__main__":
    main()
