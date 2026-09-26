#!/usr/bin/env python3
"""X2 contraction loss (A) vs readout margin (B): scored by the rules in results/breadth/x2_margin_predictions.md
(written before any margin record existed). Babel records only (retm_*.json).

    python x2_margin_analyze.py -> results/breadth/x2_margin_readout.md, x2_margin.json
"""
import glob
import json
import math
from pathlib import Path

import numpy as np

from x2_return_analyze import family, ctype, auroc, spearman, fmt

HERE = Path(__file__).resolve().parent
RES = HERE / "results/breadth/babel/x2_retm"
SIG0 = 0.05
TR_MAX = 128.0


def main():
    rows, models = [], {}
    for f in sorted(glob.glob(str(RES / "retm_*.json"))):
        b = json.load(open(f))
        cf = np.array(b["corr_f"], bool)
        models.setdefault(b["model"], {"r_f": float(np.median(np.array(b["r_f"])[cf])) if cf.any() else float(np.median(b["r_f"])),
                                       "res_f": float(np.median(np.array(b["res_f"])[cf])) if cf.any() else float(np.median(b["res_f"])),
                                       "fp32": b["fp32_acc"], "family": family(b["model"], b["family"])})
        for r in b["records"]:
            tr = np.array([np.inf if v is None else v for v in r["per"]["t_r"]], float)
            sel = cf if cf.any() else np.ones(len(tr), bool)
            rows.append({"model": b["model"], "family": family(b["model"], b["family"]), "cell": r["cell"], "ctype": ctype(r["cell"]),
                         "acc_f": r["acc_f"], "retained": r["retained"], "collapse": r["retained"] < 0.5,
                         "r_q": r["r_q_median"], "res_q": r["res_q_median"], "disp": r["disp_median"],
                         "t_r": float(np.median(tr[sel])), "t_r_cens": float(np.mean(~np.isfinite(tr[sel])))})
    rows = [r for r in rows if r["acc_f"] >= 0.10]
    main_ = [r for r in rows if r["ctype"] in ("weight-quant", "act-quant", "weight-noise")]
    col = [r for r in main_ if r["collapse"]]
    held = [r for r in main_ if not r["collapse"]]
    fa = float(np.mean([r["r_q"] >= 1 for r in col])) if col else None
    fh = float(np.mean([r["r_q"] < 1 for r in held])) if held else None
    fb = float(np.mean([r["r_q"] < 1 for r in col])) if col else None
    js = {"n_cells": len(main_), "n_collapsed": len(col), "A_frac_collapsed_r_ge1": fa, "A_frac_held_r_lt1": fh,
          "B_frac_collapsed_r_lt1": fb, "auroc_r_q": auroc([r["r_q"] for r in main_], [r["collapse"] for r in main_]),
          "auroc_res_q": auroc([r["res_q"] for r in main_], [r["collapse"] for r in main_]),
          "auroc_neg_t_r": auroc([-min(r["t_r"], 1e6) for r in main_], [r["collapse"] for r in main_])}
    js["A_verdict"] = "supported" if (fa is not None and fa >= 0.8 and fh >= 0.8) else ("partly" if (fa or 0) >= 0.8 or (fh or 0) >= 0.8 else "falsified")
    # B clause 2: zero-parameter width
    sm = json.load(open(HERE / "results/breadth/x2_smix.json"))
    laws = sm["laws"]
    tab = []
    for w in sm["rows"]:
        r = next((x for x in rows if x["model"] == w["model"] and x["cell"] == f"noise{SIG0}"), None)
        if r is None:
            continue
        tr = r["t_r"] if math.isfinite(r["t_r"]) else TR_MAX
        tab.append({"model": w["model"], "mixer": w["mixer"], "held_out": w["held_out"], "fp32": w["fp32"],
                    "sigma_half": w["sigma_half"], "t_r": r["t_r"], "t_r_cens": r["t_r_cens"], "pred": SIG0 * tr,
                    "disp": r["disp"], "r_q": r["r_q"]})
    fit = [t for t in tab if not t["held_out"] and t["fp32"] >= 0.30]
    new = [t for t in tab if t["held_out"]]

    def bias(ts, c=1.0):
        return {mx: float(math.exp(np.mean([math.log(c * t["pred"] / t["sigma_half"]) for t in ts if t["mixer"] == mx])))
                for mx in sorted({t["mixer"] for t in ts})}
    wl = {"n_fit": len(fit), "n_new": len(new)}
    if fit:
        c = math.exp(np.mean([math.log(t["sigma_half"] / t["pred"]) for t in fit]))
        wl.update({"fit_bias_param_free": bias(fit), "scale_1p": c, "fit_bias_1p": bias(fit, c)})
    if new:
        lr0 = [abs(math.log(t["pred"] / t["sigma_half"])) for t in new]
        wl.update({"new_bias_param_free": bias(new), "new_ratios_param_free": [round(t["pred"] / t["sigma_half"], 3) for t in new],
                   "new_med_param_free": float(math.exp(np.median(lr0))), "new_worst_param_free": float(math.exp(max(lr0)))})
        if fit:
            lr1 = [abs(math.log(wl["scale_1p"] * t["pred"] / t["sigma_half"])) for t in new]
            wl.update({"new_med_1p": float(math.exp(np.median(lr1))), "new_bias_1p": bias(new, wl["scale_1p"])})
        nb = wl["new_bias_param_free"]
        b2 = all(0.9 <= nb.get(mx, 0) <= 1.1 for mx in ("MLP", "attn")) and wl["new_med_param_free"] < laws["S_mix"]["new_med"] \
            and wl["new_med_param_free"] < laws["S"]["new_med"]
    else:
        b2 = None
    js["width"] = wl
    js["width_rows"] = tab
    b1 = fb is not None and fb >= 0.8
    js["B_clause1"] = b1
    js["B_clause2"] = b2
    js["B_verdict"] = "supported" if (b1 and b2) else ("partly" if (b1 or b2) else "falsified")
    # amplification: disp vs one-step residual / (1 - r_f)
    js["models"] = models
    md = ["# X2 contraction loss (A) vs readout margin (B): read-out (Babel; `python x2_margin_analyze.py`)\n",
          f"Rules: `x2_margin_predictions.md` (02:47 UTC, before any margin record). Cells {len(main_)} (weight/act quant, weight noise; "
          f"fp32 >= 10%), collapsed {len(col)}.\n",
          "| test | value | rule |\n|---|---|---|",
          f"| A: collapsed cells with late contraction ratio r >= 1 | {fmt(fa)} | >= 0.80 |",
          f"| A: held cells with r < 1 | {fmt(fh)} | >= 0.80 |",
          f"| B: collapsed cells with r < 1 | {fmt(fb)} | >= 0.80 |",
          f"| B: zero-parameter width sigma_0 x median t_r(sigma_0), new seeds: bias MLP / attn | "
          f"{' / '.join(fmt(wl.get('new_bias_param_free', {}).get(m), 2) for m in ('MLP', 'attn'))} | both in [0.9, 1.1] |",
          f"| B: new seeds median / worst ratio (zero-parameter) | x{fmt(wl.get('new_med_param_free'), 2)} / x{fmt(wl.get('new_worst_param_free'), 2)} "
          f"| < S x{laws['S']['new_med']:.2f} and S_mix x{laws['S_mix']['new_med']:.2f} |",
          f"\n**A: {js['A_verdict'].upper()}. B: {js['B_verdict'].upper()}** (clause 1 {b1}, clause 2 {b2}).\n",
          f"Pooled AUROC for collapse: r {fmt(js['auroc_r_q'])}, compressed residual {fmt(js['auroc_res_q'])}, -t_r {fmt(js['auroc_neg_t_r'])}.\n"]
    if fit:
        md.append(f"Fit set ({len(fit)}): zero-parameter bias {json.dumps({k: round(v, 2) for k, v in wl['fit_bias_param_free'].items()})}; "
                  f"1-parameter scale {wl['scale_1p']:.3f}, bias {json.dumps({k: round(v, 2) for k, v in wl['fit_bias_1p'].items()})}; "
                  f"new seeds with that scale x{fmt(wl.get('new_med_1p'), 2)}, bias {json.dumps({k: round(v, 2) for k, v in wl.get('new_bias_1p', {}).items()})}. "
                  f"S fit bias {json.dumps({k: round(v, 2) for k, v in laws['S']['bias_ratio_by_mixer'].items()})}, S_mix "
                  f"{json.dumps({k: round(v, 2) for k, v in laws['S_mix']['bias_ratio_by_mixer'].items()})}.\n")
    md.append("| model | mixer | held out | sigma_1/2 | median t_r at sigma 0.05 (censored frac) | pred 0.05 x t_r | disp at 0.05 | r at 0.05 |\n|---|---|---|---|---|---|---|---|")
    for t in tab:
        md.append(f"| {t['model']} | {t['mixer']} | {t['held_out']} | {t['sigma_half']:.3f} | {fmt(t['t_r'], 2)} ({t['t_r_cens']:.2f}) | "
                  f"{t['pred']:.3f} | {t['disp']:.4f} | {t['r_q']:.3f} |")
    md.append("\n| model | family | fp32 | fp32 late ratio L_0 | fp32 residual |\n|---|---|---|---|---|")
    for m, v in sorted(models.items()):
        md.append(f"| {m} | {v['family']} | {v['fp32']:.3f} | {v['r_f']:.3f} | {v['res_f']:.2e} |")
    md.append("\n| family | model | cell | retained | r | residual | t_r | disp |\n|---|---|---|---|---|---|---|---|")
    for r in sorted(main_, key=lambda r: (r["family"], r["model"], r["retained"])):
        md.append(f"| {r['family']} | {r['model']} | {r['cell']} | {fmt(r['retained'])} | {fmt(r['r_q'])} | {fmt(r['res_q'], 4)} | "
                  f"{fmt(r['t_r'], 2) if math.isfinite(r['t_r']) else 'inf'} | {fmt(r['disp'], 4)} |")
    (HERE / "results/breadth/x2_margin_readout.md").write_text("\n".join(md) + "\n")
    (HERE / "results/breadth/x2_margin.json").write_text(json.dumps(js, indent=1, default=float))
    print("\n".join(md[:16]))


if __name__ == "__main__":
    main()
