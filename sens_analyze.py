#!/usr/bin/env python3
"""X2 extended fit read-out: basin width sigma_1/2 against the one-step sensitivity S, across families.

    python sens_analyze.py [--sens 'results/breadth/babel/x2/sens/sens_*.json'] [--out results/breadth/x2_sens_fit.json]

sigma_1/2 sources (all: weight noise = W + sigma * std(W) * eps, median over draws, first sigma where the median
falls below half of fp32, log-linear interpolation; the same rule as contraction.task_curves):
  TRM/HRM/DT originals  results/breadth/x2_contraction_figdata.json (X2 grid, n=256 / 48 / 128)
  13 trained variants   results/breadth/p4d/variants_noise/<variant>{,_seed1,_seed2}.json (n_sup 16, n=1000)
  MDEQ-Small / XL       results/transfer/babel/deq_{small,xl}_*.json (Broyden k=100 top-1)
  DEQ-Transformer       results/transfer/babel/deqlm_noise_s*.json (Anderson cap 60); accuracy has no analogue,
                        so "half" = test perplexity doubles (the geometric-mean probability of the next token halves)
Quantizer cells (collapse = retained accuracy < 50% of fp32; DEQ-Transformer: perplexity > 2x fp32):
  originals from results/breadth/{sudoku_mlp_all,sudoku_attn_all,hrm_sudoku_all,maze_attn_n816}.json at n_sup 16 and
  the default H; variants from variants_noise; DT from the X2 contraction grid; MDEQ / DEQ-T from X6.
Fits: log sigma_1/2 = log delta* - log S (slope fixed at -1; one parameter) and a free-slope version; leave-one-
family-out and fit-on-TRM/HRM-predict-the-rest; quantizer collapse predicted by delta_T > delta* (delta* from the
noise fit on the training families only).
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B = HERE / "results" / "breadth"
TR = HERE / "results" / "transfer" / "babel"
SIG0 = (0.02, 0.05, 0.1)
PRIMARY = 0.05


def sigma_half(sig_acc, a0, higher_is_better=True):
    """sig_acc: {sigma: [values over draws]}; a0 fp32 value. Half = a0/2 (accuracy) or 2*a0 (perplexity)."""
    xs = sorted(sig_acc)
    med = [float(np.median(sig_acc[x])) for x in xs]
    if higher_is_better:
        thr, below = a0 / 2, [m < a0 / 2 for m in med]
    else:
        thr, below = 2 * a0, [m > 2 * a0 for m in med]
    prev = None
    for x, m, b in zip(xs, med, below):
        if b:
            if prev is None:
                return None, f"< {x}"
            x0, m0 = prev
            if higher_is_better:
                f = (m0 - thr) / max(m0 - m, 1e-12)
            else:
                f = (np.log(thr) - np.log(m0)) / max(np.log(m) - np.log(m0), 1e-12)
            return float(np.exp(np.log(x0) + f * (np.log(x) - np.log(x0)))), None
        prev = (x, m)
    return None, f"> {xs[-1]}" if xs else None


# ----------------------------------------------------------------------------- sigma_1/2 and quantizer outcomes
def widths():
    W, Q = {}, defaultdict(dict)                    # W[model] = (sigma_half, note, fp32, source); Q[model][spec] = retained
    fd = json.load(open(B / "x2_contraction_figdata.json"))
    for k, d in fd.items():
        if k.startswith("_") or d.get("period", 1) != 1:
            continue
        name = d["task"]
        W[name] = (d["sigma_half"]["value"], d["sigma_half"]["bound"], d["fp32_acc"], "X2 grid")
        if name.startswith("dt_"):
            for c in d["cells"]:
                if c["kind"] == "quant" and c["retained"] is not None:
                    Q[name][c["spec"]] = c["acc"] / d["fp32_acc"]
    for task, fn, H in [("sudoku_mlp", "sudoku_mlp_all", 3), ("sudoku_attn", "sudoku_attn_all", 3),
                        ("hrm_sudoku", "hrm_sudoku_all", 2), ("maze_attn", "maze_attn_n816", 3)]:
        recs = [r for r in json.load(open(B / f"{fn}.json"))["records"] if r["nsup"] == 16 and r["H_cycles"] == H]
        a0 = next(r["pexact"] for r in recs if r["quant"]["spec"] == "fp32")
        for r in recs:
            if r["quant"]["spec"] != "fp32":
                Q[task][r["quant"]["spec"]] = r["pexact"] / max(a0, 1e-9)
    for task, fn, H in [("hrm_maze", "hrm_maze_n816", 2)]:
        if (B / f"{fn}.json").exists():
            recs = [r for r in json.load(open(B / f"{fn}.json"))["records"] if r["nsup"] == 16 and r["H_cycles"] == H]
            if recs:
                a0 = next(r["pexact"] for r in recs if r["quant"]["spec"] == "fp32")
                for r in recs:
                    if r["quant"]["spec"] != "fp32":
                        Q[task][r["quant"]["spec"]] = r["pexact"] / max(a0, 1e-9)
    wid = B / "babel" / "x2" / "sens"
    hm = defaultdict(list)                         # HRM maze: X2 width sweep (noise 0.5-3.0, n=500, 3 seeds)
    for f in sorted(glob.glob(str(wid / "wid_hrm_maze_seed*.json"))):
        for r in json.load(open(f))["records"]:
            hm[r["quant"]["spec"]].append(r["pexact"])
    if hm.get("fp32"):
        a0 = float(np.median(hm["fp32"]))
        sh, note = sigma_half({float(k[5:]): v for k, v in hm.items() if k.startswith("noise")}, a0)
        W["hrm_maze"] = (sh, note, a0, "X2 width sweep, n=500")
    extra = [f for f in sorted(glob.glob(str(wid / "wid_*_seed*.json"))) if "hrm_maze" not in f]
    q8 = {Path(f).stem.split("_seed")[0] for f in glob.glob(str(B / "p4d" / "variants_noise" / "trm_sudoku_*.json"))}
    for f in sorted(glob.glob(str(B / "p4d" / "variants_noise" / "trm_sudoku_*.json"))) + extra:
        v = Path(f).stem.split("_seed")[0].replace("wid_", "")
        for r in json.load(open(f))["records"]:
            if r["nsup"] != 16:
                continue
            s = r["quant"]["spec"]
            if f in extra and s == "fp32" and v in q8:
                continue                           # fp32 comes from the Q8 files (same checkpoint, same n)
            if f in extra and s == "fp32" and len(W.get("_raw_" + v, {}).get("fp32", [])) >= 1:
                continue                           # one fp32 per model (identical across seed files)
            W.setdefault("_raw_" + v, defaultdict(list))[s].append(r["pexact"])
    for k in [k for k in W if k.startswith("_raw_")]:
        raw = W.pop(k)
        v = k[5:]
        a0 = float(np.median(raw["fp32"]))
        sa = {float(s[5:]): vals for s, vals in raw.items() if s.startswith("noise")}
        sh, note = sigma_half(sa, a0)
        W[v] = (sh, note, a0, "variants_noise (Q8), n=1000" if v in q8 else "X2 width sweep (Babel), n=1000")
        for s, vals in raw.items():
            if not s.startswith("noise") and s != "fp32":
                Q[v][s] = float(np.median(vals)) / max(a0, 1e-9)
    for f in sorted(glob.glob(str(wid / "wid_*_quant.json"))):      # quantizer cells of X2-swept models
        v = Path(f).stem.replace("wid_", "").replace("_quant", "")
        recs = [r for r in json.load(open(f))["records"] if r["nsup"] == 16]
        a0 = next((r["pexact"] for r in recs if r["quant"]["spec"] == "fp32"), None)
        for r in recs:
            if a0 and r["quant"]["spec"] != "fp32":
                Q[v][r["quant"]["spec"]] = r["pexact"] / max(a0, 1e-9)
    # MDEQ
    for model, pat in (("mdeq_small", "deq_small_*.json"), ("mdeq_xl", "deq_xl_*.json")):
        sa, a0s, qd = defaultdict(list), [], {}
        for f in sorted(glob.glob(str(TR / pat))):
            for r in json.load(open(f))["records"]:
                if r.get("solver") != "broyden":
                    continue
                top1 = next(b["top1"] for b in r["by_budget"] if b["budget"] == 100)
                if r["spec"] == "fp32":
                    a0s.append(top1)
                elif r["spec"].startswith("noise"):
                    sa[float(r["spec"][5:])].append(top1)
                else:
                    qd[r["spec"]] = top1
        a0 = float(np.median(a0s))
        sh, note = sigma_half(sa, a0)
        W[model] = (sh, note, a0, "X6 Broyden k=100")
        for s, v in qd.items():
            Q[model][s] = v / a0
    # DEQ-Transformer (perplexity)
    sa, a0s, qd = defaultdict(list), [], {}
    for f in sorted(glob.glob(str(TR / "deqlm_*.json"))):
        for r in json.load(open(f))["records"]:
            if r["solver"] != "anderson" or r["budget"] != 60:
                continue
            if r["spec"] == "fp32":
                a0s.append(r["ppl"])
            elif r["spec"].startswith("noise"):
                sa[float(r["spec"][5:])].append(r["ppl"])
            else:
                qd[r["spec"]] = r["ppl"]
    a0 = float(np.median(a0s))
    sh, note = sigma_half(sa, a0, higher_is_better=False)
    W["deq_transformer"] = (sh, note, a0, "X6 Anderson cap 60; half = ppl doubles")
    for s, v in qd.items():
        Q["deq_transformer"][s] = a0 / v                 # "retained" analogue: < 0.5 iff ppl > 2x fp32
    return W, Q


def lm_cells():
    """Looped-LM quantizer cells (X4 read-out, analyze_looped.boundary_cells): retained greedy GSM8K accuracy
    over bf16 at the same loop count, bf16 >= 10%; duplicates (Lambda grid and X4 harness) averaged."""
    import analyze_looped as AL
    rows = AL.boundary_cells(AL.load(AL.DEFAULT_ROOTS))
    agg = defaultdict(list)
    for r in rows:
        if r["acc_bf16"] >= 10:
            agg[(r["model"], int(r["loops"]), r["quant"])].append(r["acc"] / r["acc_bf16"])
    return {k: float(np.mean(v)) for k, v in agg.items()}


def family(m):
    if m.startswith(("ouro", "huginn", "olmo")):
        return "LM"
    if m.startswith("hrm"):
        return "HRM"
    if m.startswith("dt_"):
        return "DT"
    if m.startswith("mdeq") or m.startswith("deq"):
        return "DEQ"
    return "TRM"


# ----------------------------------------------------------------------------- fits
def fit_delta(rows):
    """one-parameter law: log sigma_1/2 = log delta* - log S  ->  log delta* = mean(log sigma_1/2 + log S)."""
    return float(np.exp(np.mean([np.log(r["sigma_half"] * r["S"]) for r in rows])))


def fit_free(rows):
    x = np.log([r["S"] for r in rows])
    y = np.log([r["sigma_half"] for r in rows])
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def r2(rows, delta):
    y = np.log([r["sigma_half"] for r in rows])
    p = np.log([delta / r["S"] for r in rows])
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens", default=str(B / "babel" / "x2" / "sens" / "sens_*.json"))
    ap.add_argument("--out", default=str(B / "x2_sens_fit.json"))
    ap.add_argument("--sigma0", type=float, default=PRIMARY)
    ap.add_argument("--min_fp32", type=float, default=0.3, help="exclude models whose fp32 accuracy is below this")
    a = ap.parse_args(argv)
    W, Q = widths()
    S, D = {}, {}
    for f in sorted(glob.glob(a.sens)):
        j = json.load(open(f))
        m = j["variant"]
        by = defaultdict(list)
        for r in j["records"]:
            if r["sigma"] is not None and "@" not in r["spec"]:
                by[r["sigma"]].append(r["delta_med"])
            elif r["sigma"] is None:
                D.setdefault(m, {})[r["spec"]] = r["delta_med"]
        S[m] = {s: float(np.median(v)) / s for s, v in by.items()}
        S[m]["_n"] = j["n"]
        S[m]["_fp32_here"] = j.get("fp32_acc", j.get("fp32_ppl"))
    rows, skipped = [], []
    for m, (sh, note, a0, src) in sorted(W.items()):
        if m not in S:
            skipped.append((m, "no sensitivity measured yet"))
            continue
        if a.sigma0 not in S[m]:
            skipped.append((m, f"no delta at sigma {a.sigma0}"))
            continue
        low = family(m) != "DEQ" and m != "deq_transformer" and a0 < a.min_fp32
        rows.append({"model": m, "family": family(m), "sigma_half": sh, "bound": note, "fp32": a0, "source": src,
                     "S": S[m][a.sigma0], "S_all": {str(k): v for k, v in S[m].items() if not str(k).startswith("_")},
                     "low_fp32": bool(low)})
    fitrows = [r for r in rows if r["sigma_half"] is not None and not r["low_fp32"]]
    res = {"sigma0": a.sigma0, "rows": rows, "skipped": skipped}
    # all-model fit
    if len(fitrows) >= 3:
        d_all = fit_delta(fitrows)
        res["all"] = {"n": len(fitrows), "delta_star": d_all, "r2_slope_minus1": r2(fitrows, d_all),
                      "free_fit_intercept_slope": fit_free(fitrows)}
    # leave-one-family-out and train on TRM+HRM
    res["held_out"] = {}
    fams = sorted({r["family"] for r in fitrows})
    for hold in fams + ["TRM+HRM -> DT, DEQ"]:
        if hold == "TRM+HRM -> DT, DEQ":
            train = [r for r in fitrows if r["family"] in ("TRM", "HRM")]
            test = [r for r in rows if r["family"] in ("DT", "DEQ")]
        else:
            train = [r for r in fitrows if r["family"] != hold]
            test = [r for r in rows if r["family"] == hold]
        if len(train) < 2 or not test:
            continue
        d = fit_delta(train)
        preds = []
        for r in test:
            p = d / r["S"]
            preds.append({"model": r["model"], "measured": r["sigma_half"], "bound": r["bound"], "predicted": p,
                          "ratio": (p / r["sigma_half"]) if r["sigma_half"] else None, "low_fp32": r["low_fp32"]})
        rat = [abs(np.log(x["ratio"])) for x in preds if x["ratio"] and not x["low_fp32"]]
        res["held_out"][hold] = {"train_n": len(train), "delta_star": d, "preds": preds,
                                 "median_abs_log_ratio": float(np.median(rat)) if rat else None,
                                 "max_ratio_factor": float(np.exp(max(rat))) if rat else None}
    # leave-one-model-out
    loo = []
    for r in fitrows:
        d = fit_delta([o for o in fitrows if o is not r])
        loo.append({"model": r["model"], "ratio": d / r["S"] / r["sigma_half"]})
    res["loo"] = loo
    # quantizer collapse
    train = [r for r in fitrows if r["family"] in ("TRM", "HRM")]
    dq = fit_delta(train) if len(train) >= 2 else None
    qc = []
    for m, specs in Q.items():
        for s, ret in specs.items():
            if m in D and s in D[m]:
                qc.append({"model": m, "family": family(m), "spec": s, "retained": ret, "delta": D[m][s],
                           "collapse": ret < 0.5, "pred_collapse": (D[m][s] > dq) if dq else None})
    # looped LMs: delta of loop L at the bf16 state after L loops (sens_lm.py), labels from X4
    try:
        lab = lm_cells()
    except Exception as e:                        # the X4 read-out needs its result trees
        lab, res["lm_error"] = {}, repr(e)
    for f in sorted(glob.glob(str(Path(a.sens).parent / "lm_*.json"))):
        j = json.load(open(f))
        for r in j["records"]:
            key = (j["model"], r["loops"], r["spec"])
            if key in lab:
                qc.append({"model": f"{j['model']}@{r['loops']}", "family": "LM", "spec": r["spec"],
                           "retained": lab[key], "delta": r["delta_med"], "collapse": lab[key] < 0.5,
                           "pred_collapse": (r["delta_med"] > dq) if dq else None})
    res["quant_cells"] = qc
    res["quant_delta_star_trm_hrm"] = dq
    for fam in ["TRM", "HRM", "DT", "DEQ", "LM", "ALL"]:
        cs = [c for c in qc if fam == "ALL" or c["family"] == fam]
        if not cs or dq is None:
            continue
        acc = float(np.mean([c["collapse"] == c["pred_collapse"] for c in cs]))
        pos = [c["delta"] for c in cs if c["collapse"]]
        neg = [c["delta"] for c in cs if not c["collapse"]]
        au = None
        if pos and neg:
            au = float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))
        res.setdefault("quant_by_family", {})[fam] = {"n": len(cs), "n_collapse": len(pos), "accuracy": acc,
                                                      "auroc_delta": au}
    json.dump(res, open(a.out, "w"), indent=1)
    # print
    f2 = lambda x, p=2: "--" if x is None else f"{x:.{p}f}"
    print(f"### sensitivity at sigma0 = {a.sigma0} (S = median over draws of median-over-inputs delta_T / sigma0)")
    print("| model | family | fp32 | sigma_1/2 | S(0.02) | S(0.05) | S(0.1) | S x sigma_1/2 | source |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        sa = r["S_all"]
        print(f"| {r['model']}{' (low fp32)' if r['low_fp32'] else ''} | {r['family']} | {f2(r['fp32'], 3)} | "
              f"{f2(r['sigma_half']) if r['sigma_half'] else r['bound']} | {f2(sa.get('0.02'))} | {f2(sa.get('0.05'))} | "
              f"{f2(sa.get('0.1'))} | {f2(r['S'] * r['sigma_half']) if r['sigma_half'] else '--'} | {r['source']} |")
    if "all" in res:
        A = res["all"]
        print(f"\nall fitted models (n={A['n']}): delta* = {A['delta_star']:.3f}, R^2 (log, slope -1) = "
              f"{A['r2_slope_minus1']:.2f}; free fit log sigma_1/2 = {A['free_fit_intercept_slope'][0]:.2f} + "
              f"{A['free_fit_intercept_slope'][1]:.2f} log S")
    for k, h in res["held_out"].items():
        print(f"\nheld out {k}: fit on {h['train_n']} models, delta* {h['delta_star']:.3f}; median |log ratio| "
              f"{f2(h['median_abs_log_ratio'])} (x{f2(np.exp(h['median_abs_log_ratio']) if h['median_abs_log_ratio'] is not None else None)}), "
              f"worst x{f2(h['max_ratio_factor'])}")
        for p in h["preds"]:
            print(f"   {p['model']}: predicted {p['predicted']:.3f} vs measured "
                  f"{f2(p['measured']) if p['measured'] else p['bound']} (ratio {f2(p['ratio'])})"
                  + (" [low fp32, not scored]" if p["low_fp32"] else ""))
    if loo:
        lr = [abs(np.log(x["ratio"])) for x in loo]
        print(f"\nleave-one-model-out: median x{np.exp(np.median(lr)):.2f}, worst x{np.exp(max(lr)):.2f} "
              f"({max(loo, key=lambda x: abs(np.log(x['ratio'])))['model']})")
    if res.get("quant_by_family"):
        print(f"\nquantizer collapse predicted by delta_T > delta* = {dq:.3f} (fit on TRM+HRM noise widths):")
        for fam, v in res["quant_by_family"].items():
            print(f"   {fam}: {v['n']} cells ({v['n_collapse']} collapse), accuracy {v['accuracy']:.2f}, "
                  f"AUROC(delta) {f2(v['auroc_delta'])}")
    if skipped:
        print("\nskipped:", skipped)
    print("wrote", a.out)
    return res


if __name__ == "__main__":
    main()
