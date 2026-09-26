#!/usr/bin/env python3
"""X2 S_mix read-out (hypotheses H1-H3 in reports/edgetrm_exp/X2.md, "Mixer-restricted sensitivity S_mix").

All Babel L40S. Width sigma_1/2 (3 noise draws, n=1000, n_sup 16; median-over-draws half-accuracy crossing):
  released TRM/HRM models   wid_<task>_n1000_seed{0,1,2}.json  (X2 evergreen sweeps)
  13 trained variants       wid_trm_sudoku_<v>_babel_seed{0,1,2}.json
  X5's 6 new seeds          wid_x5_<m>_seed{0,1,2}.json
Sensitivities at sigma 0.05 (median over 3 draws of median-over-inputs delta_T / 0.05):
  S      all perturbed weights (sens_*.json / sens_var_*.json)
  S_mix  token mixer only (self_attn or mlp_t), S_mlp channel SwiGLU only (smix_<model>.json)
Laws: sigma_1/2 = d/S, = d/S_mix (one parameter each), log sigma_1/2 = a - b1 log S_mix - b2 log S_mlp (three).
Scores: R^2 on log sigma_1/2 (fit set: fp32 >= 30%, the 5 released + 13 variants; the 6 new seeds held out),
leave-one-model-out on the fit set, errors on the 6 held-out seeds, and the mean log residual per mixer (bias).

    python x2_smix_analyze.py
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import sens_analyze as A

HERE = Path(__file__).resolve().parent
D = HERE / "results" / "breadth" / "babel" / "x2" / "sens"
REL = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze_attn", "hrm_maze"]
VAR = ["attn_w256_s0", "attn_w256_s1", "attn_w256_s2", "attn_w512_s0", "attn_w512_s1", "attn_w512_s2", "mlp_w256_s0",
       "mlp_w256_s1", "mlp_w256_s2", "mlp_w512_s0", "mlp_w512_s1", "mlp_w512_s2", "mlp_w768_s0"]
NEW = ["attn512_s3", "attn512_s4", "attn512_s5", "mlp512_s3", "mlp512_s4", "mlp512_s5"]


def width(stem):
    fs = sorted(glob.glob(str(D / f"wid_{stem}_seed*.json")))
    if len(fs) < 3:
        return None, None
    sa, a0 = defaultdict(list), None
    for f in fs:
        for r in json.load(open(f))["records"]:
            s = r["quant"]["spec"]
            if s == "fp32":
                a0 = a0 or r["pexact"]
            elif s.startswith("noise"):
                sa[float(s[5:])].append(r["pexact"])
    sh, note = A.sigma_half(sa, a0)
    return (sh, note), a0


def s_of(path, spec_prefix):
    j = json.load(open(path))
    v = [r["delta_med"] for r in j["records"] if r["spec"] == spec_prefix]
    return float(np.median(v)) / 0.05 if len(v) >= 3 else None


def mixer_of(name):
    return "MLP" if ("mlp" in name and "attn" not in name) else ("HRM" if name.startswith("hrm") else "attn")


def main():
    rows = []
    for m in REL + [f"trm_sudoku_{v}" for v in VAR] + [f"x5_{m}" for m in NEW]:
        stem = f"{m}_n1000" if m in REL else (f"{m}_babel" if m.startswith("trm_sudoku") else m)
        (w, a0) = width(stem)
        sf = D / (f"sens_{m}.json" if m in REL else f"sens_var_{m.replace('trm_sudoku_', '')}.json")
        if m.startswith("x5_"):
            sf = D / f"sens_var_{m}.json"
        mf = D / f"smix_{m}.json"
        if not (w and sf.exists() and mf.exists()):
            print("missing", m, bool(w), sf.exists(), mf.exists())
            continue
        rows.append({"model": m, "mixer": mixer_of(m.replace("x5_", "")), "sigma_half": w[0], "bound": w[1], "fp32": a0,
                     "S": s_of(sf, "noise0.05"), "S_mix": s_of(mf, "noise0.05@mixer"), "S_mlp": s_of(mf, "noise0.05@mlp"),
                     "held_out": m.startswith("x5_")})
    fit = [r for r in rows if not r["held_out"] and r["sigma_half"] and r["fp32"] >= 0.3]
    new = [r for r in rows if r["held_out"] and r["sigma_half"]]
    y = np.log([r["sigma_half"] for r in fit])
    out = {"rows": rows, "laws": {}}

    def one(key):
        def f(train):
            return float(np.mean([np.log(r["sigma_half"] * r[key]) for r in train]))
        return (lambda train, r: f(train) - np.log(r[key])), 1

    def two():
        def fitp(train):
            X = np.c_[np.ones(len(train)), np.log([r["S_mix"] for r in train]), np.log([r["S_mlp"] for r in train])]
            return np.linalg.lstsq(X, np.log([r["sigma_half"] for r in train]), rcond=None)[0]
        return (lambda train, r: float(np.dot(fitp(train), [1, np.log(r["S_mix"]), np.log(r["S_mlp"])]))), 3

    for name, (pred, k) in {"S": one("S"), "S_mix": one("S_mix"), "S_mix+S_mlp": two()}.items():
        p_in = np.array([pred(fit, r) for r in fit])
        r2 = 1 - np.sum((y - p_in) ** 2) / np.sum((y - y.mean()) ** 2)
        loo = np.array([pred([o for o in fit if o is not r], r) - np.log(r["sigma_half"]) for r in fit])
        nw = np.array([pred(fit, r) - np.log(r["sigma_half"]) for r in new])
        bias = {mx: float(np.mean([p - np.log(r["sigma_half"]) for p, r in zip(p_in, fit) if r["mixer"] == mx]))
                for mx in ("MLP", "attn", "HRM")}
        out["laws"][name] = {"params": k, "r2": float(r2), "loo_med": float(np.exp(np.median(np.abs(loo)))),
                             "loo_worst": float(np.exp(np.max(np.abs(loo)))), "new_ratios": np.exp(nw).round(3).tolist(),
                             "new_med": float(np.exp(np.median(np.abs(nw)))) if len(nw) else None,
                             "new_worst": float(np.exp(np.max(np.abs(nw)))) if len(nw) else None,
                             "bias_ratio_by_mixer": {kk: float(np.exp(v)) for kk, v in bias.items()}}
    json.dump(out, open(HERE / "results" / "breadth" / "x2_smix.json", "w"), indent=1)
    print("| model | mixer | fp32 | sigma_1/2 | S | S_mix | S_mlp | S_mix/S |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']}{' (held out)' if r['held_out'] else ''} | {r['mixer']} | {r['fp32']:.3f} | "
              f"{r['sigma_half']:.3f} | {r['S']:.2f} | {r['S_mix']:.2f} | {r['S_mlp']:.2f} | {r['S_mix']/r['S']:.2f} |"
              if r["sigma_half"] else f"| {r['model']} | {r['mixer']} | {r['fp32']:.3f} | {r['bound']} | {r['S']:.2f} | "
              f"{r['S_mix']:.2f} | {r['S_mlp']:.2f} | {r['S_mix']/r['S']:.2f} |")
    print(f"\nfit set n={len(fit)} (fp32 >= 30%), held-out new seeds n={len(new)}")
    print("| law | params | R^2 | LOO median / worst | new seeds: ratios, median / worst | fit bias (pred/meas) MLP / attn / HRM |")
    print("|---|---|---|---|---|---|")
    for name, L in out["laws"].items():
        b = L["bias_ratio_by_mixer"]
        print(f"| {name} | {L['params']} | {L['r2']:.2f} | x{L['loo_med']:.2f} / x{L['loo_worst']:.2f} | "
              f"{L['new_ratios']}, x{L['new_med']:.2f} / x{L['new_worst']:.2f} | {b['MLP']:.2f} / {b['attn']:.2f} / {b['HRM']:.2f} |")


if __name__ == "__main__":
    main()
