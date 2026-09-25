#!/usr/bin/env python3
"""M3 read-out: weight-allocation Pareto tables, score predictiveness, activation curves, seed transfer.

    python m3_analyze.py --weights 'results/m3/w_*.json' --acts 'results/m3/act_*.json' \
        --xfer 'results/m3/wT_*.json' --md results/m3/m3_readout.md --fig figures/m3_pareto.pdf
"""
import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a))
    r[o] = np.arange(len(a))
    # average ties
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y):
    if len(x) < 3:
        return float("nan")
    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def short(g):
    return (g.replace("inner.", "").replace("L_level.layers.", "L").replace("H_level.layers.", "H")
            .replace(".mlp_t.", ".mix.").replace(".self_attn.", ".attn.").replace("gate_up_proj", "gu")
            .replace("down_proj", "dn").replace("qkv_proj", "qkv").replace("o_proj", "o"))


SCORES = ["dT_q", "dT_noise", "dT_traj", "kl", "lin", "hawq", "hawq_full", "hawq_lbl", "dT_greedy", "small_first"]


def weights_readout(files, out):
    rows_all = []
    for f in sorted(files):
        j = json.load(open(f))
        if "evals" not in j or "fp32" not in j:
            continue
        E = j["evals"]
        name = Path(f).stem
        n = j["n"]
        fp = j["fp32"]["exact"]
        out.append(f"\n### {name}: {j['task']}, low {j['lo']} (clip {j.get('clip', 'none')}), high {j['hi']}, "
                   f"n={n}, T={j['T']}{', ckpt ' + Path(j['ckpt']).parent.name if j.get('ckpt') else ''}")
        out.append(f"fp32 exact {100*fp:.1f}. Uniform formats:")
        out.append("| format | avg bits | exact % [95% CI] | cell % | fidelity |")
        out.append("|---|---|---|---|---|")
        for k, r in sorted(E.items(), key=lambda kv: kv[1]["bits"]):
            if r.get("method") != "uniform":
                continue
            lo, hi = wilson(sum(r["correct"]), len(r["correct"]))
            out.append(f"| {r.get('fmt', k)} | {r['bits']:.3f} | {100*r['exact']:.1f} [{100*lo:.1f}, {100*hi:.1f}] | "
                       f"{100*r['cell']:.1f} | {r['fidelity_H']:.3f} |")
        for cname in ("curve", "curve_prefix"):
            weights_curve(j, E, cname, out)
        rows_all.append((name, j))
        weights_scores(j, E, out)
    return rows_all


def weights_curve(j, E, cname, out):
        curve = j.get(cname, {})
        if not curve:
            return
        budgets = sorted({b for c in curve.values() for b in c}, key=float)
        meths = [m for m in SCORES if m in curve] + sorted(m for m in curve if m.startswith("xfer_"))
        rnd = sorted(m for m in curve if m.startswith("random"))
        out.append(f"\nMixed per-tensor allocation, {'knapsack fill' if cname == 'curve' else 'strict prefix of the ranking'} "
                   "(exact %, realised bits in parentheses); random = mean [min, max] over draws:")
        out.append("| budget | " + " | ".join(meths) + " | random |")
        out.append("|---|" + "---|" * (len(meths) + 1))
        for b in budgets:
            cells = []
            for m in meths:
                k = curve[m].get(b)
                r = E.get(k) if k else None
                cells.append(f"{100*r['exact']:.1f} ({r['bits']:.2f})" if r else "-")
            rv = [E[curve[m][b]]["exact"] for m in rnd if b in curve[m] and curve[m][b] in E]
            cells.append(f"{100*np.mean(rv):.1f} [{100*min(rv):.1f}, {100*max(rv):.1f}]" if rv else "-")
            out.append(f"| {float(b):.2f} | " + " | ".join(cells) + " |")


def weights_scores(j, E, out):
        # score predictiveness against the single-tensor ground truth
        sc = j.get("scores", {})
        groups = [g["name"] for g in j["groups"]]
        lo_single = {g: E.get(f"single_lo:{g}") for g in groups}
        hi_single = {g: E.get(f"single_hi:{g}") for g in groups}
        allhi = E.get(f"uniform:{j['hi']}")
        if all(lo_single.values()) and allhi:
            drop = {g: allhi["exact"] - lo_single[g]["exact"] for g in groups}
            base_lo = E.get(f"uniform:{j['lo']}") if j.get("clip", "none") == "none" else E.get(f"uniform_lo_clip:{j['clip']}")
            gain = {g: hi_single[g]["exact"] - (base_lo["exact"] if base_lo else 0) for g in groups} if all(hi_single.values()) else None
            out.append("\nScore predictiveness (Spearman over tensors): against the accuracy drop with the tensor alone low "
                       "(rest high), and the gain with it alone high (rest low):")
            out.append("| score | rho vs drop(alone low) | rho vs gain(alone high) | top tensor |")
            out.append("|---|---|---|---|")
            for m in ["dT_q", "dT_noise", "dT_traj", "kl", "lin", "lin_mean", "hawq", "hawq_full", "hawq_lbl"]:
                if m not in sc or not isinstance(sc[m], dict):
                    continue
                x = [sc[m][g] for g in groups]
                r1 = spearman(x, [drop[g] for g in groups])
                r2 = spearman(x, [gain[g] for g in groups]) if gain else float("nan")
                top = max(groups, key=lambda g: sc[m][g])
                out.append(f"| {m} | {r1:+.2f} | {r2:+.2f} | {short(top)} |")
            out.append("\nPer tensor: size, exact with it alone low (rest high) / alone high (rest low):")
            out.append("| tensor | numel | alone low | alone high |")
            out.append("|---|---|---|---|")
            for gi in j["groups"]:
                g = gi["name"]
                out.append(f"| {short(g)} | {gi['numel']} | {100*lo_single[g]['exact']:.1f} | "
                           f"{100*hi_single[g]['exact']:.1f} |" if hi_single[g] else f"| {short(g)} | {gi['numel']} | "
                           f"{100*lo_single[g]['exact']:.1f} | - |")


def acts_readout(files, out):
    for f in sorted(files):
        j = json.load(open(f))
        E = j.get("evals", {})
        if not E:
            continue
        name = Path(f).stem
        ref16, ref8 = E.get("eval|all16"), E.get("eval|all8")
        b2 = E.get("eval|x3_B2_rms16")
        out.append(f"\n### {name}: {j['task']} activations ({j['n_tensors']} tensors, T={j['T']}, "
                   f"{j.get('device', j.get('gpu', '?'))}, {j.get('n_score', 64)} scoring puzzles"
                   f"{', weights ' + j['wfmt'] if j.get('wfmt') else ''})")
        if ref16 and ref8:
            out.append(f"all 16-bit {100*ref16['exact']:.0f}/100, all 8-bit {100*ref8['exact']:.0f}/100"
                       + (f", X3 B2 (every RMSNorm tensor at 16: {b2['n16']} tensors, {100*b2['frac16_el']:.1f}% of "
                          f"activation elements) {100*b2['exact']:.0f}/100" if b2 else ""))
        curve = j.get("curve", {})
        ks = sorted({int(k) for c in curve.values() for k in c})
        meths = [m for m in ["greedy", "a_dTrm", "a_dT", "a_noise", "a_kl", "range"] if m in curve]
        rnd = sorted(m for m in curve if m.startswith("random"))
        out.append("\nexact /100 with the top-k tensors of each score at 16 bit (rest 8); [% of activation elements at 16]:")
        out.append("| k | " + " | ".join(meths) + " | random (mean) |")
        out.append("|---|" + "---|" * (len(meths) + 1))
        for k in ks:
            cells = []
            for m in meths:
                r = E.get(curve[m].get(str(k), ""))
                cells.append(f"{100*r['exact']:.0f} [{100*r['frac16_el']:.1f}]" if r else "-")
            rv = [E[curve[m][str(k)]]["exact"] for m in rnd if str(k) in curve[m] and curve[m][str(k)] in E]
            cells.append(f"{100*np.mean(rv):.0f}" if rv else "-")
            out.append(f"| {k} | " + " | ".join(cells) + " |")
        dm = j.get("delta_of", {})
        if dm and ref16:
            xs, ys = [], []
            for key, d in dm.items():
                r = E.get(f"eval|{key}")
                if r:
                    xs.append(d)
                    ys.append(r["exact"])
            if xs:
                ok = [y >= 0.9 * ref16["exact"] for y in ys]
                out.append(f"\nlabel-free delta_T (one step, 16 scoring puzzles or more) of the evaluated allocations vs exact: "
                           f"Spearman {spearman(xs, ys):+.2f} over {len(xs)} allocations; largest delta_T among allocations "
                           f"keeping >= 90% of all-16: {max([x for x, o in zip(xs, ok) if o], default=float('nan')):.3f}; "
                           f"smallest among the rest: {min([x for x, o in zip(xs, ok) if not o], default=float('nan')):.3f}")
        conf = {k: v for k, v in E.items() if k.startswith("evalbig|")}
        if conf:
            out.append("\nconfirmation on the large split:")
            for k, r in conf.items():
                lo, hi = wilson(sum(r["correct"]), r["n"])
                out.append(f"- {k[8:]}: {100*r['exact']:.1f}% [{100*lo:.1f}, {100*hi:.1f}] of {r['n']} ({r['n16']} tensors at 16)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="")
    ap.add_argument("--acts", default="")
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    out = []
    if a.weights:
        out.append("## Weights")
        weights_readout([f for p in a.weights.split(",") for f in glob.glob(p)], out)
    if a.acts:
        out.append("\n## Activations")
        acts_readout([f for p in a.acts.split(",") for f in glob.glob(p)], out)
    txt = "\n".join(out)
    print(txt)
    if a.md:
        Path(a.md).write_text(txt + "\n")


if __name__ == "__main__":
    main()
