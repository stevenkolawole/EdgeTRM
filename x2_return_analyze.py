#!/usr/bin/env python3
"""X2 return test read-out: P1-P4 scored by the rules fixed in results/breadth/x2_return_scoring.md (written before any
record was read). Reads results/breadth/babel/x2_ret/ret_*.json (Babel L40S, one platform), M1 records
(results/breadth/babel/m1/), M2 SR cells (results/m2/cells/*sr0.npz) and the S/S_mix width table (x2_smix.json).

    python x2_return_analyze.py   -> results/breadth/x2_return_readout.md, x2_return.json
"""
import glob
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results/breadth/babel/x2_ret"
M1 = HERE / "results/breadth/babel/m1"
M2 = HERE / "results/m2/cells"
OUT_MD = HERE / "results/breadth/x2_return_readout.md"
OUT_JS = HERE / "results/breadth/x2_return.json"
GRID = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0]
BIG = 1e6   # rho for t_c = 0 (first grid point already fails)


def family(model, fam):
    if fam == "dt":
        return "DT"
    if fam == "mdeq":
        return "MDEQ"
    if model.startswith("x5_"):
        return "seeds"
    if model.startswith("qat_") or model.startswith("ft_"):
        return "QAT"
    if model.startswith("hrm") or fam == "hrm":
        return "HRM"
    if "attn" in model:
        return "TRM-attn"
    return "TRM-MLP"


def mixer(model, fam):
    f = family(model, fam)
    if f in ("seeds",):
        return "attn" if "attn" in model else "MLP"
    return {"TRM-attn": "attn", "TRM-MLP": "MLP", "QAT": "MLP", "HRM": "HRM"}.get(f, f)


def ctype(cell):
    if cell.startswith("sr"):
        return "sr"
    if "anoise" in cell:
        return "act-noise(fresh)"
    if "+a" in cell:
        return "act-quant"
    if cell.startswith("noise"):
        return "weight-noise"
    return "weight-quant"


def auroc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = score[y], score[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def bal_acc(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    parts = [np.mean(pred[y] == y[y]) if y.any() else None, np.mean(pred[~y] == y[~y]) if (~y).any() else None]
    parts = [p for p in parts if p is not None]
    return float(np.mean(parts))


def best_threshold(score, y):
    """threshold on score (collapse iff score > thr) maximizing balanced accuracy."""
    s = np.unique(np.asarray(score, float))
    cands = np.concatenate([[-np.inf], (s[:-1] + s[1:]) / 2 if len(s) > 1 else [], [np.inf]])
    best = max(cands, key=lambda t: (bal_acc(np.asarray(score) > t, y), -abs(t)))
    return float(best)


def _rank(x):
    x = np.asarray(x, float)
    o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    for v in np.unique(x):                      # average ranks for ties
        m = x == v
        r[m] = r[m].mean()
    return r


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    return pearson(_rank(a), _rank(b))


def binom_p(k, n):
    """two-sided exact binomial p (p = 0.5)."""
    pm = [math.comb(n, i) / 2 ** n for i in range(n + 1)]
    return float(min(1.0, sum(q for q in pm if q <= pm[k] * (1 + 1e-9))))


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else None


def load_records():
    rows, seen = [], set()
    for f in sorted(glob.glob(str(RES / "ret_*.json"))):
        b = json.load(open(f))
        m = b
        for r in b["records"]:
            if (m["model"], r["cell"]) in seen:       # a requeued job re-ran cells another file already holds
                continue
            seen.add((m["model"], r["cell"]))
            p = r["per"]
            cf = np.array(p["corr_f"], bool)
            cq = np.array(p["corr_q"], bool)
            tc = np.array(p["t_c"], float)
            rho = np.where(tc > 0, 1.0 / np.maximum(tc, 1e-9), BIG)
            phi = np.array(p["phi"], float)
            rows.append({"file": Path(f).name, "model": m["model"], "fam": m["family"], "family": family(m["model"], m["family"]),
                         "mixer": mixer(m["model"], m["family"]), "task": m["task"], "cell": r["cell"], "ctype": ctype(r["cell"]),
                         "n": r["n"], "acc_f": r["acc_f"], "acc_q": r["acc_q"], "retained": r["retained"],
                         "collapse": r["retained"] < 0.5, "return_rate": r["return_rate"],
                         "return_rate_fc": float(np.mean(np.array(p["ret1"], bool)[cf])) if cf.any() else None,
                         "rho_med_fc": float(np.median(rho[cf])) if cf.any() else None, "rho_med_all": float(np.median(rho)),
                         "tc_med_fc": float(np.median(tc[cf])) if cf.any() else None,
                         "tc_cens_fc": float(np.mean(tc[cf] >= GRID[-1])) if cf.any() else None,
                         "phi_fc": float(np.mean(phi[cf])) if cf.any() else None,
                         "_cf": cf, "_cq": cq, "_ret1": np.array(p["ret1"], bool), "_tc": tc,
                         "_qh": None if p.get("qhalt") is None else np.array(p["qhalt"], float)})
    return rows


def fmt(x, d=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        if x >= BIG / 2:
            return "inf"
        return f"{x:.{d}f}"
    return str(x)


# ----------------------------------------------------------------------------- P1
def p1(rows, md, js):
    cells = [r for r in rows if r["ctype"] != "sr" and r["acc_f"] >= 0.10 and r["rho_med_fc"] is not None]
    main = [r for r in cells if r["ctype"] in ("weight-quant", "act-quant", "weight-noise")]
    fresh = [r for r in cells if r["ctype"] == "act-noise(fresh)"]
    y = np.array([r["collapse"] for r in main])
    rho = np.array([r["rho_med_fc"] for r in main])
    lrho = np.log(np.minimum(rho, BIG))
    phi = np.array([r["phi_fc"] for r in main])
    rule = rho > 1.0
    res = {"n_cells": len(main), "n_collapse": int(y.sum()), "bal_acc_rule_rho_gt_1": bal_acc(rule, y),
           "acc_rule": float(np.mean(rule == y)), "auroc_rho": auroc(lrho, y), "auroc_negphi": auroc(-phi, y)}
    # the collapsed cells: rho distribution
    res["collapsed_rho_med_fc"] = sorted(float(v) for v in rho[y])
    res["best_thr_rho"] = math.exp(best_threshold(lrho, y)) if len(set(y)) > 1 else None
    res["bal_acc_best_rho"] = bal_acc(lrho > math.log(res["best_thr_rho"]), y) if res["best_thr_rho"] else None
    # leave one family out
    fams = sorted({r["family"] for r in main})
    lofo = {"rho": [], "phi": []}
    per_fam_lofo = {}
    for fam in fams:
        tr = np.array([r["family"] != fam for r in main])
        te = ~tr
        if len(set(y[tr])) < 2:
            continue
        t_r = best_threshold(lrho[tr], y[tr])
        t_p = best_threshold(-phi[tr], y[tr])
        pr, pp = lrho[te] > t_r, -phi[te] > t_p
        lofo["rho"] += list(pr == y[te])
        lofo["phi"] += list(pp == y[te])
        per_fam_lofo[fam] = {"n": int(te.sum()), "n_collapse": int(y[te].sum()), "acc_rho": float(np.mean(pr == y[te])),
                             "acc_phi": float(np.mean(pp == y[te])), "thr_rho": float(math.exp(t_r)), "thr_phi": float(-t_p)}
    res["lofo_acc_rho"] = float(np.mean(lofo["rho"])) if lofo["rho"] else None
    res["lofo_acc_phi"] = float(np.mean(lofo["phi"])) if lofo["phi"] else None
    res["lofo_by_family"] = per_fam_lofo
    # tracking: Spearman(retained, return rate)
    rr = np.array([r["return_rate"] for r in main])
    ret = np.array([r["retained"] for r in main])
    res["spearman_retained_return"] = spearman(ret, rr)
    res["pearson_retained_return"] = pearson(ret, rr)
    rrf = np.array([r["return_rate_fc"] for r in main])
    res["spearman_retained_return_fc"] = spearman(ret, rrf)   # secondary: return rate over fp32-correct examples
    # per family: rule accuracy and phi collapse point
    fam_tab = {}
    for fam in fams:
        sel = np.array([r["family"] == fam for r in main])
        fam_tab[fam] = {"n": int(sel.sum()), "n_collapse": int(y[sel].sum()), "bal_acc_rule": bal_acc(rule[sel], y[sel]),
                        "acc_rule": float(np.mean(rule[sel] == y[sel])), "auroc_rho": auroc(lrho[sel], y[sel]),
                        "auroc_negphi": auroc(-phi[sel], y[sel]),
                        "spearman_retained_return": spearman(ret[sel], rr[sel]) if sel.sum() > 2 else None}
    res["by_family"] = fam_tab
    by_type = {}
    for ct in ("weight-quant", "act-quant", "weight-noise"):
        sel = np.array([r["ctype"] == ct for r in main])
        if sel.any():
            by_type[ct] = {"n": int(sel.sum()), "n_collapse": int(y[sel].sum()), "bal_acc_rule": bal_acc(rule[sel], y[sel]),
                           "auroc_rho": auroc(lrho[sel], y[sel])}
    res["by_type"] = by_type
    # per-example agreement ret1 vs corr_q among fp32-correct examples
    agree = {}
    for fam in fams:
        a, b = [], []
        for r in main:
            if r["family"] == fam:
                a.append(r["_ret1"][r["_cf"]])
                b.append(r["_cq"][r["_cf"]])
        a, b = np.concatenate(a), np.concatenate(b)
        agree[fam] = {"n_examples": int(len(a)), "agree": float(np.mean(a == b)), "kappa": kappa(a, b),
                      "P(correct|return)": float(b[a].mean()) if a.any() else None,
                      "P(correct|no return)": float(b[~a].mean()) if (~a).any() else None, "return_rate": float(a.mean())}
    res["per_example"] = agree
    res["fresh_e"] = [{k: r[k] for k in ("model", "cell", "acc_f", "retained", "return_rate", "rho_med_fc", "phi_fc")} for r in fresh]
    ok = [res["bal_acc_rule_rho_gt_1"] >= 0.80,
          res["auroc_rho"] is not None and res["auroc_negphi"] is not None and res["auroc_rho"] > res["auroc_negphi"],
          res["lofo_acc_rho"] is not None and res["lofo_acc_rho"] >= (res["lofo_acc_phi"] or 0),
          (res["spearman_retained_return"] or 0) >= 0.7]
    res["criteria_abcd"] = ok
    res["verdict"] = "supported" if all(ok) else ("falsified" if not ok[0] else "partly")
    js["P1"] = res

    md.append("## P1: collapse at median rho ~ 1, one threshold, rho beats phi\n")
    md.append(f"Cells (weight quant, activation quant, weight noise; fp32 acc >= 10%): {res['n_cells']}, collapsed "
              f"(retained < 0.5): {res['n_collapse']}.\n")
    md.append("| test | value | rule |\n|---|---|---|")
    md.append(f"| (a) balanced accuracy of 'collapse iff median rho > 1' | {fmt(res['bal_acc_rule_rho_gt_1'])} (plain {fmt(res['acc_rule'])}) | >= 0.80 |")
    md.append(f"| (b) AUROC rho vs AUROC -phi | {fmt(res['auroc_rho'])} vs {fmt(res['auroc_negphi'])} | rho > phi |")
    md.append(f"| (c) leave-one-family-out accuracy rho vs phi | {fmt(res['lofo_acc_rho'])} vs {fmt(res['lofo_acc_phi'])} | rho >= phi |")
    md.append(f"| (d) Spearman(retained, t=1 return rate) | {fmt(res['spearman_retained_return'])} (Pearson {fmt(res['pearson_retained_return'])}); secondary, fp32-correct examples: {fmt(res['spearman_retained_return_fc'])} | >= 0.7 |")
    md.append(f"| best pooled rho threshold (post hoc, for reference) | {fmt(res['best_thr_rho'])} (bal. acc {fmt(res['bal_acc_best_rho'])}) | - |")
    md.append(f"\n**Verdict: {res['verdict'].upper()}** (criteria a-d: {ok}).\n")
    md.append("Collapsed cells' median rho (fp32-correct examples): " + ", ".join(fmt(v, 2) for v in res["collapsed_rho_med_fc"]) + "\n")
    md.append("| family | cells | collapsed | rule bal.acc | AUROC rho | AUROC -phi | Spearman | LOFO acc rho / phi (thr rho / phi) |\n|---|---|---|---|---|---|---|---|")
    for fam, v in fam_tab.items():
        lf = per_fam_lofo.get(fam)
        lfs = f"{fmt(lf['acc_rho'],2)} / {fmt(lf['acc_phi'],2)} ({fmt(lf['thr_rho'],2)} / {fmt(lf['thr_phi'],2)})" if lf else "-"
        md.append(f"| {fam} | {v['n']} | {v['n_collapse']} | {fmt(v['bal_acc_rule'],2)} | {fmt(v['auroc_rho'],2)} | {fmt(v['auroc_negphi'],2)} | {fmt(v['spearman_retained_return'],2)} | {lfs} |")
    md.append("\n| compression type | cells | collapsed | rule bal.acc | AUROC rho |\n|---|---|---|---|---|")
    for ct, v in by_type.items():
        md.append(f"| {ct} | {v['n']} | {v['n_collapse']} | {fmt(v['bal_acc_rule'],2)} | {fmt(v['auroc_rho'],2)} |")
    md.append("\nPer-example agreement (fp32-correct examples, all P1 cells pooled per family): t=1 return vs compressed correct\n")
    md.append("| family | examples | return rate | agree | kappa | P(correct given return) | P(correct given no return) |\n|---|---|---|---|---|---|---|")
    for fam, v in agree.items():
        md.append(f"| {fam} | {v['n_examples']} | {fmt(v['return_rate'])} | {fmt(v['agree'])} | {fmt(v['kappa'])} | {fmt(v['P(correct|return)'])} | {fmt(v['P(correct|no return)'])} |")
    md.append("\nFresh-e cells (activation noise redrawn every call):\n")
    md.append("| model | cell | fp32 | retained | return rate | median rho | phi |\n|---|---|---|---|---|---|---|")
    for r in res["fresh_e"]:
        md.append(f"| {r['model']} | {r['cell']} | {fmt(r['acc_f'])} | {fmt(r['retained'])} | {fmt(r['return_rate'])} | {fmt(r['rho_med_fc'])} | {fmt(r['phi_fc'])} |")
    md.append("\nAll cells:\n")
    md.append("| family | model | cell | type | n | fp32 | retained | return t=1 (all / fp32-correct) | median rho (fp32-correct / all) | frac t_c>=16 | phi |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(cells, key=lambda r: (r["family"], r["model"], r["ctype"], r["cell"])):
        md.append(f"| {r['family']} | {r['model']} | {r['cell']} | {r['ctype']} | {r['n']} | {fmt(r['acc_f'])} | {fmt(r['retained'])} | "
                  f"{fmt(r['return_rate'])} / {fmt(r['return_rate_fc'])} | {fmt(r['rho_med_fc'])} / {fmt(r['rho_med_all'])} | {fmt(r['tc_cens_fc'],2)} | {fmt(r['phi_fc'])} |")
    md.append("")


# ----------------------------------------------------------------------------- P2
def p2(rows, md, js):
    out = []
    by = {(r["model"], r["cell"]): r for r in rows if r["ctype"] != "sr"}
    for f in sorted(glob.glob(str(M1 / "p_*.json"))):
        d = json.load(open(f))
        if d.get("T") != 16 or d.get("offset") != 0 or d.get("ckpt"):
            continue
        for rec in d["records"]:
            if rec["polish"] != "fp32" or rec["draw"] != 0:
                continue
            r = by.get((d["task"], rec["cell"]))
            if r is None or any(o["model"] == r["model"] and o["cell"] == r["cell"] for o in out):
                continue
            n = r["n"]
            sc = rec["schedules"]
            if "c16.P8" not in sc or "c16" not in sc:
                continue
            c16 = np.array([c == "1" for c in sc["c16"]["correct"][:n]])
            p8 = np.array([c == "1" for c in sc["c16.P8"]["correct"][:n]])
            match = float(np.mean(c16 == r["_cq"]))
            obs = float(p8.mean() - c16.mean())
            pred = float(np.mean(r["_ret1"] * (r["_cf"].astype(int) - r["_cq"].astype(int))))
            ex = float(np.mean(r["_ret1"] & ~r["_cq"] & r["_cf"]))
            out.append({"model": r["model"], "family": r["family"], "cell": r["cell"], "m1_file": Path(f).name, "match_c16": match,
                        "matched": match >= 0.98, "obs_gain": obs, "pred_gain": pred, "gap": r["acc_f"] - r["acc_q"],
                        "return_rate": r["return_rate"], "rho_med_fc": r["rho_med_fc"], "retained": r["retained"],
                        "ok": abs(pred - obs) <= max(0.02, 0.25 * abs(obs))})
    # MDEQ: M1 deq hybrid records ('26+k' = 26 compressed Broyden iterations, then k fp32)
    for f, mdl, kp in (("deq_small_dev.json", "mdeq_small", "26+26"), ("deq_xl_dense_dev.json", "mdeq_xl", "26+16")):
        p = M1 / f
        if not p.exists():
            continue
        d = json.load(open(p))
        for key, sch in d["hybrid"].items():
            cell, pol = key.split("|")
            if pol != "fp32" or kp not in sch:
                continue
            r = by.get((mdl, cell))
            if r is None:
                continue
            n = r["n"]
            cq = np.array([c == "1" for c in d["curve"][cell]["correct"]["26"][:n]])   # 26 compressed iterations, no polish
            hy = np.array([c == "1" for c in sch[kp]["correct"][:n]])
            match = float(np.mean(cq == r["_cq"]))
            obs = float(hy.mean() - cq.mean())
            pred = float(np.mean(r["_ret1"] * (r["_cf"].astype(int) - r["_cq"].astype(int))))
            out.append({"model": mdl, "family": "MDEQ", "cell": cell, "m1_file": f + ":" + kp, "match_c16": match,
                        "matched": match >= 0.98, "obs_gain": obs, "pred_gain": pred, "gap": r["acc_f"] - r["acc_q"],
                        "return_rate": r["return_rate"], "rho_med_fc": r["rho_med_fc"], "retained": r["retained"],
                        "ok": abs(pred - obs) <= max(0.02, 0.25 * abs(obs))})
    m = [o for o in out if o["matched"]]
    frac = float(np.mean([o["ok"] for o in m])) if m else None
    js["P2"] = {"rows": out, "n_matched": len(m), "frac_within_rule": frac,
                "verdict": None if not m else ("supported" if frac >= 0.8 else ("partly" if frac >= 0.5 else "falsified"))}
    md.append("## P2: polish gain ~ return rate x gap on returned examples\n")
    md.append(f"Matched cells (compressed per-example correctness agrees with M1's >= 98%): {len(m)}; within "
              f"max(2 pts, 25%): {fmt(frac)} -> **{str(js['P2']['verdict']).upper()}**\n")
    md.append("| family | model | cell | M1 record | c16 match | fp32-cmp gap | return t=1 | median rho | predicted gain | observed gain (c16.P8 - c16) | within rule |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for o in out:
        md.append(f"| {o['family']} | {o['model']} | {o['cell']} | {o['m1_file']} | {fmt(o['match_c16'])} | {fmt(o['gap'])} | {fmt(o['return_rate'])} | "
                  f"{fmt(o['rho_med_fc'])} | {fmt(o['pred_gain'])} | {fmt(o['obs_gain'])} | {'yes' if o['ok'] else 'no'}{'' if o['matched'] else ' (unmatched)'} |")
    md.append("")


# ----------------------------------------------------------------------------- P3
def p3(rows, md, js, m2_gain=True):
    res = {"head_pick": {}, "gain_vs_return": []}
    srs = [r for r in rows if r["ctype"] == "sr"]
    for task in sorted({r["task"] for r in srs}):
        rs = sorted([r for r in srs if r["task"] == task], key=lambda r: int(r["cell"].split("~")[1]))
        ks = [int(r["cell"].split("~")[1]) for r in rs]
        if len(rs) < 2 or any(r["_qh"] is None for r in rs):
            continue
        spec = rs[0]["cell"][2:].split("~")[0]
        tc = np.stack([r["_tc"] for r in rs])
        qh = np.stack([r["_qh"] for r in rs])
        cq = np.stack([r["_cq"] for r in rs])
        n = tc.shape[1]
        pick = qh.argmax(0)
        tcp = tc[pick, np.arange(n)]
        med = np.median(tc, 0)
        up, dn = int((tcp > med).sum()), int((tcp < med).sum())
        p = binom_p(up, up + dn) if up + dn else None
        # consistency with M2's own draws on the same examples
        m2f = M2 / f"{task}__{spec}__none0__D16_K16_sr0.npz"
        agree_m2 = None
        if m2f.exists():
            c2 = np.load(m2f)["corr"][:, :n, 15].astype(bool)
            agree_m2 = float(np.mean([np.mean(c2[k] == cq[i]) for i, k in enumerate(ks) if k < c2.shape[0]]))
        res["head_pick"][f"{task} {spec}"] = {"draws": len(ks), "n": n, "picked_tc_gt_median": up, "picked_tc_lt_median": dn,
                                             "sign_p": p, "mean_tc_picked": float(tcp.mean()), "mean_tc_all": float(tc.mean()),
                                             "acc_rtn": float(cq[ks.index(0)].mean()) if 0 in ks else None,
                                             "acc_bestq": float(cq[pick, np.arange(n)].mean()),
                                             "return_rate_by_draw": [float(r["return_rate"]) for r in rs],
                                             "agree_with_m2_corr": agree_m2,
                                             "ok": p is not None and up > dn and p < 0.05}
    # gain vs return rate across M2's SR cells (first n of each)
    by = {(r["model"], r["cell"]): r for r in rows if r["ctype"] in ("weight-quant",)}
    for f in (sorted(glob.glob(str(M2 / "*__none0__D16_K16_sr0.npz"))) if m2_gain else []):
        task, spec = Path(f).name.split("__")[:2]
        r = by.get((task, spec))
        if r is None or "@" in spec:
            continue
        d = np.load(f)
        n = r["n"]
        c = d["corr"][:, :n, 15].astype(bool)
        q = d["q"][:, :n, 15].astype(np.float32)
        gain = float(c[q.argmax(0), np.arange(n)].mean() - c[0].mean())
        res["gain_vs_return"].append({"task": task, "spec": spec, "rtn_acc_m2": float(c[0].mean()), "rtn_acc_mine": r["acc_q"],
                                      "gain_bestq16": gain, "return_rate": r["return_rate"], "return_rate_fc": r["return_rate_fc"], "rho_med_fc": r["rho_med_fc"],
                                      "retained": r["retained"]})
    g = res["gain_vs_return"]
    peak_ok = None
    if len(g) >= 3:
        gs = sorted(g, key=lambda x: x["return_rate"])
        best = max(gs, key=lambda x: x["gain_bestq16"])
        lo, hi = gs[0], gs[-1]
        peak_ok = (0.2 <= best["return_rate"] <= 0.8) and best is not lo and best is not hi
    hp_ok = all(v["ok"] for v in res["head_pick"].values()) if res["head_pick"] else None
    res["peak_intermediate"] = peak_ok
    if not m2_gain:
        res["verdict"] = None if hp_ok is None else ("head-pick part " + ("supported" if hp_ok else "falsified"))
    else:
        res["verdict"] = None if hp_ok is None or peak_ok is None else (
            "supported" if (hp_ok and peak_ok) else ("partly" if (hp_ok or peak_ok) else "falsified"))
    js["P3"] = res
    md.append("## P3: sampling gain peaks at intermediate return rates; the head picks draws with smaller rho\n")
    md.append(f"**Verdict: {str(res['verdict']).upper()}** (head picks larger t_c in every model: {hp_ok}; gain peaks at "
              f"intermediate return rate: {peak_ok}).\n")
    md.append("| model | draws | n | picked t_c > / < median-over-draws | sign-test p | mean t_c picked / all | acc RTN / best-Q | agreement with M2 per-example correctness |\n|---|---|---|---|---|---|---|---|")
    for k, v in res["head_pick"].items():
        md.append(f"| {k} | {v['draws']} | {v['n']} | {v['picked_tc_gt_median']} / {v['picked_tc_lt_median']} | {fmt(v['sign_p'],4)} | "
                  f"{fmt(v['mean_tc_picked'],2)} / {fmt(v['mean_tc_all'],2)} | {fmt(v['acc_rtn'])} / {fmt(v['acc_bestq'])} | {fmt(v['agree_with_m2_corr'])} |")
    md.append("\n| task | spec | RTN acc (M2 / mine) | best-Q@16 gain (M2) | RTN return rate (all / fp32-correct) | median rho | retained |\n|---|---|---|---|---|---|---|")
    for v in sorted(g, key=lambda x: x["return_rate"]):
        md.append(f"| {v['task']} | {v['spec']} | {fmt(v['rtn_acc_m2'])} / {fmt(v['rtn_acc_mine'])} | {fmt(v['gain_bestq16'])} | {fmt(v['return_rate'])} / {fmt(v['return_rate_fc'])} | {fmt(v['rho_med_fc'])} | {fmt(v['retained'])} |")
    md.append("")


# ----------------------------------------------------------------------------- P4
def p4(rows, md, js):
    sm = json.load(open(HERE / "results/breadth/x2_smix.json"))
    by = {(r["model"], r["cell"]): r for r in rows}
    tab = []
    for w in sm["rows"]:
        r = by.get((w["model"], "noise0.1"))
        if r is None or r["tc_med_fc"] is None:
            continue
        tab.append({"model": w["model"], "mixer": w["mixer"], "held_out": w["held_out"], "fp32": w["fp32"],
                    "sigma_half": w["sigma_half"], "S": w["S"], "S_mix": w["S_mix"], "tc_med": r["tc_med_fc"],
                    "censored": r["tc_cens_fc"], "pred0": 0.1 * max(r["tc_med_fc"], GRID[0] / 2)})
    fit = [t for t in tab if not t["held_out"] and t["fp32"] >= 0.30]
    new = [t for t in tab if t["held_out"]]
    res = {"rows": tab, "n_fit": len(fit), "n_new": len(new)}
    if fit:
        c = math.exp(np.mean([math.log(t["sigma_half"] / t["pred0"]) for t in fit]))
        res["scale"] = c
        lr = [math.log(c * t["pred0"] / t["sigma_half"]) for t in fit]
        res["r2_log"] = float(1 - np.sum(np.square(lr)) / np.sum(np.square(
            [math.log(t["sigma_half"]) - np.mean([math.log(u["sigma_half"]) for u in fit]) for t in fit])))
        res["bias_by_mixer"] = {mx: float(math.exp(np.mean([math.log(c * t["pred0"] / t["sigma_half"]) for t in fit if t["mixer"] == mx])))
                                for mx in sorted({t["mixer"] for t in fit})}
        res["param_free_bias_by_mixer"] = {mx: float(math.exp(np.mean([math.log(t["pred0"] / t["sigma_half"]) for t in fit if t["mixer"] == mx])))
                                           for mx in sorted({t["mixer"] for t in fit})}
        res["new_ratios"] = [round(c * t["pred0"] / t["sigma_half"], 3) for t in new]
        if new:
            ab = [abs(math.log(x)) for x in res["new_ratios"]]
            res["new_med"] = float(math.exp(np.median(ab)))
            res["new_worst"] = float(math.exp(max(ab)))
        b = res["bias_by_mixer"]
        ok = all(0.9 <= b.get(mx, 1.0) <= 1.1 for mx in ("MLP", "attn")) and (not new or res["new_med"] <= 1.157)
        res["verdict"] = "supported" if ok else ("partly" if all(0.8 <= b.get(mx, 1.0) <= 1.25 for mx in ("MLP", "attn")) else "falsified")
    js["P4"] = res
    md.append("## P4: sigma_1/2 from the radius along the compression direction removes the mixer bias\n")
    if not fit:
        md.append("No noise0.1 records matched the width table.\n")
        return
    md.append(f"Fit set {len(fit)} models (fp32 >= 30%), new seeds {len(new)}. Radius prediction sigma_pred = c x 0.1 x median t_c "
              f"(fp32-correct examples, noise0.1). Pooled c = {res['scale']:.3f}; R^2 (log) {res['r2_log']:.2f}.\n")
    md.append("| law | bias pred/meas MLP / attn / HRM | new seeds median / worst |\n|---|---|---|")
    for name, L in sm["laws"].items():
        if name in ("S", "S_mix"):
            bb = L["bias_ratio_by_mixer"]
            md.append(f"| {name} (from x2_smix.json) | {bb['MLP']:.2f} / {bb['attn']:.2f} / {bb['HRM']:.2f} | x{L['new_med']:.2f} / x{L['new_worst']:.2f} |")
    b = res["bias_by_mixer"]
    pf = res["param_free_bias_by_mixer"]
    md.append(f"| radius, 1 parameter | {b.get('MLP', float('nan')):.2f} / {b.get('attn', float('nan')):.2f} / {b.get('HRM', float('nan')):.2f} | "
              f"x{res.get('new_med', float('nan')):.2f} / x{res.get('new_worst', float('nan')):.2f} |")
    md.append(f"| radius, parameter-free (c = 1) | {pf.get('MLP', float('nan')):.2f} / {pf.get('attn', float('nan')):.2f} / {pf.get('HRM', float('nan')):.2f} | - |")
    md.append(f"\n**Verdict: {res['verdict'].upper()}** (rule: 1-parameter bias within [0.9, 1.1] for MLP and attention, new-seed median <= x1.16).\n")
    md.append("| model | mixer | held out | sigma_1/2 measured | median t_c at noise0.1 | frac censored (t_c = 16) | radius pred (c=1) | S | S_mix |\n|---|---|---|---|---|---|---|---|---|")
    for t in tab:
        md.append(f"| {t['model']} | {t['mixer']} | {t['held_out']} | {t['sigma_half']:.3f} | {t['tc_med']:.2f} | {t['censored']:.2f} | {t['pred0']:.3f} | {t['S']:.2f} | {t['S_mix']:.2f} |")
    md.append("")


def main():
    global RES, OUT_MD, OUT_JS
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default="babel", choices=["babel", "p4d"])
    a = ap.parse_args()
    if a.platform == "p4d":
        # within-platform replicate on the released models; never pooled with Babel. P2 (M1 records) and P4 (width
        # table) and M2's gain table come from other platforms' records, so only P1 and P3's head-pick test are scored.
        RES = HERE / "results/breadth/p4d/x2_ret"
        OUT_MD = HERE / "results/breadth/x2_return_readout_p4d.md"
        OUT_JS = HERE / "results/breadth/x2_return_p4d.json"
    rows = load_records()
    plat = "Babel L40S" if a.platform == "babel" else "p4d A100-40GB"
    md = ["# X2 return test: read-out (`python x2_return_analyze.py --platform " + a.platform + "`; rules in `x2_return_scoring.md`, fixed before any record was read)\n",
          f"Records: {len({r['file'] for r in rows})} files, {len(rows)} cells; platform {plat} only.\n"]
    js = {"platform": a.platform}
    p1(rows, md, js)
    if a.platform == "babel":
        p2(rows, md, js)
    p3(rows, md, js, m2_gain=(a.platform == "babel"))
    if a.platform == "babel":
        p4(rows, md, js)
    OUT_MD.write_text("\n".join(md) + "\n")

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, list):
            return [clean(v) for v in o]
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        return o
    OUT_JS.write_text(json.dumps(clean(js), indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
