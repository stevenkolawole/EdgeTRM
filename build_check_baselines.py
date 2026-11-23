#!/usr/bin/env python3
"""Compare simpler and prior-art detectors with the 2-loop build check, from existing records only.

Builds: the 80 gate rows of results/devices/e3_gate_phi_step2.json (label = row["vv"]: PASS / WARN / FAIL).
Device records: results/devices/ctl/<task>/infer/dev_<prec>_<device>[_<acct>].{json,npz} (final carry zH/zL,
per-step preds and q for steps 1..8, per-step exact / cell in the json).
Host references: ctl/<task>/infer/host_fp32*.npz and host_<prec>[_ref].npz (final carry and preds).
Per-step device carries at loops 1 and 2 survive only as cosines in e3_gate_phi_step2.json (the sidecar npz files
were lost with the sk-p4d wipe), so state distances at loops 3..7 and an L2 / projected distance at loop 2 cannot be
computed.  No device job, no network, read-only.  Output: results/devices/build_check_baselines.json.
"""
import json
import os
import numpy as np

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "devices")
OUT = os.path.join(D, "build_check_baselines.json")
GATE = "e3_gate_phi_step2.json"
FAITH = "device_faithful_acc.json"
KS = (2, 8, 32)
SEEDS = (0, 1, 2, 3, 4)


def auroc(pos, neg):
    """P(score_pos > score_neg) + 0.5 P(tie); a higher score means more suspicious."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def tag_of(r):
    dev = r["device"].replace(" ", "_")
    return f"dev_{r['prec']}_{dev}" + ("" if r["acct"] == "main" else "_" + r["acct"])


def cos_mean(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    c = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-30)
    return float(c.mean())


def proj(k, seed, dim=512):
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return q[:k]


def main():
    gate = json.load(open(os.path.join(D, GATE)))
    rows = [r for r in gate["rows"] if r["in_gate"]]
    faith = json.load(open(os.path.join(D, FAITH)))
    host_acc = {}
    for x in faith:
        if x["prec"] == "fp32" and x["source"].startswith("host sim of compiled QDQ graph, test 0-"):
            host_acc[(x["task"], x["step"])] = (x["exact"], x["cell"])
    host_cache = {}

    def host(task, name):
        key = (task, name)
        if key not in host_cache:
            p = os.path.join(D, "ctl", task, "infer", name)
            host_cache[key] = np.load(p) if os.path.exists(p) else None
        return host_cache[key]

    builds = []
    for r in rows:
        task, prec, t = r["task"], r["prec"], tag_of(r)
        base = os.path.join(D, "ctl", task, "infer", t)
        b = dict(name=f"{task} {prec} {r['device']} {r['acct']}", task=task, prec=prec, label=r["vv"], sc={})
        if not os.path.exists(base + ".npz"):
            b["missing"] = "no dev npz"
            builds.append(b)
            continue
        z = np.load(base + ".npz")
        rec = json.load(open(base + ".json"))
        dev_pred = z["preds"][:8]
        hf = host(task, "host_fp32_n8.npz") if host(task, "host_fp32_n8.npz") is not None else host(task, "host_fp32.npz")
        hpred = hf["preds"][:8]
        N = dev_pred.shape[1]
        sc = b["sc"]
        rows_u = len({tuple(x) for x in dev_pred[7]})
        sc["const_unique_rows"] = -rows_u / N
        b["n_unique_final_rows"] = rows_u
        zH = z["zH"].reshape(N, -1, 512)
        hz = hf["zH"].reshape(N, -1, 512)
        vr = float(zH.std(axis=0).mean() / (hz.std(axis=0).mean() + 1e-12))
        sc["const_state_var_ratio"] = abs(np.log(vr + 1e-12))
        b["state_std_ratio_dev_over_host"] = vr
        agree = [float((dev_pred[k] == hpred[k]).mean()) for k in range(8)]
        b["pred_agree_by_step"] = agree
        sc["final_agree"] = 1 - agree[7]
        for k in range(8):
            sc[f"pred_agree_s{k+1}"] = 1 - agree[k]
        st = rec["steps"]
        ex8, ce8 = st[7]["exact"], st[7]["cell"]
        hx8, hc8 = host_acc[(task, 8)]
        sc["exact_gap_s8"] = hx8 - ex8
        sc["cell_gap_s8"] = hc8 - ce8
        b["exact_ratio_s8"] = ex8 / hx8 if hx8 else None
        sc["labelled_probe_s2"] = 1 - r["probe"]
        c8H, c8L = cos_mean(zH, hz), cos_mean(z["zL"].reshape(N, -1, 512), hf["zL"].reshape(N, -1, 512))
        b["phiH_s8_recomputed"], b["phiH_s8_row"] = c8H, r["phiH_s8"]
        for k, key in ((1, "s1_fp32_H"), (2, "s2_fp32_H")):
            sc[f"cosdist_fp32_s{k}"] = 1 - r[key]
        sc["cosdist_fp32_s8"] = 1 - c8H
        sc["cosdist_fp32_s8_L"] = 1 - c8L
        b["cos_fp32_H"] = {"s1": r["s1_fp32_H"], "s2": r["s2_fp32_H"], "s8": c8H}
        sc["rel_l2_fp32_s8"] = float(np.linalg.norm(zH - hz) / np.linalg.norm(hz))
        for k, key in ((1, "s1_ref_H"), (2, "s2_ref_H")):
            if r.get(key) is not None:
                sc[f"cosdist_ref_s{k}"] = 1 - r[key]
        lows = {}
        if prec != "fp32":
            for kind, name in (("ref", f"host_{prec}_ref.npz"), ("fused", f"host_{prec}.npz")):
                h = host(task, name)
                if h is not None:
                    lows[kind] = h["zH"].reshape(N, -1, 512)
        if "ref" in lows:
            sc["cosdist_ref_s8"] = 1 - cos_mean(zH, lows["ref"])
        for kind, hl in lows.items():
            sc[f"ekka_ratio_{kind}_s8"] = float(np.linalg.norm(zH - hz) / (np.linalg.norm(hl - hz) + 1e-12))
        diff = (zH - hz).reshape(-1, 512)
        for k in KS:
            for seed in SEEDS:
                P = proj(k, seed)
                dd = np.linalg.norm(diff @ P.T, axis=1).mean()
                hn = np.linalg.norm(hz.reshape(-1, 512) @ P.T, axis=1).mean()
                if seed == 0:
                    sc[f"difr_k{k}_raw_s8"] = float(dd)
                    sc[f"difr_k{k}_rel_s8"] = float(dd / hn)
                sc[f"difr_k{k}_rel_s8_seed{seed}"] = float(dd / hn)
        builds.append(b)

    have = [b for b in builds if b["sc"]]
    subsets = {
        "all": lambda b: True,
        "maze_only": lambda b: b["task"] == "maze",
        "quantized_only": lambda b: b["prec"] != "fp32",
    }
    det_names = sorted({k for b in have for k in b["sc"]})
    table = {}
    for d in det_names:
        table[d] = {}
        for sn, f in subsets.items():
            xs = [(b["label"], b["sc"][d]) for b in have if f(b) and d in b["sc"]]
            p = [s for l, s in xs if l == "PASS"]
            fl = [s for l, s in xs if l == "FAIL"]
            npas = [s for l, s in xs if l != "PASS"]
            table[d][sn] = dict(n=len(xs), n_pass=len(p), n_fail=len(fl), n_notpass=len(npas),
                                auroc_pass_vs_fail=auroc(fl, p), auroc_pass_vs_notpass=auroc(npas, p))
    spread = {}
    for k in KS:
        v = []
        for seed in SEEDS:
            d = f"difr_k{k}_rel_s8_seed{seed}"
            t = table[d]["all"]
            v.append(t["auroc_pass_vs_fail"])
        spread[f"k{k}"] = dict(min=min(v), max=max(v))

    rule = {}
    for kind in ("ref", "fused"):
        d = f"ekka_ratio_{kind}_s8"
        xs = [(b["label"], b["sc"][d]) for b in have if d in b["sc"]]
        out = {}
        for tgt, cls in (("pass_vs_fail", ("PASS", "FAIL")), ("pass_vs_notpass", ("PASS", "WARN", "FAIL"))):
            sel = [(l, s) for l, s in xs if l in cls]
            truth = [l != "PASS" for l, s in sel]
            pred = [s > 1.5 for l, s in sel]
            tp = sum(t and p for t, p in zip(truth, pred)); fp = sum((not t) and p for t, p in zip(truth, pred))
            fn = sum(t and (not p) for t, p in zip(truth, pred)); tn = sum((not t) and (not p) for t, p in zip(truth, pred))
            out[tgt] = dict(n=len(sel), tp=tp, fp=fp, fn=fn, tn=tn, accuracy=(tp + tn) / len(sel) if sel else None)
        out["ratio_range_by_label"] = {l: [min(s for ll, s in xs if ll == l), max(s for ll, s in xs if ll == l)]
                                       for l in ("PASS", "WARN", "FAIL") if any(ll == l for ll, s in xs)}
        rule[kind] = out

    traj = []
    for b in have:
        if b["label"] != "FAIL":
            continue
        c = b["cos_fp32_H"]
        d1, d2, d8 = 1 - c["s1"], 1 - c["s2"], 1 - c["s8"]
        if d1 >= 0.8 * d8:
            cls = "offset from loop 1 (loop-1 distance already >= 0.8 of loop-8; no further growth)"
        elif d8 > 1.25 * d2:
            cls = "grows"
        else:
            cls = "plateau by loop 2"
        a = b["pred_agree_by_step"]
        traj.append(dict(build=b["name"], dist_s1=d1, dist_s2=d2, dist_s8=d8, cls=cls.split(" (")[0],
                         pred_agree_s1_s2_s4_s8=[a[0], a[1], a[3], a[7]]))
    pass_range = {}
    for g in sorted({(b["task"], b["prec"]) for b in have if b["label"] == "FAIL"}):
        v = [b["cos_fp32_H"] for b in have if (b["task"], b["prec"]) == g and b["label"] == "PASS"]
        pass_range["%s %s" % g] = {k: [1 - max(c[k] for c in v), 1 - min(c[k] for c in v)] for k in ("s1", "s2", "s8")} if v else None

    res = dict(
        description=__doc__.strip().split("\n")[0],
        label_definition="vv: device exact at step 8 / host fp32 exact at step 8, PASS >= 0.95, FAIL < 0.5, else WARN "
                         "(E3_FAULT_TABLE s3); exact_gap_s8 and exact_ratio_s8 therefore reproduce the label",
        records=dict(gate=f"results/devices/{GATE}", device_acc="results/devices/device_faithful_acc.json",
                     device_state="results/devices/ctl/<task>/infer/dev_*.{json,npz}",
                     host="results/devices/ctl/<task>/infer/host_*.npz"),
        n_builds=len(rows), n_with_npz=len(have), missing=[b["name"] for b in builds if not b["sc"]],
        label_counts={l: sum(b["label"] == l for b in builds) for l in ("PASS", "WARN", "FAIL")},
        not_computable=["state distance at loops 3..7 (only loops 1, 2 as cosine, and 8)",
                        "L2 / DiFR / Ekka ratio at loop 2 (per-step carries lost with sk-p4d wipe, 2026-10-22)"],
        auroc=table, difr_seed_spread_pass_vs_fail_all=spread, ekka_rule_1p5=rule,
        trajectory_fail_builds=traj, pass_distance_range_by_fail_graph=pass_range,
        phi8_check_max_abs_diff=max(abs(b["phiH_s8_recomputed"] - b["phiH_s8_row"]) for b in have),
        label_reproduced_from_exact_ratio=sum(
            (("PASS" if b["exact_ratio_s8"] >= .95 else "FAIL" if b["exact_ratio_s8"] < .5 else "WARN") == b["label"])
            for b in have),
        n_unique_final_rows_fail=[b["n_unique_final_rows"] for b in have if b["label"] == "FAIL"],
        builds=[dict(name=b["name"], label=b["label"], **{k: v for k, v in b.items()
                if k in ("n_unique_final_rows", "exact_ratio_s8", "state_std_ratio_dev_over_host")}, scores=b["sc"])
                for b in have],
    )
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", OUT, "builds", len(have), "of", len(rows))


if __name__ == "__main__":
    main()
