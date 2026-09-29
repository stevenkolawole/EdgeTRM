#!/usr/bin/env python3
"""Score the label-free gate at steps 1 and 2 (E3_GATE_PHI_STEP2.md): carry fidelity phi of the recovered step-1 /
step-2 device carries (aihub_e3_steps12.py phi -> e3_steps12_phi.json, pulled from p4d) against the labelled
step-8 verdict of E3_FAULT_TABLE.md section 3 (PASS / WARN / FAIL, from results/devices/e3_gate_phi.json rows).

    python tools/e3_gate_phi_step2.py --phi results/devices/e3_steps12_phi.json
"""
import argparse
import collections
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_readout as B  # noqa: E402
from e3_gate_phi import auroc, pick_threshold  # noqa: E402

DEV = B.DEV
REFS = ("fp32", "ref", "fused")
V79 = {"Samsung Galaxy S25", "Snapdragon 8 Elite QRD"}


def key(r):
    return (r["task"], r["prec"], r["device"], r.get("acct", "main"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi", default=str(DEV / "e3_steps12_phi.json"))
    ap.add_argument("--step8", default=str(DEV / "e3_gate_phi.json"))
    ap.add_argument("--out", default=str(DEV / "e3_gate_phi_step2.json"))
    ap.add_argument("--train_acct", default="acct2")
    a = ap.parse_args()
    phi = json.load(open(a.phi))
    s8 = {key(r): r for r in json.load(open(a.step8))["rows"]}
    devs = json.load(open(DEV / "aihub_devices_sep29.json"))["devices"]
    rows = []
    for r in phi:
        p = r.get("phi") or {}
        row = dict(task=r["task"], prec=r["prec"], device=r["device"], acct=r.get("acct", "main"),
                   hexagon=devs.get(r["device"], {}).get("hexagon", "?"), in_gate=r.get("in_gate", False), vv=r.get("vv"), pv=r.get("pv"),
                   followup=r.get("followup", False), source=r.get("source"))
        e = s8.get(key(r), {})
        row["phiH_s8"], row["phiL_s8"], row["s8"], row["probe"] = e.get("phiH"), e.get("phiL"), e.get("s8"), e.get("probe")
        for s in (1, 2):
            for ref in REFS:
                q = p.get(f"s{s}_{ref}")
                for m in ("H", "L", "H_min", "L_min"):
                    row[f"s{s}_{ref}_{m}"] = q[m] if q else None
        rows.append(row)
    G = [r for r in rows if r["in_gate"]]
    print(f"gate chains in the phi file: {len(G)} (PASS {sum(r['vv'] == 'PASS' for r in G)}, WARN {sum(r['vv'] == 'WARN' for r in G)}, "
          f"FAIL {sum(r['vv'] == 'FAIL' for r in G)}); with a step-2 fp32 phi: {sum(r['s2_fp32_H'] is not None for r in G)}; "
          f"with a step-1 fp32 phi: {sum(r['s1_fp32_H'] is not None for r in G)}")
    for r in G:
        if r["s2_fp32_H"] is None:
            print("  no step-2 phi:", r["task"], r["prec"], r["device"], r["acct"], r["source"])
    res = {"n_gate": len(G), "auroc": {}, "cross_account": {}, "rows": rows}

    def fmt(v, n=3):
        return "-" if v is None else f"{v:.{n}f}"

    # ---- 1. AUROC per step / reference / statistic
    print("\n## 1. AUROC of phi at steps 1 and 2 against the labelled step-8 verdict (80 pinned chains)\n")
    print("| step | reference | score | n scored | AUROC PASS vs FAIL | PASS vs WARN | PASS vs not-PASS | PASS range | WARN values | FAIL range |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in (1, 2):
        for ref in REFS:
            for m in ("H", "L", "H_min", "L_min"):
                k = f"s{s}_{ref}_{m}"
                sc = [r for r in G if r[k] is not None]
                if ref != "fp32":
                    sc = [r for r in sc if r["prec"] != "fp32"]
                P = [r[k] for r in sc if r["vv"] == "PASS"]; W = [r[k] for r in sc if r["vv"] == "WARN"]; F = [r[k] for r in sc if r["vv"] == "FAIL"]
                if not P or not F:
                    continue
                d = dict(n=len(sc), auroc_pass_vs_fail=auroc(P, F), auroc_pass_vs_warn=auroc(P, W), auroc_pass_vs_notpass=auroc(P, W + F),
                         pass_min=min(P), pass_max=max(P), warn=sorted(W), fail_min=min(F), fail_max=max(F))
                res["auroc"][k] = d
                print(f"| {s} | {ref} | phi_{m} | {len(sc)} | {d['auroc_pass_vs_fail']:.3f} | {fmt(d['auroc_pass_vs_warn'])} | {d['auroc_pass_vs_notpass']:.3f} | "
                      f"{d['pass_min']:.4f}-{d['pass_max']:.4f} | {', '.join(f'{w:.3f}' for w in sorted(W))} | {d['fail_min']:.4f}-{d['fail_max']:.4f} |")
    # the step-8 numbers on the same chains for reference
    for m in ("H", "L"):
        k = f"phi{m}_s8"
        P = [r[k] for r in G if r["vv"] == "PASS" and r[k] is not None]; W = [r[k] for r in G if r["vv"] == "WARN" and r[k] is not None]
        F = [r[k] for r in G if r["vv"] == "FAIL" and r[k] is not None]
        print(f"| 8 (Part A) | fp32 | phi_{m} | {len(P) + len(W) + len(F)} | {auroc(P, F):.3f} | {auroc(P, W):.3f} | {auroc(P, W + F):.3f} | "
              f"{min(P):.4f}-{max(P):.4f} | {', '.join(f'{w:.3f}' for w in sorted(W))} | {min(F):.4f}-{max(F):.4f} |")

    # ---- 2. per graph: PASS floor and FAIL ceiling at each step
    print("\n## 2. Per graph: PASS floor / FAIL ceiling of phi_H by step and reference\n")
    print("| task | graph | n PASS / WARN / FAIL | s1 fp32 PASS min / FAIL max | s2 fp32 | s8 fp32 | s1 same-prec ref | s2 same-prec ref | s2 same-prec fused |")
    print("|---|---|---|---|---|---|---|---|---|")
    graphs = sorted({(r["task"], r["prec"]) for r in G})
    for task, prec in graphs:
        gg = [r for r in G if r["task"] == task and r["prec"] == prec]

        def cell(k):
            P = [r[k] for r in gg if r["vv"] == "PASS" and r[k] is not None]; F = [r[k] for r in gg if r["vv"] == "FAIL" and r[k] is not None]
            W = [r[k] for r in gg if r["vv"] == "WARN" and r[k] is not None]
            return (fmt(min(P), 4) if P else "-") + " / " + (fmt(max(F), 4) if F else "-") + (f" (WARN {', '.join(fmt(w, 3) for w in W)})" if W else "")
        n = f"{sum(r['vv'] == 'PASS' for r in gg)} / {sum(r['vv'] == 'WARN' for r in gg)} / {sum(r['vv'] == 'FAIL' for r in gg)}"
        print(f"| {task} | {prec} | {n} | {cell('s1_fp32_H')} | {cell('s2_fp32_H')} | {cell('phiH_s8')} | {cell('s1_ref_H')} | {cell('s2_ref_H')} | {cell('s2_fused_H')} |")

    # ---- 3. cross-account threshold at steps 1 and 2
    print(f"\n## 3. Cross-account threshold (cut chosen on {a.train_acct}, tested on the rest, and the reverse; PASS = phi >= cut)\n")
    print("| step | reference | score | split | rule | cut | test PASS / FAIL | missed FAIL | false alarms | WARN flagged |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s in (1, 2):
        for ref in ("fp32", "ref"):
            for m in ("H", "L"):
                k = f"s{s}_{ref}_{m}"
                sc = [r for r in G if r[k] is not None and (ref == "fp32" or r["prec"] != "fp32")]
                tr = [r for r in sc if r["acct"] == a.train_acct]; te = [r for r in sc if r["acct"] != a.train_acct]
                for name, trs, tes in ((f"{a.train_acct} -> rest", tr, te), (f"rest -> {a.train_acct}", te, tr)):
                    P = [r[k] for r in trs if r["vv"] == "PASS"]; F = [r[k] for r in trs if r["vv"] == "FAIL"]
                    if not P or not F:
                        continue
                    for rule in ("midpoint", "min_pass"):
                        cut, sep = pick_threshold(P, F) if rule == "midpoint" else (float(min(P)) - 1e-9, min(P) > max(F))
                        miss = [f"{r['task']} {r['prec']} {r['device']} ({r[k]:.3f})" for r in tes if r["vv"] == "FAIL" and r[k] >= cut]
                        fa = [f"{r['task']} {r['prec']} {r['device']} ({r[k]:.3f})" for r in tes if r["vv"] == "PASS" and r[k] < cut]
                        wf = [f"{r['task']} {r['prec']} {r['device']} ({r[k]:.3f})" for r in tes if r["vv"] == "WARN" and r[k] < cut]
                        res["cross_account"][f"{k} | {name} | {rule}"] = dict(cut=cut, separable_on_train=sep, n_train=len(trs), n_test=len(tes),
                                                                             missed_fail=miss, false_alarm=fa, warn_flagged=wf)
                        print(f"| {s} | {ref} | phi_{m} | {name} | {rule} | {cut:.4f} | {sum(r['vv'] == 'PASS' for r in tes)} / {sum(r['vv'] == 'FAIL' for r in tes)} | "
                              f"{len(miss)} {miss if miss else ''} | {len(fa)} {fa if fa else ''} | {len(wf)} {wf if wf else ''} |")

    # ---- 4. the v79 rows
    print("\n## 4. The v79 rows (S25, 8 Elite QRD) against the rest of their graph\n")
    print("| task | graph | device | acct | verdict | s1 fp32 phi_H | s2 fp32 | s8 fp32 | s1 ref | s2 ref | graph PASS floor s1 fp32 / s2 fp32 / s1 ref / s2 ref |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    v79 = []
    for r in sorted(G, key=lambda r: (r["task"], r["prec"], r["device"])):
        if r["device"] not in V79 or r["vv"] == "PASS" and r["prec"] == "fp32":
            continue
        gg = [x for x in G if x["task"] == r["task"] and x["prec"] == r["prec"] and x["vv"] == "PASS" and x["device"] not in V79]
        floors = [min([x[k] for x in gg if x[k] is not None], default=None) for k in ("s1_fp32_H", "s2_fp32_H", "s1_ref_H", "s2_ref_H")]
        below = {k: (r[k] is not None and f is not None and r[k] < f) for k, f in zip(("s1_fp32_H", "s2_fp32_H", "s1_ref_H", "s2_ref_H"), floors)}
        v79.append(dict(task=r["task"], prec=r["prec"], device=r["device"], acct=r["acct"], vv=r["vv"], below_pass_floor=below,
                        **{k: r[k] for k in ("s1_fp32_H", "s2_fp32_H", "phiH_s8", "s1_ref_H", "s2_ref_H")}))
        print(f"| {r['task']} | {r['prec']} | {r['device']} | {r['acct']} | {r['vv']} | {fmt(r['s1_fp32_H'], 4)} | {fmt(r['s2_fp32_H'], 4)} | {fmt(r['phiH_s8'], 4)} | "
              f"{fmt(r['s1_ref_H'], 4)} | {fmt(r['s2_ref_H'], 4)} | {' / '.join(fmt(f, 4) for f in floors)} |")
    res["v79"] = v79

    # ---- 5. per chain
    print("\n## 5. Per-chain table (80 pinned chains; phi_H / phi_L)\n")
    print("| task | graph | device | acct | Hexagon | s8 verdict | s1 vs fp32 | s2 vs fp32 | s8 vs fp32 | s1 vs ref | s2 vs ref | s2 vs fused |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(G, key=lambda r: (r["vv"], r["task"], r["prec"], r["s2_fp32_H"] if r["s2_fp32_H"] is not None else 9)):
        pair = lambda h, l: "-" if r[h] is None else f"{r[h]:.4f} / {r[l]:.4f}"   # noqa: E731
        print(f"| {r['task']} | {r['prec']} | {r['device']} | {r['acct']} | {r['hexagon']} | {r['vv']} | {pair('s1_fp32_H', 's1_fp32_L')} | {pair('s2_fp32_H', 's2_fp32_L')} | "
              f"{pair('phiH_s8', 'phiL_s8')} | {pair('s1_ref_H', 's1_ref_L')} | {pair('s2_ref_H', 's2_ref_L')} | {pair('s2_fused_H', 's2_fused_L')} |")

    # ---- 6. chains outside the 80
    print("\n## 6. Chains outside the 80 (Sep 25 S24 / 8 Elite QRD variants; E3_FOLLOWUP chains)\n")
    print("| task | graph | device | acct | source | s1 vs fp32 phi_H / phi_L | s2 vs fp32 | s1 vs ref | s2 vs ref |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["in_gate"]:
            continue
        pair = lambda h, l: "-" if r[h] is None else f"{r[h]:.4f} / {r[l]:.4f}"   # noqa: E731
        print(f"| {r['task']} | {r['prec']} | {r['device']} | {r['acct']} | {r['source']} | {pair('s1_fp32_H', 's1_fp32_L')} | {pair('s2_fp32_H', 's2_fp32_L')} | "
              f"{pair('s1_ref_H', 's1_ref_L')} | {pair('s2_ref_H', 's2_ref_L')} |")
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
