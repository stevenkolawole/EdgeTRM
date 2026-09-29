#!/usr/bin/env python3
"""Label-free gate on the device carries: carry fidelity phi (cosine of the device carry with the host reference
carry at the same step, App. F of the ICLR paper, eq. fidelity) against the labelled gate of E3_FAULT_TABLE.md
section 3 on the pinned (QAIRT 2.45) chains. Records only; no device job, no host run.

What the records hold. Every chain record dev_<prec>_<device>_<acct>.npz is a rolling checkpoint written by
aihub_acct.Chain / aihub_infer.stage_run after each device step: `preds` and `q` for every step, but `zH` / `zL`
only for the LAST step (shape (N, 1, seq, 512)); the earlier carries were overwritten. The host references
(host_fp32_n8.npz for the 8-step Sudoku chains, host_fp32.npz for Maze, host_<prec>_ref.npz in reference
semantics) likewise keep the carry after their last step only. So phi is computable exactly at the step where a
device chain and a host file end together: step 8 on every completed pinned chain (Sudoku host_fp32_n8, Maze
host_fp32; same-precision host files are 8-step on Maze and 16-step on Sudoku, so the same-precision phi exists
on Maze only). phi at steps 1 and 2 does not exist in the records on either side; the script says so per chain
rather than substituting.

Gate proxy and verdicts exactly as breadth_readout.gate: probe = device cell s2 / host-fp32 cell s2, verdict =
device exact s8 / host-fp32 exact s8, PASS >= 0.95, FAIL < 0.5 (release_check THRESHOLDS), WARN between; the 80
chains are the 2.45 chains with both a step-2 and a step-8 row on the 7 breadth graphs + Sudoku-attn fp32 / w8a16.

    python tools/e3_gate_phi.py --root <dir with <task>/infer/*.npz>   (default results/devices/ctl)
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

DEV = B.DEV
HOST_FP32 = {"sudoku_mlp": {8: "host_fp32_n8.npz", 16: "host_fp32.npz"},
             "sudoku_attn": {8: "host_fp32_n8.npz", 16: "host_fp32.npz"},
             "maze": {8: "host_fp32.npz"}}
# same-precision host carries exist at step 8 only where the host chain is 8 steps (Maze); Sudoku host_<prec> files are 16-step
SAME_PREC_STEPS = {"sudoku_mlp": 16, "sudoku_attn": 16, "maze": 8}
# Sep 29 follow-up chains (E3_FOLLOWUP.md), not in breadth_chain_meta.json; verdicts from that note's tables
FOLLOWUP = [("maze", "w8a8_nofuse2", "Samsung Galaxy S23", "acct2"), ("maze", "w8a8_nofuse2", "Samsung Galaxy S25", "acct2"),
            ("maze", "w8a8_nofuse2_attn16", "Samsung Galaxy S25", "acct2"), ("maze", "w8a16_nofuse2", "Samsung Galaxy S25", "acct2"),
            ("maze", "maze_v79_ropefp", "Samsung Galaxy S25", "acct2"),
            ("sudoku_mlp", "w8a8_nofuse2", "Samsung Galaxy S21", "acct2"), ("sudoku_mlp", "w8a8_nofuse", "Samsung Galaxy S21", "acct2"),
            ("sudoku_mlp", "w8a8_nofuse2", "Snapdragon 7 Gen 4 QRD", "acct2"),
            ("sudoku_mlp", "w8a8_nofuse2", "Dragonwing RB3 Gen 2 Vision Kit", "acct3"), ("sudoku_mlp", "w8a8_nofuse2", "Dragonwing Q-6690 MTP", "acct3")]


def cos_rows(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def auroc(pos, neg):
    """P(score_pos > score_neg) with ties at 1/2 (Mann-Whitney). pos = the class the score should rank HIGH."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def load_dev(path):
    z = np.load(path)
    k = int(z["preds"].shape[0])
    return z["zH"][:, 0], z["zL"][:, 0], k


def phi_of(root, task, prec, npz_name):
    """phi_H, phi_L (mean, min, 10th pct over puzzles) vs host fp32 and vs the same-precision reference at the
    chain's final step; None where no host carry exists at that step."""
    d = root / task / "infer"
    p = d / npz_name
    if not p.exists():
        return None
    zH, zL, k = load_dev(p)
    out = {"steps": k, "fp32": None, "same": None, "same_file": None}
    hf = HOST_FP32[task].get(k)
    if hf and (d / hf).exists():
        h = np.load(d / hf)
        cH, cL = cos_rows(zH, h["zH"]), cos_rows(zL, h["zL"])
        out["fp32"] = dict(H=float(cH.mean()), L=float(cL.mean()), H_min=float(cH.min()), H_p10=float(np.percentile(cH, 10)),
                           L_min=float(cL.min()), per_puzzle_H=cH.tolist())
    if prec != "fp32" and k == SAME_PREC_STEPS[task]:
        for cand in (f"host_{prec}_ref.npz", f"host_{prec}.npz"):
            if (d / cand).exists():
                h = np.load(d / cand)
                cH, cL = cos_rows(zH, h["zH"]), cos_rows(zL, h["zL"])
                out["same"] = dict(H=float(cH.mean()), L=float(cL.mean()), H_min=float(cH.min()), L_min=float(cL.min()))
                out["same_file"] = cand
                break
    return out


def pick_threshold(pos_scores, neg_scores):
    """Cut on phi such that phi >= cut -> PASS. Chosen on the training account: if separable, the midpoint of the
    gap; otherwise the cut with the best balanced accuracy (ties -> higher cut)."""
    pos, neg = np.sort(pos_scores), np.sort(neg_scores)
    if len(neg) == 0:
        return float(pos.min()) - 1e-6, True
    if len(pos) == 0:
        return float(neg.max()) + 1e-6, True
    if pos.min() > neg.max():
        return float((pos.min() + neg.max()) / 2), True
    cands = np.unique(np.concatenate([pos, neg]))
    best, bc = -1, None
    for c in cands:
        ba = 0.5 * ((pos >= c).mean() + (neg < c).mean())
        if ba >= best:
            best, bc = ba, float(c)
    return bc, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEV / "ctl"), help="directory holding <task>/infer/*.npz (pulled from p4d)")
    ap.add_argument("--out", default=str(DEV / "e3_gate_phi.json"))
    ap.add_argument("--train_acct", default="acct2", help="account whose chains choose the phi cut; tested on the rest")
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    tab, meta, devs, meta_all = B.load()
    gate = {(r["task"], r["prec"], r["device"]): r for r in B.gate(tab, meta, devs)}   # the 80 pinned chains

    rows, missing = [], []
    for k, m in meta.items():
        task, prec, src = k
        g = gate.get((task, prec, B.dev_label(m)))
        r = phi_of(root, task, prec, m["file"].replace(".json", ".npz"))
        if r is None:
            missing.append(m["file"])
            continue
        rows.append(dict(task=task, prec=prec, device=m["device"], acct=m["acct"], hexagon=devs.get(m["device"], {}).get("hexagon", "?"),
                         steps=r["steps"], in_gate=g is not None, probe=g["probe"] if g else None, pv=g["pv"] if g else None,
                         s8=g["s8"] if g else None, vv=g["vv"] if g else None,
                         phiH=r["fp32"]["H"] if r["fp32"] else None, phiL=r["fp32"]["L"] if r["fp32"] else None,
                         phiH_min=r["fp32"]["H_min"] if r["fp32"] else None, phiH_p10=r["fp32"]["H_p10"] if r["fp32"] else None,
                         phiL_min=r["fp32"]["L_min"] if r["fp32"] else None,
                         sameH=r["same"]["H"] if r["same"] else None, sameL=r["same"]["L"] if r["same"] else None,
                         same_file=r["same_file"], per_puzzle_H=r["fp32"]["per_puzzle_H"] if r["fp32"] else None))
    # follow-up chains (not in the meta file)
    fu = []
    for task, prec, dev, acct in FOLLOWUP:
        name = f"dev_{prec}_{dev.replace(' ', '_')}_{acct}.npz"
        r = phi_of(root, task, prec, name)
        if r is None:
            missing.append(name)
            continue
        fu.append(dict(task=task, prec=prec, device=dev, acct=acct, hexagon=devs.get(dev, {}).get("hexagon", "?"), steps=r["steps"],
                       phiH=r["fp32"]["H"] if r["fp32"] else None, phiL=r["fp32"]["L"] if r["fp32"] else None,
                       phiH_min=r["fp32"]["H_min"] if r["fp32"] else None,
                       sameH=r["same"]["H"] if r["same"] else None, sameL=r["same"]["L"] if r["same"] else None, same_file=r["same_file"]))

    G = [r for r in rows if r["in_gate"]]
    scored = [r for r in G if r["phiH"] is not None]
    print(f"pinned gate chains: {len(G)}; with a step-8 host carry: {len(scored)}; records not found under {root}: {len(missing)}")
    for r in G:
        if r["phiH"] is None:
            print("  no host carry at the chain's final step:", r["task"], r["prec"], r["device"], r["acct"], "steps", r["steps"])

    def cls(r, which):
        return r[which]

    res = {"n_gate": len(G), "n_scored": len(scored), "missing": missing}
    for score in ("phiH", "phiL", "phiH_min", "phiH_p10", "phiL_min"):
        P = [r[score] for r in scored if r["vv"] == "PASS"]
        W = [r[score] for r in scored if r["vv"] == "WARN"]
        F = [r[score] for r in scored if r["vv"] == "FAIL"]
        res[score] = dict(auroc_pass_vs_fail=auroc(P, F), auroc_pass_vs_warn=auroc(P, W), auroc_pass_vs_notpass=auroc(P, W + F),
                          pass_min=min(P), pass_med=float(np.median(P)), warn=sorted(W), fail_max=max(F) if F else None, fail=sorted(F))
    # the labelled two-step proxy on the same chains, for comparison
    P = [r["probe"] for r in scored if r["vv"] == "PASS"]; W = [r["probe"] for r in scored if r["vv"] == "WARN"]; F = [r["probe"] for r in scored if r["vv"] == "FAIL"]
    res["probe_cell_s2"] = dict(auroc_pass_vs_fail=auroc(P, F), auroc_pass_vs_warn=auroc(P, W), auroc_pass_vs_notpass=auroc(P, W + F))

    # cross-account threshold on phiH (step 8): choose on train_acct, test on the others; PASS = phi >= cut
    def split(acct_sel):
        return [r for r in scored if acct_sel(r["acct"])]
    tr = split(lambda x: x == a.train_acct); te = split(lambda x: x != a.train_acct)
    xa = {}
    for name, trs, tes in (("train=%s test=rest" % a.train_acct, tr, te), ("train=rest test=%s" % a.train_acct, te, tr)):
        for rule in ("midpoint", "min_pass"):
            for score in ("phiH", "phiL"):
                P = [r[score] for r in trs if r["vv"] == "PASS"]; F = [r[score] for r in trs if r["vv"] == "FAIL"]
                if rule == "midpoint":
                    cut, sep = pick_threshold(P, F)
                else:   # the highest cut with no false alarm on the training chains: phi >= min PASS phi
                    cut, sep = float(min(P)) - 1e-9, (min(P) > max(F)) if F else True
                miss_fail = [f"{r['task']} {r['prec']} {r['device']}" for r in tes if r["vv"] == "FAIL" and r[score] >= cut]
                false_alarm = [f"{r['task']} {r['prec']} {r['device']} ({r[score]:.3f})" for r in tes if r["vv"] == "PASS" and r[score] < cut]
                warn_flagged = [f"{r['task']} {r['prec']} {r['device']} ({r[score]:.3f})" for r in tes if r["vv"] == "WARN" and r[score] < cut]
                warn_passed = [f"{r['task']} {r['prec']} {r['device']} ({r[score]:.3f})" for r in tes if r["vv"] == "WARN" and r[score] >= cut]
                xa[f"{name} | {score} | {rule}"] = dict(cut=cut, separable_on_train=sep, n_train=len(trs), n_test=len(tes),
                                                       test_fail=sum(r["vv"] == "FAIL" for r in tes), test_pass=sum(r["vv"] == "PASS" for r in tes),
                                                       missed_fail=miss_fail, false_alarm=false_alarm, warn_flagged=warn_flagged, warn_passed=warn_passed)
    res["cross_account"] = xa
    # same-precision phi where it exists (Maze)
    res["same_prec"] = [dict(task=r["task"], prec=r["prec"], device=r["device"], vv=r["vv"], sameH=r["sameH"], sameL=r["sameL"], file=r["same_file"])
                        for r in scored if r["sameH"] is not None]
    res["rows"] = [{k: v for k, v in r.items() if k != "per_puzzle_H"} for r in rows]
    res["followup"] = fu
    pathlib.Path(a.out).write_text(json.dumps(res, indent=1))

    # ---- markdown
    print("\n## phi at step 8 vs the labelled step-8 verdict (80 pinned chains)\n")
    for score in ("phiH", "phiL", "phiH_min", "phiL_min", "probe_cell_s2"):
        s = res[score]
        print(f"| {score} | AUROC PASS vs FAIL {s['auroc_pass_vs_fail']:.3f} | PASS vs WARN {s['auroc_pass_vs_warn']:.3f} | PASS vs not-PASS {s['auroc_pass_vs_notpass']:.3f} |")
    print("\n## Cross-account threshold (PASS = phi >= cut)\n")
    print("| split | score | rule | cut | separable on train | test PASS / FAIL | missed FAIL | false alarms | WARN flagged | WARN passed |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, x in xa.items():
        split, score, rule = [s.strip() for s in name.split("|")]
        print(f"| {split} | {score} | {rule} | {x['cut']:.4f} | {'yes' if x['separable_on_train'] else 'no'} | {x['test_pass']} / {x['test_fail']} | "
              f"{len(x['missed_fail'])} {x['missed_fail']} | {len(x['false_alarm'])} {x['false_alarm']} | {x['warn_flagged']} | {x['warn_passed']} |")
    print("\n## Per-chain table\n")
    print("| task | prec | device | acct | Hexagon | steps | probe cell s2 | probe verdict | exact s8 ratio | s8 verdict | phi_H s8 | phi_L s8 | phi_H min | same-prec phi_H / phi_L (file) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(G, key=lambda r: (r["vv"], r["task"], r["prec"], r["phiH"] if r["phiH"] is not None else 9)):
        f = lambda v, n=4: "-" if v is None else f"{v:.{n}f}"
        same = "-" if r["sameH"] is None else f"{r['sameH']:.4f} / {r['sameL']:.4f} ({r['same_file']})"
        print(f"| {r['task']} | {r['prec']} | {r['device']} | {r['acct']} | {r['hexagon']} | {r['steps']} | {f(r['probe'], 2)} | {r['pv']} | {f(r['s8'], 2)} | {r['vv']} | {f(r['phiH'])} | {f(r['phiL'])} | {f(r['phiH_min'])} | {same} |")
    print("\n## Follow-up chains (E3_FOLLOWUP.md; verdicts in that note)\n")
    print("| task | prec | device | acct | Hexagon | steps | phi_H vs fp32 | phi_L vs fp32 | phi_H min | same-prec phi_H / phi_L (file) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in fu:
        f = lambda v: "-" if v is None else f"{v:.4f}"
        same = "-" if r["sameH"] is None else f"{r['sameH']:.4f} / {r['sameL']:.4f} ({r['same_file']})"
        print(f"| {r['task']} | {r['prec']} | {r['device']} | {r['acct']} | {r['hexagon']} | {r['steps']} | {f(r['phiH'])} | {f(r['phiL'])} | {f(r['phiH_min'])} | {same} |")
    print("\n## Other pinned chains outside the 80 (no step-2 + step-8 pair, or non-breadth graph)\n")
    for r in rows:
        if not r["in_gate"]:
            f = lambda v: "-" if v is None else f"{v:.4f}"
            print(f"- {r['task']} {r['prec']} {r['device']} ({r['acct']}): steps {r['steps']}, phi_H {f(r['phiH'])}, phi_L {f(r['phiL'])}, same-prec {f(r['sameH'])}")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
