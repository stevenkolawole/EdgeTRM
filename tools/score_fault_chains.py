#!/usr/bin/env python3
"""Score the 2-loop build check on the finished aihub_faults.py chains, out of sample.

Statistic (as tools/aihub_e3_steps12.py phi, s2_fp32 H): mean over puzzles of the cosine between the device z_H after
step 2 and the host fp32 z_H after step 2. Label (as E3_FAULT_TABLE s3): device exact at step 8 / host fp32 exact at
step 8, PASS >= 0.95, FAIL < 0.5, else WARN. Cuts come from the 80 gate builds (results/devices/e3_gate_phi_step2.json,
auroc.s2_fp32_H): the PASS floor and the midpoint between that floor and the FAIL ceiling; neither saw these chains.

Reads the packed chains under /data, so it runs on a compute node:
    python score_fault_chains.py --out ~/aihub_faults/check/fault_chains_scored.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_faults as F  # noqa: E402

PASS_FLOOR, FAIL_CEIL = 0.8206705575659003, 0.7685424976192253
CUTS = {"pass_floor": PASS_FLOOR, "midpoint": (PASS_FLOOR + FAIL_CEIL) / 2}


def cos_rows(x, y):
    x = x.reshape(x.shape[0], -1).astype(np.float64); y = y.reshape(y.shape[0], -1).astype(np.float64)
    return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)


def auroc(pos, neg):
    """pos = FAIL scores, neg = PASS scores; a lower cosine is more suspicious"""
    if not pos or not neg:
        return None
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float((p < n).mean() + 0.5 * (p == n).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for task in ("sudoku_mlp", "sudoku_attn", "maze", "hrm_sudoku"):
        tdir = F.FAULTS / task
        hj = tdir / "packed" / "host_fp32_n8.json"
        hpath = json.loads(hj.read_text())["path"] if hj.exists() else str(tdir / "host_fp32_n8.npz")
        h = np.load(hpath)
        hzH, hpreds = h["zH"][1], h["preds"]
        hpred = hpreds[7]
        _, lab, _ = F.data(task)
        N = hzH.shape[0]
        lab = lab[:N]
        host_exact = float((hpred == lab).all(-1).mean())
        for m in sorted(tdir.glob("dev_*.json")):
            meta = json.loads(m.read_text())
            if len(meta["steps"]) != 8:
                continue
            tag = m.stem[len("dev_"):]
            pk = tdir / "packed" / f"{tag}.json"
            parts = json.loads(pk.read_text())["parts"] if pk.exists() else []
            src = next((p["path"] for p in parts if 1 in p["steps"]), None)
            r = dict(task=task, prec=meta["prec"], device=meta["device"], acct=meta["account"], tag=tag,
                     dev_exact_s8=meta["steps"][-1]["exact"], dev_cell_s8=meta["steps"][-1]["cell"],
                     host_exact_s8=host_exact)
            ratio = r["dev_exact_s8"] / host_exact if host_exact > 0 else None
            r["exact_ratio_s8"] = ratio
            r["label"] = None if ratio is None else ("PASS" if ratio >= 0.95 else "FAIL" if ratio < 0.5 else "WARN")
            if src is None:
                r["phi_s2_H"] = None; r["note"] = "no packed step-2 carry"; rows.append(r); continue
            z = np.load(src)
            steps = list(z["carry_steps"]) if "carry_steps" in z.files else list(range(z["zH_steps"].shape[0]))
            dz = z["zH_steps"][steps.index(1)][:, 0]
            assert dz.shape[0] == N, (tag, dz.shape, N)
            c = cos_rows(dz, hzH)
            r["phi_s2_H"], r["phi_s2_H_min"] = float(c.mean()), float(c.min())
            if "preds" in z.files and z["preds"].shape[0] == 8:
                r["dev_exact_s8_recomputed"] = float((z["preds"][7] == lab).all(-1).mean())
                for k in (1, 2, 8):
                    r[f"pred_agree_s{k}"] = float((z["preds"][k - 1] == hpreds[k - 1]).mean())
            rows.append(r)
            print(f"{task} {tag}: phi_s2 {r['phi_s2_H']:.4f} exact {r['dev_exact_s8']:.3f}/{host_exact:.3f} {r['label']}",
                  flush=True)
    sc = [r for r in rows if r.get("phi_s2_H") is not None and r["label"]]
    P = [r["phi_s2_H"] for r in sc if r["label"] == "PASS"]
    Fl = [r["phi_s2_H"] for r in sc if r["label"] == "FAIL"]
    W = [r["phi_s2_H"] for r in sc if r["label"] == "WARN"]
    summ = dict(n_scored=len(sc), n_pass=len(P), n_warn=len(W), n_fail=len(Fl),
                auroc_fail_vs_pass=auroc(Fl, P), auroc_notpass_vs_pass=auroc(Fl + W, P),
                auroc_pred_agree={k: auroc([r[k] for r in sc if r["label"] == "FAIL" and k in r],
                                           [r[k] for r in sc if r["label"] == "PASS" and k in r])
                                  for k in ("pred_agree_s1", "pred_agree_s2", "pred_agree_s8")},
                pass_range=[min(P), max(P)] if P else None, fail_range=[min(Fl), max(Fl)] if Fl else None, warn=sorted(W),
                cuts={})
    for name, cut in CUTS.items():
        summ["cuts"][name] = dict(cut=cut,
                                  missed_fail=[r["tag"] + " " + r["task"] for r in sc if r["label"] == "FAIL" and r["phi_s2_H"] >= cut],
                                  false_alarm=[r["tag"] + " " + r["task"] for r in sc if r["label"] == "PASS" and r["phi_s2_H"] < cut],
                                  warn_flagged=sum(r["phi_s2_H"] < cut for r in sc if r["label"] == "WARN"))
    json.dump(dict(description=__doc__.strip().split("\n")[0], summary=summ, rows=rows), open(a.out, "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
