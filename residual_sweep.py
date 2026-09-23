#!/usr/bin/env python3
"""Open question Q3: can the model tell, per puzzle and without the fp32 reference, that its
reasoning broke?  Records, after every supervision step k = 1..nsup, for fp32 and for each
quantizer:
  correct[k]   puzzle-exact correctness of the step-k read-out            (bool per puzzle)
  resid_H[k]   ||z_H(k) - z_H(k-1)|| / ||z_H(k)||  (convergence residual)  (per puzzle)
  resid_L[k]   the same for z_L
  qhalt[k]     the model's own halting-head logit (trained to predict "answer is correct")
  fid_H[k]     cosine(z_H(k), z_H^fp32(k)) on the same puzzle             (reference-based; for
               comparison, needs the fp32 run in the same process)
and reports, per quantizer and step, the AUROC of each signal for predicting an incorrect
answer.  Contraction predicts the residual shrinks inside the basin and does not outside; the
halting head is the learned alternative; fidelity is the reference-based upper bound.

    cd ~/edgetrm_src && python residual_sweep.py --task sudoku_mlp --nsup 16 \
        --quants fp32,w8c,w4c,w4t,w3g32,noise0.1,noise0.2,noise0.3 --n 1000 \
        --out ~/breadth/results/residual_sudoku_mlp.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breadth_sweep as bs  # noqa: E402
from breadth_sweep import DEV, TRM_TASKS, HRM_TASKS, _ActQuant, _ActNoise, _linears, build, load_qa, parse_quant, quantize_weights_  # noqa: E402


def auroc(score, positive):
    """AUROC of `score` for the binary label `positive` (rank-based, ties averaged)."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    n_pos, n_neg = positive.sum(), (~positive).sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and score[order[j + 1]] == score[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def run_steps(m, inp, lab, batch, nsup, act_quant=None, act_noise=None, ref=None):
    m = m.to(DEV)
    n = inp.shape[0]
    handles, obs = [], []
    if act_noise is not None:
        for mod in _linears(m):
            handles.append(mod.register_forward_pre_hook(_ActNoise(act_noise)))
    if act_quant is not None:
        for mod in _linears(m):
            q = _ActQuant(act_quant)
            obs.append(q)
            handles.append(mod.register_forward_pre_hook(q))
    K = list(range(1, nsup + 1))
    rec = {k: {"correct": [], "resid_H": [], "resid_L": [], "qhalt": [], "fid_H": [], "fid_L": []} for k in K}
    states = {k: {"zH": [], "zL": []} for k in K} if ref is None else None
    off = 0
    for ci, s in enumerate(range(0, n, batch)):
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        B = x.shape[0]
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}

        def fresh():
            c = m.initial_carry(bt)
            ic = c.inner_carry
            return type(c)(inner_carry=type(ic)(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)),
                           steps=c.steps.to(DEV), halted=c.halted.to(DEV),
                           current_data={k: v.to(DEV) for k, v in c.current_data.items()})

        carry = fresh()
        if act_quant is not None and ci == 0:               # calibrate static scales at full depth
            cc = carry
            for _ in range(nsup):
                cc, _ = m(cc, bt)
            for q in obs:
                q.freeze()
            carry = fresh()
        prevH, prevL = carry.inner_carry.z_H.float(), carry.inner_carry.z_L.float()
        mask = y != 0
        for k in K:
            carry, out = m(carry, bt)
            zH, zL = carry.inner_carry.z_H.float(), carry.inner_carry.z_L.float()
            preds = out["logits"].argmax(-1)
            corr = ((preds == y) & mask).sum(-1) == mask.sum(-1)
            rH = (zH - prevH).flatten(1).norm(dim=1) / zH.flatten(1).norm(dim=1).clamp_min(1e-8)
            rL = (zL - prevL).flatten(1).norm(dim=1) / zL.flatten(1).norm(dim=1).clamp_min(1e-8)
            qh = out.get("q_halt_logits")
            r = rec[k]
            r["correct"].extend(corr.cpu().tolist())
            r["resid_H"].extend(rH.cpu().tolist())
            r["resid_L"].extend(rL.cpu().tolist())
            r["qhalt"].extend(qh.float().cpu().tolist() if qh is not None else [None] * B)
            if ref is None:
                states[k]["zH"].append(zH.half().cpu())
                states[k]["zL"].append(zL.half().cpu())
            else:
                refH = ref[k]["zH"][off:off + B].to(DEV).float()
                refL = ref[k]["zL"][off:off + B].to(DEV).float()
                r["fid_H"].extend(F.cosine_similarity(zH.flatten(1), refH.flatten(1), dim=-1).cpu().tolist())
                r["fid_L"].extend(F.cosine_similarity(zL.flatten(1), refL.flatten(1), dim=-1).cpu().tolist())
            prevH, prevL = zH, zL
        off += B
    for h in handles:
        h.remove()
    m.cpu()
    torch.cuda.empty_cache()
    if states is not None:
        states = {k: {"zH": torch.cat(v["zH"]), "zL": torch.cat(v["zL"])} for k, v in states.items()}
    return rec, states


def summarize(rec, nsup):
    out = {}
    for k in range(1, nsup + 1):
        r = rec[k]
        wrong = ~np.asarray(r["correct"], bool)
        s = {"acc": float(np.mean(r["correct"])),
             "resid_H_mean": float(np.mean(r["resid_H"])), "resid_L_mean": float(np.mean(r["resid_L"])),
             "auroc_resid_H": auroc(r["resid_H"], wrong), "auroc_resid_L": auroc(r["resid_L"], wrong)}
        if r["qhalt"] and r["qhalt"][0] is not None:
            s["qhalt_mean"] = float(np.mean(r["qhalt"]))
            s["auroc_qhalt"] = auroc([-v for v in r["qhalt"]], wrong)      # low halt logit -> wrong
        if r["fid_H"]:
            s["fid_H_mean"] = float(np.mean(r["fid_H"]))
            s["auroc_fid_H"] = auroc([-v for v in r["fid_H"]], wrong)
            s["auroc_fid_L"] = auroc([-v for v in r["fid_L"]], wrong)
        out[k] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TRM_TASKS) + list(HRM_TASKS))
    ap.add_argument("--quants", default="fp32,w8c,w4c,w4t,w3g32,noise0.1,noise0.2,noise0.3")
    ap.add_argument("--H", default="")
    ap.add_argument("--nsup", type=int, default=16)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--variant", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.ckpt:
        bs.CKPT_OVERRIDE = a.ckpt
    np.random.seed(0)
    torch.manual_seed(0)
    t = TRM_TASKS.get(a.task) or HRM_TASKS[a.task]
    batch = a.batch or t["batch"]
    inp, lab = load_qa(t["data"], a.n)
    H = int(a.H) if a.H else None
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quants = [q.strip() for q in a.quants.split(",") if q.strip()]
    t0 = time.time()
    m, meta = build(a.task, H, a.nsup, batch)
    ref_rec, ref_states = run_steps(m, inp, lab, batch, a.nsup)
    del m
    records = [{"task": a.task, "H_cycles": meta["H_cycles"], "nsup": a.nsup, "quant": parse_quant("fp32"),
                "n": int(len(inp)), "variant": a.variant, "summary": summarize(ref_rec, a.nsup),
                "per_step": {k: {kk: vv for kk, vv in v.items() if kk in ("correct", "resid_H", "resid_L", "qhalt")}
                             for k, v in ref_rec.items()}, "secs": time.time() - t0}]
    print(f"{a.task} fp32: " + " ".join(f"k{k}={s['acc']*100:.1f}/rH{s['resid_H_mean']:.3f}/auR{s['auroc_resid_H'] or 0:.2f}"
                                       + (f"/auQ{s['auroc_qhalt']:.2f}" if "auroc_qhalt" in s else "")
                                       for k, s in records[0]["summary"].items() if k in (1, 2, 4, 8, a.nsup)), flush=True)
    json.dump({"task": a.task, "records": records}, open(out_path, "w"))
    for spec in quants:
        if spec == "fp32":
            continue
        q = parse_quant(spec)
        t1 = time.time()
        mq, _ = build(a.task, H, a.nsup, batch)
        if q["gran"] is not None:
            quantize_weights_(mq, q["bits"], q["gran"], q.get("sigma"))
        rec, _ = run_steps(mq, inp, lab, batch, a.nsup, act_quant=q["act"], act_noise=q.get("anoise"), ref=ref_states)
        del mq
        summ = summarize(rec, a.nsup)
        records.append({"task": a.task, "H_cycles": meta["H_cycles"], "nsup": a.nsup, "quant": q, "n": int(len(inp)),
                        "variant": a.variant, "summary": summ,
                        "per_step": {k: {kk: ([round(x, 4) if isinstance(x, float) else x for x in vv])
                                         for kk, vv in v.items()} for k, v in rec.items()},
                        "secs": time.time() - t1})
        print(f"  {spec:10s}: " + " ".join(f"k{k}={s['acc']*100:.1f}/rH{s['resid_H_mean']:.3f}/auR{s['auroc_resid_H'] or 0:.2f}"
                                          + (f"/auQ{s['auroc_qhalt']:.2f}" if "auroc_qhalt" in s else "")
                                          + (f"/fid{s['fid_H_mean']:.2f}/auF{s['auroc_fid_H'] or 0:.2f}" if "fid_H_mean" in s else "")
                                          for k, s in summ.items() if k in (1, 2, 4, 8, a.nsup)), flush=True)
        json.dump({"task": a.task, "records": records}, open(out_path, "w"))
    print("wrote", out_path, len(records))


if __name__ == "__main__":
    main()
