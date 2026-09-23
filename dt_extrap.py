#!/usr/bin/env python3
"""Deep-thinking nets: extrapolation under compression (open question Q1) and
answer-bearing fidelity (Q2).

Extends dt_breadth.py in two ways.
  1. Any iteration count (e.g. up to 500, past the checkpoint's max_iters), keeping
     only the requested iterations' feature maps, so the large-maze curves can be
     followed to saturation: does compression *tax* the iteration budget (the curve
     shifts right) or *cap* it (the curve plateaus lower)?
  2. Fidelity is reported four ways per cell, all cosine similarities between the
     compressed and fp32 recurrent feature maps on the same inputs:
       fidelity        whole flattened map (what dt_breadth.py reports)
       fid_target      restricted to the target-path cells (spatial mask = target==1)
       fid_pred        restricted to the cells fp32 itself predicts as path at that iteration
       fid_cell_target per-cell cosine over channels, averaged over target-path cells
     For prefix sums every position is answer-bearing, so the masked variants equal the
     whole-map one and are reported as null.

    cd ~/breadth/deep-thinking && ~/venvs/kvcache/bin/python ~/edgetrm_src/dt_extrap.py \
        --task mazes --sizes 33 --iters 10,20,40,60,100,150,200,300,500 \
        --quants fp32,w8c,w4g32,w4c,w4t,w3g32,noise0.1,noise0.2,noise0.3 \
        --n 1000 --out ~/breadth/results/dt_extrap_mazes_33.json

Records carry per-example correctness and per-example whole/target cosines so the
scatter of Fig 4.9 can be redrawn with any variant.
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
from dt_breadth import DEV, _ActQuant, build, dataset, parse_quant, quantize_  # noqa: E402


def _masked_cos(fq, fr, m):
    """fq, fr: [B, C, P]; m: [B, 1, P] float mask. Cosine over the masked (c, p) entries."""
    num = (fq * fr * m).sum(dim=(1, 2))
    den = torch.sqrt((fq * fq * m).sum(dim=(1, 2)) * (fr * fr * m).sum(dim=(1, 2))).clamp_min(1e-12)
    return num / den


def _cell_cos(fq, fr, m):
    """Per-position cosine over channels, averaged over masked positions. m: [B, P] bool."""
    cos = F.cosine_similarity(fq, fr, dim=1)                      # [B, P]
    mf = m.float()
    return (cos * mf).sum(1) / mf.sum(1).clamp_min(1.0)


@torch.no_grad()
def run(net, loader, task, iters, act=None, ref=None):
    """Returns per-iteration dicts. With ref=None this is the fp32 pass and returns the
    feature maps ([N, C, P] float16 on CPU), the predicted-path masks and the target masks;
    with ref given it returns the four fidelity variants per example instead."""
    from deepthinking.utils.testing import get_predicted
    net = net.to(DEV)
    max_it = max(iters)
    keep = set(iters)
    store, counter = {}, [0]

    def hook(m, i, o):
        counter[0] += 1
        if counter[0] in keep:
            store[counter[0]] = o.detach()

    h = net.recur_block.register_forward_hook(hook)
    handles, obs = [], []
    if act is not None:
        for mod in net.modules():
            if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d)):
                q = _ActQuant(act)
                obs.append(q)
                handles.append(mod.register_forward_pre_hook(q))
    correct = {k: [] for k in iters}
    feats = {k: [] for k in iters}
    pmask = {k: [] for k in iters}
    tmask = []
    fid = {k: {"whole": [], "target": [], "pred": [], "cell": []} for k in iters}
    off = 0
    for bi, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(DEV), targets.to(DEV)
        B = inputs.size(0)
        if act is not None and bi == 0:
            counter[0] = 0
            store.clear()
            net(inputs, iters_to_do=max_it)          # calibration pass at full depth
            for q in obs:
                q.freeze()
        counter[0] = 0
        store.clear()
        all_out = net(inputs, iters_to_do=max_it)
        tg = targets.view(B, -1)
        tmask_b = (tg == 1)
        if ref is None:
            tmask.append(tmask_b.cpu())
        for k in iters:
            pred = get_predicted(inputs, all_out[:, k - 1], task)
            correct[k].extend(torch.amin(pred == tg, dim=[1]).cpu().tolist())
            f = store[k].float().reshape(B, store[k].shape[1], -1)      # [B, C, P]
            pm = (pred == 1)
            if ref is None:
                feats[k].append(f.half().cpu())
                pmask[k].append(pm.cpu())
            else:
                fr = ref["feats"][k][off:off + B].to(DEV).float()
                fid[k]["whole"].extend(F.cosine_similarity(f.flatten(1), fr.flatten(1), dim=-1).cpu().tolist())
                if task == "prefix_sums":
                    continue
                m_t = tmask_b.unsqueeze(1).float()
                m_p = ref["pmask"][k][off:off + B].to(DEV).unsqueeze(1).float()
                fid[k]["target"].extend(_masked_cos(f, fr, m_t).cpu().tolist())
                fid[k]["pred"].extend(_masked_cos(f, fr, m_p).cpu().tolist())
                fid[k]["cell"].extend(_cell_cos(f, fr, tmask_b).cpu().tolist())
        off += B
    h.remove()
    for hh in handles:
        hh.remove()
    net.cpu()
    torch.cuda.empty_cache()
    out = {"correct": {k: np.array(v, dtype=bool) for k, v in correct.items()}}
    if ref is None:
        out["feats"] = {k: torch.cat(v) for k, v in feats.items()}
        out["pmask"] = {k: torch.cat(v) for k, v in pmask.items()}
        out["tmask"] = torch.cat(tmask)
    else:
        out["fid"] = fid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["prefix_sums", "mazes", "chess"])
    ap.add_argument("--sizes", required=True)
    ap.add_argument("--iters", default="10,20,40,60,100,150,200,300,500")
    ap.add_argument("--quants", default="fp32,w8c,w4g32,w4c,w4t,w3g32,noise0.1,noise0.2,noise0.3")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    iters = [int(x) for x in a.iters.split(",")]
    quants = [q for q in a.quants.split(",") if q]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():
        try:
            records = json.load(open(out_path))["records"]
        except (json.JSONDecodeError, ValueError):
            print(f"  {out_path} unreadable; starting fresh", flush=True)
    done = {(r["size"], r["iters"], r["quant"]["spec"]) for r in records}

    def save():
        tmp = out_path.with_suffix(".tmp")
        json.dump({"task": a.task, "protocol": "dt_extrap", "records": records}, open(tmp, "w"))
        tmp.replace(out_path)

    for size in a.sizes.split(","):
        ds = dataset(a.task, size)
        sub = torch.utils.data.Subset(ds, list(range(min(a.n, len(ds)))))
        loader = torch.utils.data.DataLoader(sub, batch_size=a.batch, shuffle=False)
        net, meta = build(a.task)
        t0 = time.time()
        ref = run(net, loader, a.task, iters)
        n_path = float(ref["tmask"].float().sum(1).mean())
        for k in iters:
            if (size, k, "fp32") in done:
                continue
            records.append({"task": a.task, "size": size, "iters": k, "quant": parse_quant("fp32"), **meta,
                            "n": int(len(ref["correct"][k])), "acc": float(ref["correct"][k].mean()),
                            "fidelity": 1.0, "fid_target": None if a.task == "prefix_sums" else 1.0,
                            "fid_pred": None if a.task == "prefix_sums" else 1.0,
                            "fid_cell_target": None if a.task == "prefix_sums" else 1.0,
                            "mean_target_cells": n_path,
                            "correct": ref["correct"][k].astype(int).tolist(), "secs": time.time() - t0})
        print(f"{a.task} size={size}: fp32 " + " ".join(f"it{k}={ref['correct'][k].mean()*100:.1f}" for k in iters)
              + f"  ({time.time()-t0:.0f}s, {n_path:.0f} target cells)", flush=True)
        save()
        for spec in quants:
            if spec == "fp32" or all((size, k, spec) in done for k in iters):
                continue
            q = parse_quant(spec)
            net, meta = build(a.task)
            nq = quantize_(net, q["bits"], q["gran"], q.get("sigma"))
            t1 = time.time()
            r = run(net, loader, a.task, iters, act=q["act"], ref=ref)
            for k in iters:
                fk = r["fid"][k]
                mean = lambda v: (float(np.mean(v)) if len(v) else None)
                records.append({"task": a.task, "size": size, "iters": k, "quant": q, **meta,
                                "n": int(len(r["correct"][k])), "acc": float(r["correct"][k].mean()),
                                "fidelity": mean(fk["whole"]), "fid_target": mean(fk["target"]),
                                "fid_pred": mean(fk["pred"]), "fid_cell_target": mean(fk["cell"]),
                                "mean_target_cells": n_path,
                                "correct": r["correct"][k].astype(int).tolist(),
                                "cos_whole": [round(x, 4) for x in fk["whole"]],
                                "cos_target": [round(x, 4) for x in fk["target"]],
                                "n_quantized_convs": nq, "secs": time.time() - t1})
            print(f"  {spec:9s} " + " ".join(
                f"it{k}={r['correct'][k].mean()*100:.1f}/f{np.mean(r['fid'][k]['whole']):.3f}"
                + (f"/t{np.mean(r['fid'][k]['target']):.3f}" if r["fid"][k]["target"] else "") for k in iters),
                flush=True)
            save()
    print("wrote", out_path, len(records))


if __name__ == "__main__":
    main()
