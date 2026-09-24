#!/usr/bin/env python3
"""Re-score Maze-30x30-Hard under three correctness definitions, per ACT step, with the halting
logit, for fp32 and every quantizer (X1; reports/edgetrm_exp/X1.md).

Why: exact match to one reference path mis-scores Maze-Hard. Every one of the 1000 test mazes
has more than one shortest path (min 96, median ~3.8e6; checked from test.csv), so a model can
return a correct shortest path that is not the reference. TRM's reported "failures" are mostly
valid paths (2609.26487 App. K). This script records, after every ACT step k = 1..nsup:
  exact[k]     every cell equals the reference (the metric used so far)
  simple[k]    a valid path: every non-path cell reproduces the input, path cells sit on open
               cells, and path + S + G form one simple 4-connected path S -> G (S and G have one
               path neighbour, every path cell exactly two, one component)
  conn[k]      lenient: non-path cells reproduce the input and S, G are 4-connected through the
               marked cells (extra marked cells allowed)
  shortest[k]  simple and exactly as long as the reference (the reference is a shortest path for
               all 1000 test mazes), i.e. "valid and optimal", HRM's own definition of correct
  qhalt[k]     the halting-head logit (for the Q3 AUROC under each definition)
and, after the last step, the predicted path mask and the count of wrong cells.

    cd ~/edgetrm_src && python maze_rescore.py --task maze_attn --quants fp32,w8c,w4c,w4t \
        --out ~/breadth_results/x1/maze_rescore_maze_attn.json
Resumable per spec. Needs scipy (component labelling).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from breadth_sweep import (DEV, HRM_TASKS, TRM_TASKS, _ActNoise, _ActQuant, _linears, build,  # noqa: E402
                           load_qa, parse_quant, quantize_weights_)

WALL, OPEN, S_, G_, PATH = 1, 2, 3, 4, 5          # "# SGo" index + 1
N = 30
FOUR = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def score(x, p, ref_len):
    """x, p: (B, 900) int arrays (input, prediction); ref_len: (B,) reference path length.
    Returns dict of (B,) bool arrays: simple, conn, shortest."""
    B = x.shape[0]
    P = p == PATH
    nonpath_ok = ((p == x) | (P & (x == OPEN))).all(1)
    Q = P | (x == S_) | (x == G_)
    Qg = Q.reshape(B, N, N)
    Qp = np.pad(Qg, ((0, 0), (1, 1), (1, 1)))
    deg = (Qp[:, :-2, 1:-1].astype(int) + Qp[:, 2:, 1:-1] + Qp[:, 1:-1, :-2] + Qp[:, 1:-1, 2:]).reshape(B, -1)
    simple = np.zeros(B, bool)
    conn = np.zeros(B, bool)
    for i in range(B):
        if not nonpath_ok[i]:
            continue
        s, g = int(np.argmax(x[i] == S_)), int(np.argmax(x[i] == G_))
        lab, ncomp = ndimage.label(Qg[i], structure=FOUR)
        lab = lab.reshape(-1)
        conn[i] = lab[s] == lab[g] and lab[s] > 0
        if conn[i] and ncomp == 1 and deg[i, s] == 1 and deg[i, g] == 1 and (deg[i][P[i]] == 2).all():
            simple[i] = True
    shortest = simple & (P.sum(1) == ref_len)
    return {"simple": simple, "conn": conn, "shortest": shortest}


def _fresh(m, bt):
    c = m.initial_carry(bt)
    ic = c.inner_carry
    return type(c)(inner_carry=type(ic)(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)),
                   steps=c.steps.to(DEV), halted=c.halted.to(DEV),
                   current_data={k: v.to(DEV) for k, v in c.current_data.items()})


@torch.no_grad()
def run(m, inp, lab, batch, K, act_quant=None, act_noise=None):
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
    ref_len = (lab == PATH).sum(1)
    flags = {k: np.zeros((n, K), bool) for k in ("exact", "simple", "conn", "shortest")}
    qhalt = np.zeros((n, K), np.float32)
    final_path = np.zeros((n, inp.shape[1]), bool)
    wrong_cells = np.zeros(n, np.int32)
    for ci, s in enumerate(range(0, n, batch)):
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        B = x.shape[0]
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
        carry = _fresh(m, bt)
        if act_quant is not None and ci == 0:
            cc = carry
            for _ in range(K):
                cc, _ = m(cc, bt)
            for q in obs:
                q.freeze()
            carry = _fresh(m, bt)
        xn = inp[s:s + B]
        for k in range(K):
            carry, out = m(carry, bt)
            pred = out["logits"].argmax(-1)
            flags["exact"][s:s + B, k] = (pred == y).all(-1).cpu().numpy()
            pn = pred.cpu().numpy()
            sc = score(xn, pn, ref_len[s:s + B])
            for key in ("simple", "conn", "shortest"):
                flags[key][s:s + B, k] = sc[key]
            qh = out.get("q_halt_logits")
            qhalt[s:s + B, k] = qh.float().cpu().numpy() if qh is not None else np.nan
            if k == K - 1:
                final_path[s:s + B] = pn == PATH
                wrong_cells[s:s + B] = (pn != lab[s:s + B]).sum(1)
    for h in handles:
        h.remove()
    m.cpu()
    torch.cuda.empty_cache()
    return flags, qhalt, final_path, wrong_cells


def pack_rows(bits):
    w = (bits.shape[1] + 3) // 4
    out = []
    for row in bits:
        v = 0
        for j in np.nonzero(row)[0]:
            v |= 1 << int(j)
        out.append(format(v, f"0{w}x"))
    return out


def atomic_dump(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["maze_attn", "hrm_maze"])
    ap.add_argument("--quants", default="fp32,w8c,w4c,w4t")
    ap.add_argument("--nsup", type=int, default=16)
    ap.add_argument("--H", default="")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t = TRM_TASKS.get(a.task) or HRM_TASKS[a.task]
    batch = a.batch or t["batch"]
    inp, lab = load_qa("maze", a.n)
    # sanity: the reference is a valid shortest path under our own scorer
    sc = score(inp, lab, (lab == PATH).sum(1))
    assert sc["shortest"].all(), f"scorer rejects {int((~sc['shortest']).sum())} reference paths"
    H = int(a.H) if a.H else None
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"task": a.task, "n": int(len(inp)), "nsup": a.nsup, "records": []}
    if out_path.exists():
        try:
            doc["records"] = json.load(open(out_path))["records"]
        except (json.JSONDecodeError, ValueError):
            pass
    done = {r["spec"] for r in doc["records"]}
    quants = [q.strip() for q in a.quants.split(",") if q.strip()]
    print(f"{a.task}: {len(inp)} mazes, nsup {a.nsup}, specs {quants}; {len(done)} done; "
          f"{torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'}", flush=True)
    for spec in quants:
        if spec in done:
            continue
        base, seed = (spec.split("@s")[0], int(spec.split("@s")[1])) if "@s" in spec else (spec, 0)
        q = parse_quant(base)
        t1 = time.time()
        m, meta = build(a.task, H, a.nsup, batch)
        torch.manual_seed(seed)
        if q["gran"] is not None:
            quantize_weights_(m, q["bits"], q["gran"], q.get("sigma"))
        torch.manual_seed(seed)
        flags, qh, fpath, wc = run(m, inp, lab, batch, a.nsup, act_quant=q["act"], act_noise=q.get("anoise"))
        del m
        rec = {"spec": spec, "quant": q, "seed": seed, "H_cycles": meta["H_cycles"], "nsup": a.nsup,
               "n": int(len(inp)), "secs": time.time() - t1,
               "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu",
               "acc": {k: v.mean(0).tolist() for k, v in flags.items()},
               "flags": {k: pack_rows(v) for k, v in flags.items()},
               "qhalt": np.round(qh.astype(float), 3).tolist(),
               "final_path": pack_rows(fpath), "wrong_cells": wc.tolist()}
        doc["records"].append(rec)
        atomic_dump(doc, out_path)
        a16 = {k: 100 * v[-1] for k, v in rec["acc"].items()}
        print(f"  {spec:12s} ({rec['secs']:.0f}s) step {a.nsup}: exact {a16['exact']:.1f} shortest "
              f"{a16['shortest']:.1f} simple {a16['simple']:.1f} conn {a16['conn']:.1f}", flush=True)
    print("wrote", out_path, len(doc["records"]), flush=True)


if __name__ == "__main__":
    main()
