#!/usr/bin/env python3
"""M5 x M1 depth-for-precision: does the halting head keep the hybrid's accuracy at a fraction of the steps?

M1's schedule c(T-k).P(k): T-k supervision steps with a cheap low-bit model (e.g. w3c, nested n8c3), then k
steps with an INT8 model (w8c); on TRM Sudoku-attention it beats fp32 at any depth from T=64 on (M1.md round 2).
Here each puzzle follows a per-instance schedule simulated online (only the rows in a phase are run):
  cheap phase   until the cheap model's head fires (exit rule) or T-k cheap steps
  polish phase  k polish steps, or until the polish model's head fires (at most k)
Rules (all R rules evaluated in one pass, one copy of the batch per rule):
  fix<k>            M1's c(T-k).P(k) (no exit)
  <e>-><k>          cheap until exit rule e (nat = q>0, pat2 = two consecutive q>0), then k polish steps
  <e>->h<p><k>      cheap until e, then polish until the polish head fires (rule p), at most k
  <e>->0            cheap until e (capped at T), no polish (the exit alone)
Per puzzle and rule: final exact correctness, cheap steps, polish steps.  Weight transforms from polish.py
(quant_2d: w<b><t|c|a|gN>, n<B><t|c|a><b>, fp32), applied to every CastedLinear, as M1 does.

    python exit_hybrid.py --task sudoku_attn --cheap w3c,n8c3 --polish w8c --T 256 --out ~/breadth_results/m5/hybrid/attn_T256.json
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breadth_sweep as bs  # noqa: E402
from breadth_sweep import DEV, TRM_TASKS, build, load_qa, _linears  # noqa: E402
from detector_sweep import explicit_carry  # noqa: E402
from polish import quant_2d  # noqa: E402

RULES = ["fix16", "fix32", "pat2->16", "pat2->32", "pat2->hpat2_16", "pat2->hpat2_32", "nat->hnat_32", "pat2->hnat_32", "pat2->0"]


def parse_rule(r):
    if r.startswith("fix"):
        return {"exit": None, "k": int(r[3:]), "pstop": None}
    e, rest = r.split("->")
    if rest.startswith("h"):
        p, k = rest[1:].split("_")
        return {"exit": e, "k": int(k), "pstop": p}
    return {"exit": e, "k": int(rest), "pstop": None}


def make(task, spec, batch):
    m, _ = build(task, None, 10 ** 6, batch)
    with torch.no_grad():
        for mod in _linears(m):
            mod.weight.data = quant_2d(mod.weight.data.float(), spec).to(mod.weight.dtype)
    return m.to(DEV).eval()


def sub_carry(carry, idx):
    ic = carry.inner_carry
    return type(carry)(inner_carry=type(ic)(z_H=ic.z_H[idx], z_L=ic.z_L[idx]), steps=carry.steps[idx],
                       halted=torch.zeros_like(carry.halted[idx]), current_data={k: v[idx] for k, v in carry.current_data.items()})


@torch.no_grad()
def run_chunk(mc, mp, x, y, T, rules):
    R, B = len(rules), x.shape[0]
    X, Y = x.repeat(R, 1), y.repeat(R, 1)
    N = R * B
    bt = {"inputs": X, "labels": Y, "puzzle_identifiers": torch.zeros(N, dtype=torch.int32, device=DEV)}
    carry = explicit_carry(mc, bt)
    zH, zL = carry.inner_carry.z_H.clone(), carry.inner_carry.z_L.clone()
    P = [parse_rule(r) for r in rules]
    k = torch.tensor([p["k"] for p in P], device=DEV).repeat_interleave(B)
    ex = [p["exit"] for p in P]
    ps = [p["pstop"] for p in P]
    ex_code = torch.tensor([{None: 0, "nat": 1, "pat2": 2}[e] for e in ex], device=DEV).repeat_interleave(B)
    ps_code = torch.tensor([{None: 0, "nat": 1, "pat2": 2}[e] for e in ps], device=DEV).repeat_interleave(B)
    phase = torch.zeros(N, dtype=torch.long, device=DEV)          # 0 cheap, 1 polish, 2 done
    ncheap = torch.zeros(N, dtype=torch.long, device=DEV)
    npol = torch.zeros(N, dtype=torch.long, device=DEV)
    run_c = torch.zeros(N, dtype=torch.long, device=DEV)          # consecutive halts of the current phase's head
    correct = torch.zeros(N, dtype=torch.bool, device=DEV)
    for _ in range(T):
        for ph, m in ((0, mc), (1, mp)):
            idx = torch.nonzero(phase == ph).squeeze(1)
            if idx.numel() == 0:
                continue
            c = type(carry)(inner_carry=type(carry.inner_carry)(z_H=zH, z_L=zL), steps=carry.steps,
                            halted=carry.halted, current_data=bt)
            cs = sub_carry(c, idx)
            cs2, out = m(cs, {kk: v[idx] for kk, v in bt.items()})
            zH[idx], zL[idx] = cs2.inner_carry.z_H, cs2.inner_carry.z_L
            correct[idx] = (out["logits"].argmax(-1) == Y[idx]).all(1)
            q = out["q_halt_logits"].float() > 0
            run_c[idx] = torch.where(q, run_c[idx] + 1, torch.zeros_like(run_c[idx]))
            if ph == 0:
                ncheap[idx] += 1
                code = ex_code[idx]
                fired = ((code == 1) & (run_c[idx] >= 1)) | ((code == 2) & (run_c[idx] >= 2))
                switch = fired | (ncheap[idx] >= T - k[idx])
                sw = idx[switch]
                phase[sw] = torch.where(k[sw] > 0, torch.ones_like(sw), torch.full_like(sw, 2))
                run_c[sw] = 0
            else:
                npol[idx] += 1
                code = ps_code[idx]
                fired = ((code == 1) & (run_c[idx] >= 1)) | ((code == 2) & (run_c[idx] >= 2))
                stop = fired | (npol[idx] >= k[idx])
                phase[idx[stop]] = 2
        if (phase == 2).all():
            break
    f = lambda t: t.view(R, B).cpu().numpy()
    return f(correct), f(ncheap), f(npol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sudoku_attn", choices=list(TRM_TASKS))
    ap.add_argument("--cheap", default="w3c,n8c3")
    ap.add_argument("--polish", default="w8c")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow_cpu", action="store_true")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA"
    if a.ckpt:
        bs.CKPT_OVERRIDE = a.ckpt
    rules = [r for r in a.rules.split(",") if r]
    inp, lab = load_qa(TRM_TASKS[a.task]["data"], a.n)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    recs = json.load(open(out_path))["records"] if out_path.exists() else []
    done = {(r["cheap"], r["polish"]) for r in recs}
    mp = make(a.task, a.polish, a.batch * len(rules))
    for cs in [c for c in a.cheap.split(",") if c]:
        if (cs, a.polish) in done:
            print(f"  {cs}: exists, skipped", flush=True)
            continue
        t0 = time.time()
        mc = make(a.task, cs, a.batch * len(rules))
        parts = []
        for s in range(0, len(inp), a.batch):
            x = torch.from_numpy(inp[s:s + a.batch]).to(DEV)
            y = torch.from_numpy(lab[s:s + a.batch]).to(DEV)
            parts.append(run_chunk(mc, mp, x, y, a.T, rules))
            print(f"    {cs}: {s + x.shape[0]}/{len(inp)} ({time.time() - t0:.0f}s)", flush=True)
        del mc
        torch.cuda.empty_cache()
        corr, nc, npl = (np.concatenate([p[i] for p in parts], 1) for i in range(3))
        rec = {"task": a.task, "ckpt": a.ckpt, "cheap": cs, "polish": a.polish, "T": a.T, "n": int(len(inp)), "rules": rules,
               "correct": corr.astype(int).tolist(), "cheap_steps": nc.tolist(), "polish_steps": npl.tolist(),
               "secs": time.time() - t0, "host": socket.gethostname(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        recs.append(rec)
        tmp = str(out_path) + ".tmp"
        json.dump({"records": recs}, open(tmp, "w"))
        os.replace(tmp, out_path)
        for i, r in enumerate(rules):
            print(f"  {cs} -> {a.polish} {r:16s} acc {100 * corr[i].mean():5.1f}  cheap {nc[i].mean():6.1f}  polish {npl[i].mean():5.1f}",
                  flush=True)


if __name__ == "__main__":
    main()
