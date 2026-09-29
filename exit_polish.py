#!/usr/bin/env python3
"""M5 x M1: exit with the compressed model's halting head, then polish with a more precise model.

For one (cheap spec, polish spec) pair: the cheap (compressed) model runs K default-H supervision
steps from the model's own initial carry; after EVERY cheap step t = 1..K the carry is cloned and the
polish model runs J steps from it. Per puzzle we store
  c_cheap[t], q_cheap[t]         exact / halting logit of the cheap trajectory after step t
  c_pol[t, j], q_pol[t, j]       exact / polish-model halting logit after j polish steps started at t
so every rule "cheap until exit rule R fires at t, then j polish steps (or until the polish head
halts)" is scored offline, together with M1's fixed schedules c(t).P(j).
Default H per model (TRM 3, HRM 2); first n test rows; noise specs seeded as detector_sweep / exit_sweep
(torch.manual_seed(seed) right before quantize_weights_).

Dither (E_COMBINED, Sep 29): --dither <master> re-draws the cheap weights by stochastic rounding onto the cheap spec's
grid before EVERY cheap step, from the master's weights (w8c = the resident INT8 copy the finishing model is; fp32 = the
full-precision weights). This is M2's arm 6 (m2_bestofk --wsample dither --master w8c) with the same draw seeds
(seed k, step t: m2_bestofk's step_hook formula), so the cheap trajectory of seed k equals arm 6's rollout k; the
finishing model is untouched (round-to-nearest). --seeds K writes one file per re-draw sequence,
<cheap>__<polish>__dither_m<master>_s<k>.npz, same layout as the deterministic record.

    python exit_polish.py --task sudoku_mlp --cheap w3g32,w4t,noise0.15@s0 --polish fp32,w8c --K 16 --J 4 \
        --outdir ~/breadth_results/m5/polish/sudoku_mlp
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
from breadth_sweep import DEV, TRM_TASKS, HRM_TASKS, _linears, build, load_qa, parse_quant, quantize_weights_  # noqa: E402
from detector_sweep import explicit_carry, split_spec  # noqa: E402
from exit_sweep import DEFAULT_H  # noqa: E402


def make(task, spec, K, batch, dither=""):
    base, seed = split_spec(spec)
    q = parse_quant("fp32" if base.startswith("gq") else base)          # gq<fmt>: GPTQ weights (m5_gptq.py)
    assert q["act"] is None and q.get("anoise") is None, "weight specs only"
    m, _ = build(task, None, 10 ** 6, batch)            # default H; never reaches halt_max_steps
    if dither:                                          # master weights (RTN of the master spec); the grid is re-drawn per step
        assert q["bits"] is not None and q["gran"] not in (None, "noise"), "dither needs a quantizer cheap spec"
        m = m.to(DEV).eval()
        if dither != "fp32":
            qm = parse_quant(dither)
            quantize_weights_(m, qm["bits"], qm["gran"])
        master = [mod.weight.data.clone() for mod in _linears(m)]
        return m, q, (master, 7919 * (seed or 0) + 17)
    torch.manual_seed(seed if seed is not None else 0)
    if q["gran"] is not None:
        quantize_weights_(m, q["bits"], q["gran"], q.get("sigma"))
    if base.startswith("gq"):
        import m5_gptq
        m5_gptq.load_if_gptq(m, task, base)
    return m.to(DEV).eval(), q, None


def dither_hook(mc, q, master, dseed, k):
    """before cheap step t of re-draw sequence k: SR draw of the master onto the cheap grid, m2_bestofk's seed formula."""
    from m2_bestofk import sr_quantize_
    ms = master

    def hook(t):
        sr_quantize_(mc, ms, q["bits"], q["gran"], torch.Generator(device=DEV).manual_seed(dseed * 7907 + 100003 * (k + 1) + t))
    return hook


def clone(carry):
    ic = carry.inner_carry
    return type(carry)(inner_carry=type(ic)(z_H=ic.z_H.clone(), z_L=ic.z_L.clone()), steps=carry.steps.clone(),
                       halted=torch.zeros_like(carry.halted), current_data=carry.current_data)


@torch.no_grad()
def run(mc, mp, inp, lab, batch, K, J, hook=None):
    n = inp.shape[0]
    out = {"c_cheap": np.zeros((n, K), bool), "q_cheap": np.zeros((n, K), np.float16), "qc_cheap": np.zeros((n, K), np.float16),
           "c_pol": np.zeros((n, K, J), bool), "q_pol": np.zeros((n, K, J), np.float16), "qc_pol": np.zeros((n, K, J), np.float16)}
    for s in range(0, n, batch):
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        B = x.shape[0]
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
        carry = explicit_carry(mc, bt)
        for t in range(K):
            if hook is not None:
                hook(t)
            carry, o = mc(carry, bt)
            carry.halted.zero_()
            out["c_cheap"][s:s + B, t] = (o["logits"].argmax(-1) == y).all(1).cpu().numpy()
            out["q_cheap"][s:s + B, t] = o["q_halt_logits"].float().cpu().numpy()
            out["qc_cheap"][s:s + B, t] = o["q_continue_logits"].float().cpu().numpy()
            pc = clone(carry)
            for j in range(J):
                pc, po = mp(pc, bt)
                pc.halted.zero_()
                out["c_pol"][s:s + B, t, j] = (po["logits"].argmax(-1) == y).all(1).cpu().numpy()
                out["q_pol"][s:s + B, t, j] = po["q_halt_logits"].float().cpu().numpy()
                out["qc_pol"][s:s + B, t, j] = po["q_continue_logits"].float().cpu().numpy()
        print(f"    {s + B}/{n}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TRM_TASKS) + list(HRM_TASKS))
    ap.add_argument("--cheap", required=True)
    ap.add_argument("--polish", default="fp32")
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--J", type=int, default=4)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--offset", type=int, default=0, help="first test row (held-out rows: e.g. 1000)")
    ap.add_argument("--ckpt", default="", help="TRM: a trained variant's step_N file; cheap and polish models both use it")
    ap.add_argument("--dither", default="", help="re-draw the cheap weights (SR onto the cheap grid) before every cheap step from "
                                                 "this master: w8c = the resident INT8 copy, fp32 = full precision (M2 arm 6 / dither)")
    ap.add_argument("--seeds", default="1", help="dither: re-draw sequences, 'K' = seeds 0..K-1 or 'a:b' = seeds a..b-1 "
                                                 "(one file per seed, suffix _s<k>; one seed per backlog line avoids duplicate work)")
    ap.add_argument("--allow_cpu", action="store_true")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA"
    if a.ckpt:
        import breadth_sweep as bsw
        bsw.CKPT_OVERRIDE = a.ckpt
    t = TRM_TASKS.get(a.task) or HRM_TASKS[a.task]
    batch = a.batch or t["batch"]
    inp, lab = load_qa(t["data"], a.n + a.offset)
    inp, lab = inp[a.offset:], lab[a.offset:]
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    seeds = (list(range(*map(int, a.seeds.split(":")))) if ":" in a.seeds else list(range(int(a.seeds)))) if a.dither else [None]
    for pspec in [p for p in a.polish.split(",") if p]:
        mp, _, _ = make(a.task, pspec, a.K, batch)
        for cspec in [c for c in a.cheap.split(",") if c]:
            for k in seeds:
                suf = f"__dither_m{a.dither}_s{k}" if a.dither else ""
                path = od / f"{cspec}__{pspec}{suf}.npz"
                if path.exists():
                    print(f"  {cspec} -> {pspec}{suf}: exists, skipped", flush=True)
                    continue
                t0 = time.time()
                mc, q, dm = make(a.task, cspec, a.K, batch, a.dither)
                hook = dither_hook(mc, q, dm[0], dm[1], k) if a.dither else None
                r = run(mc, mp, inp, lab, batch, a.K, a.J, hook)
                del mc
                torch.cuda.empty_cache()
                info = {"task": a.task, "cheap": cspec, "polish": pspec, "H": DEFAULT_H[a.task], "K": a.K, "J": a.J,
                        "n": int(len(inp)), "offset": a.offset, "ckpt": a.ckpt, "dither": a.dither, "seed_k": k,
                        "secs": time.time() - t0, "host": socket.gethostname(),
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
                tmp = f"{path}.tmp{os.getpid()}.npz"           # per-process name: two lines on one record cannot collide
                np.savez_compressed(tmp, info=json.dumps(info), **r)
                if path.exists():                                # another process finished the same record meanwhile
                    os.remove(tmp)
                    print(f"  {path.name}: written by another process meanwhile, this copy dropped", flush=True)
                else:
                    os.replace(tmp, path)
                print(f"  {cspec:14s} -> {pspec:6s}{suf} cheap@K {100 * r['c_cheap'][:, -1].mean():5.1f}  "
                      f"cK.P1 {100 * r['c_pol'][:, -1, 0].mean():5.1f}  cK.P{a.J} {100 * r['c_pol'][:, -1, -1].mean():5.1f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        del mp
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
