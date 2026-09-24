#!/usr/bin/env python3
"""E0 of the device plan (reports/edgetrm_lit/06_edge_hardware.md section 5): the quantizers the
chips can actually run, in the accuracy harness, plus the precision at which the carry (z_H, z_L)
is stored between supervision steps (the host-loop deployment pattern: the carry leaves the NPU
after every step).

Spec grammar:  <weights>[+a8|+a16][+cs_<carry>]
  weights  fp32 | w<bits><gran>, gran as in breadth_sweep (t, c, a, g<k>) plus
           l<k>  Qualcomm LPBQ-style block INT4: per-output-channel float scale s_c = max|W_c|/qmax,
                 per-block integer multiplier k_b = ceil(16 * max|W_block| / (qmax * s_c)) in [1, 16],
                 effective block scale s_c * k_b / 16, symmetric, blocks of k along the input dim
                 (k must divide in_features; QNN block-expansion rule)
  +a8/+a16 static per-tensor symmetric activation fake-quant at every linear input, scales frozen on
           the first chunk (identical to breadth_sweep's +a8)
  +cs_<m>  carry storage after every supervision step:
           fp16 | bf16                     cast and back
           i8t | i4t                       static per-tensor symmetric (scale from calibration puzzles)
           i8c | i4c                       static per-hidden-channel symmetric
           i8td | i4td                     dynamic per-tensor per sample (absmax of that carry)
           i8tok | i4tok                   dynamic per token (absmax of each token's 512-vector)
           Static scales come from 64 calibration puzzles drawn after the evaluated n (test indices
           n..n+63; Maze-Hard has only 1000 test puzzles, so there the first 64 evaluated ones, the
           +a8 convention), run through the same weight-quantized model for nsup steps.

Records match breadth_sweep's (pexact, cell, fidelity_H/L vs fp32 at the same H and nsup, per-puzzle
`correct`), with quant.carry set. Resumable: finished (H, nsup, spec) cells are skipped.

    python chipq_sweep.py --task sudoku_mlp --H 1,3 --nsup 8,16 --quants fp32,w4l64,w4c+a16 \
        --out ~/breadth_results/chipq_sudoku_mlp.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import breadth_sweep as B


def parse(spec):
    parts = spec.split("+")
    w = parts[0]
    act, carry = None, None
    for p in parts[1:]:
        if p in ("a8", "a16"):
            act = int(p[1:])
        elif p.startswith("cs_"):
            carry = p[3:]
        else:
            raise ValueError(spec)
    if w == "fp32":
        bits, gran = None, None
    else:
        assert w.startswith("w"), spec
        i = 1
        while w[i].isdigit():
            i += 1
        bits, gran = int(w[1:i]), w[i:]
        assert gran in ("t", "c", "a") or gran[0] in "gl", spec
    return {"spec": spec, "bits": bits, "gran": gran, "act": act, "carry": carry}


@torch.no_grad()
def quantize_weights(m, bits, gran):
    if gran is None:
        return
    if not gran.startswith("l"):
        return B.quantize_weights_(m, bits, gran)
    g = int(gran[1:])
    qmax = 2 ** (bits - 1) - 1
    for mod in B._linears(m):
        W = mod.weight.data
        out_f, in_f = W.shape
        if in_f % g:
            # QNN requires the block to divide the input dim; such layers stay per-channel.
            s = W.abs().amax(dim=1, keepdim=True) / qmax
            mod.weight.data = torch.round(W / s.clamp_min(1e-12)).clamp(-qmax, qmax) * s.clamp_min(1e-12)
            continue
        sc = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-12)          # [out, 1]
        Wb = W.view(out_f, in_f // g, g)
        sb = Wb.abs().amax(dim=2, keepdim=True) / qmax                             # [out, nb, 1]
        k = torch.ceil(16 * sb / sc.unsqueeze(-1)).clamp(1, 16)
        s = sc.unsqueeze(-1) * k / 16
        mod.weight.data = (torch.round(Wb / s).clamp(-qmax, qmax) * s).view(out_f, in_f)


class CarryQ:
    def __init__(self, mode):
        self.mode = mode
        self.bits = 8 if "8" in mode else 4 if "4" in mode else None
        self.qmax = 2 ** (self.bits - 1) - 1 if self.bits else None
        self.static = mode in ("i8t", "i4t", "i8c", "i4c")
        self.amax = {}      # name -> running absmax (tensor or per-channel)
        self.frozen = False

    def observe(self, name, z):
        if self.mode.endswith("c"):
            a = z.detach().abs().amax(dim=(0, 1))
        else:
            a = z.detach().abs().amax()
        self.amax[name] = torch.maximum(self.amax[name], a) if name in self.amax else a

    def __call__(self, name, z):
        m = self.mode
        if m == "fp16":
            return z.half().float()
        if m == "bf16":
            return z.bfloat16().float()
        q = self.qmax
        if self.static:
            if not self.frozen:
                self.observe(name, z)
                return z
            s = (self.amax[name] / q).clamp_min(1e-12)
        elif m.endswith("td"):
            s = (z.abs().amax(dim=(1, 2), keepdim=True) / q).clamp_min(1e-12)
        elif m.endswith("tok"):
            s = (z.abs().amax(dim=2, keepdim=True) / q).clamp_min(1e-12)
        else:
            raise ValueError(m)
        return torch.round(z / s).clamp(-q, q) * s


def attach_carry(m, cq):
    """Quantize the carry the step hands back (logits were computed on-chip from the
    unquantized z_H, as on a device whose carry leaves the NPU after the step)."""
    orig = m.forward

    def fwd(carry, batch):
        new_carry, out = orig(carry, batch)
        ic = new_carry.inner_carry
        new_carry.inner_carry = type(ic)(z_H=cq("z_H", ic.z_H), z_L=cq("z_L", ic.z_L))
        return new_carry, out
    m.forward = fwd
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(B.TRM_TASKS) + list(B.HRM_TASKS))
    ap.add_argument("--quants", required=True)
    ap.add_argument("--H", default="")
    ap.add_argument("--nsup", default="16")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_cal", type=int, default=64)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    spec_t = B.TRM_TASKS.get(a.task) or B.HRM_TASKS[a.task]
    batch = a.batch or spec_t["batch"]
    inp_all, lab_all = B.load_qa(spec_t["data"], a.n + a.n_cal)
    inp, lab = inp_all[:a.n], lab_all[:a.n]
    cal_inp, cal_lab = inp_all[a.n:], lab_all[a.n:]
    cal_src = "heldout"
    if len(cal_inp) < a.n_cal:
        # Maze-Hard has exactly 1000 test puzzles: calibrate on the first n_cal evaluated ones,
        # the convention breadth_sweep's +a8 already uses (it calibrates on the first chunk).
        cal_inp, cal_lab, cal_src = inp[:a.n_cal], lab[:a.n_cal], "test_head"
    Hs = [None] if not a.H else [int(h) for h in a.H.split(",")]
    nsups = [int(x) for x in a.nsup.split(",")]
    quants = [q for q in a.quants.split(",") if q]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out.exists():
        records = json.load(open(out)).get("records", [])
    done = {(r["H_cycles"], r["nsup"], r["quant"]["spec"]) for r in records}
    print(f"{a.task}: n={len(inp)} cal={len(cal_inp)} H={Hs} nsup={nsups} {len(quants)} specs, "
          f"{len(records)} done", flush=True)

    def save():
        tmp = out.with_suffix(".tmp")
        json.dump({"task": a.task, "n": int(len(inp)), "records": records}, open(tmp, "w"))
        tmp.replace(out)

    for H in Hs:
        for nsup in nsups:
            m, meta = B.build(a.task, H, nsup, batch)
            Hc = meta["H_cycles"]
            ref_c, ref_cell, ref_zH, ref_zL = B.run(m, inp, lab, batch, nsup)
            del m
            if (Hc, nsup, "fp32") not in done:
                records.append({"task": a.task, "H_cycles": Hc, "L_cycles": meta["L_cycles"], "nsup": nsup,
                                "quant": parse("fp32"), "n": int(len(inp)), "pexact": float(ref_c.mean()),
                                "cell": ref_cell, "fidelity_H": 1.0, "fidelity_L": 1.0,
                                "correct": ref_c.astype(int).tolist(), "harness": "chipq_sweep"})
                save()
                print(f"  H={Hc} nsup={nsup} fp32  pexact {ref_c.mean()*100:6.2f} cell {ref_cell*100:6.2f}", flush=True)
            for spec in quants:
                if spec == "fp32" or (Hc, nsup, spec) in done:
                    continue
                q = parse(spec)
                t1 = time.time()
                mq, _ = B.build(a.task, H, nsup, batch)
                quantize_weights(mq, q["bits"], q["gran"])
                cq = None
                if q["carry"]:
                    cq = CarryQ(q["carry"])
                    attach_carry(mq, cq)
                    if cq.static:   # calibrate the carry scales on held-out puzzles, then freeze
                        B.run(mq, cal_inp, cal_lab, batch, nsup, act_quant=q["act"])
                        cq.frozen = True
                c, cell, zH, zL = B.run(mq, inp, lab, batch, nsup, act_quant=q["act"])
                del mq
                fH, fL = B.fidelity(zH, ref_zH), B.fidelity(zL, ref_zL)
                records.append({"task": a.task, "H_cycles": Hc, "L_cycles": meta["L_cycles"], "nsup": nsup,
                                "quant": q, "n": int(len(inp)), "pexact": float(c.mean()), "cell": cell,
                                "fidelity_H": fH, "fidelity_L": fL, "correct": c.astype(int).tolist(),
                                "secs": time.time() - t1, "harness": "chipq_sweep",
                                "carry_cal": cal_src if (q["carry"] and cq.static) else None})
                save()
                print(f"  H={Hc} nsup={nsup} {spec:18s} pexact {c.mean()*100:6.2f} cell {cell*100:6.2f} "
                      f"fidH {fH:.4f} fidL {fL:.4f} ({time.time()-t1:.0f}s)", flush=True)
    save()
    print("wrote", out, len(records), "records", flush=True)


if __name__ == "__main__":
    main()
