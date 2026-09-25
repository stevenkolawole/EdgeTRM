#!/usr/bin/env python3
"""M5 on looped LMs: Ouro's learned per-token exit gate under weight compression (GSM8K, greedy).

Ouro (ByteDance/Ouro-1.4B, -2.6B) applies its whole layer stack total_ut_steps = 4 times and has a trained
exit gate: after loop i a per-token hazard lambda_i = sigmoid(gate(h_i)), exit pdf p_i = lambda_i prod_{j<i}(1-lambda_j),
and at inference the model can read the token from the first loop whose cumulative exit probability reaches
a threshold tau (modeling_ouro.OuroForCausalLM.forward, `exit_threshold`; config default 1.0 = the last loop).
Question (M5 "past-edge recovery"): under compression, does the learned exit keep (or recover) accuracy while
using fewer loops, as the TRM halting head does?

Per (model, weight spec, tau): greedy generation on GSM8K test[:n] with looped_decode.gen_pass (X4's harness:
batched, left padding, 512 new tokens, #### extraction), plus the loop at which each decoded token exits
(forward hook on the inner model, last position of every forward call; rows that already stopped still count,
so the mean is over all rows of each batch step). Note: the HF implementation still computes all loops (the
KV cache of every loop is filled); the exit changes which loop's state is read, so "loops used" is the compute
an early-exit runtime could skip, not a measured saving.
Weight specs: X4's (bf16, w8c, w4g32, w4c, w3a, ...; scope 'core' = every Linear in the shared stack; the
exit gate and LM head stay bf16).

    PYTHONPATH=~/pylibs/tf4554 python ouro_exit.py --model ouro14 --quants bf16,w4g32,w4c --taus 1.0,0.7,0.5 \
        --n 200 --outdir ~/breadth_results/m5/ouro
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import looped_decode as LD  # noqa: E402
import looped_lm_eval as L  # noqa: E402


class ExitRecorder:
    def __init__(self, model):
        self.model = model
        self.steps = []
        self.h = model.model.register_forward_hook(self.hook)

    def hook(self, mod, inp, out):
        gates = out[2]
        if not gates:
            return
        rem = None
        pdf = []
        for i, g in enumerate(gates):
            lam = torch.sigmoid(g[:, -1, 0].float())
            if rem is None:
                rem = torch.ones_like(lam)
            p = lam * rem if i < len(gates) - 1 else rem
            rem = rem * (1 - lam)
            pdf.append(p)
        cdf = torch.cumsum(torch.stack(pdf, 1), 1)
        tau = self.model.early_exit_threshold
        tau = 1.0 if tau is None else float(tau)
        m = cdf >= tau
        st = torch.where(m.any(1), m.float().argmax(1), torch.full_like(m[:, 0], len(gates) - 1, dtype=torch.float))
        self.steps.append((st + 1).float().mean().item())

    def close(self):
        self.h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro14", choices=["ouro14", "ouro26"])
    ap.add_argument("--quants", default="bf16,w4g32,w4c")
    ap.add_argument("--taus", default="1.0,0.7,0.5")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    fam = LD.family(a.model)
    data = L.load_gsm8k(a.n)
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    summ_path = od / f"{a.model}_summary.json"
    summ = json.load(open(summ_path)) if summ_path.exists() else {}
    log = lambda s: print(time.strftime("%H:%M:%S ") + s, flush=True)  # noqa: E731
    for spec in [q for q in a.quants.split(",") if q]:
        model = tok = None
        for tau in [float(t) for t in a.taus.split(",") if t]:
            key = f"{spec}|tau{tau}"
            if key in summ:
                log(f"{key}: done")
                continue
            if model is None:
                model, tok = fam.build(a.steps)
                nq = LD.quantize_(fam, model, L.parse_quant(spec), "core")
                log(f"{a.model} {spec}: {nq} linears quantized")
            model.early_exit_threshold = tau
            rec = ExitRecorder(model)
            t0 = time.time()
            recs, secs = LD.gen_pass(fam, model, tok, data, "greedy", a.max_new, a.batch,
                                     od / f"{a.model}_{spec}_tau{tau}.jsonl", log)
            rec.close()
            s = LD.summarize(recs)
            s.update({"mean_exit_loop": (sum(rec.steps) / len(rec.steps)) if rec.steps else None, "tau": tau,
                      "spec": spec, "loops": a.steps, "secs": time.time() - t0})
            summ[key] = s
            json.dump(summ, open(str(summ_path) + ".tmp", "w"), indent=1)
            os.replace(str(summ_path) + ".tmp", summ_path)
            log(f"{a.model} {spec} tau {tau}: acc {100 * s['acc']:.1f}  mean exit loop {s['mean_exit_loop']:.2f}  "
                f"degenerate {100 * s['degenerate']:.1f}%")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
