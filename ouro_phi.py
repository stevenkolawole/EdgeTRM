#!/usr/bin/env python3
"""M5 looped-LM fidelity for the teacher-forced control (ouro_tf.py): phi = cos(final latent of the quantized model,
final latent of the bf16 model) at every gold-solution position, same inputs, both models in one process.

The ouro_tf.py cells are deterministic (RTN weight quantizers, teacher-forced gold solution, 4 loops read at loop 4), so a
second pass over the same (model, spec, problem) reproduces them. The check: token accuracy at 4 loops recomputed here must
equal the stored tok_acc of <model>_<spec>_tf.json (printed as "tok_acc here / stored").
Latent = output of model.model.norm on its last call (the 4th loop), looped_decode.Ouro.latents.
Per problem: phi = mean over solution positions of the cosine; per cell: mean over problems.

    python ouro_phi.py --model ouro14 --quants w8c,w4g32,w4c --n 1319 --tfdir .../results/m5/ouro_tf --out .../ouro14_phi.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import looped_decode as LD  # noqa: E402
import looped_lm_eval as L  # noqa: E402


def build(fam, spec):
    model, tok = fam.build(4)
    LD.quantize_(fam, model, L.parse_quant(spec), "core")
    model.early_exit_step = 3
    model.early_exit_threshold = 1.0
    return model, tok


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro14", choices=["ouro14", "ouro26"])
    ap.add_argument("--quants", default="w8c,w4g32,w4c")
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--tfdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    print("device", torch.cuda.get_device_name(0), flush=True)
    fam = LD.family(a.model)
    data = L.load_gsm8k(a.n)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {"model": a.model, "n": len(data), "cells": {}}
    ref, tok = build(fam, "bf16")
    specs = ["bf16"] + [s for s in a.quants.split(",") if s]
    for spec in specs:
        if spec in res["cells"]:
            print(f"{spec}: done", flush=True)
            continue
        t0 = time.time()
        m = ref if spec == "bf16" else build(fam, spec)[0]
        phis, oks, ntok = [], 0, 0
        for i, (q, gold, sol) in enumerate(data):
            ids = tok.encode(fam.prompt_text(tok, q), return_tensors="pt", add_special_tokens=False).to(LD.DEV)
            gtxt = fam.gold_text(tok, sol)
            k = gtxt.index("####") + 4
            tail = tok.eos_token if fam.style == "chat" else "\n\n"
            ans = gtxt[k:len(gtxt) - len(tail)]
            parts = [tok.encode(x, add_special_tokens=False) for x in (gtxt[:k], ans, tail)]
            sol_ids = torch.tensor([parts[0] + parts[1] + parts[2]], device=LD.DEV)
            full = torch.cat([ids, sol_ids], dim=1)
            sl = slice(ids.shape[1] - 1, full.shape[1] - 1)
            torch.manual_seed(1234 + i)
            zr = fam.latents(ref, full)[sl].float()
            zq = zr if spec == "bf16" else fam.latents(m, full)[sl].float()
            phis.append(float(F.cosine_similarity(zq, zr, dim=-1).mean()))
            lg = fam.logits(m, full)[sl]
            oks += float((lg.argmax(-1) == sol_ids[0]).float().mean())       # per-problem share, as ouro_tf.py
            ntok += 1
        stored = None
        tf = Path(a.tfdir) / f"{a.model}_{spec}_tf.json"
        if tf.exists():
            bits = json.load(open(tf))["settings"].get("L4", {}).get("tok_bits")
            if bits:
                acc = []
                for h in bits[:len(data)]:
                    hx, ln = h.split(":")
                    b = np.unpackbits(np.frombuffer(bytes.fromhex(hx), np.uint8))[:int(ln)]
                    acc.append(b.mean())
                stored = float(np.mean(acc))                 # the stored per-problem token accuracy, same problems
        res["cells"][spec] = {"phi": phis, "phi_mean": float(np.mean(phis)), "tok_acc_here": oks / ntok, "tok_acc_stored": stored,
                              "secs": time.time() - t0}
        json.dump(res, open(a.out + ".tmp", "w"))
        os.replace(a.out + ".tmp", a.out)
        print(f"{a.model} {spec}: phi {np.mean(phis):.4f}  tok_acc here {100 * oks / ntok:.2f} / stored "
              f"{100 * stored if stored is not None else float('nan'):.2f} ({time.time() - t0:.0f}s)", flush=True)
        if spec != "bf16":
            del m
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
