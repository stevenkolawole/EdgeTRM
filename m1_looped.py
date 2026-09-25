#!/usr/bin/env python3
"""M1 on looped LMs: compressed loops, then the last loop(s) in bf16 (precision switching per loop).

Thin wrapper around X4's `looped_decode.py` (unchanged).  Extra quant specs:
  <q>~P<ij...>     loops i, j, ... (0-based current_ut, one digit each) use the bf16 weights for every token;
                   e.g. w4c~P3 on Ouro with 4 loops = loops 0-2 w4c, the last loop bf16
  <q>~D<ij...>     the same, but only on decode steps (one new token); the prompt (prefill) runs fully compressed
  <q>~P3@w8c       polish loops use per-channel INT8 weights (of the bf16 original) instead of bf16
Every Linear in the looped core holds both weights; a pre-hook on each decoder layer reads `current_ut`
(Ouro passes it to every layer) and whether the call is a prefill (sequence length > 1), and the
Linear's pre-hook points its weight at the bf16 or the compressed tensor.  Memory: both copies.

    PYTHONPATH=$HOME/pylibs/tf4571 python m1_looped.py --model ouro14 --steps 4 \
        --quants bf16,w4c,w4c~P3,w4c~D3 --decodes greedy --n 200 --outdir ~/breadth_results/lm/m1
"""
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import looped_decode as D  # noqa: E402
import looped_lm_eval as L  # noqa: E402

STATE = {"ut": 0, "prefill": True}
_orig_parse = L.parse_quant
_orig_quantize = D.quantize_


def parse_quant(spec):
    m = re.fullmatch(r"(.+)~([PD])(\d+)(?:@(\w+))?", spec)   # each digit is one loop index (~P23 = loops 2 and 3)
    if not m:
        return _orig_parse(spec)
    q = dict(_orig_parse(m.group(1)))
    q.update(spec=spec, polish_loops=[int(i) for i in m.group(3)], polish_decode_only=m.group(2) == "D",
             polish_q=_orig_parse(m.group(4)) if m.group(4) else None)   # @w8c: polish weights INT8 instead of bf16
    return q


def quantize_(fam, model, q, scope):
    if "polish_loops" not in q:
        return _orig_quantize(fam, model, q, scope)
    loops, dec_only = set(q["polish_loops"]), q["polish_decode_only"]
    layers = model.model.layers

    def layer_pre(mod, args, kwargs):
        STATE["ut"] = int(kwargs.get("current_ut", 0))
        hs = args[0] if args else kwargs["hidden_states"]
        STATE["prefill"] = hs.shape[1] > 1
        return None
    for lay in layers:
        lay.register_forward_pre_hook(layer_pre, with_kwargs=True)
    n = 0
    for m in fam.core_modules(model, scope):
        if not isinstance(m, torch.nn.Linear):
            continue
        W0 = m.weight.data.clone()                         # bf16 original
        if q.get("polish_q"):                              # @w8c: polish weights = INT8 of the original
            L.quantize_linear_(m, q["polish_q"]["bits"], q["polish_q"]["gran"])
            Wb = m.weight.data.clone()
            m.weight.data = W0.clone()
        else:
            Wb = W0
        L.quantize_linear_(m, q["bits"], q["gran"])        # cheap loops: the compressed original
        Wq = m.weight.data

        def lin_pre(mod, args, Wb=Wb, Wq=Wq):
            use_b = STATE["ut"] in loops and not (dec_only and STATE["prefill"])
            mod.weight.data = Wb if use_b else Wq
            return None
        m.register_forward_pre_hook(lin_pre)
        n += 1
    return n


L.parse_quant = parse_quant
D.quantize_ = quantize_

if __name__ == "__main__":
    D.main()
