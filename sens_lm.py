#!/usr/bin/env python3
"""X2 extended fit, looped LMs: one-loop map perturbation delta of a weight format at the bf16 state.

For a looped LM run for L loops (Ouro: the whole layer stack is the loop; Huginn / retrofitted recurrence:
the core block after its adapter), the recursion map F is one loop.  With h_L the bf16 state entering loop L
(0-based: after L bf16 loops) and b = F(h_L) the bf16 output of that loop,

    delta = || F~(h_L) - F(h_L) || / || F(h_L) ||

where F~ is the same loop with the format's weights on the looped Linears (X4's 'core' scope).  One model
holds both weight copies and a pre-hook swaps them per loop (the m1_looped.py mechanism), so loops 0..L-1
run in bf16 in both passes and h_L is identical.  The model is run with L+1 loops; the state is read at the
output of the loop's last layer.  Prompts: GSM8K test[:n] in X4's format (per-token state, norm over the
whole prompt sequence).  Also records the same for weight noise (sigma x std(W) x eps on the looped
Linears), for the sensitivity S.

    PYTHONPATH=$HOME/pylibs/tf4571 python sens_lm.py --model ouro14 --loops 1,2,3,4,6,8 \
        --specs w8c,w4g32,w4c,noise0.05:3 --n 16 --out ~/breadth_results/x2/sens/lm_ouro14.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "lm"))
import looped_decode as D  # noqa: E402
import looped_lm_eval as L  # noqa: E402

STATE = {"ut": -1, "qloop": None, "cap": None, "count": 0}


def parse_specs(s):
    out = []
    for tok in [t for t in s.split(",") if t]:
        if ":" in tok:
            sp, k = tok.split(":")
            out += [(sp, i) for i in range(int(k))]
        else:
            out.append((tok, 0))
    return out


def install(fam, model):
    """Hooks: loop counter, capture of the last layer's output at loop STATE['qloop']."""
    if isinstance(fam, D.Ouro):
        layers = model.model.layers
        first, last = layers[0], layers[-1]

        def pre(mod, args, kwargs):
            STATE["ut"] = int(kwargs.get("current_ut", 0))
            return None
        first.register_forward_pre_hook(pre, with_kwargs=True)
    else:
        blocks = model.transformer.core_block
        first, last = blocks[0], blocks[-1]

        def pre(mod, args, kwargs):                # Huginn: one call of the first core layer per loop
            STATE["ut"] = STATE["count"]
            STATE["count"] += 1
            return None
        first.register_forward_pre_hook(pre, with_kwargs=True)

    def post(mod, args, kwargs, out):
        if STATE["ut"] == STATE["qloop"]:
            o = out[0] if isinstance(out, (tuple, list)) else out
            STATE["cap"] = o.detach().float().clone()
        return None
    last.register_forward_hook(post, with_kwargs=True)


def prepare_swap(fam, model):
    lins = [m for m in fam.core_modules(model, "core") if isinstance(m, torch.nn.Linear)]
    for m in lins:
        m._w0 = m.weight.data.clone()
        m._wq = None

        def lin_pre(mod, args):
            use_q = mod._wq is not None and STATE["ut"] == STATE["qloop"]
            mod.weight.data = mod._wq if use_q else mod._w0
            return None
        m.register_forward_pre_hook(lin_pre)
    return lins


@torch.no_grad()
def set_format(lins, spec, draw):
    if spec in ("bf16", None):
        for m in lins:
            m._wq = None
        return {}
    if spec.startswith("noise"):
        sig = float(spec[5:])
        g = torch.Generator(device=lins[0]._w0.device).manual_seed(20_000 + 1000 * draw + int(round(sig * 1000)))
        for m in lins:
            W = m._w0.float()
            m._wq = (W + sig * W.std() * torch.randn(W.shape, generator=g, device=W.device)).to(m._w0.dtype)
        return {"sigma": sig}
    q = L.parse_quant(spec)
    for m in lins:
        m.weight.data = m._w0.clone()
        L.quantize_linear_(m, q["bits"], q["gran"])
        m._wq = m.weight.data.clone()
        m.weight.data = m._w0
    return {"bits": q["bits"], "gran": q["gran"]}


@torch.no_grad()
def run_state(fam, model, ids, loops, qloop, seed):
    STATE.update(qloop=qloop, cap=None, count=0, ut=-1)
    torch.manual_seed(seed)                        # Huginn draws its initial latent state at random
    fam.latents(model, ids)
    return STATE["cap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--loops", required=True, help="comma list L: delta of loop L (0-based) at the state after L loops")
    ap.add_argument("--specs", default="w8c,w4g32,w4c,noise0.05:3")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    torch.backends.cuda.matmul.allow_tf32 = False
    fam = D.family(a.model)
    loops = [int(x) for x in a.loops.split(",")]
    out = Path(a.out)
    blob = {"model": a.model, "n": a.n, "records": []}
    if out.exists():
        try:
            blob = json.load(open(out))
        except Exception:
            pass
    done = {(r["loops"], r["spec"], r["draw"]) for r in blob["records"]}
    data = L.load_gsm8k(a.n)
    t0 = time.time()
    model, tok = fam.build(max(loops) + 1)
    install(fam, model)
    lins = prepare_swap(fam, model)
    ids = [tok.encode(fam.prompt_text(tok, q), return_tensors="pt", add_special_tokens=False).to(D.DEV)
           for q, _, _ in data]
    print(f"{a.model}: built with {max(loops)+1} loops, {len(lins)} looped Linears, n={len(ids)} ({time.time()-t0:.0f}s)",
          flush=True)
    for Lp in loops:
        fam.steps = Lp + 1
        if isinstance(fam, D.Ouro):
            model.config.total_ut_steps = Lp + 1
            if hasattr(model.model, "total_ut_steps"):
                model.model.total_ut_steps = Lp + 1
        set_format(lins, "bf16", 0)
        ref = [run_state(fam, model, x, Lp + 1, Lp, 1234 + i) for i, x in enumerate(ids)]
        assert all(r is not None for r in ref), f"no state captured at loop {Lp}"
        for spec, draw in parse_specs(a.specs):
            if (Lp, spec, draw) in done:
                continue
            t1 = time.time()
            info = set_format(lins, spec, draw)
            ds = []
            for i, x in enumerate(ids):
                b = run_state(fam, model, x, Lp + 1, Lp, 1234 + i)
                ds.append(float((b - ref[i]).norm() / ref[i].norm()))
            set_format(lins, "bf16", 0)
            blob["records"].append({"loops": Lp, "spec": spec, "draw": draw, **info, "delta_med": float(np.median(ds)),
                                    "delta": [round(d, 6) for d in ds], "secs": time.time() - t1})
            print(f"  loops {Lp} {spec}#{draw}: delta med {np.median(ds):.4f} ({time.time()-t1:.0f}s)", flush=True)
            json.dump(blob, open(out, "w"))
    json.dump(blob, open(out, "w"))
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
