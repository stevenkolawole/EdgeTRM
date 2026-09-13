#!/usr/bin/env python3
"""Ouro looped language models (ByteDance, 1.4B / 2.6B; 24 shared layers
applied `total_ut_steps` times) under the same weight quantizers, across
loop counts, on GSM8K: accuracy (greedy), teacher-forced gold NLL, and the
fidelity of the final hidden state on the prompt against the bf16 model.

Needs transformers<4.56:   ~/venvs/ouro/bin/python ouro_eval.py --model ouro14 \
    --quants bf16,w8c,w4t,w4a,w4g128,w3a --steps 1,2,3,4,6,8 --n 100 --out ~/breadth/results/ouro14_gsm8k.json
"""
import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HOME = Path.home()
os.environ.setdefault("HF_HOME", str(HOME / "hf_cache"))
DEV = "cuda"
MODELS = {"ouro14": "ByteDance/Ouro-1.4B", "ouro26": "ByteDance/Ouro-2.6B",
          "ouro14t": "ByteDance/Ouro-1.4B-Thinking", "ouro26t": "ByteDance/Ouro-2.6B-Thinking"}

import looped_lm_eval as L  # noqa: E402  (shared: load_gsm8k, extract_answer, parse_quant, quantize_linear_)


def build(model_id, steps):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    cfg.total_ut_steps = steps
    model = AutoModelForCausalLM.from_pretrained(model_id, config=cfg, torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
    if hasattr(model, "model") and hasattr(model.model, "total_ut_steps"):
        model.model.total_ut_steps = steps
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model.to(DEV).eval(), tok


def quantize_model_(model, q, scope):
    if q["bits"] is None:
        return 0
    mods = list(model.model.layers.modules())
    if scope == "all":
        mods += [model.lm_head]
    n = 0
    for m in mods:
        if isinstance(m, torch.nn.Linear):
            L.quantize_linear_(m, q["bits"], q["gran"])
            n += 1
    return n


def prompt_ids(tok, question):
    msgs = [{"role": "user", "content": question + "\nSolve this step by step. End with the line '#### <final numeric answer>'."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(text, return_tensors="pt", add_special_tokens=False).to(DEV)


@torch.no_grad()
def latents(model, ids):
    """Final hidden state (after the last loop, before the LM head), read
    through a hook on the final norm: transformers' output-capture wrapper
    cannot attach hidden states to this remote model's tuple outputs."""
    cap = []
    h = model.model.norm.register_forward_hook(lambda m, i, o: cap.append(o.detach()))
    model(ids, use_cache=False)
    h.remove()
    return cap[-1][0].float().cpu()


@torch.no_grad()
def gold_nll(model, tok, ids, solution):
    sol = tok.encode(solution + tok.eos_token, return_tensors="pt", add_special_tokens=False).to(DEV)
    full = torch.cat([ids, sol], dim=1)
    logits = model(full, return_dict=True, use_cache=False).logits[0, ids.shape[1] - 1:-1].float()
    return float(F.cross_entropy(logits, sol[0], reduction="mean"))


@torch.no_grad()
def evaluate(model, tok, data, max_new, ref_lat=None, gen=True):
    correct, lats, fids, gens, nlls = [], [], [], [], []
    for i, (q, gold, sol) in enumerate(data):
        ids = prompt_ids(tok, q)
        lat = latents(model, ids)
        lats.append(lat)
        if ref_lat is not None:
            fids.append(float(F.cosine_similarity(lat, ref_lat[i], dim=-1).mean()))
        nlls.append(gold_nll(model, tok, ids, sol))
        if not gen:
            continue
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        correct.append(int(L.extract_answer(text) == gold))
        gens.append(int(out.shape[1] - ids.shape[1]))
    return (correct, lats, (sum(fids) / len(fids) if fids else 1.0),
            (sum(gens) / len(gens) if gens else 0.0), sum(nlls) / len(nlls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro14", choices=list(MODELS))
    ap.add_argument("--quants", default="bf16,w8c,w4t,w4a,w4g128,w3a")
    ap.add_argument("--scope", default="core", choices=["core", "all"])
    ap.add_argument("--steps", default="1,2,3,4,6,8")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--no_gen", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_id = MODELS[a.model]
    data = L.load_gsm8k(a.n)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():
        try:
            records = json.load(open(out_path))["records"]
        except (json.JSONDecodeError, ValueError):
            pass
    done = {(r["steps"], r["quant"]["spec"], r["scope"]) for r in records}
    quants = [q for q in a.quants.split(",") if q]
    for steps in [int(s) for s in a.steps.split(",")]:
        model, tok = build(model_id, steps)
        t0 = time.time()
        c, lats, _, glen, nll = evaluate(model, tok, data, a.max_new, gen=not a.no_gen)
        acc = sum(c) / len(c) if c else None
        if (steps, "bf16", a.scope) not in done:
            records.append({"model": a.model, "steps": steps, "quant": L.parse_quant("bf16"), "scope": a.scope,
                            "n": len(data), "acc": acc, "nll": nll, "fidelity": 1.0, "gen_len": glen,
                            "correct": c, "secs": time.time() - t0})
            json.dump({"model": a.model, "records": records}, open(out_path, "w"))
        print(f"  steps={steps} bf16      acc {acc if acc is None else round(acc*100,1)}  nll {nll:.3f}  gen {glen:.0f} ({time.time()-t0:.0f}s)", flush=True)
        for spec in quants:
            if spec == "bf16" or (steps, spec, a.scope) in done:
                continue
            q = L.parse_quant(spec)
            t1 = time.time()
            nq = quantize_model_(model, q, a.scope)
            c, _, fid, glen, nll = evaluate(model, tok, data, a.max_new, ref_lat=lats, gen=not a.no_gen)
            acc = sum(c) / len(c) if c else None
            records.append({"model": a.model, "steps": steps, "quant": q, "scope": a.scope, "n": len(data),
                            "acc": acc, "nll": nll, "fidelity": fid, "gen_len": glen, "correct": c,
                            "n_quantized_linears": nq, "secs": time.time() - t1})
            json.dump({"model": a.model, "records": records}, open(out_path, "w"))
            print(f"  steps={steps} {spec:9s} acc {acc if acc is None else round(acc*100,1)}  nll {nll:.3f}  fid {fid:.4f}  gen {glen:.0f} ({time.time()-t1:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
            model, tok = build(model_id, steps)
        del model
        torch.cuda.empty_cache()
    print("wrote", out_path, len(records))


if __name__ == "__main__":
    main()
