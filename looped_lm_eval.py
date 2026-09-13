#!/usr/bin/env python3
"""Looped language models under the same quantizers as the puzzle solvers:
Huginn-0125 (3.5B, recurrent-depth; Geiping et al. 2025) on GSM8K at a sweep
of recurrence counts, with weight fake-quant of the recurrent core block and
the carry-trajectory fidelity read from the recurrent latent state.

    ~/venvs/kvcache/bin/python looped_lm_eval.py --model huginn --quants bf16,w8c,w4t,w4a,w4g128 \
        --steps 4,8,16,32,64 --n 250 --out ~/breadth/results/huginn_gsm8k.json

Quant specs as in breadth_sweep.py (w<bits>{t,c,a,g32,g128}); --scope core
quantizes the Linear layers of the recurrent block only (the paper's
analogue of quantizing the recursion), --scope all also the prelude, coda,
adapter and lm_head.  Fidelity: cosine between the recurrent latent state
after `steps` recurrences on the prompt tokens (teacher-forced) and the bf16
model's, averaged over tokens and prompts.
"""
import argparse
import glob
import json
import os
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HOME = Path.home()
os.environ.setdefault("HF_HOME", str(HOME / "hf_cache"))
DEV = "cuda"


def load_gsm8k(n):
    import pandas as pd
    files = sorted(glob.glob(str(HOME / "hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/test-*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files]).reset_index(drop=True)
    rows = df.iloc[:n]
    return [(q, a.split("####")[-1].strip().replace(",", ""), a) for q, a in zip(rows["question"], rows["answer"])]


def extract_answer(text):
    m = re.search(r"####\s*([-\d.,/]+)", text)
    if m:
        s = m.group(1)
    else:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "").rstrip(".")
    if len(s) > 40:                       # a runaway digit string is never a GSM8K answer
        return None
    try:
        v = float(s)
        if v != v or v in (float("inf"), float("-inf")):
            return None
        return str(int(v)) if v == int(v) else s
    except (ValueError, OverflowError):
        return s


def parse_quant(spec):
    if spec in ("bf16", "fp32"):
        return {"spec": spec, "bits": None, "gran": None}
    body = spec[1:]
    i = 0
    while i < len(body) and body[i].isdigit():
        i += 1
    return {"spec": spec, "bits": int(body[:i]), "gran": body[i:]}


@torch.no_grad()
def quantize_linear_(lin, bits, gran):
    W = lin.weight.data
    Wf = W.float()
    if gran == "a":
        qmax = 2 ** bits - 1
        wmin, wmax = Wf.amin(1, keepdim=True), Wf.amax(1, keepdim=True)
        scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
        zp = torch.round(-wmin / scale)
        Wq = (torch.clamp(torch.round(Wf / scale) + zp, 0, qmax) - zp) * scale
    else:
        qmax = 2 ** (bits - 1) - 1
        if gran == "t":
            s = (Wf.abs().max() / qmax).clamp_min(1e-8)
            Wq = torch.round(Wf / s).clamp(-qmax, qmax) * s
        elif gran == "c":
            s = (Wf.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-8)
            Wq = torch.round(Wf / s).clamp(-qmax, qmax) * s
        else:
            g = int(gran[1:])
            o, i_ = Wf.shape
            pad = (-i_) % g
            Wp = F.pad(Wf, (0, pad)).view(o, -1, g)
            s = (Wp.abs().amax(2, keepdim=True) / qmax).clamp_min(1e-8)
            Wq = (torch.round(Wp / s).clamp(-qmax, qmax) * s).view(o, -1)[:, :i_]
    lin.weight.data = Wq.to(W.dtype)


def quantize_model_(model, q, scope):
    if q["bits"] is None:
        return 0
    tf = model.transformer
    mods = list(tf.core_block.modules())
    if scope == "all":
        mods += list(tf.prelude.modules()) + list(tf.coda.modules()) + [tf.adapter, model.lm_head]
    n = 0
    for m in mods:
        if isinstance(m, torch.nn.Linear):
            quantize_linear_(m, q["bits"], q["gran"])
            n += 1
    return n


def build(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    return model.to(DEV).eval(), tok


def prompt_ids(tok, question):
    msgs = [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": question + "\nSolve this step by step. End with the line '#### <final numeric answer>'."}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(text, return_tensors="pt", add_special_tokens=False).to(DEV)


@torch.no_grad()
def latents(model, ids, steps):
    out = model(ids, num_steps=steps, output_details={"return_logits": False, "return_latents": True,
                                                      "return_head": False, "return_stats": False})
    return out.latent_states[0].float().cpu()          # (T, d)


@torch.no_grad()
def gold_nll(model, tok, ids, solution, steps):
    """Teacher-forced mean NLL (nats/token) of the gold solution after the prompt."""
    sol = tok.encode(solution + tok.eos_token, return_tensors="pt", add_special_tokens=False).to(DEV)
    full = torch.cat([ids, sol], dim=1)
    out = model(full, num_steps=steps, output_details={"return_logits": True, "return_latents": False,
                                                       "return_head": False, "return_stats": False})
    logits = out.logits[0, ids.shape[1] - 1:-1].float()
    return float(F.cross_entropy(logits, sol[0], reduction="mean"))


@torch.no_grad()
def evaluate(model, tok, data, steps, max_new, ref_lat=None, gen=True):
    from transformers import GenerationConfig
    cfg = GenerationConfig(max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                           pad_token_id=tok.pad_token_id or tok.eos_token_id, eos_token_id=tok.eos_token_id)
    correct, lats, fids, gens, nlls = [], [], [], [], []
    for i, (q, gold, sol) in enumerate(data):
        ids = prompt_ids(tok, q)
        lat = latents(model, ids, steps)
        lats.append(lat)
        if ref_lat is not None:
            fids.append(float(F.cosine_similarity(lat, ref_lat[i], dim=-1).mean()))
        nlls.append(gold_nll(model, tok, ids, sol, steps))
        if not gen:
            continue
        out = model.generate(ids, cfg, num_steps=steps, tokenizer=tok)
        text = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        pred = extract_answer(text)
        correct.append(int(pred == gold))
        gens.append(len(out[0]) - ids.shape[1])
    return (correct, lats, (sum(fids) / len(fids) if fids else 1.0),
            (sum(gens) / len(gens) if gens else 0.0), sum(nlls) / len(nlls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="huginn")
    ap.add_argument("--quants", default="bf16,w8c,w4t,w4a,w4g128")
    ap.add_argument("--scope", default="core", choices=["core", "all"])
    ap.add_argument("--steps", default="4,8,16,32,64")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--no_gen", action="store_true", help="teacher-forced NLL and fidelity only (fast)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    model_id = {"huginn": "tomg-group-umd/huginn-0125"}[a.model]
    data = load_gsm8k(a.n)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():
        try:
            records = json.load(open(out_path))["records"]
        except (json.JSONDecodeError, ValueError):
            print(f"  {out_path} unreadable (truncated write); starting fresh", flush=True)
    done = {(r["steps"], r["quant"]["spec"], r["scope"]) for r in records}
    steps_list = [int(s) for s in a.steps.split(",")]
    quants = [q for q in a.quants.split(",") if q]
    print(f"{a.model}: {len(data)} GSM8K problems, steps {steps_list}, quants {quants}, scope {a.scope}", flush=True)
    ref = {}
    for steps in steps_list:
        model, tok = build(model_id)
        t0 = time.time()
        c, lats, _, glen, nll = evaluate(model, tok, data, steps, a.max_new, gen=not a.no_gen)
        ref[steps] = lats
        acc = sum(c) / len(c) if c else None
        if (steps, "bf16", a.scope) not in done:
            records.append({"model": a.model, "steps": steps, "quant": parse_quant("bf16"), "scope": a.scope,
                            "n": len(data), "acc": acc, "nll": nll, "fidelity": 1.0, "gen_len": glen,
                            "correct": c, "secs": time.time() - t0})
            json.dump({"model": a.model, "records": records}, open(out_path, "w"))
        print(f"  steps={steps:3d} bf16      acc {acc if acc is None else round(acc*100,1)}  nll {nll:.3f}  gen {glen:.0f} ({time.time()-t0:.0f}s)", flush=True)
        for spec in quants:
            if spec == "bf16" or (steps, spec, a.scope) in done:
                continue
            q = parse_quant(spec)
            t1 = time.time()
            nq = quantize_model_(model, q, a.scope)
            c, _, fid, glen, nll = evaluate(model, tok, data, steps, a.max_new, ref_lat=ref[steps], gen=not a.no_gen)
            acc = sum(c) / len(c) if c else None
            records.append({"model": a.model, "steps": steps, "quant": q, "scope": a.scope, "n": len(data),
                            "acc": acc, "nll": nll, "fidelity": fid, "gen_len": glen, "correct": c,
                            "n_quantized_linears": nq, "secs": time.time() - t1})
            json.dump({"model": a.model, "records": records}, open(out_path, "w"))
            print(f"  steps={steps:3d} {spec:9s} acc {acc if acc is None else round(acc*100,1)}  nll {nll:.3f}  fid {fid:.4f}  gen {glen:.0f} ({time.time()-t1:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()
            model, tok = build(model_id)           # fresh weights for the next quantizer
        del model
        torch.cuda.empty_cache()
    print("wrote", out_path, len(records))


if __name__ == "__main__":
    main()
