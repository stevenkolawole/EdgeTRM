"""CPU functional check of looped_decode.py for a raven-family model: load, teacher-forced pass on
one problem, and an 8-token batched greedy generation (batch 2, left padding).  Not a numerical test."""
import os, sys, time
os.environ["X4_DEVICE"] = "cpu"
sys.path.insert(0, os.path.expanduser("~/breadth/lm"))
import torch
torch.set_num_threads(16)
import looped_decode as D
D.L.DEV = "cpu"
name, steps = sys.argv[1], int(sys.argv[2])
fam = D.family(name)
t = time.time()
model, tok = fam.build(steps)
fam.steps = steps
print("loaded", fam.model_id, f"{time.time()-t:.0f}s", flush=True)
data = D.L.load_gsm8k(2)
recs, ref = D.tf_pass(fam, model, tok, data[:1])
print("tf", {k: round(v, 3) if isinstance(v, float) else v for k, v in recs[0].items()}, flush=True)
prompts = [fam.prompt_text(tok, q) for q, _, _ in data]
enc = [tok.encode(p, add_special_tokens=False) for p in prompts]
T = max(map(len, enc)); pad = fam.pad_id(tok)
ids = torch.full((2, T), pad); attn = torch.zeros((2, T), dtype=torch.long)
for r, e in enumerate(enc):
    ids[r, T - len(e):] = torch.tensor(e); attn[r, T - len(e):] = 1
extra = dict(fam.gen_extra(tok))
if fam.stop_strings:
    extra.update(stop_strings=fam.stop_strings, tokenizer=tok)
t = time.time()
out = fam.generate(model, ids, attn, max_new_tokens=8, pad_token_id=pad, eos_token_id=fam.stop_ids(tok), do_sample=False, **extra)
print("gen", f"{time.time()-t:.0f}s", [tok.decode(o[T:]) for o in out], flush=True)
# single-row reference for row 1 (the shorter, left-padded prompt)
r = 0 if len(enc[0]) < len(enc[1]) else 1
o1 = fam.generate(model, torch.tensor([enc[r]]), torch.ones((1, len(enc[r])), dtype=torch.long), max_new_tokens=8,
                  pad_token_id=pad, eos_token_id=fam.stop_ids(tok), do_sample=False, **extra)
print("padded row", r, "batched:", out[r, T:].tolist(), "alone:", o1[0, len(enc[r]):].tolist(), flush=True)
