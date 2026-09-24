#!/usr/bin/env python3
"""EdgeTRM Q6 and looped-LM transfer (angle F): can decoding rescue a compressed
looped generator, and does "loop count is not the risk; the quantizer is;
teacher-forced metrics miss generation collapse" hold on more looped LMs?

One process = one model at one or more loop counts x one or more weight
quantizers x one or more decoding configs, GSM8K test[:n].  For every
(model, loops, quant) it first measures, per problem and at batch 1 (tf2.json):
  * teacher-forced mean NLL of the gold solution (as looped_lm_eval/ouro_eval),
  * teacher-forced next-token agreement with the bf16 model on the gold solution
    (argmax_q == argmax_bf16) and teacher-forced top-1 accuracy,
  * teacher-forced KL(bf16 || q) per gold position on the bf16 top-64 support with
    the remaining mass lumped into one bin (a lower bound on the full KL), and the
    bf16 next-token entropy there (Lotfi et al. 2606.00206: KL concentrates at
    high-entropy positions),
  * latent fidelity: cosine between this model's latent on the prompt and the bf16
    model's, averaged over prompt tokens (Ouro: final-norm output after the last
    loop; Huginn family: recurrent state after `steps` recurrences), as before.
Then it generates (batched, left padding, 512 new tokens by default) under each
decoding config and stores every output text with degeneracy statistics.

Resumable at batch granularity: per-problem records go to
<outdir>/<tag>/<decode>.jsonl (one line per problem, fsynced per batch);
teacher-forced records to <outdir>/<tag>/tf2.json; the bf16 reference to
<refdir>/<model>_s<steps>_n<n>_v2.pt.  <outdir>/<tag>/summary.json holds the
full config ("meta") and is merged, not overwritten, by later runs on the same
tag.  <tag> = <model>_s<steps>_<quant>_<scope>.  (v1 of this script wrote tf.json
and <model>_s<steps>_n<n>.pt without KL; v2 ignores them.)

Decoding configs.  HF semantics: repetition penalty and no-repeat-ngram act on the
whole input_ids, prompt included (as HF and vLLM do).
  greedy            argmax
  rep<x>            greedy + repetition penalty x (HF, CTRL-style)
  nrng<k>           greedy + no-repeat-ngram-size k (HF)
  topk<k>t<T>s<s>   top-k sampling at temperature T, seed s
  mark<d>           greedy, logit - d on single-token overthinking markers
                    (wait/but/alternatively/hmm, any case, with and without a
                    leading space; Lotfi et al. 2606.00206)
  loopstop          greedy; force a stop token once the generated tail is periodic
                    (last 96 tokens repeat with period <= 32, the degeneracy test)
  loopban<n>        greedy; ban any token that would repeat an n-gram already in
                    the *generated* text (prompt excluded): a back-off that only
                    touches long verbatim repeats
  dry               greedy + DRY penalty (multiplier 0.8, base 1.75, allowed length 2,
                    breakers = tokens containing newline : " *), over prompt + output
Teacher-forced extras: tf_ans_acc / tf_ans_nll score the gold final answer (tokens
after '####') by likelihood given the gold reasoning: the evaluation a
loglikelihood harness would run, beside free generation of the same answer.

    PYTHONPATH=~/pylibs/tf4554 ~/venvs/kvcache/bin/python looped_decode.py --model ouro14 \
        --steps 4 --quants bf16,w4c --decodes greedy,rep1.1,loopstop --n 200 \
        --outdir ~/breadth/lm/results/q6
"""
import argparse
import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
import zlib
from pathlib import Path

import torch
import torch.nn.functional as F

HOME = Path.home()
os.environ.setdefault("HF_HOME", str(HOME / "hf_cache"))
DEV = "cuda"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import looped_lm_eval as L  # noqa: E402  (load_gsm8k, extract_answer, parse_quant, quantize_linear_)

MODELS = {   # name: (family, repo id, prompt style)
    "ouro14": ("ouro", "ByteDance/Ouro-1.4B", "chat"),
    "ouro26": ("ouro", "ByteDance/Ouro-2.6B", "chat"),
    "ouro14t": ("ouro", "ByteDance/Ouro-1.4B-Thinking", "chat"),
    "ouro26t": ("ouro", "ByteDance/Ouro-2.6B-Thinking", "chat"),
    "huginn": ("raven", "tomg-group-umd/huginn-0125", "chat"),
    # retrofitted recurrence (McLeish, Geiping et al., arXiv 2511.07384): Huginn code, base models
    "olmo_rec32": ("raven", "smcleish/Recurrent-OLMo-2-0425-train-recurrence-32", "fewshot"),
    "llama_rec32": ("raven", "smcleish/Recurrent-Llama-3.2-train-recurrence-32", "fewshot"),
    "tinyllama_rec32": ("raven", "smcleish/Recurrent-TinyLlama-3T-train-recurrence-32", "fewshot"),
}
SUFFIX = "\nSolve this step by step. End with the line '#### <final numeric answer>'."
N_SHOT = 2
TOPK = 64
MARKERS = ("wait", "but", "alternatively", "hmm")


def fewshot_prefix():
    import pandas as pd
    import glob
    files = sorted(glob.glob(str(HOME / "hf_cache/hub/datasets--openai--gsm8k/snapshots/*/main/train-*.parquet")))
    df = pd.read_parquet(files[0])
    return "".join(f"Question: {q}\nAnswer: {a}\n\n" for q, a in zip(df["question"][:N_SHOT], df["answer"][:N_SHOT]))


# ----------------------------------------------------------------------------- families
def _stop_ids(tok, names):
    """eos plus the named end-of-turn tokens that exist in this vocabulary."""
    ids = {tok.eos_token_id}
    vocab = tok.get_vocab()
    for t in names:
        if t in vocab:
            ids.add(vocab[t])
    return sorted(i for i in ids if i is not None)


class Family:
    style = "chat"
    stop_strings = None

    def __init__(self, model_id, style):
        self.model_id, self.style = model_id, style
        if style == "fewshot":
            self.prefix = fewshot_prefix()
            self.stop_strings = ["Question:", "\n\n\n"]

    def prompt_text(self, tok, question):
        if self.style == "fewshot":
            return self.prefix + f"Question: {question}\nAnswer:"
        return tok.apply_chat_template(self.messages(question), tokenize=False, add_generation_prompt=True)

    def gold_text(self, tok, sol):
        return (" " + sol + "\n\n") if self.style == "fewshot" else (sol + tok.eos_token)

    def clean(self, text):
        return text.split("Question:")[0] if self.style == "fewshot" else text


class Ouro(Family):
    """24 (1.4B) / 48 (2.6B) shared layers applied total_ut_steps times; the whole
    stack is the loop, so scope 'core' quantizes every Linear in model.model.layers
    (not the embeddings, the exit gate or the LM head)."""

    def build(self, steps):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        cfg.total_ut_steps = steps
        model = AutoModelForCausalLM.from_pretrained(self.model_id, config=cfg, torch_dtype=torch.bfloat16,
                                                     trust_remote_code=True)
        if hasattr(model, "model") and hasattr(model.model, "total_ut_steps"):
            model.model.total_ut_steps = steps
        tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.steps = steps
        return model.to(DEV).eval(), tok

    def core_modules(self, model, scope):
        mods = list(model.model.layers.modules())
        if scope == "all":
            mods += [model.lm_head]
        return mods

    def messages(self, question):
        return [{"role": "user", "content": question + SUFFIX}]

    @torch.no_grad()
    def latents(self, model, ids):
        cap = []
        h = model.model.norm.register_forward_hook(lambda m, i, o: cap.append(o.detach()))
        model(ids, use_cache=False)
        h.remove()
        return cap[-1][0]

    @torch.no_grad()
    def logits(self, model, ids):
        return model(ids, return_dict=True, use_cache=False).logits[0]

    def stop_ids(self, tok):
        return _stop_ids(tok, ("<|im_end|>", "<|endoftext|>"))

    def pad_id(self, tok):
        return tok.convert_tokens_to_ids("<empty_output>")

    def gen_extra(self, tok):
        return {}

    def generate(self, model, input_ids, attn, **kw):
        return model.generate(input_ids, attention_mask=attn, **kw)


# Huginn's code ignores attention masks (prepared_attn_mask = None), so left padding
# would be attended to.  During batched generation we route its SDPA calls through a
# wrapper that adds the causal + key-padding mask (plus the diagonal, so pad queries
# never have an empty row).  RoPE is relative, so left-shifted positions are harmless.
_PAD = {"valid": None}
_ORIG_SDPA = F.scaled_dot_product_attention


def _sdpa_padded(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    pv = _PAD["valid"]
    if pv is None or q.shape[0] != pv.shape[0]:
        return _ORIG_SDPA(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale, **kw)
    B, S, K, T = q.shape[0], q.shape[2], k.shape[2], pv.shape[1]
    kv_valid = torch.ones(B, K, dtype=torch.bool, device=q.device)
    kv_valid[:, :min(T, K)] = pv[:, :min(T, K)]
    qpos = torch.arange(K - S, K, device=q.device)
    kpos = torch.arange(K, device=q.device)
    m = (kpos[None, :] <= qpos[:, None])[None] & kv_valid[:, None, :]
    m = m | (kpos[None, None, :] == qpos[None, :, None])
    return _ORIG_SDPA(q, k, v, attn_mask=m[:, None], dropout_p=dropout_p, scale=scale, **kw)


def _fast_update(self, key_states, value_states, step_idx_tensor, lookup_strategy=None):
    """Drop-in for HuginnDynamicCache.update with lookup_strategy 'full' and a fixed number of
    recurrences: keep one concatenated tensor per block index instead of one tensor per token
    and step (the original re-stacks every cached token at every layer call, which dominates
    decode time).  Returns exactly what the original returns in that setting."""
    step_idx = int(step_idx_tensor)
    if not hasattr(self, "_x4k"):
        self._x4k, self._x4v = {}, {}
    if step_idx == 0:
        self._seen_tokens += key_states.shape[-2]
    if step_idx in self._x4k:
        self._x4k[step_idx] = torch.cat([self._x4k[step_idx], key_states], dim=-2)
        self._x4v[step_idx] = torch.cat([self._x4v[step_idx], value_states], dim=-2)
    else:
        self._x4k[step_idx], self._x4v[step_idx] = key_states, value_states
    return self._x4k[step_idx], self._x4v[step_idx]


class Raven(Family):
    """Huginn-0125 and the retrofitted-recurrence models (same code): prelude ->
    core block iterated num_steps times (with an adapter that re-injects the prelude
    output) -> coda.  Scope 'core' quantizes the core block's Linears (and, as in
    looped_lm_eval, not the adapter)."""

    def build(self, steps):
        model, tok = L.build(self.model_id)
        self.steps = steps
        mod = sys.modules[type(model).__module__]
        F.scaled_dot_product_attention = _sdpa_padded       # inert unless _PAD["valid"] is set
        cache_cls = getattr(mod, "HuginnDynamicCache", None)
        if cache_cls is not None and os.environ.get("X4_SLOW_CACHE") != "1":
            cache_cls.update = _fast_update
        return model, tok

    def core_modules(self, model, scope):
        tf = model.transformer
        mods = list(tf.core_block.modules())
        if scope == "all":
            mods += list(tf.prelude.modules()) + list(tf.coda.modules()) + [tf.adapter, model.lm_head]
        return mods

    def messages(self, question):
        return [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question + SUFFIX}]

    init_scale = None     # None: the model's own random state init; 0.0: zero init (deterministic checks)

    def _ikw(self):
        return {} if self.init_scale is None else {"init_scale": self.init_scale}

    @torch.no_grad()
    def latents(self, model, ids):
        out = model(ids, num_steps=self.steps, output_details={"return_logits": False, "return_latents": True,
                                                               "return_head": False, "return_stats": False},
                    **self._ikw())
        return out.latent_states[0]

    @torch.no_grad()
    def logits(self, model, ids):
        out = model(ids, num_steps=self.steps, output_details={"return_logits": True, "return_latents": False,
                                                               "return_head": False, "return_stats": False},
                    **self._ikw())
        return out.logits[0]

    def stop_ids(self, tok):
        return _stop_ids(tok, ("<|end_turn|>", "<|end_text|>"))

    def pad_id(self, tok):
        return tok.convert_tokens_to_ids("<|pad|>") if "<|pad|>" in tok.get_vocab() else tok.eos_token_id

    def gen_extra(self, tok):
        return {"num_steps": self.steps, "tokenizer": tok, **self._ikw()}

    def generate(self, model, input_ids, attn, **kw):
        _PAD["valid"] = attn.bool()
        try:
            return model.generate(input_ids, attention_mask=attn, **kw)
        finally:
            _PAD["valid"] = None


def family(name):
    """A bf16 copy under ~/hf_models/<name> (made by convert_bf16.py where disk is short)
    takes precedence over the hub id; meta records which was used."""
    fam, mid, style = MODELS[name]
    local = HOME / "hf_models" / name
    f = {"ouro": Ouro, "raven": Raven}[fam](str(local) if local.exists() else mid, style)
    f.hub_id = mid
    return f


# ----------------------------------------------------------------------------- quantization
def quantize_(fam, model, q, scope):
    if q["bits"] is None:
        return 0
    n = 0
    for m in fam.core_modules(model, scope):
        if isinstance(m, torch.nn.Linear):
            L.quantize_linear_(m, q["bits"], q["gran"])
            n += 1
    return n


# ----------------------------------------------------------------------------- decoding
def tail_period(ids, window=96, pmax=32):
    """Smallest period p <= pmax such that the last `window` tokens repeat with period p."""
    if len(ids) < window:
        return 0
    t = ids[-window:]
    for p in range(1, pmax + 1):
        if all(t[j] == t[j - p] for j in range(p, window)):
            return p
    return 0


def marker_ids(tok):
    ids = set()
    for w in MARKERS:
        for v in (w, w.capitalize(), w.upper()):
            for s in (v, " " + v):
                e = tok.encode(s, add_special_tokens=False)
                if len(e) == 1:
                    ids.add(e[0])
    return sorted(ids)


class _Proc:
    """Logits processors for the non-HF controls; `start` = prompt length (all rows
    are left-padded to the same length, so generated tokens start at column `start`)."""

    def __init__(self, kind, start, arg, stop, tok, pad=None):
        self.kind, self.start, self.arg, self.stop, self.pad = kind, start, arg, stop, pad
        self.markers = torch.tensor(marker_ids(tok), dtype=torch.long) if kind == "mark" else None
        if kind == "dry":   # DRY sequence breakers: any token whose text contains a newline, ':', '"' or '*'
            self.breakers = {i for t, i in tok.get_vocab().items()
                             if any(c in tok.convert_tokens_to_string([t]) for c in ("\n", ":", '"', "*"))}

    def __call__(self, input_ids, scores):
        if self.kind == "mark":
            scores[:, self.markers.to(scores.device)] -= self.arg
            return scores
        g = input_ids[:, self.start:]
        if self.kind == "loopstop":
            W, P = 96, 32
            if g.shape[1] >= W:
                t = g[:, -W:]
                hit = torch.zeros(t.shape[0], dtype=torch.bool, device=t.device)
                for p in range(1, P + 1):
                    hit |= (t[:, p:] == t[:, :-p]).all(1)
                if hit.any():
                    forced = torch.full_like(scores[hit], float("-inf"))
                    forced[:, self.stop[0]] = 0.0
                    scores[hit] = forced
            return scores
        if self.kind == "loopban":
            n = self.arg
            if g.shape[1] >= n:
                rows = g.tolist()
                for r, seq in enumerate(rows):
                    pre, last = seq[-(n - 1):], seq[-1]
                    banned = {seq[j + 1] for j in range(n - 2, len(seq) - 1)
                              if seq[j] == last and seq[j - n + 2:j + 1] == pre}
                    if banned:
                        scores[r, list(banned)] = float("-inf")
            return scores
        if self.kind == "dry":
            mult, base, allowed, cap = self.arg
            rows = input_ids.tolist()
            for r, seq in enumerate(rows):
                seq = [t for t in seq if t != self.pad]          # prompt + generated, padding removed
                last = seq[-1]
                if last in self.breakers:
                    continue
                pen = {}
                for j in range(len(seq) - 2, -1, -1):
                    if seq[j] != last:
                        continue
                    ln = 1
                    while ln < cap and j - ln >= 0 and seq[j - ln] == seq[-1 - ln] and seq[j - ln] not in self.breakers:
                        ln += 1
                    if ln >= allowed:
                        nxt = seq[j + 1]
                        pen[nxt] = max(pen.get(nxt, 0.0), mult * base ** (ln - allowed))
                if pen:
                    ids_ = torch.tensor(list(pen), device=scores.device)
                    scores[r, ids_] -= torch.tensor(list(pen.values()), device=scores.device, dtype=scores.dtype)
            return scores
        raise ValueError(self.kind)


def decode_kwargs(name):
    """(HF generate kwargs, custom processor spec or None).  'seed' is popped by the caller."""
    if name == "greedy":
        return {"do_sample": False}, None
    m = re.fullmatch(r"rep([\d.]+)", name)
    if m:
        return {"do_sample": False, "repetition_penalty": float(m.group(1))}, None
    m = re.fullmatch(r"nrng(\d+)", name)
    if m:
        return {"do_sample": False, "no_repeat_ngram_size": int(m.group(1))}, None
    m = re.fullmatch(r"topk(\d+)t([\d.]+)s(\d+)", name)
    if m:
        return {"do_sample": True, "top_k": int(m.group(1)), "temperature": float(m.group(2)), "top_p": 1.0,
                "seed": int(m.group(3))}, None
    m = re.fullmatch(r"mark([\d.]+)", name)
    if m:
        return {"do_sample": False}, ("mark", float(m.group(1)))
    if name == "loopstop":
        return {"do_sample": False}, ("loopstop", None)
    m = re.fullmatch(r"loopban(\d+)", name)
    if m:
        return {"do_sample": False}, ("loopban", int(m.group(1)))
    if name == "dry":   # DRY defaults (multiplier 0.8, base 1.75, allowed length 2), max match 64
        return {"do_sample": False}, ("dry", (0.8, 1.75, 2, 64))
    raise ValueError(name)


def degeneracy(ids, text, max_new, hit_eos):
    grams = [tuple(ids[i:i + 4]) for i in range(max(0, len(ids) - 3))]
    rep4 = 1.0 - len(set(grams)) / len(grams) if grams else 0.0
    raw = text.encode("utf-8", "ignore")
    zr = len(zlib.compress(raw)) / max(1, len(raw))
    period = tail_period(ids)
    has_marker = "####" in text
    capped = (not hit_eos) and len(ids) >= max_new
    return {"n_gen": len(ids), "hit_eos": bool(hit_eos), "capped": bool(capped), "tail_period": period,
            "rep4": round(rep4, 4), "zlib_ratio": round(zr, 4), "has_marker": has_marker,
            "degenerate": bool(period > 0 or (capped and not has_marker))}


# ----------------------------------------------------------------------------- io helpers
def atomic_json(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def read_jsonl(path):
    recs = {}
    if not path.exists():
        return recs
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue                       # a torn last line from a crash
        recs[r["idx"]] = r
    return recs


def snapshot_rev(model_id):
    d = Path(os.environ["HF_HOME"]) / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    try:
        return sorted(p.name for p in d.iterdir())
    except FileNotFoundError:
        return None


# ----------------------------------------------------------------------------- passes
@torch.no_grad()
def tf_pass(fam, model, tok, data, ref=None):
    """Teacher-forced pass at batch 1.  Returns per-problem records and the tensors a
    reference needs (prompt latents in bf16, argmax / top-64 over the gold positions)."""
    recs, lats, amaxs, tops = [], [], [], []
    for i, (q, gold, sol) in enumerate(data):
        ids = tok.encode(fam.prompt_text(tok, q), return_tensors="pt", add_special_tokens=False).to(DEV)
        torch.manual_seed(1234 + i)          # Huginn draws its initial latent state at random
        lat = fam.latents(model, ids).float()
        gtxt = fam.gold_text(tok, sol)
        k = gtxt.index("####") + 4
        tail = tok.eos_token if fam.style == "chat" else "\n\n"
        ans = gtxt[k:len(gtxt) - len(tail)]
        parts = [tok.encode(x, add_special_tokens=False) for x in (gtxt[:k], ans, tail)]
        sol_ids = torch.tensor([parts[0] + parts[1] + parts[2]], device=DEV)
        a0, a1 = len(parts[0]), len(parts[0]) + len(parts[1])     # answer-token positions within the solution
        full = torch.cat([ids, sol_ids], dim=1)
        torch.manual_seed(1234 + i)
        lg = fam.logits(model, full)[ids.shape[1] - 1:-1].float()
        tgt = sol_ids[0]
        lp = F.log_softmax(lg, -1)
        nll = float(-lp.gather(1, tgt[:, None]).mean())
        am = lg.argmax(-1)
        r = {"idx": i, "nll": nll, "tf_acc": float((am == tgt).float().mean()), "n_sol_tokens": int(tgt.numel()),
             "n_prompt_tokens": int(ids.shape[1]),
             # likelihood-style scoring of the gold answer given the gold reasoning (teacher forced)
             "tf_ans_acc": int(bool((am[a0:a1] == tgt[a0:a1]).all())),
             "tf_ans_nll": float(-lp[a0:a1].gather(1, tgt[a0:a1, None]).mean()),
             "tf_ans_tokens": a1 - a0}
        if ref is not None:
            r["fidelity"] = float(F.cosine_similarity(lat, ref["lat"][i].to(DEV).float(), dim=-1).mean())
            r["tf_agree"] = float((am.cpu() == ref["argmax"][i]).float().mean())
            tid, tlp = ref["top_ids"][i].to(DEV), ref["top_lp"][i].to(DEV).float()
            p = tlp.exp()
            qlp = lp.gather(1, tid)
            ptail = (1 - p.sum(1)).clamp_min(1e-12)
            qtail = (1 - qlp.exp().sum(1)).clamp_min(1e-12)
            kl = (p * (tlp - qlp)).sum(1) + ptail * (ptail.log() - qtail.log())
            r["kl"] = float(kl.mean())
            r["kl_max"] = float(kl.max())
            r["entropy_bf16"] = ref["entropy"][i]
        else:
            r["fidelity"], r["tf_agree"], r["kl"], r["kl_max"] = 1.0, 1.0, 0.0, 0.0
            r["entropy_bf16"] = float(-(lp.exp() * lp).sum(1).mean())
        recs.append(r)
        lats.append(lat.to(torch.bfloat16).cpu())
        amaxs.append(am.cpu())
        tl, ti = lp.topk(TOPK, dim=-1)
        tops.append((ti.cpu(), tl.cpu(), r["entropy_bf16"]))
    return recs, {"lat": lats, "argmax": amaxs, "top_ids": [t[0] for t in tops], "top_lp": [t[1] for t in tops],
                  "entropy": [t[2] for t in tops]}


@torch.no_grad()
def gen_pass(fam, model, tok, data, decode, max_new, batch, out_jsonl, log):
    from transformers import LogitsProcessorList
    kw, proc = decode_kwargs(decode)
    seed = kw.pop("seed", None)
    stop = fam.stop_ids(tok)
    pad = fam.pad_id(tok)
    done = read_jsonl(out_jsonl)
    prompts = [fam.prompt_text(tok, q) for q, _, _ in data]
    extra = dict(fam.gen_extra(tok))
    if fam.stop_strings:
        extra["stop_strings"] = fam.stop_strings
        extra["tokenizer"] = tok
    f = open(out_jsonl, "a")
    t0 = time.time()
    for b0 in range(0, len(data), batch):
        idxs = list(range(b0, min(len(data), b0 + batch)))
        if all(i in done for i in idxs):
            continue
        enc = [tok.encode(prompts[i], add_special_tokens=False) for i in idxs]
        T = max(len(e) for e in enc)
        input_ids = torch.full((len(idxs), T), pad, dtype=torch.long)
        attn = torch.zeros((len(idxs), T), dtype=torch.long)
        for r, e in enumerate(enc):
            input_ids[r, T - len(e):] = torch.tensor(e)
            attn[r, T - len(e):] = 1
        input_ids, attn = input_ids.to(DEV), attn.to(DEV)
        s0 = (seed if seed is not None else 0) * 100003 + b0   # fixed per batch: sampling and Huginn's random init
        torch.manual_seed(s0)
        torch.cuda.manual_seed_all(s0)
        lp = LogitsProcessorList([_Proc(proc[0], T, proc[1], stop, tok, pad)]) if proc else None
        tb = time.time()
        out = fam.generate(model, input_ids, attn, max_new_tokens=max_new, pad_token_id=pad, eos_token_id=stop,
                           logits_processor=lp, **kw, **extra)
        gen = out[:, T:].tolist()
        lines = []
        ends = set(stop) | {pad}
        for r, i in enumerate(idxs):
            g = gen[r]
            cut = next((k for k, t in enumerate(g) if t in ends), None)
            hit_eos = cut is not None
            g = g[:cut] if hit_eos else g
            full_text = tok.decode(g, skip_special_tokens=True)
            text = fam.clean(full_text)
            if text != full_text:
                hit_eos = True                     # stopped by a stop string (few-shot prompts)
            pred = L.extract_answer(text)
            rec = {"idx": i, "gold": data[i][1], "pred": pred, "correct": int(pred == data[i][1]),
                   **degeneracy(g, text, max_new, hit_eos), "text": text, "ids": g}
            lines.append(json.dumps(rec))
        f.write("\n".join(lines) + "\n")
        f.flush()
        os.fsync(f.fileno())
        log(f"    {decode} batch {b0 // batch}: {len(idxs)} problems in {time.time() - tb:.0f}s "
            f"(max gen {max(len(x) for x in gen)})")
    f.close()
    recs = read_jsonl(out_jsonl)
    return [recs[i] for i in sorted(recs) if i < len(data)], time.time() - t0


def summarize(gen_recs):
    n = len(gen_recs)
    if not n:
        return {}
    m = lambda k: sum(float(r[k]) for r in gen_recs) / n  # noqa: E731
    return {"n": n, "acc": m("correct"), "degenerate": m("degenerate"),
            "loop": sum(r["tail_period"] > 0 for r in gen_recs) / n, "capped": m("capped"),
            "has_marker": m("has_marker"), "mean_gen_len": m("n_gen"), "rep4": m("rep4"), "zlib_ratio": m("zlib_ratio")}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--steps", default="4")
    ap.add_argument("--quants", default="bf16,w4c")
    ap.add_argument("--scope", default="core", choices=["core", "all"])
    ap.add_argument("--decodes", default="greedy,rep1.1,rep1.3,nrng3,topk20t0.7s0,topk20t0.7s1,topk20t0.7s2")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--no_gen", action="store_true")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--refdir", default=None, help="where bf16 references live (default <outdir>/ref)")
    ap.add_argument("--slow_cache", action="store_true", help="raven: keep the original HuginnDynamicCache.update")
    ap.add_argument("--init_scale", type=float, default=None, help="raven: latent-state init scale (0 = zeros)")
    a = ap.parse_args()
    if a.slow_cache:
        os.environ["X4_SLOW_CACHE"] = "1"

    fam = family(a.model)
    if a.init_scale is not None:
        fam.init_scale = a.init_scale
    data = L.load_gsm8k(a.n)
    outdir = Path(a.outdir)
    refdir = Path(a.refdir) if a.refdir else outdir / "ref"
    refdir.mkdir(parents=True, exist_ok=True)
    script_sha = hashlib.sha1(open(__file__, "rb").read()).hexdigest()[:12]
    import transformers
    base_meta = {"script": "looped_decode.py (v2)", "script_sha1": script_sha, "argv": sys.argv,
                 "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                 "transformers": transformers.__version__, "python": platform.python_version(),
                 "model": a.model, "model_id": fam.hub_id, "model_path": fam.model_id,
                 "snapshot": snapshot_rev(fam.hub_id),
                 "dataset": "openai/gsm8k main test[:n]", "n": len(data), "max_new_tokens": a.max_new,
                 "batch": a.batch, "padding": "left", "dtype": "bfloat16", "scope": a.scope,
                 "prompt_style": fam.style, "prompt_suffix": SUFFIX if fam.style == "chat" else None,
                 "n_shot": N_SHOT if fam.style == "fewshot" else 0, "stop_strings": fam.stop_strings,
                 "raven_cache": "slow (original)" if os.environ.get("X4_SLOW_CACHE") == "1" else
                                "fast concatenating update (_fast_update)",
                 "init_scale": a.init_scale, "tf_batch": 1, "kl": f"KL(bf16||q) on bf16 top-{TOPK} + one tail bin, mean over gold positions",
                 "seeding": "tf pass: torch.manual_seed(1234+idx) before each forward; generation: "
                            "torch.manual_seed(seed*100003+first_idx_of_batch) before each batch (seed 0 for "
                            "deterministic decodes)",
                 "answer_extraction": "looped_lm_eval.extract_answer (#### else last number)",
                 "degenerate_def": "tail_period>0 (last 96 tokens periodic with p<=32) or (hit max_new without "
                                   "a stop token and no '####')"}

    def log(s):
        print(time.strftime("%H:%M:%S ") + s, flush=True)

    for steps in [int(s) for s in a.steps.split(",")]:
        ref_path = refdir / f"{a.model}_s{steps}_n{len(data)}_v2.pt"
        for spec in [q for q in a.quants.split(",") if q]:
            q = L.parse_quant(spec)
            tag = f"{a.model}_s{steps}_{spec}_{a.scope}"
            tdir = outdir / tag
            tdir.mkdir(parents=True, exist_ok=True)
            model, tok = fam.build(steps)
            fam.steps = steps
            ref = torch.load(ref_path) if ref_path.exists() else None
            ref_recs = None
            if ref is None:
                log(f"{tag}: computing bf16 reference ({len(data)} problems)")
                t0 = time.time()
                recs, ref = tf_pass(fam, model, tok, data)
                ref_recs = recs
                torch.save(ref, str(ref_path) + f".tmp{os.getpid()}")
                os.replace(str(ref_path) + f".tmp{os.getpid()}", ref_path)
                log(f"  reference done in {time.time() - t0:.0f}s")
            nq = quantize_(fam, model, q, a.scope)
            tf_path = tdir / "tf2.json"
            if tf_path.exists():
                tf = json.load(open(tf_path))
            else:
                t0 = time.time()
                if q["bits"] is None and ref_recs is not None:
                    recs = ref_recs                   # bf16 against itself: the reference pass already is this
                else:
                    recs, _ = tf_pass(fam, model, tok, data, ref=ref)
                m = lambda k: sum(r[k] for r in recs) / len(recs)  # noqa: E731
                tf = {"summary": {k: m(k) for k in ("nll", "tf_acc", "tf_agree", "kl", "fidelity", "entropy_bf16",
                                                    "tf_ans_acc", "tf_ans_nll")},
                      "records": recs, "secs": time.time() - t0, "n_quantized_linears": nq,
                      "meta": {**base_meta, "steps": steps, "quant": q}}
                atomic_json(tf, tf_path)
            s = tf["summary"]
            log(f"{tag}: tf nll {s['nll']:.3f}  tf_acc {s['tf_acc']:.3f}  agree {s['tf_agree']:.3f}  "
                f"kl {s['kl']:.3f}  fid {s['fidelity']:.4f}  ans_acc {s['tf_ans_acc']:.3f}  ({nq} linears quantized)")
            sp = tdir / "summary.json"
            old = json.load(open(sp)) if sp.exists() else {}
            summary = {"meta": {**base_meta, "steps": steps, "quant": q, "n_quantized_linears": nq,
                                "stop_token_ids": fam.stop_ids(tok), "pad_token_id": fam.pad_id(tok),
                                "marker_token_ids": marker_ids(tok),
                                "decodes": old.get("meta", {}).get("decodes", {})},
                       "tf": s, "gen": old.get("gen", {})}
            if not a.no_gen:
                for dec in [d for d in a.decodes.split(",") if d]:
                    kw, proc = decode_kwargs(dec)
                    recs, secs = gen_pass(fam, model, tok, data, dec, a.max_new, a.batch, tdir / f"{dec}.jsonl", log)
                    summary["meta"]["decodes"][dec] = {**kw, "processor": proc}
                    summary["gen"][dec] = {**summarize(recs), "secs_this_run": secs, "batch": a.batch,
                                           "script_sha1": script_sha}
                    g = summary["gen"][dec]
                    log(f"{tag} {dec:14s} acc {g['acc'] * 100:.1f}  degen {g['degenerate'] * 100:.1f}  "
                        f"loop {g['loop'] * 100:.1f}  capped {g['capped'] * 100:.1f}  len {g['mean_gen_len']:.0f}  "
                        f"(n={g['n']}, {secs:.0f}s)")
                    atomic_json(summary, sp)
            atomic_json(summary, sp)
            del model
            torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
