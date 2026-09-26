#!/usr/bin/env python3
"""Return test on looped LMs (X4, pre-registered in prereg/X4_return_test_prereg.md, commit 8a14e1f).

For one model and a list of weight quantizers (on the looped block only), at hand-off loop k:
  * bf16 ("exact" map), compressed, and hybrid (compressed loops 0..k-1, bf16 loops k..L-1) greedy generation on
    GSM8K test[:n] (batch B, 512 new tokens), per-example answers;
  * teacher forced, batch 1, per example: prompt latent fidelity (as looped_decode), bf16 / compressed / hybrid top-1
    on the gold solution, and the interpolation z = z_bf16 + t (z_q - z_bf16) at the hand-off followed by bf16
    loops k..L-1, for t in T_GRID: top-1 agreement with bf16 and final-latent cosine.
Per-cell records go to <outdir>/<model>_k<k>_<quant>.json (bf16 to <model>_k<k>_bf16.json); resumable per cell.

The loop is switched between the two weight copies at loop granularity:
  Ouro:  every decoder layer is wrapped; the wrapper picks the bf16 or compressed copy from `current_ut`;
  Raven (Huginn, OLMo-rec): model.core_block_forward is replaced by one that picks the block list from `current_step`.
A controller holds the mode ('fp', 'q', 'hybrid'), k, and an optional state override applied at loop k.
"""
import argparse
import copy
import json
import os
import socket
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import looped_decode as D  # noqa: E402

DEV = "cuda"
T_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
AGREE = 0.95


class Ctl:
    def __init__(self, k):
        self.mode, self.k, self.override, self.capture, self.captured = "fp", k, None, False, None

    def use_q(self, step):
        return self.mode == "q" or (self.mode == "hybrid" and step < self.k)


class OuroSwitch(torch.nn.Module):
    def __init__(self, fp, q, ctl, first):
        super().__init__()
        self.fp, self.q, self.ctl, self.first = fp, q, ctl, first
        self.attention_type = getattr(fp, "attention_type", "full_attention")

    def forward(self, hidden_states, *args, **kw):
        ut = int(kw.get("current_ut", 0))
        c = self.ctl
        if self.first and ut == c.k:
            if c.capture:
                c.captured = hidden_states.detach().clone()
            if c.override is not None:
                hidden_states = c.override.to(hidden_states.dtype)
        layer = self.q if c.use_q(ut) else self.fp
        return layer(hidden_states, *args, **kw)


def install_ouro(model, q, scope, ctl):
    layers = model.model.layers
    qlayers = copy.deepcopy(layers)
    for m in qlayers.modules():
        if isinstance(m, torch.nn.Linear):
            D.L.quantize_linear_(m, q["bits"], q["gran"])
    for i in range(len(layers)):
        layers[i] = OuroSwitch(layers[i], qlayers[i], ctl, first=(i == 0))
    return sum(isinstance(m, torch.nn.Linear) for m in qlayers.modules())


def install_raven(model, q, scope, ctl):
    tf = model.transformer
    fp_blocks = tf.core_block
    q_blocks = copy.deepcopy(fp_blocks)
    for m in q_blocks.modules():
        if isinstance(m, torch.nn.Linear):
            D.L.quantize_linear_(m, q["bits"], q["gran"])

    def cbf(x, input_embeds, freqs_cis, mask, past_key_values, block_idx, current_step):
        step = int(current_step)
        if step == ctl.k:
            if ctl.capture:
                ctl.captured = x.detach().clone()
            if ctl.override is not None:
                x = ctl.override.to(x.dtype)
        block_idx = block_idx.detach().clone()
        x = model._maybe_inject_noise(x, current_step)
        x = tf.adapter(torch.cat([x, input_embeds.to(x.device)], dim=-1))
        for block in (q_blocks if ctl.use_q(step) else fp_blocks):
            block_idx += 1
            x = block(x, freqs_cis, block_idx, mask, past_key_values)
        return x, block_idx

    model.core_block_forward = cbf
    return sum(isinstance(m, torch.nn.Linear) for m in q_blocks.modules())


@torch.no_grad()
def tf_one(fam, model, tok, q_text, sol, seed, steps):
    """ONE teacher-forced forward on prompt+gold: (logits over the gold positions, final latent of the prompt
    positions from the same pass; causal, so equal to a prompt-only pass up to numerics)."""
    ids = tok.encode(fam.prompt_text(tok, q_text), return_tensors="pt", add_special_tokens=False).to(DEV)
    sol_ids = tok.encode(fam.gold_text(tok, sol), return_tensors="pt", add_special_tokens=False).to(DEV)
    full = torch.cat([ids, sol_ids], dim=1)
    P = ids.shape[1]
    torch.manual_seed(seed)
    if isinstance(fam, D.Ouro):
        cap = []
        h = model.model.norm.register_forward_hook(lambda m, i, o: cap.append(o.detach()))
        o = model(full, return_dict=True, use_cache=False)
        h.remove()
        lg, lat = o.logits[0], cap[-1][0][:P]
    else:
        o = model(full, num_steps=steps, output_details={"return_logits": True, "return_latents": True,
                                                         "return_head": False, "return_stats": False})
        lg, lat = o.logits[0], o.latent_states[0][:P]
    return lg[P - 1:-1].float(), lat.float(), sol_ids[0]


def gen_answers(fam, model, tok, data, batch, max_new, log, tag):
    import re as _re
    stop, pad = fam.stop_ids(tok), fam.pad_id(tok)
    prompts = [fam.prompt_text(tok, q) for q, _, _ in data]
    extra = dict(fam.gen_extra(tok))
    if fam.stop_strings:
        extra.update(stop_strings=fam.stop_strings, tokenizer=tok)
    out = []
    for b0 in range(0, len(data), batch):
        idxs = list(range(b0, min(len(data), b0 + batch)))
        enc = [tok.encode(prompts[i], add_special_tokens=False) for i in idxs]
        T = max(len(e) for e in enc)
        x = torch.full((len(idxs), T), pad, dtype=torch.long)
        a = torch.zeros((len(idxs), T), dtype=torch.long)
        for r, e in enumerate(enc):
            x[r, T - len(e):] = torch.tensor(e)
            a[r, T - len(e):] = 1
        torch.manual_seed(b0)
        torch.cuda.manual_seed_all(b0)
        o = fam.generate(model, x.to(DEV), a.to(DEV), max_new_tokens=max_new, pad_token_id=pad, eos_token_id=stop,
                         do_sample=False, **extra)
        ends = set(stop) | {pad}
        for r, i in enumerate(idxs):
            g = o[r, T:].tolist()
            cut = next((j for j, t in enumerate(g) if t in ends), None)
            g = g[:cut] if cut is not None else g
            text = fam.clean(tok.decode(g, skip_special_tokens=True))
            pred = D.L.extract_answer(text)
            out.append({"idx": i, "pred": pred, "correct": int(pred == data[i][1]), "n_gen": len(g)})
    log(f"  {tag}: gen acc {sum(r['correct'] for r in out) / len(out):.3f}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(D.MODELS))
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--quants", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    data = D.L.load_gsm8k(a.n)

    def log(s):
        print(time.strftime("%H:%M:%S ") + s, flush=True)

    import transformers
    meta = {"script": "ret_test.py", "argv": sys.argv, "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
            "transformers": transformers.__version__, "torch": torch.__version__, "t_grid": T_GRID, "agree_thr": AGREE,
            "n": len(data), "batch": a.batch, "max_new": a.max_new, "k": a.k, "steps": a.steps,
            "prereg": "prereg/X4_return_test_prereg.md (commit 8a14e1f)"}
    fam = D.family(a.model)
    fam.steps = a.steps
    gen_fp_cache = None
    for spec in [s for s in a.quants.split(",") if s]:
        path = out / f"{a.model}_k{a.k}_{spec}.json"
        bf_path = out / f"{a.model}_k{a.k}_bf16.json"
        if path.exists():
            log(f"{path.name} exists, skipping")
            continue
        model, tok = fam.build(a.steps)
        fam.steps = a.steps
        ctl = Ctl(a.k)
        q = D.parse_spec(spec)
        nq = (install_ouro if isinstance(fam, D.Ouro) else install_raven)(model, q, "core", ctl)
        t0 = time.time()
        # bf16 reference (recomputed per job so every job is self-contained; written once)
        ctl.mode, ctl.override = "fp", None
        if gen_fp_cache is None:            # once per job; every cell of the job is scored against the same bf16 run
            gen_fp_cache = gen_answers(fam, model, tok, data, a.batch, a.max_new, log, "bf16")
        gen_fp = gen_fp_cache
        ctl.mode = "q"
        gen_q = gen_answers(fam, model, tok, data, a.batch, a.max_new, log, spec)
        ctl.mode = "hybrid"
        gen_h = gen_answers(fam, model, tok, data, a.batch, a.max_new, log, f"{spec} hybrid k={a.k}")
        recs = []
        for i, (qt, gold, sol) in enumerate(data):
            seed = 1234 + i
            ctl.mode, ctl.override, ctl.capture = "fp", None, True
            lg_fp, lat_fp, tgt = tf_one(fam, model, tok, qt, sol, seed, a.steps)
            z_fp = ctl.captured
            ctl.capture = False
            am_fp = lg_fp.argmax(-1)
            # final latent of the full sequence under bf16 (for distance), via a second capture of the last layer output
            ctl.mode, ctl.capture = "q", True
            lg_q, lat_q, _ = tf_one(fam, model, tok, qt, sol, seed, a.steps)
            z_q = ctl.captured
            ctl.capture = False
            rec = {"idx": i, "fid": float(F.cosine_similarity(lat_q, lat_fp, dim=-1).mean()),
                   "agree_q": float((lg_q.argmax(-1) == am_fp).float().mean()),
                   "nll_fp": float(F.cross_entropy(lg_fp, tgt)), "nll_q": float(F.cross_entropy(lg_q, tgt)),
                   "state_rel_dist": float((z_q - z_fp).float().norm() / z_fp.float().norm())}
            curve = []
            for t in T_GRID:
                ctl.mode = "hybrid"
                ctl.override = z_fp + t * (z_q - z_fp)
                lg_t, lat_t, _ = tf_one(fam, model, tok, qt, sol, seed, a.steps)
                ctl.override = None
                curve.append({"t": t, "agree": float((lg_t.argmax(-1) == am_fp).float().mean()),
                              "cos": float(F.cosine_similarity(lat_t, lat_fp, dim=-1).mean())})
            tc = T_GRID[-1]
            for j in range(1, len(curve)):
                if curve[j]["agree"] < AGREE:
                    a0, a1 = curve[j - 1]["agree"], curve[j]["agree"]
                    t0_, t1_ = curve[j - 1]["t"], curve[j]["t"]
                    tc = t0_ + (t1_ - t0_) * ((a0 - AGREE) / (a0 - a1)) if a0 >= AGREE and a0 != a1 else t0_
                    break
            else:
                tc = T_GRID[-1] if curve[0]["agree"] >= AGREE else 0.0
            rec.update({"curve": curve, "t_c": tc, "rho": (1.0 / tc) if tc > 0 else float("inf"),
                        "return_tf": int(next(c["agree"] for c in curve if c["t"] == 1.0) >= AGREE),
                        "latent_dist_t1": 1 - next(c["cos"] for c in curve if c["t"] == 1.0),
                        "fp_correct": gen_fp[i]["correct"], "q_correct": gen_q[i]["correct"],
                        "h_correct": gen_h[i]["correct"],
                        "return_answer": int(gen_h[i]["pred"] == gen_fp[i]["pred"]),
                        "q_matches_fp": int(gen_q[i]["pred"] == gen_fp[i]["pred"])})
            recs.append(rec)
        acc = lambda g: sum(r["correct"] for r in g) / len(g)  # noqa: E731
        import statistics as st
        summ = {"acc_fp": acc(gen_fp), "acc_q": acc(gen_q), "acc_hybrid": acc(gen_h),
                "retained": acc(gen_q) / acc(gen_fp) if acc(gen_fp) else None,
                "fid": st.mean(r["fid"] for r in recs), "rho_median": st.median(r["rho"] for r in recs),
                "tc_median": st.median(r["t_c"] for r in recs), "return_tf_rate": st.mean(r["return_tf"] for r in recs),
                "return_answer_rate": st.mean(r["return_answer"] for r in recs),
                "state_rel_dist": st.mean(r["state_rel_dist"] for r in recs), "secs": time.time() - t0}
        json.dump({"meta": {**meta, "model": a.model, "quant": q, "n_quantized_linears": nq}, "summary": summ,
                   "gen_fp": gen_fp, "gen_q": gen_q, "gen_hybrid": gen_h, "records": recs}, open(str(path) + ".tmp", "w"))
        os.replace(str(path) + ".tmp", path)
        if not bf_path.exists():
            json.dump({"meta": {**meta, "model": a.model}, "gen_fp": gen_fp}, open(bf_path, "w"))
        log(f"{a.model} {spec}: {json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ.items()})}")
        del model
        torch.cuda.empty_cache()
    log("done")


if __name__ == "__main__":
    main()
