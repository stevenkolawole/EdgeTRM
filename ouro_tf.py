#!/usr/bin/env python3
"""M5 looped-LM control without the generation-diversity confound (coordinator, REWRITE_NOTES 88).

Free generation at different loop counts produces different token sequences, so "right with fewer loops, wrong
with 4" mixes overthinking with ordinary sampling diversity. Here the token sequence is fixed: the gold GSM8K
solution is teacher-forced (prompt + gold reasoning + "#### answer"), and a problem counts as right at a loop
setting when the argmax at every gold answer-token position reproduces the gold answer (looped_decode.tf_pass,
`tf_ans_acc`, X4's likelihood-style score). Settings per (model, weight spec): L = 1, 2, 3, 4 loops (total_ut_steps)
and the learned exit gate at thresholds tau (4 loops computed, per-token exit). One JSON per (model, spec) with
per-problem scores per setting.
Because the answer tokens are nearly always copied once the gold reasoning is given (bf16 smoke: 100% at every L),
the per-token score over the whole gold solution is stored too: for every solution position, whether the argmax
equals the gold token (bit-packed per problem). Token-level visit-then-lose on a fixed input = positions wrong at 4
loops that some L < 4 predicts correctly, the looped-LM analogue of the TRM per-cell measure.

    HOME=/opt/dlami/nvme/edgetrm python ouro_tf.py --model ouro14 --quants bf16 --n 1319 --outdir .../results/m5/ouro_tf
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


@torch.no_grad()
def tf_tokens(fam, model, tok, data):
    """looped_decode.tf_pass's teacher-forced scoring, plus the per-position correctness bits over the solution."""
    recs, bits = [], []
    for i, (q, gold, sol) in enumerate(data):
        ids = tok.encode(fam.prompt_text(tok, q), return_tensors="pt", add_special_tokens=False).to(LD.DEV)
        gtxt = fam.gold_text(tok, sol)
        k = gtxt.index("####") + 4
        tail = tok.eos_token if fam.style == "chat" else "\n\n"
        ans = gtxt[k:len(gtxt) - len(tail)]
        parts = [tok.encode(x, add_special_tokens=False) for x in (gtxt[:k], ans, tail)]
        sol_ids = torch.tensor([parts[0] + parts[1] + parts[2]], device=LD.DEV)
        a0, a1 = len(parts[0]), len(parts[0]) + len(parts[1])
        full = torch.cat([ids, sol_ids], dim=1)
        torch.manual_seed(1234 + i)
        lg = fam.logits(model, full)[ids.shape[1] - 1:-1].float()
        tgt = sol_ids[0]
        lp = F.log_softmax(lg, -1)
        am = lg.argmax(-1)
        ok = (am == tgt)
        recs.append({"nll": float(-lp.gather(1, tgt[:, None]).mean()), "tf_acc": float(ok.float().mean()),
                     "tf_ans_acc": int(bool(ok[a0:a1].all())), "tf_ans_nll": float(-lp[a0:a1].gather(1, tgt[a0:a1, None]).mean())})
        bits.append(np.packbits(ok.cpu().numpy().astype(np.uint8)).tobytes().hex() + f":{int(tgt.numel())}")
    return recs, bits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ouro14", choices=["ouro14", "ouro26"])
    ap.add_argument("--quants", default="bf16")
    ap.add_argument("--loops", default="1,2,3,4")
    ap.add_argument("--taus", default="0.7,0.5")
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    print("device", torch.cuda.get_device_name(0), flush=True)
    fam = LD.family(a.model)
    data = L.load_gsm8k(a.n)
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    for spec in [q for q in a.quants.split(",") if q]:
        path = od / f"{a.model}_{spec}_tf.json"
        res = json.load(open(path)) if path.exists() else {"model": a.model, "spec": spec, "n": len(data), "settings": {}}
        runs = [(int(x), None) for x in a.loops.split(",") if x] + [(4, float(t)) for t in a.taus.split(",") if t]
        for steps, tau in runs:
            key = f"L{steps}" if tau is None else f"tau{tau}"
            if key in res["settings"]:
                print(f"{spec} {key}: done", flush=True)
                continue
            t0 = time.time()
            model, tok = fam.build(steps)
            LD.quantize_(fam, model, L.parse_quant(spec), "core")
            # fixed L: read loop L exactly (exit_at_step overrides the threshold); gate: per-token threshold exit
            model.early_exit_step = (steps - 1) if tau is None else None
            model.early_exit_threshold = 1.0 if tau is None else tau
            recs, bits = tf_tokens(fam, model, tok, data)
            res["settings"][key] = {"tf_ans_acc": [r["tf_ans_acc"] for r in recs], "tf_ans_nll": [round(r["tf_ans_nll"], 4) for r in recs],
                                    "nll": [round(r["nll"], 4) for r in recs], "tok_bits": bits,
                                    "tok_acc": float(np.mean([r["tf_acc"] for r in recs])), "secs": time.time() - t0}
            json.dump(res, open(str(path) + ".tmp", "w"))
            os.replace(str(path) + ".tmp", path)
            acc = sum(res["settings"][key]["tf_ans_acc"]) / len(recs)
            print(f"{a.model} {spec} {key}: teacher-forced answer accuracy {100 * acc:.1f} ({time.time() - t0:.0f}s)", flush=True)
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
