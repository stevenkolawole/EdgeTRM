#!/usr/bin/env python3
"""Breadth grid for EdgeTRM: every recursive checkpoint x every quantizer x
every recursion depth, with puzzle-exact, cell accuracy and carry-trajectory
fidelity (final z_H against FP32 at the SAME depth, same inputs) per cell.

    python breadth_sweep.py --task sudoku_mlp --quants fp32,w8c,w4t,w4a \
        --H 1,3 --nsup 1,4,16 --n 1000 --out ~/breadth/results/sudoku_mlp.json

Tasks (checkpoint, task):
  sudoku_mlp   TRM MLP-mixing, Sudoku-Extreme (community reproduction, 60k epochs)
  sudoku_attn  TRM attention, Sudoku-Extreme (same recipe, attn_type)
  maze_attn    TRM attention, Maze-30x30-Hard (official-style checkpoint)
  hrm_sudoku   HRM official checkpoint, Sudoku-Extreme
  hrm_maze     HRM official checkpoint, Maze-30x30-Hard
Quantizer specs (weights of every CastedLinear; embeddings untouched, as in
the paper): fp32 | w<bits><gran>[+a8]
  gran: t = per-tensor symmetric ("naive"), c = per-channel symmetric,
        a = per-channel asymmetric min/max ("calibrated"), g32/g128 = group-wise
        symmetric along the input dimension.
  +a8:  static per-tensor INT8 activation fake-quant at every linear input,
        scales frozen on the first chunk (the QDQ-toolchain behaviour).
Depth: --H overrides H_cycles (outer cycles per ACT step), --nsup sets
halt_max_steps and the number of ACT steps run (the paper's n_sup).
Every record carries the per-example correctness so any pair of cells can be
compared on the same puzzles.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "TinyRecursiveModels"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HOME = Path.home()
CKPT = HOME / "edgetrm_ckpt"
MAZE_CHARSET = "# SGo"

TRM_TASKS = {
    "sudoku_mlp": dict(ckpt=CKPT / "sudoku_extreme" / "step_39060_sudoku_epoch_60k", data="sudoku",
                       seq=81, vocab=11, H=3, L=6, mlp_t=True, pos="none", batch=256),
    "sudoku_attn": dict(ckpt=CKPT / "sudoku_extreme" / "step_39060_sudoku_60k_epoch_attn_type", data="sudoku",
                        seq=81, vocab=11, H=3, L=6, mlp_t=False, pos="rope", batch=256),
    "maze_attn": dict(ckpt=CKPT / "maze_hard" / "model.pt", data="maze",
                      seq=900, vocab=6, H=3, L=4, mlp_t=False, pos="rope", batch=32),
}
HRM_TASKS = {
    "hrm_sudoku": dict(repo="sapientinc/HRM-checkpoint-sudoku-extreme", data="sudoku", seq=81, batch=256),
    "hrm_maze": dict(repo="sapientinc/HRM-checkpoint-maze-30x30-hard", data="maze", seq=900, batch=32),
}


# ----------------------------------------------------------------------------- data
def load_qa(kind, n):
    from huggingface_hub import hf_hub_download
    if kind == "sudoku":
        p = hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset")
    else:
        p = hf_hub_download("sapientinc/maze-30x30-hard-1k", "test.csv", repo_type="dataset")
    q, a = [], []
    with open(p, newline="") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            qq, aa = row[1], row[2]
            if kind == "sudoku":
                q.append((np.frombuffer(qq.replace(".", "0").encode(), np.uint8) - ord("0")).astype(np.int64) + 1)
                a.append((np.frombuffer(aa.encode(), np.uint8) - ord("0")).astype(np.int64) + 1)
            else:
                q.append(np.array([MAZE_CHARSET.index(c) + 1 for c in qq], dtype=np.int64))
                a.append(np.array([MAZE_CHARSET.index(c) + 1 for c in aa], dtype=np.int64))
            if len(q) >= n:
                break
    return np.stack(q), np.stack(a)


# ----------------------------------------------------------------------------- models
def _clean(sd):
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return {k.replace("_orig_mod.model.", "").replace("model.", ""): v for k, v in sd.items()}


CKPT_OVERRIDE = None   # set from --ckpt: a trained variant's step_N file (all_config.yaml beside it)


def _variant_arch(ckpt):
    """arch fields of a checkpoint trained with TRM's pretrain.py (all_config.yaml in the run dir)."""
    import yaml
    cfg_path = Path(ckpt).parent / "all_config.yaml"
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text()).get("arch", {})


def build_trm(task, H, nsup, batch):
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    t = dict(TRM_TASKS[task])
    ckpt = CKPT_OVERRIDE or t["ckpt"]
    if CKPT_OVERRIDE:
        arch = _variant_arch(ckpt)
        t.update(L=arch.get("L_cycles", t["L"]), mlp_t=arch.get("mlp_t", t["mlp_t"]),
                 pos=arch.get("pos_encodings", t["pos"]))
        heads, plen, exp = arch.get("num_heads", 8), arch.get("puzzle_emb_len", 16), arch.get("expansion", 4)
    else:
        heads, plen, exp = 8, 16, 4
    sd = _clean(torch.load(ckpt, map_location="cpu", weights_only=False))
    hidden = sd["inner.embed_tokens.embedding_weight"].shape[1]
    npid = sd["inner.puzzle_emb.weights"].shape[0]
    cfg = dict(batch_size=batch, seq_len=t["seq"], vocab_size=t["vocab"], num_puzzle_identifiers=npid,
               H_cycles=H, L_cycles=t["L"], H_layers=0, L_layers=2, hidden_size=hidden, expansion=exp,
               num_heads=heads, pos_encodings=t["pos"], halt_max_steps=nsup, halt_exploration_prob=0.1,
               forward_dtype="float32", mlp_t=t["mlp_t"], puzzle_emb_ndim=hidden, puzzle_emb_len=plen,
               no_ACT_continue=True)
    m = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss[:3], unexp[:3])
    return m, {"H_cycles": H, "L_cycles": t["L"], "nsup": nsup}


def build_hrm(task, H, nsup, batch, L=None):
    import yaml
    from huggingface_hub import snapshot_download
    from models.recursive_reasoning.hrm import HierarchicalReasoningModel_ACTV1 as HRM
    t = HRM_TASKS[task]
    snap = snapshot_download(t["repo"])
    ck = glob.glob(os.path.join(snap, "checkpoint*"))[0]
    arch = yaml.safe_load(open(os.path.join(snap, "all_config.yaml")))["arch"]
    sd = _clean(torch.load(ck, map_location="cpu", weights_only=False))
    npid = sd["inner.puzzle_emb.weights"].shape[0]
    vocab = sd["inner.embed_tokens.embedding_weight"].shape[0]
    cfg = dict(batch_size=batch, seq_len=t["seq"], vocab_size=vocab, num_puzzle_identifiers=npid,
               H_cycles=H if H is not None else arch["H_cycles"],
               L_cycles=L if L is not None else arch["L_cycles"],
               H_layers=arch["H_layers"], L_layers=arch["L_layers"], hidden_size=arch["hidden_size"],
               expansion=arch["expansion"], num_heads=arch["num_heads"], pos_encodings=arch["pos_encodings"],
               halt_max_steps=nsup, halt_exploration_prob=arch["halt_exploration_prob"],
               forward_dtype="float32", puzzle_emb_ndim=arch["puzzle_emb_ndim"])
    m = HRM(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss[:3], unexp[:3])
    return m, {"H_cycles": cfg["H_cycles"], "L_cycles": cfg["L_cycles"], "nsup": nsup,
               "arch_default": {"H": arch["H_cycles"], "L": arch["L_cycles"], "halt": arch["halt_max_steps"]}}


def build(task, H, nsup, batch):
    if task in TRM_TASKS:
        return build_trm(task, H if H is not None else TRM_TASKS[task]["H"], nsup, batch)
    return build_hrm(task, H, nsup, batch)


# ----------------------------------------------------------------------------- quantizers
def _linears(m):
    return [mod for mod in m.modules() if mod.__class__.__name__ == "CastedLinear"]


def parse_quant(spec):
    """'fp32' | 'w4t' | 'w4a+a8' | 'w3g128' -> dict."""
    if spec == "fp32":
        return {"spec": spec, "bits": None, "gran": None, "act": None}
    if spec.startswith("noise"):          # Gaussian weight noise, std = sigma x per-tensor weight std
        return {"spec": spec, "bits": None, "gran": "noise", "sigma": float(spec[5:]), "act": None}
    s = spec
    act = None
    if "+a8" in s:
        s, act = s.replace("+a8", ""), 8
    assert s.startswith("w"), spec
    body = s[1:]
    i = 0
    while i < len(body) and body[i].isdigit():
        i += 1
    bits, gran = int(body[:i]), body[i:]
    assert gran in ("t", "c", "a") or gran.startswith("g"), spec
    return {"spec": spec, "bits": bits, "gran": gran, "act": act}


@torch.no_grad()
def quantize_weights_(m, bits, gran, sigma=None):
    for mod in _linears(m):
        W = mod.weight.data
        if gran == "noise":                                 # control: additive Gaussian noise, no rounding
            mod.weight.data = W + torch.randn_like(W) * (sigma * W.std())
            continue
        if gran == "a":                                     # per-channel asymmetric min/max
            qmax = 2 ** bits - 1
            wmin = W.amin(dim=1, keepdim=True)
            wmax = W.amax(dim=1, keepdim=True)
            scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
            zp = torch.round(-wmin / scale)
            mod.weight.data = (torch.clamp(torch.round(W / scale) + zp, 0, qmax) - zp) * scale
            continue
        qmax = 2 ** (bits - 1) - 1
        if gran == "t":
            s = W.abs().max() / qmax
            Wq = torch.round(W / s.clamp_min(1e-8)).clamp(-qmax, qmax) * s.clamp_min(1e-8)
        elif gran == "c":
            s = W.abs().amax(dim=1, keepdim=True) / qmax
            Wq = torch.round(W / s.clamp_min(1e-8)).clamp(-qmax, qmax) * s.clamp_min(1e-8)
        else:                                               # group-wise along input dim
            g = int(gran[1:])
            out_f, in_f = W.shape
            pad = (-in_f) % g
            Wp = F.pad(W, (0, pad)).view(out_f, -1, g)
            s = Wp.abs().amax(dim=2, keepdim=True) / qmax
            Wq = (torch.round(Wp / s.clamp_min(1e-8)).clamp(-qmax, qmax) * s.clamp_min(1e-8)).view(out_f, -1)[:, :in_f]
        mod.weight.data = Wq


class _ActQuant:
    """Static per-tensor symmetric activation fake-quant at a linear's input."""

    def __init__(self, bits):
        self.qmax = 2 ** (bits - 1) - 1
        self.absmax = 0.0
        self.scale = None

    def __call__(self, mod, args):
        x = args[0]
        if self.scale is None:
            self.absmax = max(self.absmax, float(x.detach().abs().max()))
            return None
        xq = torch.round(x / self.scale).clamp(-self.qmax, self.qmax) * self.scale
        return (xq,) + tuple(args[1:])

    def freeze(self):
        self.scale = max(self.absmax, 1e-8) / self.qmax


# ----------------------------------------------------------------------------- run
@torch.no_grad()
def run(m, inp, lab, batch, nsup, act_quant=None, calib_chunks=1):
    """puzzle-exact per example, cell accuracy, final z_H (and z_L) per example."""
    m = m.to(DEV)
    n = inp.shape[0]
    correct, cell_c, cell_n, zH, zL = [], 0, 0, [], []
    # activation calibration: observe on the first chunks, then freeze
    handles = []
    if act_quant is not None:
        obs = []
        for mod in _linears(m):
            q = _ActQuant(act_quant)
            obs.append(q)
            handles.append(mod.register_forward_pre_hook(q))
    for ci, s in enumerate(range(0, n, batch)):
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        B = x.shape[0]
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
        c = m.initial_carry(bt)
        ic = c.inner_carry
        carry = type(c)(inner_carry=type(ic)(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)),
                        steps=c.steps.to(DEV), halted=c.halted.to(DEV),
                        current_data={k: v.to(DEV) for k, v in c.current_data.items()})
        if act_quant is not None and ci < calib_chunks:
            # calibration pass (observe), then freeze and re-run this chunk quantized
            cc, out = carry, None
            for _ in range(nsup):
                cc, out = m(cc, bt)
                if cc.halted.all():
                    break
            if ci == calib_chunks - 1:
                for q in obs:
                    q.freeze()
            if any(q.scale is None for q in obs):
                continue  # still calibrating (only when calib_chunks > 1)
            c = m.initial_carry(bt)
            ic = c.inner_carry
            carry = type(c)(inner_carry=type(ic)(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)),
                            steps=c.steps.to(DEV), halted=c.halted.to(DEV),
                            current_data={k: v.to(DEV) for k, v in c.current_data.items()})
        out = None
        for _ in range(nsup):
            carry, out = m(carry, bt)
            if carry.halted.all():
                break
        preds = out["logits"].argmax(-1)
        mask = y != 0
        corr = (preds == y) & mask
        correct.extend((corr.sum(-1) == mask.sum(-1)).cpu().tolist())
        cell_c += corr.sum().item()
        cell_n += mask.sum().item()
        zH.append(carry.inner_carry.z_H.detach().float().cpu())
        zL.append(carry.inner_carry.z_L.detach().float().cpu())
    for h in handles:
        h.remove()
    m.cpu()
    torch.cuda.empty_cache()
    return np.array(correct, dtype=bool), cell_c / max(cell_n, 1), torch.cat(zH), torch.cat(zL)


def fidelity(z, ref):
    return float(F.cosine_similarity(z.flatten(1), ref.flatten(1), dim=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TRM_TASKS) + list(HRM_TASKS))
    ap.add_argument("--quants", default="fp32,w8c,w4t,w4c,w4a")
    ap.add_argument("--H", default="", help="comma list of H_cycles overrides; empty = checkpoint default")
    ap.add_argument("--nsup", default="16", help="comma list of ACT step counts (halt_max_steps)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default="", help="TRM tasks: evaluate this trained checkpoint (step_N file of a "
                                                "pretrain.py run; heads/mixer/L read from its all_config.yaml)")
    ap.add_argument("--variant", default="", help="label stored in every record (e.g. sudoku_mlp_w256_s0_step6510)")
    a = ap.parse_args()
    global CKPT_OVERRIDE
    if a.ckpt:
        assert a.task in TRM_TASKS, "--ckpt applies to TRM tasks only"
        CKPT_OVERRIDE = a.ckpt
        arch = _variant_arch(a.ckpt)
        print(f"variant checkpoint {a.ckpt}: arch {arch}", flush=True)

    np.random.seed(0)
    torch.manual_seed(0)
    kind = (TRM_TASKS.get(a.task) or HRM_TASKS[a.task])["data"]
    batch = a.batch or (TRM_TASKS.get(a.task) or HRM_TASKS[a.task])["batch"]
    inp, lab = load_qa(kind, a.n)
    Hs = [None] if not a.H else [int(h) for h in a.H.split(",")]
    nsups = [int(x) for x in a.nsup.split(",")]
    quants = [q.strip() for q in a.quants.split(",") if q.strip()]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():
        try:
            records = json.load(open(out_path)).get("records", [])
        except (json.JSONDecodeError, ValueError):
            print(f"  {out_path} unreadable (truncated write); starting fresh", flush=True)
    done = {(r["H_cycles"], r["nsup"], r["quant"]["spec"]) for r in records}
    extra = {"variant": a.variant, "ckpt": a.ckpt} if a.ckpt else {}
    print(f"{a.task}: {len(inp)} puzzles, batch {batch}, H={Hs}, nsup={nsups}, quants={quants}; "
          f"{len(records)} records already", flush=True)

    for H in Hs:
        for nsup in nsups:
            t0 = time.time()
            m, meta = build(a.task, H, nsup, batch)
            Hc = meta["H_cycles"]
            ref_c, ref_cell, ref_zH, ref_zL = run(m, inp, lab, batch, nsup)
            del m
            if (Hc, nsup, "fp32") not in done:
                records.append({"task": a.task, "H_cycles": Hc, "L_cycles": meta["L_cycles"], "nsup": nsup,
                                "quant": parse_quant("fp32"), "n": int(len(inp)),
                                "pexact": float(ref_c.mean()), "cell": ref_cell, "fidelity_H": 1.0,
                                "fidelity_L": 1.0, "correct": ref_c.astype(int).tolist(),
                                "secs": time.time() - t0, **extra})
                print(f"  H={Hc} nsup={nsup} fp32       pexact {ref_c.mean()*100:6.2f} cell {ref_cell*100:6.2f}", flush=True)
            for spec in quants:
                if spec == "fp32" or (Hc, nsup, spec) in done:
                    continue
                q = parse_quant(spec)
                t1 = time.time()
                mq, _ = build(a.task, H, nsup, batch)
                quantize_weights_(mq, q["bits"], q["gran"], q.get("sigma"))
                c, cell, zH, zL = run(mq, inp, lab, batch, nsup, act_quant=q["act"])
                del mq
                fH, fL = fidelity(zH, ref_zH), fidelity(zL, ref_zL)
                rec = {"task": a.task, "H_cycles": Hc, "L_cycles": meta["L_cycles"], "nsup": nsup, "quant": q,
                       "n": int(len(inp)), "pexact": float(c.mean()), "cell": cell, "fidelity_H": fH,
                       "fidelity_L": fL, "correct": c.astype(int).tolist(), "secs": time.time() - t1, **extra}
                records.append(rec)
                print(f"  H={Hc} nsup={nsup} {spec:10s} pexact {c.mean()*100:6.2f} cell {cell*100:6.2f} "
                      f"fidH {fH:.4f} fidL {fL:.4f} ({time.time()-t1:.0f}s)", flush=True)
                json.dump({"task": a.task, "n": int(len(inp)), "records": records}, open(out_path, "w"))
            json.dump({"task": a.task, "n": int(len(inp)), "records": records}, open(out_path, "w"))
    print("wrote", out_path, len(records), "records")


if __name__ == "__main__":
    main()
