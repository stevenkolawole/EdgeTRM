#!/usr/bin/env python3
"""M3 activations: which activation tensors of the AI Hub step graph must stay 16-bit when the rest runs
at the vendor's 8-bit per-tensor encoding (QNN HTP A8; Ethos-U's A8 constraint for INT4 weights)?

Graphs: the exact QDQ graphs AI Hub compiles (`stepc`, H=1; w8a16 and w8a8 of the same export, INT8
per-channel weights, AIMET encodings), executed by qdq_torch.QDQRunner (validated against ONNX Runtime).
The allocation keeps a set P of activation tensors at their 16-bit encoding and puts every other one at its
8-bit encoding. P is chosen by a label-free per-tensor sensitivity measured on UNLABELED puzzles (the
`score` split, disjoint from the evaluation split and from X3's calibration puzzles):

  a_dT      delta_T with tensor t alone at 8 bit (rest 16): ||F'(s_T) - F(s_T)|| / ||s_T|| at the 16-bit
            graph's own end state s_T (X2's one-step perturbation; X3's "A" direction)
  a_dTrm    delta_T(all 8) - delta_T(all 8 but t at 16): how much keeping t at 16 removes (X3's "B" direction)
  a_noise   X2's S_t = delta_T / sigma with Gaussian noise (sigma x std) at t, times t's 8-bit rms error in std
            units (step8 / sqrt(12) / std): the delta_T/sigma recipe applied to one activation tensor
  a_kl      KL(p_16 || p) of the answer after the full rollout with t alone at 8 bit
  range     8-bit step / std of t (a data-light range heuristic: coarse encodings first)
  greedy    iterative: from all-8, promote the tensor whose promotion lowers delta_T most (interactions)
  random    random order (several draws)
Group-level versions use wa_collapse's tensor groups (RMSNorm internals, residual, SwiGLU, ...); X3's hand
found fix (B2: every RMSNorm tensor at 16) is the reference allocation.

    python m3_act.py --task sudoku_mlp --graphs ~/m3_graphs --out ~/breadth_results/m3/act_sudoku_mlp.json
Resumable (every evaluated allocation is cached by its key).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from qdq_torch import QDQRunner

RMS = {"rms_sq", "rms_var", "rms_rsqrt", "rms_out", "rms_const", "const"}


def exact_of(preds_last, lab):
    return ((preds_last == lab) | (lab == 0)).all(-1)


def cell_of(preds_last, lab):
    m = lab != 0
    return float(((preds_last == lab) & m).sum() / m.sum())


class Act:
    def __init__(self, task, gdir, device="cuda", batch=100, export=None, n_score=0):
        root = Path(gdir)
        exp = Path(export) if export else root / "export"
        self.task = task
        self.r = QDQRunner(root / task / "infer" / "qdq_w8a16" / "model.onnx",
                           root / task / "infer" / "qdq_w8a8" / "model.onnx", device=device)
        self.D = dict(np.load(exp / f"data_{task}.npz"))
        if n_score:
            for k in ("inp", "lab", "emb", "idx"):
                self.D[f"score_{k}"] = self.D[f"score_{k}"][:n_score]
        self.grp = json.load(open(exp / f"groups_{task}.json"))
        self.T = int(self.D["nsteps"])
        self.names = [t for t in self.r.act_names if (t, 8) in self.r._enc_t]
        self.batch = batch
        # tensor sizes, stds, 8-bit steps (one sample of the scoring split at its initial carry)
        f = {"z_H": torch.as_tensor(self.D["zH0"], device=self.r.dev), "z_L": torch.as_tensor(self.D["zL0"], device=self.r.dev),
             "inputs": torch.as_tensor(self.D["score_inp"][:1], device=self.r.dev).to(torch.int32),
             "puzzle_embedding": torch.as_tensor(self.D["score_emb"][:1], device=self.r.dev)}
        st = self.r.tensor_stats(f)
        self.numel = {t: st[t][0] for t in self.names}
        self.std0 = {t: st[t][1] for t in self.names}
        self.N_el = sum(self.numel.values())

    def init(self, split):
        return (self.D["zH0"], self.D["zL0"], self.D[f"{split}_emb"])

    def run(self, split, act8, nsteps=None, carry0=None, logits=False, idx=None):
        self.r.set_act8(act8)
        inp = self.D[f"{split}_inp"] if idx is None else self.D[f"{split}_inp"][idx]
        init = self.init(split) if idx is None else (self.D["zH0"], self.D["zL0"], self.D[f"{split}_emb"][idx])
        return self.r.rollout(inp, init, nsteps or self.T, batch=self.batch, keep_logits_last=logits, carry0=carry0)

    def frac16(self, keep16):
        return sum(self.numel[t] for t in keep16) / self.N_el


class Scorer:
    """16-bit reference trajectory on the scoring split; one-step perturbations at its end state."""

    def __init__(self, A):
        self.A = A
        allset = set(A.names)
        ref = A.run("score", set(), logits=True)
        self.sT = (ref["zH"], ref["zL"])
        self.lg = ref["logits"]
        one = A.run("score", set(), nsteps=1, carry0=self.sT)
        self.sT1 = (one["zH"], one["zL"])
        self.nrm = np.sqrt((self.sT[0].reshape(len(self.sT[0]), -1) ** 2).sum(1) + (self.sT[1].reshape(len(self.sT[1]), -1) ** 2).sum(1))
        self.all = allset

    def delta(self, act8, noise=None):
        A = self.A
        A.r.act_noise = noise or {}
        o = A.run("score", act8, nsteps=1, carry0=self.sT)
        A.r.act_noise = {}
        dH = (o["zH"] - self.sT1[0]).reshape(len(self.nrm), -1)
        dL = (o["zL"] - self.sT1[1]).reshape(len(self.nrm), -1)
        return float((np.sqrt((dH ** 2).sum(1) + (dL ** 2).sum(1)) / self.nrm).mean())

    def kl(self, act8):
        o = self.A.run("score", act8, logits=True)
        p = torch.log_softmax(torch.as_tensor(self.lg), -1)
        q = torch.log_softmax(torch.as_tensor(o["logits"]), -1)
        k = (p.exp() * (p - q)).sum(-1)
        m = torch.as_tensor(self.A.D["score_lab"] != 0).float()
        return float(((k * m).sum(-1) / m.sum(-1)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--graphs", default=str(Path.home() / "m3_graphs"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--scores", default="a_dT,a_dTrm,a_noise,a_kl,range,greedy")
    ap.add_argument("--sigma0", type=float, default=0.02)
    ap.add_argument("--ks", default="0,5,10,15,20,30,40,60,80,120,160,240,360")
    ap.add_argument("--greedy_max", type=int, default=60)
    ap.add_argument("--random_draws", type=int, default=3)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--export", default="", help="dir with data_<task>.npz and groups_<task>.json (default <graphs>/export)")
    ap.add_argument("--n_score", type=int, default=0, help="use only the first n scoring puzzles (0 = all 64)")
    ap.add_argument("--greedy_cand", type=int, default=0, help="greedy over the top-M tensors by a_dTrm (0 = all)")
    ap.add_argument("--confirm", default="", help="comma list of curve keys (score@k) to re-evaluate on evalbig")
    a = ap.parse_args()
    assert a.device == "cpu" or torch.cuda.is_available(), "NO CUDA"
    torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.load(open(out)) if out.exists() else {}
    blob.setdefault("evals", {})

    def save():
        tmp = out.with_suffix(".tmp")
        json.dump(blob, open(tmp, "w"))
        tmp.replace(out)

    t0 = time.time()
    A = Act(a.task, a.graphs, device=a.device, batch=a.batch, export=a.export or None, n_score=a.n_score)
    names = A.names
    blob.update({"task": a.task, "T": A.T, "n_tensors": len(names), "n_elements": A.N_el,
                 "device": torch.cuda.get_device_name(0) if a.device != "cpu" else "cpu", "n_score": len(A.D["score_inp"]),
                 "numel": A.numel,
                 "group_of": {t: A.grp.get(t, "?") for t in names}})
    print(f"{a.task}: {len(names)} activation tensors ({A.N_el} elements), T={A.T} ({time.time()-t0:.0f}s)", flush=True)

    def evaluate(key, keep16, split="eval", meta=None):
        k2 = f"{split}|{key}"
        if k2 in blob["evals"]:
            return blob["evals"][k2]
        t1 = time.time()
        act8 = set(names) - set(keep16)
        o = A.run(split, act8)
        lab = A.D[f"{split}_lab"]
        ex = exact_of(o["preds"][-1], lab)
        per_step = [float(exact_of(o["preds"][s], lab).mean()) for s in range(A.T)]
        r = {"exact": float(ex.mean()), "cell": cell_of(o["preds"][-1], lab), "n": int(len(lab)),
             "n16": len(set(keep16)), "frac16_el": A.frac16(set(keep16)), "exact_per_step": per_step,
             "correct": ex.astype(int).tolist(), "secs": time.time() - t1, "split": split, **(meta or {})}
        blob["evals"][k2] = r
        save()
        print(f"  [{split}] {key:32s} n16 {r['n16']:3d} ({100*r['frac16_el']:5.1f}% el) exact {100*r['exact']:5.1f} "
              f"cell {100*r['cell']:5.2f} ({r['secs']:.0f}s)", flush=True)
        return r

    # references
    evaluate("all16", names, meta={"method": "ref"})
    evaluate("all8", [], meta={"method": "ref"})
    b2 = [t for t in names if A.grp.get(t) in RMS]
    evaluate("x3_B2_rms16", b2, meta={"method": "x3", "groups": sorted(RMS)})

    S = None
    sc = blob.setdefault("scores", {})
    want = a.scores.split(",")

    def need(k):
        return k in want and k not in sc

    if any(need(k) for k in want):
        S = Scorer(A)
        sc["delta_all8"] = S.delta(set(names))
        sc["delta_all16"] = 0.0
        print(f"  delta_T all-8 {sc['delta_all8']:.4f}", flush=True)
    if need("a_dT"):
        ts = time.time()
        sc["a_dT"] = {t: S.delta({t}) for t in names}
        save()
        print(f"  a_dT done ({time.time()-ts:.0f}s); top {sorted(sc['a_dT'].items(), key=lambda kv: -kv[1])[:5]}", flush=True)
    if need("a_dTrm"):
        ts = time.time()
        base = sc["delta_all8"]
        sc["a_dTrm"] = {t: base - S.delta(set(names) - {t}) for t in names}
        save()
        print(f"  a_dTrm done ({time.time()-ts:.0f}s); top {sorted(sc['a_dTrm'].items(), key=lambda kv: -kv[1])[:5]}", flush=True)
    if need("a_noise"):
        ts = time.time()
        s8 = {t: float(np.asarray(A.r.enc8[t][0]).reshape(-1)[0]) for t in names}
        sens = {t: S.delta(set(), noise={t: a.sigma0}) / a.sigma0 for t in names}
        err = {t: (s8[t] / np.sqrt(12)) / max(A.std0[t], 1e-12) for t in names}
        sc["S_act"], sc["err8_std"] = sens, err
        sc["a_noise"] = {t: sens[t] * err[t] for t in names}
        save()
        print(f"  a_noise done ({time.time()-ts:.0f}s); top {sorted(sc['a_noise'].items(), key=lambda kv: -kv[1])[:5]}", flush=True)
    if need("range"):
        s8 = {t: float(np.asarray(A.r.enc8[t][0]).reshape(-1)[0]) for t in names}
        sc["range"] = {t: s8[t] / max(A.std0[t], 1e-12) for t in names}
        save()
    if need("a_kl"):
        ts = time.time()
        sc["a_kl"] = {t: S.kl({t}) for t in names}
        save()
        print(f"  a_kl done ({time.time()-ts:.0f}s); top {sorted(sc['a_kl'].items(), key=lambda kv: -kv[1])[:5]}", flush=True)
    if need("greedy"):
        ts = time.time()
        keep, trace = [], [sc["delta_all8"]]
        cand = list(names)
        if a.greedy_cand and "a_dTrm" in sc:
            cand = sorted(names, key=lambda t: -sc["a_dTrm"][t])[:a.greedy_cand]
        for it in range(a.greedy_max):
            best, bd = None, float("inf")
            for t in cand:
                d = S.delta(set(names) - set(keep) - {t})
                if d < bd:
                    best, bd = t, d
            keep.append(best)
            cand.remove(best)
            trace.append(bd)
            if it % 5 == 0:
                print(f"    greedy {it+1}: +{best} ({A.grp.get(best)}) delta_T {bd:.4f} ({time.time()-ts:.0f}s)", flush=True)
                sc["greedy_partial"] = {"order": keep, "trace": trace}
                save()
        sc["greedy"] = {"order": keep, "trace": trace}
        sc.pop("greedy_partial", None)
        save()

    # curves: keep the top-k tensors of each score at 16 bit
    ks = [int(k) for k in a.ks.split(",")]
    orders = {}
    for name in ("a_dT", "a_dTrm", "a_noise", "a_kl", "range"):
        if name in sc:
            orders[name] = sorted(names, key=lambda t: -sc[name][t])
    if "greedy" in sc:
        g = sc["greedy"]["order"]
        orders["greedy"] = g + [t for t in sorted(names, key=lambda t: -sc.get("a_dTrm", {}).get(t, 0)) if t not in g]
    rng = np.random.default_rng(7)
    for d in range(a.random_draws):
        orders[f"random{d}"] = [names[i] for i in rng.permutation(len(names))]
    blob["orders"] = orders
    for k in ks:
        for name, order in orders.items():
            if name == "greedy" and k > len(sc["greedy"]["order"]):
                continue
            keep = order[:k]
            r = evaluate(f"{name}@{k}", keep, meta={"method": name, "k": k})
            blob.setdefault("curve", {}).setdefault(name, {})[str(k)] = f"eval|{name}@{k}"
    # label-free delta_T of every evaluated allocation (the success predictor)
    if True:
        S = S or Scorer(A)
        dmap = blob.setdefault("delta_of", {})
        for k in ks:
            for name, order in orders.items():
                key = f"{name}@{k}"
                if key in dmap or (name == "greedy" and k > len(sc["greedy"]["order"])):
                    continue
                dmap[key] = S.delta(set(names) - set(order[:k]))
        dmap["x3_B2_rms16"] = S.delta(set(names) - set(b2))
        save()
    for key in [c for c in a.confirm.split(",") if c]:
        if key == "x3_B2_rms16":
            evaluate(key, b2, split="evalbig", meta={"method": "x3"})
        elif key in ("all16", "all8"):
            evaluate(key, names if key == "all16" else [], split="evalbig", meta={"method": "ref"})
        else:
            name, k = key.split("@")
            evaluate(key, orders[name][:int(k)], split="evalbig", meta={"method": name, "k": int(k)})
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
