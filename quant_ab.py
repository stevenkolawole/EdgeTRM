#!/usr/bin/env python3
"""
Attribute the ARC INT8 gap: 35.25% in the notebook harness against 26.0% on
device, both single-pass, on the same checkpoint at the same recursion depth.

The paper localises the gap to "the quantizer" but never attributes the
magnitude, and that footnote is the largest open hole in EdgeTRM. The harness
number comes from `_fake_quant_(m, 8, per_ch=True)`, which rounds *weights*
per output channel and leaves activations in fp32. The device runs static QDQ,
which also quantizes activations, with per-tensor scales fixed at compile time.

So the hypothesis is not "some quantizer difference": it is that activation
quantization accounts for essentially all 9.3 points. This runs three variants
on identical rows to test it.

    fp32            no quantization                     expect ~36.0
    int8_w_only     weights per-channel symmetric       expect ~35.25 (harness)
    int8_w_and_a    weights + activations, static       expect ~26.0 (device)

If the third lands near 26, the gap is attributed and the paper can say which
half of the quantizer costs the accuracy instead of pointing at the whole thing.

Run on p4d, where arc1concept-aug-1000 already lives:
    python3 EdgeTRM/quant_ab.py --out ~/quant_ab.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

CKPT_DIR = Path.home() / "edgetrm_ckpt" / "arc_v1_public"
DATA = Path.home() / "edgetrm_data" / "arc1concept-aug-1000"
TRM_SRC = Path(__file__).parent / "TinyRecursiveModels"


def _setup():
    if str(TRM_SRC) not in sys.path:
        sys.path.insert(0, str(TRM_SRC))


def _load_sd(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return {k.replace("_orig_mod.model.", "").replace("model.", ""): v
            for k, v in sd.items()}


def _canonical_rows(identifiers):
    """One row per (task, test-pair); the dataset repeats each under augmentation."""
    seen, idx, pids, names = set(), [], [], []
    puzzle_ids = np.load(DATA / "test" / "all__puzzle_identifiers.npy")
    for i, pid in enumerate(puzzle_ids):
        if pid == 0 or pid in seen:
            continue
        seen.add(pid)
        idx.append(i)
        pids.append(pid)
        names.append(identifiers[pid])
    return np.array(idx), np.array(pids), names


def _build(sd, cfg_yaml, batch_size, forward_dtype=None):
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    arch = cfg_yaml["arch"]
    vocab = sd["inner.embed_tokens.embedding_weight"].shape[0]
    hidden = sd["inner.embed_tokens.embedding_weight"].shape[1]
    npid = sd["inner.puzzle_emb.weights"].shape[0]
    cfg = dict(
        batch_size=batch_size, seq_len=900, vocab_size=vocab,
        num_puzzle_identifiers=npid,
        H_cycles=arch["H_cycles"], L_cycles=arch["L_cycles"],
        H_layers=arch["H_layers"], L_layers=arch["L_layers"],
        hidden_size=hidden, expansion=arch["expansion"], num_heads=arch["num_heads"],
        pos_encodings=arch["pos_encodings"], halt_max_steps=arch["halt_max_steps"],
        halt_exploration_prob=arch["halt_exploration_prob"],
        forward_dtype=forward_dtype or arch["forward_dtype"],
        mlp_t=arch.get("mlp_t", False),
        puzzle_emb_ndim=arch["puzzle_emb_ndim"],
        puzzle_emb_len=arch.get("puzzle_emb_len", 16),
        no_ACT_continue=arch.get("no_ACT_continue", True),
    )
    m = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert len(miss) == 0 and len(unexp) == 0, (miss[:5], unexp[:5])
    return m


def _linears(m):
    return [mod for mod in m.modules() if mod.__class__.__name__ == "CastedLinear"]


def _quant_weights_(m, bits=8, per_ch=True):
    """Exactly the harness path: weights only, no activation handling."""
    qmax = 2 ** (bits - 1) - 1
    for mod in _linears(m):
        W = mod.weight.data
        s = (W.abs().amax(dim=1, keepdim=True) if per_ch else W.abs().max()) / qmax
        s = s.clamp_min(1e-8)
        mod.weight.data = torch.round(W / s).clamp(-qmax, qmax) * s


class _ActQuant:
    """Static per-tensor activation fake-quant, the half the harness omits.

    Two phases, matching how a QDQ toolchain actually works: observe ranges on a
    calibration set, freeze the scale, then quantize every later input with it.
    Per-tensor and frozen is the point -- per-token or dynamic scales would be a
    different (and easier) quantizer than the one the NPU runs.
    """

    def __init__(self, bits=8):
        self.qmax = 2 ** (bits - 1) - 1
        self.absmax = 0.0
        self.scale = None
        self.calibrating = True

    def __call__(self, mod, args):
        x = args[0]
        if self.calibrating:
            self.absmax = max(self.absmax, float(x.detach().abs().max()))
            return None
        if self.scale is None:
            return None
        xq = torch.round(x / self.scale).clamp(-self.qmax, self.qmax) * self.scale
        return (xq,) + args[1:]

    def freeze(self):
        self.calibrating = False
        self.scale = max(self.absmax, 1e-8) / self.qmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path.home() / "quant_ab.json"))
    ap.add_argument("--chunk", type=int, default=48)
    ap.add_argument("--calib_rows", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", default="fp32,int8_w_only,int8_w_and_a",
                    help="comma list; run variants in parallel on separate cards with separate --out")
    args = ap.parse_args()

    _setup()
    dev = args.device
    t0 = time.time()

    ckpt = CKPT_DIR / "step_518071"
    cfg_yaml = yaml.safe_load((CKPT_DIR / "all_config.yaml").read_text())
    n_sup = cfg_yaml["arch"]["halt_max_steps"]
    print(f"config: n_sup={n_sup} H={cfg_yaml['arch']['H_cycles']} "
          f"L={cfg_yaml['arch']['L_cycles']} dtype={cfg_yaml['arch']['forward_dtype']}")

    sd = _load_sd(ckpt)
    identifiers = json.loads((DATA / "identifiers.json").read_text())
    idx, pids, names = _canonical_rows(identifiers)
    inputs = np.load(DATA / "test" / "all__inputs.npy", mmap_mode="r")[idx].astype(np.int64)
    labels = np.load(DATA / "test" / "all__labels.npy", mmap_mode="r")[idx].astype(np.int64)
    print(f"canonical rows: {len(idx)}")

    test_puzzles = json.loads((DATA / "test_puzzles.json").read_text())
    pair_inputs = {
        name: [(np.array(p["input"], dtype=np.uint8), np.array(p["output"], dtype=np.uint8))
               for p in puz["test"]]
        for name, puz in test_puzzles.items()
    }

    def crop(tokens):
        g = np.asarray(tokens).reshape(30, 30)
        rows = np.where((g >= 2).any(axis=1))[0]
        cols = np.where((g >= 2).any(axis=0))[0]
        if len(rows) == 0 or len(cols) == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        return (g[: rows.max() + 1, : cols.max() + 1] - 2).astype(np.uint8)

    row_match, seen = [], set()
    for i, name in enumerate(names):
        if name not in pair_inputs:
            continue
        in_grid = crop(inputs[i])
        for j, (pin, _) in enumerate(pair_inputs[name]):
            if pin.shape == in_grid.shape and np.array_equal(pin, in_grid):
                if (name, j) not in seen:
                    seen.add((name, j))
                    row_match.append((i, name, j))
                break
    print(f"matched {len(row_match)} unique (task, test-pair) rows")

    @torch.no_grad()
    def forward_rows(m, lo, hi):
        from models.recursive_reasoning.trm import (
            TinyRecursiveReasoningModel_ACTV1Carry as Carry,
            TinyRecursiveReasoningModel_ACTV1InnerCarry as IC,
        )
        binp = torch.from_numpy(inputs[lo:hi]).to(dev)
        blab = torch.from_numpy(labels[lo:hi]).to(dev)
        bpid = torch.from_numpy(pids[lo:hi]).to(torch.int32).to(dev)
        batch = {"inputs": binp, "labels": blab, "puzzle_identifiers": bpid}
        c = m.initial_carry(batch)
        ic = c.inner_carry
        carry = Carry(inner_carry=IC(z_H=ic.z_H.to(dev), z_L=ic.z_L.to(dev)),
                      steps=c.steps.to(dev), halted=c.halted.to(dev),
                      current_data={k: v.to(dev) for k, v in c.current_data.items()})
        out = None
        for _ in range(n_sup):
            carry, out = m(carry, batch)
            if carry.halted.all():
                break
        return out["logits"].argmax(-1).cpu().numpy()

    @torch.no_grad()
    def evaluate(m):
        m = m.to(dev).eval()
        preds = np.zeros_like(labels)
        for i in range(0, len(inputs), args.chunk):
            preds[i:i + args.chunk] = forward_rows(m, i, min(i + args.chunk, len(inputs)))
        mask = labels != 0
        corr = (preds == labels) & mask
        per_task = {}
        for i, name, j in row_match:
            pred_grid = crop(preds[i])
            gt = pair_inputs[name][j][1]
            per_task.setdefault(name, {})[j] = bool(
                pred_grid.shape == gt.shape and np.array_equal(pred_grid, gt))
        scores = [sum(per_task.get(n, {}).get(j, False) for j in range(len(p))) / len(p)
                  for n, p in pair_inputs.items()]
        return {
            "arc_pass1_single_pass": float(np.mean(scores)),
            "pexact_token": float((corr.sum(-1) == mask.sum(-1)).mean()),
            "cell": float(corr.sum() / mask.sum()),
            "n_tasks": len(pair_inputs),
        }

    def build_w_only():
        m = _build(sd, cfg_yaml, batch_size=args.chunk)
        _quant_weights_(m, 8, per_ch=True)
        return m

    def build_w_and_a():
        """Weights as above, plus static activation quant calibrated on real rows."""
        m = _build(sd, cfg_yaml, batch_size=args.chunk).to(dev).eval()
        _quant_weights_(m, 8, per_ch=True)
        obs, handles = [], []
        for mod in _linears(m):
            q = _ActQuant(8)
            obs.append(q)
            handles.append(mod.register_forward_pre_hook(q))
        n_cal = min(args.calib_rows, len(inputs))
        for i in range(0, n_cal, args.chunk):
            forward_rows(m, i, min(i + args.chunk, n_cal))
        for q in obs:
            q.freeze()
        print(f"  calibrated {len(obs)} linears on {n_cal} rows; "
              f"median scale {np.median([q.scale for q in obs]):.3e}")
        return m

    variants = [
        ("fp32", lambda: _build(sd, cfg_yaml, batch_size=args.chunk)),
        ("int8_w_only", build_w_only),
        ("int8_w_and_a", build_w_and_a),
    ]

    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
    variants = [(n, b) for n, b in variants if n in wanted]
    # Per-variant checkpoint: a 25 h run was lost to a CUDA fault minutes
    # before the host's daily reboot, with nothing written. Finished variants
    # are written to <out>.partial.json and skipped on restart.
    partial = Path(str(args.out) + ".partial.json")
    results = json.loads(partial.read_text()) if partial.exists() else []
    done = {r["variant"] for r in results}
    for vname, builder in variants:
        if vname in done:
            print(f"  {vname:14s} already in {partial}, skipping")
            continue
        t1 = time.time()
        m = builder()
        metrics = evaluate(m)
        rec = {"variant": vname, **metrics, "wall_s": round(time.time() - t1)}
        results.append(rec)
        partial.write_text(json.dumps(results, indent=2))
        print(f"  {vname:14s} pass@1={metrics['arc_pass1_single_pass']:.4f} "
              f"tok_pexact={metrics['pexact_token']:.4f} ({rec['wall_s']}s)", flush=True)
        del m
        torch.cuda.empty_cache()

    payload = {
        "meta": {
            "experiment": "ARC INT8 gap attribution: weight-only vs weight+activation",
            "checkpoint": str(ckpt),
            "dataset": str(DATA),
            "n_sup": n_sup,
            "chunk": args.chunk,
            "calib_rows": args.calib_rows,
            "canonical_rows": int(len(idx)),
            "matched_pairs": len(row_match),
            "wall_s": round(time.time() - t0),
        },
        "results": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
