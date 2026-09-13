#!/usr/bin/env python3
"""Third recursive family: DeepThinking networks (Bansal, Schwarzschild et al.
2022; public checkpoints) on prefix sums, mazes and chess puzzles, under the
same weight quantizers as the puzzle solvers, across test-time iteration
counts, with the carry-trajectory fidelity read from the recurrent feature
map.

    cd ~/breadth/deep-thinking && ~/venvs/kvcache/bin/python ~/edgetrm_src/dt_breadth.py \
        --task mazes --sizes 9,13,33 --iters 10,20,30,40,60 --quants fp32,w8c,w4t,w4a,w4g128 \
        --n 1000 --out ~/breadth/results/dt_mazes.json

Tasks: prefix_sums (bits: 32 in-distribution, 512 the checkpoint's test
size), mazes (size 9 in-distribution, 33 the checkpoint's test size), chess
(puzzle index ranges; 600k-700k is the checkpoint's test range).
Quantizers act on every Conv (1d/2d) weight of the network: bits x
granularity (t per-tensor, c per-output-channel, a per-channel asymmetric,
gN group-wise along the flattened input).
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
from omegaconf import OmegaConf

HOME = Path.home()
DT = HOME / "breadth" / "deep-thinking"
sys.path.insert(0, str(DT))
DEV = "cuda"
CKPTS = {"prefix_sums": DT.parent / "dt_ckpts" / "training-roupy-Ambr",
         "mazes": DT.parent / "dt_ckpts" / "training-rusty-Tayla",
         "chess": DT.parent / "dt_ckpts" / "training-mansard-Janean"}
IN_CH = {"prefix_sums": 1, "mazes": 3, "chess": 12}
DATA_ROOT = str(DT.parent / "dt_data")


def dataset(task, size):
    import easy_to_hard_data as e
    if task == "mazes":
        return e.MazeDataset(DATA_ROOT, train=False, size=int(size), download=True)
    if task == "prefix_sums":
        return e.PrefixSumDataset(DATA_ROOT, num_bits=int(size), download=True)
    from deepthinking.utils.chess_data import FlippedChessPuzzleDataset
    hi = int(size)
    return FlippedChessPuzzleDataset(DATA_ROOT, idx_start=hi - 100000, idx_end=hi, who_moves=False, download=True)


def build(task):
    from deepthinking.utils.tools import get_model
    cfg = OmegaConf.load(CKPTS[task] / ".hydra" / "config.yaml")
    m = cfg.problem.model
    net = get_model(m.model, m.width, m.max_iters, in_channels=IN_CH[task])
    sd = torch.load(CKPTS[task] / "model_best.pth", map_location="cpu")
    sd = sd.get("net", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    net.load_state_dict(sd)
    return net.eval(), {"model": m.model, "width": int(m.width), "max_iters": int(m.max_iters)}


def parse_quant(spec):
    if spec == "fp32":
        return {"spec": spec, "bits": None, "gran": None}
    body = spec[1:]
    i = 0
    while i < len(body) and body[i].isdigit():
        i += 1
    return {"spec": spec, "bits": int(body[:i]), "gran": body[i:]}


@torch.no_grad()
def quantize_(net, bits, gran):
    n = 0
    for mod in net.modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Conv2d)):
            W = mod.weight.data
            o = W.shape[0]
            Wf = W.reshape(o, -1)
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
                    i_ = Wf.shape[1]
                    pad = (-i_) % g
                    Wp = F.pad(Wf, (0, pad)).view(o, -1, g)
                    s = (Wp.abs().amax(2, keepdim=True) / qmax).clamp_min(1e-8)
                    Wq = (torch.round(Wp / s).clamp(-qmax, qmax) * s).view(o, -1)[:, :i_]
            mod.weight.data = Wq.reshape(W.shape)
            n += 1
    return n


@torch.no_grad()
def run(net, loader, task, iters, batch_ref=None):
    """accuracy at every iteration count in `iters`, per example (bool matrix),
    and the recurrent feature map at each of those counts (for fidelity)."""
    from deepthinking.utils.testing import get_predicted
    net = net.to(DEV)
    max_it = max(iters)
    thoughts = []
    hook = net.recur_block.register_forward_hook(lambda m, i, o: thoughts.append(o))
    correct = {k: [] for k in iters}
    feats = {k: [] for k in iters}
    for inputs, targets in loader:
        thoughts.clear()
        inputs, targets = inputs.to(DEV), targets.to(DEV)
        all_out = net(inputs, iters_to_do=max_it)
        tg = targets.view(targets.size(0), -1)
        for k in iters:
            pred = get_predicted(inputs, all_out[:, k - 1], task)
            correct[k].extend(torch.amin(pred == tg, dim=[1]).cpu().tolist())
            feats[k].append(thoughts[k - 1].flatten(1).float().cpu())
    hook.remove()
    net.cpu()
    torch.cuda.empty_cache()
    return {k: np.array(v, dtype=bool) for k, v in correct.items()}, {k: torch.cat(v) for k, v in feats.items()}


def fidelity(z, ref):
    return float(F.cosine_similarity(z, ref, dim=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(CKPTS))
    ap.add_argument("--sizes", required=True, help="maze sizes / prefix-sum bits / chess idx_end, comma list")
    ap.add_argument("--iters", default="10,20,30,40,60")
    ap.add_argument("--quants", default="fp32,w8c,w4t,w4a,w4g128")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    iters = [int(x) for x in a.iters.split(",")]
    quants = [q for q in a.quants.split(",") if q]
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():
        try:
            records = json.load(open(out_path))["records"]
        except (json.JSONDecodeError, ValueError):
            print(f"  {out_path} unreadable (truncated write); starting fresh", flush=True)
    done = {(r["size"], r["iters"], r["quant"]["spec"]) for r in records}
    for size in a.sizes.split(","):
        ds = dataset(a.task, size)
        sub = torch.utils.data.Subset(ds, list(range(min(a.n, len(ds)))))
        loader = torch.utils.data.DataLoader(sub, batch_size=a.batch, shuffle=False)
        net, meta = build(a.task)
        t0 = time.time()
        ref_c, ref_f = run(net, loader, a.task, iters)
        for k in iters:
            if (size, k, "fp32") in done:
                continue
            records.append({"task": a.task, "size": size, "iters": k, "quant": parse_quant("fp32"), **meta,
                            "n": int(len(ref_c[k])), "acc": float(ref_c[k].mean()), "fidelity": 1.0,
                            "correct": ref_c[k].astype(int).tolist(), "secs": time.time() - t0})
        print(f"{a.task} size={size}: fp32 " + " ".join(f"it{k}={ref_c[k].mean()*100:.1f}" for k in iters), flush=True)
        json.dump({"task": a.task, "records": records}, open(out_path, "w"))
        for spec in quants:
            if spec == "fp32" or all((size, k, spec) in done for k in iters):
                continue
            q = parse_quant(spec)
            net, meta = build(a.task)
            nq = quantize_(net, q["bits"], q["gran"])
            t1 = time.time()
            c, f = run(net, loader, a.task, iters)
            for k in iters:
                records.append({"task": a.task, "size": size, "iters": k, "quant": q, **meta, "n": int(len(c[k])),
                                "acc": float(c[k].mean()), "fidelity": fidelity(f[k], ref_f[k]),
                                "correct": c[k].astype(int).tolist(), "n_quantized_convs": nq,
                                "secs": time.time() - t1})
            print(f"  {spec:8s} " + " ".join(f"it{k}={c[k].mean()*100:.1f}/f{fidelity(f[k], ref_f[k]):.3f}" for k in iters),
                  flush=True)
            json.dump({"task": a.task, "records": records}, open(out_path, "w"))
    print("wrote", out_path, len(records))


if __name__ == "__main__":
    main()
