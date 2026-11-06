#!/usr/bin/env python3
"""Host-side: the DeepThinking maze net (dt_net_recall_2d, width 128, public checkpoint training-rusty-Tayla, via
contraction.build_dt) as board graphs for dt_loop.py, at the maze size of dt_exit.py's size-33 runs (72x72 images).

  dt_proj.onnx   x (1, 3, H, W) -> h0 (1, 128, H, W)                         the input projection, once per maze
  dt_step.onnx   h, x -> h_out = recur_block(cat[h, x]), logits = head(h_out)   one recurrent iteration (dt_exit.run)

Both are exported at opset 13 (static batch 1) and folded with fold_constants.py (<name>_fold.onnx: what
build_engines.sh and dt_loop.py use). Also writes the first --n test mazes (uint8, as dt_exit's data) and the host
reference of dt_loop's check pass: PyTorch fp32 on CPU over --T iterations, per maze and iteration the exact flag,
the residual ||h_t - h_{t-1}|| / ||h_t|| and the number of changed cells (dt_exit's definitions), plus the masked
predictions at --keep iterations. Then checks the folded graphs (onnxruntime CPU, carry fed back) against PyTorch
over --check_n mazes and --T iterations and prints the largest differences.

    python export_dt.py --size 33 --n 50 --T 300 --out assets/dt_mazes
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from contraction import build_dt  # noqa: E402
from dt_exit import DATA  # noqa: E402


class Step(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, h, x):
        h = self.net.recur_block(torch.cat([h, x], 1))
        return h, self.net.head(h)


@torch.no_grad()
def reference(net, x, y, T, keep, batch=10):
    """dt_exit.run's per-iteration records (fp32 kept for res) plus masked predictions at the iterations in keep."""
    n = len(x)
    out = {"correct": np.zeros((n, T), bool), "res": np.zeros((n, T), np.float32), "chg": np.zeros((n, T), np.int32),
           "preds": np.zeros((len(keep), n) + x.shape[2:], np.uint8)}
    for s in range(0, n, batch):
        xb = torch.from_numpy(x[s:s + batch]).float()
        yb = torch.from_numpy(y[s:s + batch]).long().view(len(xb), -1)
        mask = xb.max(1)[0].view(len(xb), -1)
        h = net.projection(xb)
        prevP, prevH = None, h
        for t in range(T):
            h = net.recur_block(torch.cat([h, xb], 1))
            pred = net.head(h).argmax(1).view(len(xb), -1) * mask.long()
            out["correct"][s:s + batch, t] = (pred == yb).all(1).numpy()
            out["res"][s:s + batch, t] = ((h - prevH).flatten(1).norm(dim=1) / h.flatten(1).norm(dim=1).clamp_min(1e-8)).numpy()
            out["chg"][s:s + batch, t] = ((pred != prevP).sum(1) if prevP is not None else mask.sum(1)).numpy()
            if t + 1 in keep:
                out["preds"][keep.index(t + 1), s:s + batch] = pred.view((len(xb),) + x.shape[2:]).numpy()
            prevP, prevH = pred, h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=33)
    ap.add_argument("--n", type=int, default=50, help="test mazes staged for the board (the first n)")
    ap.add_argument("--T", type=int, default=300, help="iterations of the host reference and the check")
    ap.add_argument("--keep", default="30,100,300", help="iterations whose masked predictions the reference keeps")
    ap.add_argument("--check_n", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "assets" / "dt_mazes"))
    a = ap.parse_args()
    od = Path(a.out)
    od.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    x = np.load(DATA / ("maze%d_inputs.npy" % a.size))[:a.n].astype(np.uint8)
    y = np.load(DATA / ("maze%d_solutions.npy" % a.size))[:a.n].astype(np.uint8)
    np.savez_compressed(od / ("mazes%d_0_%d.npz" % (a.size, a.n)), inputs=x, labels=y, index=np.arange(a.n))
    net = build_dt()
    H, W = x.shape[2:]
    xs = torch.zeros(1, 3, H, W)
    hs = torch.zeros(1, net.width, H, W)
    kw = dict(opset_version=13, do_constant_folding=True, dynamo=False)
    torch.onnx.export(net.projection, (xs,), str(od / "dt_proj.onnx"), input_names=["x"], output_names=["h0"], **kw)
    torch.onnx.export(Step(net).eval(), (hs, xs), str(od / "dt_step.onnx"), input_names=["h", "x"],
                      output_names=["h_out", "logits"], **kw)
    subprocess.check_call([sys.executable, str(HERE / "fold_constants.py"), str(od / "dt_proj.onnx"), str(od / "dt_step.onnx")])

    keep = [int(v) for v in a.keep.split(",")]
    t0 = time.time()
    ref = reference(net, x, y, a.T, keep)
    print("host reference: %d mazes x %d iterations in %.0f s; solved at %s: %s" % (
        a.n, a.T, time.time() - t0, keep, [int(ref["correct"][:, k - 1].sum()) for k in keep]))
    np.savez_compressed(od / "host_fp32.npz", keep=np.array(keep), **ref)

    # folded graphs on onnxruntime CPU, carry fed back, against PyTorch over check_n mazes x T iterations
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    proj = ort.InferenceSession(str(od / "dt_proj_fold.onnx"), so, providers=["CPUExecutionProvider"])
    step = ort.InferenceSession(str(od / "dt_step_fold.onnx"), so, providers=["CPUExecutionProvider"])
    worst = {"h0": 0.0, "h": 0.0, "h_rel": 0.0, "logits": 0.0}
    pred_mismatch = exact_mismatch = 0
    ms = []
    with torch.no_grad():
        for i in range(a.check_n):
            xt = torch.from_numpy(x[i:i + 1]).float()
            xn = xt.numpy()
            mask = xn.max(1)[0]
            hp = net.projection(xt)
            ho = proj.run(None, {"x": xn})[0]
            worst["h0"] = max(worst["h0"], float(np.abs(ho - hp.numpy()).max()))
            for t in range(a.T):
                hp = net.recur_block(torch.cat([hp, xt], 1))
                lp = net.head(hp).numpy()
                t1 = time.perf_counter()
                ho, lo = step.run(None, {"h": ho, "x": xn})
                ms.append((time.perf_counter() - t1) * 1e3)
                d = np.abs(ho - hp.numpy()).max()
                worst["h"] = max(worst["h"], float(d))
                worst["h_rel"] = max(worst["h_rel"], float(d / max(np.abs(hp.numpy()).max(), 1e-12)))
                worst["logits"] = max(worst["logits"], float(np.abs(lo - lp).max()))
                po, pp = (lo[0].argmax(0) * mask[0]), (lp[0].argmax(0) * mask[0])
                pred_mismatch += int((po != pp).sum())
                exact_mismatch += int((po == y[i]).all() != (pp == y[i]).all())
    rep = {"check_n": a.check_n, "T": a.T, "max_abs_diff": worst, "pred_cells_differing": pred_mismatch,
           "exact_flags_differing": exact_mismatch, "ort_cpu_ms_per_iter_median": float(np.median(ms)),
           "torch": torch.__version__, "onnxruntime": ort.__version__, "image": [int(H), int(W)]}
    print(json.dumps(rep))
    with open(od / "export_check.json", "w") as f:
        json.dump(rep, f, indent=1)


if __name__ == "__main__":
    main()
