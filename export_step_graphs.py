"""Export one iteration of DeepThinking (maze) and MDEQ-Small (ImageNet) as ONNX graphs for AI Hub profiling.

Each graph is exactly the work a device repeats per loop, at batch 1:
  dt_maze33_step   (thought[1,128,72,72], x[1,3,72,72]) -> (thought', logits[1,2,72,72])
                   recall conv on cat(thought, x), the recurrent block, and the output head
                   (the head runs every iteration so a stopping rule can read the answer).
  mdeq_small_step  (z0..z3, x0) -> (z0'..z3'): one evaluation of the fixed-point map f(z; x) at 224x224,
                   the unit a solver repeats (injection x0 = downsample + stage0 of the image).
Calibration inputs (one .npz per graph, for AI Hub quantize jobs) are real states:
DeepThinking thoughts after 1..8 iterations on test mazes; MDEQ iterates z from zero on Gaussian images
(MDEQ calibration therefore sets activation ranges only, which is all a latency profile needs).

    python export_step_graphs.py --dt-ckpt <model_best.pth> --dt-data <maze33_inputs.npy> \
        --mdeq-root <MDEQ-Vision> --mdeq-ckpt <mdeq_small_imagenet.pkl> --out <dir>
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


class DTStep(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, thought, x):
        thought = self.net.recur_block(torch.cat([thought, x], 1))
        return thought, self.net.head(thought)


def export_dt(ckpt, data, out, n_cal):
    from dt_models.dt_net_2d import dt_net_recall_2d
    net = dt_net_recall_2d(128, in_channels=3)
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("net", sd)
    net.load_state_dict({k.replace("module.", ""): v for k, v in sd.items()})
    net.eval()
    x = torch.from_numpy(np.load(data, mmap_mode="r")[:n_cal].astype(np.float32))
    step = DTStep(net).eval()
    cal_t, cal_x = [], []
    with torch.no_grad():
        t = net.projection(x)
        for _ in range(8):
            cal_t.append(t.numpy()), cal_x.append(x.numpy())
            t, _ = step(t, x)
    path = out / "dt_maze33_step.onnx"
    torch.onnx.export(step, (t[:1], x[:1]), path, input_names=["thought", "x"],
                      output_names=["thought_out", "logits"], opset_version=17, dynamo=False)
    np.savez(out / "dt_maze33_step_cal.npz", thought=np.concatenate(cal_t)[:, None].reshape(-1, 1, 128, 72, 72),
             x=np.concatenate(cal_x).reshape(-1, 1, 3, 72, 72))
    return path


class MDEQStep(nn.Module):
    def __init__(self, m, extra_shapes):
        super().__init__()
        self.m = m
        self.extra_shapes = extra_shapes

    def forward(self, z0, z1, z2, z3, x0):
        x_list = [x0] + [torch.zeros(s, dtype=x0.dtype) for s in self.extra_shapes]
        return tuple(self.m.fullstage([z0, z1, z2, z3], x_list))


def export_mdeq(root, ckpt, out, n_cal):
    root = Path(root)
    for p in [str(root.parent), str(root / "lib" / "models"), str(root / "lib")]:
        sys.path.insert(0, p)
    cwd = os.getcwd()
    os.chdir(root)
    try:
        from config import config as CFG
        from models.mdeq import get_cls_net
        CFG.defrost()
        CFG.merge_from_file(str(root / "experiments" / "imagenet" / "cls_mdeq_SMALL.yaml"))
        CFG.freeze()
        m = get_cls_net(CFG)
    finally:
        os.chdir(cwd)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing = [k for k in m.load_state_dict(sd, strict=False).missing_keys if not k.endswith("num_batches_tracked")]
    assert not missing, missing[:5]
    m.eval()
    with torch.no_grad():
        img = torch.randn(n_cal, 3, 224, 224)
        x = m.downsample(img)
        x0 = m.stage0(x) if m.stage0 is not None else x
        shapes = [tuple(x0.shape[1:])]
        h, w = x0.shape[2], x0.shape[3]
        for i in range(1, m.num_branches):
            h, w = h // 2, w // 2
            shapes.append((m.num_channels[i], h, w))
        zs = [torch.zeros(n_cal, *s) for s in shapes]
        m.fullstage._reset(zs)
        step = MDEQStep(m, [(1, *s) for s in shapes[1:]]).eval()
        cal = {f"z{i}": [] for i in range(4)}
        cal["x0"] = []
        xl = [x0] + [torch.zeros(n_cal, *s) for s in shapes[1:]]
        for _ in range(6):
            for i in range(4):
                cal[f"z{i}"].append(zs[i].numpy())
            cal["x0"].append(x0.numpy())
            zs = list(m.fullstage(zs, xl))
        m.fullstage._reset([z[:1] for z in zs])
    path = out / "mdeq_small_step.onnx"
    torch.onnx.export(step, tuple(z[:1] for z in zs) + (x0[:1],), path,
                      input_names=["z0", "z1", "z2", "z3", "x0"], output_names=["z0o", "z1o", "z2o", "z3o"],
                      opset_version=17, dynamo=False)
    np.savez(out / "mdeq_small_step_cal.npz",
             **{k: np.concatenate(v)[:, None].reshape(-1, 1, *v[0].shape[1:]) for k, v in cal.items()})
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt-ckpt"), ap.add_argument("--dt-data")
    ap.add_argument("--mdeq-root"), ap.add_argument("--mdeq-ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-cal", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.dt_ckpt:
        print("wrote", export_dt(a.dt_ckpt, a.dt_data, out, a.n_cal), flush=True)
    if a.mdeq_ckpt:
        print("wrote", export_mdeq(a.mdeq_root, a.mdeq_ckpt, out, a.n_cal), flush=True)


if __name__ == "__main__":
    main()
