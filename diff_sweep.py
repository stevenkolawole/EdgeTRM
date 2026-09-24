#!/usr/bin/env python3
"""X6 control family: does the phase boundary carry over to a diffusion sampler (predicted: weakly)?

google/ddpm-cifar10-32 (diffusers UNet2DModel, 35.7M parameters) under the breadth-sweep weight
perturbations (same formulas and scope rules as deq_sweep.py: every Conv2d/Linear weight, norms and
biases untouched; noise<s> = W + s*std(W)*eps with one eps per --seed shared by every s), sampled with
  ddim<K>  DDIM, eta = 0 (deterministic ODE sampler, no fresh noise), K steps
  ddpm<K>  DDPM ancestral (fresh Gaussian noise injected every step), K steps (leading spacing)
Common random numbers: every cell of one job starts from the same x_T per sample index and (for DDPM)
draws the same per-step noise, so a perturbed sample is paired with the fp32 sample of the same index.

Metrics per (cell, sampler): FID of --n samples against CIFAR-10 train (50k, pytorch-fid Inception
pool3; FID-n with n = --n, biased upward against FID-50k, so compare cells, not papers), FID against the
fp32 samples of the same sampler ("distribution displacement"), per-sample paired pixel RMSE to the
fp32 sample (in [0,1] units), and per-sample paired Inception-feature cosine.

    python diff_sweep.py --specs noise0.05,noise0.1 --seed 0 --n 5000 --samplers ddim10,ddim50,ddpm100 \
        --out ~/x6/results/diff_noise_s0.json
"""
import argparse
import glob
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from deq_sweep import parse_spec, quantize

CACHE = Path(os.environ.get("X6_CACHE", "/data/user_data/skolawol/x6"))
DEV = "cuda"
REPO = "google/ddpm-cifar10-32"


@torch.no_grad()
def perturb_unet_(m, spec, seed):
    q = parse_spec(spec)
    if q["gran"] is None:
        return q
    gen = torch.Generator(device=DEV).manual_seed(int(seed))
    rel = []
    for name, mod in m.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            continue
        W = mod.weight.data
        if q["gran"] == "noise":
            eps = torch.randn(W.shape, generator=gen, device=W.device, dtype=W.dtype)
            Wp = W + q["sigma"] * W.std() * eps
        else:
            Wp = quantize(W, q["bits"], q["gran"])
        rel.append(float((Wp - W).norm() / W.norm().clamp_min(1e-12)))
        mod.weight.data = Wp
    q["weight_rel_err_mean"] = float(np.mean(rel))
    q["n_tensors"] = len(rel)
    return q


def load_unet():
    from diffusers import UNet2DModel
    return UNet2DModel.from_pretrained(REPO).to(DEV).eval()


def make_sched(name):
    from diffusers import DDIMScheduler, DDPMScheduler
    base = DDPMScheduler.from_pretrained(REPO)
    kind, K = name[:4], int(name[4:])
    s = DDIMScheduler.from_config(base.config) if kind == "ddim" else DDPMScheduler.from_config(base.config)
    s.set_timesteps(K)
    return kind, s


@torch.no_grad()
def sample(unet, sampler, n0, n1, seed_base):
    kind, sched = make_sched(sampler)
    g0 = torch.Generator(device=DEV).manual_seed(seed_base * 1_000_003 + n0)
    gs = torch.Generator(device=DEV).manual_seed(seed_base * 1_000_033 + n0 + 7)
    x = torch.randn(n1 - n0, 3, 32, 32, generator=g0, device=DEV)
    for t in sched.timesteps:
        eps = unet(x, t).sample
        if kind == "ddim":
            x = sched.step(eps, t, x, eta=0.0).prev_sample
        else:
            x = sched.step(eps, t, x, generator=gs).prev_sample
    return ((x.clamp(-1, 1) + 1) / 2)          # [0, 1]


class Incep:
    def __init__(self):
        from pytorch_fid.inception import InceptionV3
        self.m = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(DEV).eval()

    @torch.no_grad()
    def __call__(self, x01):
        out = []
        for i in range(0, len(x01), 500):
            out.append(self.m(x01[i:i + 500].to(DEV))[0].squeeze(-1).squeeze(-1).double().cpu())
        return torch.cat(out).numpy()


def stats(feat):
    return feat.mean(0), np.cov(feat, rowvar=False)


def fid(s1, s2):
    from pytorch_fid.fid_score import calculate_frechet_distance
    return float(calculate_frechet_distance(s1[0], s1[1], s2[0], s2[1]))


def cifar_train_stats(incep):
    path = CACHE / "cifar10_train_fid_stats.npz"
    if path.exists():
        d = np.load(path)
        return d["mu"], d["sigma"]
    import pyarrow.parquet as pq
    from PIL import Image
    hf = os.environ.get("HF_HOME", "/data/hf_cache/skolawol")
    pat = hf + "/hub/datasets--uoft-cs--cifar10/snapshots/*/plain_text/train-*.parquet"
    if not glob.glob(pat):
        from huggingface_hub import snapshot_download
        snapshot_download("uoft-cs/cifar10", repo_type="dataset", allow_patterns=["plain_text/train-*"])
    files = sorted(glob.glob(pat))
    assert files, "CIFAR-10 train parquet not found"
    xs = []
    for fpath in files:
        t = pq.read_table(fpath)
        col = [c for c in t.column_names if "im" in c][0]
        for im in t.column(col).to_pylist():
            xs.append(torch.from_numpy(np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"))).permute(2, 0, 1))
    x = torch.stack(xs).float() / 255.0
    assert len(x) == 50000, len(x)
    mu, sigma = stats(incep(x))
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(path, mu=mu, sigma=sigma)
    return mu, sigma


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True)
    ap.add_argument("--seed", type=int, default=0, help="weight-noise direction")
    ap.add_argument("--sample_seed", type=int, default=0, help="x_T and per-step noise (shared by all cells)")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=1000)
    ap.add_argument("--samplers", default="ddim10,ddim25,ddim50,ddim100,ddpm50,ddpm100,ddpm250")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_tf32", action="store_true", help="strict fp32 matmuls/convs (default: TF32 for every "
                                                          "cell alike, fp32 reference included)")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    torch.backends.cudnn.allow_tf32 = not a.no_tf32
    torch.backends.cuda.matmul.allow_tf32 = not a.no_tf32
    torch.backends.cudnn.benchmark = True
    t0 = time.time()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    specs = ["fp32"] + [s for s in a.specs.split(",") if s and s != "fp32"]
    samplers = a.samplers.split(",")
    incep = Incep()
    train_stats = cifar_train_stats(incep)
    print(f"specs {specs}, samplers {samplers}, n {a.n} ({time.time()-t0:.0f}s setup)", flush=True)

    records = []
    if out.exists():
        try:
            records = json.load(open(out))["records"]
        except Exception:
            records = []
    done = {(r["spec"], r["sampler"]) for r in records}
    samp_dir = CACHE / "diff_samples"
    samp_dir.mkdir(parents=True, exist_ok=True)
    for sampler in samplers:
        ref_path = samp_dir / f"fp32_{sampler}_ss{a.sample_seed}_n{a.n}_b{a.batch}_tf{int(not a.no_tf32)}.pt"
        ref_x = None
        for spec in specs:
            if (spec, sampler) in done:
                continue
            t1 = time.time()
            qinfo = None
            cell_path = samp_dir / f"{spec}_s{a.seed}_{sampler}_ss{a.sample_seed}_n{a.n}_b{a.batch}_tf{int(not a.no_tf32)}.pt"
            if spec == "fp32" and ref_path.exists():
                x = torch.load(ref_path).float()
            elif cell_path.exists():
                x = torch.load(cell_path).float()
            else:
                unet = load_unet()
                qinfo = perturb_unet_(unet, spec, a.seed)
                x = torch.cat([sample(unet, sampler, i, min(a.n, i + a.batch), a.sample_seed).cpu()
                               for i in range(0, a.n, a.batch)])
                del unet
                dst = ref_path if spec == "fp32" else cell_path
                tmp = dst.with_suffix(f".tmp{os.getpid()}")
                torch.save(x.half(), tmp)
                os.replace(tmp, dst)                  # atomic: parallel jobs share the fp32 samples
            if ref_x is None:
                ref_x = torch.load(ref_path).float() if spec != "fp32" else x.half().float()
                ref_feat = incep(ref_x)
                ref_stats = stats(ref_feat)
            x = x.half().float()                      # the stored precision, for every cell alike
            feat = incep(x)
            st = stats(feat)
            rmse = (x - ref_x).pow(2).flatten(1).mean(1).sqrt().numpy()
            fn = feat / np.linalg.norm(feat, axis=1, keepdims=True)
            rn = ref_feat / np.linalg.norm(ref_feat, axis=1, keepdims=True)
            cos = (fn * rn).sum(1)
            rec = {"model": REPO, "tf32": not a.no_tf32, "spec": spec, "quant": parse_spec(spec), "seed": a.seed,
                   "sample_seed": a.sample_seed, "sampler": sampler, "n": a.n,
                   "fid_train": fid(st, train_stats), "fid_vs_fp32": fid(st, ref_stats) if spec != "fp32" else 0.0,
                   "rmse_mean": float(rmse.mean()), "rmse_median": float(np.median(rmse)),
                   "rmse_p90": float(np.percentile(rmse, 90)), "featcos_mean": float(cos.mean()),
                   "featcos_median": float(np.median(cos)), "frac_featcos_ge_0.9": float((cos >= 0.9).mean()),
                   "pixel_std": float(x.std()), "secs": time.time() - t1}
            if qinfo is not None:
                rec["quant"] = qinfo
            records.append(rec)
            json.dump({"records": records}, open(out, "w"))
            print(f"  {sampler:8s} {spec:10s} FID {rec['fid_train']:7.2f}  vs-fp32 {rec['fid_vs_fp32']:7.2f}  "
                  f"rmse {rec['rmse_mean']:.4f}  cos {rec['featcos_mean']:.3f}  ({rec['secs']:.0f}s)", flush=True)
    print("wrote", out, len(records), f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
