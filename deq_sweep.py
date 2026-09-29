#!/usr/bin/env python3
"""X6 (angle A transfer): does the phase boundary carry over to deep equilibrium models?

Published MDEQ ImageNet classifiers (locuslab/deq, MDEQ-Vision; MDEQ-Small / MDEQ-XL .pkl) under the
breadth-sweep perturbations, with the solver iteration budget varied:

    python deq_sweep.py --model small --specs noise0.02,...,noise1.0 --seed 0 --n 5000 \
        --out ~/x6/results/deq_small_noise_s0.json

Perturbations (same formulas as breadth_sweep.quantize_weights_, applied to every Conv2d/Linear weight
of the whole network, weight-norm folded into the effective weight first; norm layers and biases
untouched, as TRM's norms are untouched):
  noise<s>  W + s * std(W) * eps, eps ~ N(0, 1) drawn from one generator seeded by --seed, walked over
            the tensors in module order. The same eps is used for every s of one seed (common random
            numbers: one seed = one random direction in weight space, s = the distance along it).
  w8c / w4c per-output-channel symmetric; w4t per-tensor symmetric; w4g32 / w4g128 groups of the
            flattened (in, kh, kw) input dimension; w4a per-channel asymmetric min/max.
Solvers, both run per image batch from z0 = 0 to --T function evaluations, snapshotted at --budgets:
  broyden  the repo's Broyden (lib/solvers.broyden, line search off, as the MDEQ configs use), ported
           with per-sample bookkeeping: the returned iterate at budget k is each sample's lowest-residual
           iterate within the first k steps (the repo keeps the batch-level lowest); no early stop.
  picard   plain fixed-point iteration z <- f(z); the returned iterate at budget k is the k-th iterate
           (the TRM-style unrolled recursion).
Per (cell, solver, budget), per image: top-1/top-5 correctness, relative residual ||f(z)-z||/||f(z)||,
displacement ||z - z_ref|| / ||z_ref|| from the fp32 equilibrium z_ref (fp32 Broyden, per-sample lowest
residual over T steps), and cosine to the fp32 state at the same solver and budget (the breadth-sweep
"fidelity"). Per (cell, solver): the per-image residual trace over all T steps.
Output: JSON with aggregates (+ AUROC of the residual as a wrong-answer detector) and an .npz with the
per-image arrays. Resumable: accumulators are checkpointed to --state every --ckpt_every batches.

E6 (MLSys, perturb every step): --wsample dither[,gdither] --dseeds K adds, for every quantizer spec, the arms
<spec>@dither<k> (every Conv2d/Linear effective weight re-drawn by stochastic rounding from the master onto the
spec's grid BEFORE EVERY FUNCTION EVALUATION f(z) of the solver: draw 0 before the injection / evaluation 0, draw e
before evaluation e; wdither.sr_round, the m2_bestofk formula) and <spec>@gdither<k> (the control: master + fresh
zero-mean Gaussian at the SR variance, no lattice). Master = fp32 or an RTN spec (--master w8c). Seeds as
m2_bestofk (wdither.draw_seed(17, k, e)); sequence k is one weight trajectory shared by every image. The classifier
head is read at every budget with the draw current at the end of the solve (draw T). Record layout unchanged; the
record and array keys carry the arm name, the record adds base / wsample / dseed / master. --limit N keeps the
first N cached images (smoke).
"""
import argparse
import glob
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wdither import make_hook, parse_arm  # noqa: E402

DEQ_ROOT = Path(os.environ.get("DEQ_ROOT", Path.home() / "x6" / "deq"))
MV = DEQ_ROOT / "MDEQ-Vision"
CKPT_DIR = Path(os.environ.get("X6_CKPT", "/data/hf_cache/skolawol/x6/ckpt"))
CACHE = Path(os.environ.get("X6_CACHE", "/data/user_data/skolawol/x6"))
DEV = "cuda"
MODELS = {"small": ("cls_mdeq_SMALL.yaml", "mdeq_small_imagenet.pkl"),
          "xl": ("cls_mdeq_XL.yaml", "mdeq_xl_imagenet.pkl")}
TAUS = [1e-1, 3e-2, 1e-2, 3e-3, 1e-3]


# ----------------------------------------------------------------------------- model
def load_mdeq(which):
    for p in [str(DEQ_ROOT), str(MV / "lib" / "models"), str(MV / "lib")]:
        if p not in sys.path:
            sys.path.insert(0, p)
    cwd = os.getcwd()
    os.chdir(MV)                                   # the repo appends relative paths at import time
    try:
        from config import config as CFG
        import models  # noqa: F401  (MDEQ-Vision/lib/models)
        from models.mdeq import get_cls_net
        CFG.defrost()
        CFG.merge_from_file(str(MV / "experiments" / "imagenet" / MODELS[which][0]))
        CFG.freeze()
        m = get_cls_net(CFG)
    finally:
        os.chdir(cwd)
    sd = torch.load(CKPT_DIR / MODELS[which][1], map_location="cpu", weights_only=False)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    assert not missing, f"missing keys: {missing[:10]}"
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:4]}", flush=True)
    return m.to(DEV).eval()


def _wn_norm(p):
    return p.contiguous().view(p.size(0), -1).norm(dim=1).view(p.size(0), *([1] * (p.dim() - 1)))


def weight_slots(m):
    """Every Conv2d/Linear weight as (name, get, set) on the effective weight (weight norm folded)."""
    slots = []
    for name, mod in m.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            continue
        if hasattr(mod, "weight_v"):
            def get(mod=mod):
                return mod.weight_v.data * (mod.weight_g.data / _wn_norm(mod.weight_v.data))

            def put(W, mod=mod):                       # g * v / ||v|| == W exactly; a row that
                g = _wn_norm(W)                            # quantized to all zeros gets g = 0 and a
                v = W.clone()                              # dummy nonzero v (else 0/0 = NaN)
                v[g.flatten() == 0] = 1.0
                mod.weight_v.data = v
                mod.weight_g.data = g
        else:
            def get(mod=mod):
                return mod.weight.data.clone()

            def put(W, mod=mod):
                mod.weight.data = W
        slots.append((name, get, put))
    return slots


def parse_spec(spec):
    if spec == "fp32":
        return {"spec": spec, "bits": None, "gran": None}
    if spec.startswith("noise"):
        return {"spec": spec, "bits": None, "gran": "noise", "sigma": float(spec[5:])}
    body = spec[1:]
    i = 0
    while i < len(body) and body[i].isdigit():
        i += 1
    bits, gran = int(body[:i]), body[i:]
    assert gran in ("t", "c", "a") or gran.startswith("g"), spec
    return {"spec": spec, "bits": bits, "gran": gran}


@torch.no_grad()
def quantize(W, bits, gran):
    """breadth_sweep.quantize_weights_ on W viewed as (out, fan_in)."""
    shape = W.shape
    W = W.reshape(shape[0], -1)
    if gran == "a":
        qmax = 2 ** bits - 1
        wmin, wmax = W.amin(dim=1, keepdim=True), W.amax(dim=1, keepdim=True)
        scale = ((wmax - wmin) / qmax).clamp_min(1e-8)
        zp = torch.round(-wmin / scale)
        return ((torch.clamp(torch.round(W / scale) + zp, 0, qmax) - zp) * scale).reshape(shape)
    qmax = 2 ** (bits - 1) - 1
    if gran == "t":
        s = (W.abs().max() / qmax).clamp_min(1e-8)
        Wq = torch.round(W / s).clamp(-qmax, qmax) * s
    elif gran == "c":
        s = (W.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-8)
        Wq = torch.round(W / s).clamp(-qmax, qmax) * s
    else:
        g = int(gran[1:])
        out_f, in_f = W.shape
        Wp = F.pad(W, (0, (-in_f) % g)).view(out_f, -1, g)
        s = (Wp.abs().amax(dim=2, keepdim=True) / qmax).clamp_min(1e-8)
        Wq = (torch.round(Wp / s).clamp(-qmax, qmax) * s).view(out_f, -1)[:, :in_f]
    return Wq.reshape(shape)


@torch.no_grad()
def perturb_(m, spec, seed):
    q = parse_spec(spec)
    if q["gran"] is None:
        return q
    gen = torch.Generator(device=DEV).manual_seed(int(seed))
    rel = []
    for name, get, put in weight_slots(m):
        W = get()
        if q["gran"] == "noise":
            eps = torch.randn(W.shape, generator=gen, device=W.device, dtype=W.dtype)
            Wp = W + q["sigma"] * W.std() * eps
        else:
            Wp = quantize(W, q["bits"], q["gran"])
        rel.append(float((Wp - W).norm() / W.norm().clamp_min(1e-12)))
        put(Wp)
    q["weight_rel_err_mean"] = float(np.mean(rel))
    q["n_tensors"] = len(rel)
    return q


# ----------------------------------------------------------------------------- DEQ pieces
def prepare(m, x):
    """MDEQNet._forward up to the solver: injection, zero init, the map f."""
    from lib.layer_utils import list2vec, vec2list
    x = m.downsample(x)
    x_list = [m.stage0(x) if m.stage0 is not None else x]
    for i in range(1, m.num_branches):
        b, _, H, W = x_list[-1].shape
        x_list.append(torch.zeros(b, m.num_channels[i], H // 2, W // 2, device=x.device, dtype=x.dtype))
    z_list = [torch.zeros_like(e) for e in x_list]
    cutoffs = [(e.size(1), e.size(2), e.size(3)) for e in z_list]
    m.fullstage._reset(z_list)
    f = lambda z: list2vec(m.fullstage(vec2list(z, cutoffs), x_list))  # noqa: E731
    return f, list2vec(z_list), cutoffs


def _dithered(f, hook):
    """Wrap the DEQ map so that evaluation e (1-based within one solve) runs after re-draw hook(e)."""
    cnt = [0]

    def g(z):
        if cnt[0] > 0:
            hook(cnt[0])
        cnt[0] += 1
        return f(z)
    return g


def head(m, z, cutoffs):
    from lib.layer_utils import vec2list
    return m.predict(vec2list(z, cutoffs))


def _rel(fz, z):
    b = z.shape[0]
    return (fz - z).reshape(b, -1).norm(dim=1) / (fz.reshape(b, -1).norm(dim=1) + 1e-9)


def _clean(t):
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


@torch.no_grad()
def solve(f, z0, T, snaps, method):
    """Returns ({k: (z_k, res_k)}, trace (B, T+1) of the current iterate's residual)."""
    from lib.solvers import matvec, rmatvec
    B = z0.shape[0]
    out, trace = {}, torch.full((B, T + 1), float("nan"), device=z0.device)
    x = z0
    fx = f(x)
    gx = fx - x
    r = _rel(fx, x)
    trace[:, 0] = r
    if method == "picard":
        for k in range(1, T + 1):
            x = fx
            fx = f(x)
            r = _rel(fx, x)
            r = torch.where(torch.isfinite(r), r, torch.full_like(r, float("inf")))
            trace[:, k] = r
            if k in snaps:
                out[k] = (_clean(x).clone(), r.clone())
        return out, trace
    assert method == "broyden"
    best_x, best_r = x.clone(), r.clone()
    D, L = z0.shape[1], z0.shape[2]
    Us = torch.zeros(B, D, L, T, device=z0.device)
    VTs = torch.zeros(B, T, D, L, device=z0.device)
    update = gx                                     # -matvec(empty, gx) = gx
    for k in range(1, T + 1):
        x_new = x + update
        fx = f(x_new)
        gx_new = fx - x_new
        dx, dg = x_new - x, gx_new - gx
        x, gx = x_new, gx_new
        r = _rel(fx, x)
        r = torch.where(torch.isfinite(r), r, torch.full_like(r, float("inf")))
        trace[:, k] = r
        better = r < best_r
        best_x[better] = x[better]
        best_r = torch.where(better, r, best_r)
        if k in snaps:
            out[k] = (best_x.clone(), best_r.clone())
        if k == T:
            break
        pU, pV = Us[:, :, :, :k - 1], VTs[:, :k - 1]
        vT = rmatvec(pU, pV, dx)
        u = (dx - matvec(pU, pV, dg)) / torch.einsum("bij, bij -> b", vT, dg)[:, None, None]
        vT = _clean(vT)
        u = _clean(u)
        VTs[:, k - 1] = vT
        Us[:, :, :, k - 1] = u
        update = _clean(-matvec(Us[:, :, :, :k], VTs[:, :k], gx))
    return out, trace


# ----------------------------------------------------------------------------- data
def load_imagenet_orig(per):
    """Original ILSVRC2012 val JPEGs read straight from the mlx-vision/imagenet-1k val.zip (no unzip):
    per class (synsets sorted = the standard label order) the first `per` files in sorted name order."""
    import zipfile
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()])
    hf = os.environ.get("HF_HOME", "/data/hf_cache/skolawol")
    zp = glob.glob(hf + "/hub/datasets--mlx-vision--imagenet-1k/snapshots/*/val.zip")
    if not zp:
        from huggingface_hub import hf_hub_download
        zp = [hf_hub_download("mlx-vision/imagenet-1k", "val.zip", repo_type="dataset")]
    z = zipfile.ZipFile(zp[0])
    names = [nm for nm in z.namelist() if nm.lower().endswith((".jpeg", ".jpg", ".png"))
             and "__MACOSX" not in nm and not nm.split("/")[-1].startswith("._")]
    print(f"  val.zip: {len(names)} images, e.g. {names[:2]}", flush=True)
    by = {}
    for nm in names:
        syn = [part for part in nm.split("/") if part.startswith("n") and len(part) == 9 and part[1:].isdigit()]
        assert syn, f"no synset directory in {nm}"
        by.setdefault(syn[0], []).append(nm)
    syns = sorted(by)
    assert len(syns) == 1000, len(syns)
    xs, ys = [], []
    for y, sn in enumerate(syns):
        for nm in sorted(by[sn])[:per]:
            with z.open(nm) as fh:
                xs.append(tf(Image.open(fh).convert("RGB")))
            ys.append(y)
    return torch.stack(xs), torch.tensor(ys)


def load_imagenet_val(n, source="resized"):
    """Stratified val subset: n/1000 images per class, preprocessed as tools/cls_valid.py (Resize 256,
    CenterCrop 224, ImageNet normalisation); uint8 cache on /data. source 'resized' = the
    evanarlian/imagenet_1k_resized_256 parquet (re-encoded JPEG), 'orig' = the original val JPEGs."""
    per = n // 1000
    cache = CACHE / (f"imagenet_val_{per}pc.pt" if source == "resized" else f"imagenet_origval_{per}pc.pt")
    if cache.exists():
        d = torch.load(cache)
        return d["x"], d["y"]
    if source == "orig":
        x, y = load_imagenet_orig(per)
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".tmp{os.getpid()}")
        torch.save({"x": x, "y": y}, tmp)
        os.replace(tmp, cache)
        return x, y
    import pyarrow.parquet as pq
    from PIL import Image
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()])
    hf = os.environ.get("HF_HOME", "/data/hf_cache/skolawol")
    files = sorted(glob.glob(hf + "/hub/datasets--evanarlian--imagenet_1k_resized_256/snapshots/*/data/val-*.parquet") or
                   glob.glob(hf + "/datasets--evanarlian--imagenet_1k_resized_256/snapshots/*/data/val-*.parquet"))
    assert files, "ImageNet val parquet not found"
    counts = np.zeros(1000, int)
    xs, ys = [], []
    for fpath in files:
        pf = pq.ParquetFile(fpath)
        for rb in pf.iter_batches(batch_size=1000, columns=["image", "label"]):
            labels = rb.column("label").to_pylist()
            imgs = rb.column("image").to_pylist()
            for im, y in zip(imgs, labels):
                if counts[y] >= per:
                    continue
                counts[y] += 1
                xs.append(tf(Image.open(io.BytesIO(im["bytes"])).convert("RGB")))
                ys.append(y)
    x, y = torch.stack(xs), torch.tensor(ys)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(f".tmp{os.getpid()}")
    torch.save({"x": x, "y": y}, tmp)
    os.replace(tmp, cache)                         # atomic: jobs starting together build it in parallel
    return x, y


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def auroc(score, wrong):
    """P(score of a wrong item > score of a right item); ties count half."""
    score, wrong = np.asarray(score, float), np.asarray(wrong, bool)
    if wrong.all() or (~wrong).all():
        return None
    from scipy.stats import rankdata
    rk = rankdata(score)
    n1, n0 = wrong.sum(), (~wrong).sum()
    return float((rk[wrong].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", choices=list(MODELS))
    ap.add_argument("--specs", required=True, help="comma list; fp32 is always the reference cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--T", type=int, default=100)
    ap.add_argument("--budgets", default="1,2,4,8,12,16,26,40,60,100")
    ap.add_argument("--solvers", default="broyden,picard")
    ap.add_argument("--data", default="resized", choices=["resized", "orig"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--state", default="")
    ap.add_argument("--ckpt_every", type=int, default=10)
    ap.add_argument("--wsample", default="none", help="E6: comma list of dither,gdither (per-evaluation re-draw arms) or none")
    ap.add_argument("--dseeds", type=int, default=3, help="E6: number of re-draw sequences per arm")
    ap.add_argument("--master", default="fp32", help="E6: fp32 or an RTN spec (e.g. w8c) the draws are made from")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N cached images (smoke); 0 = all")
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    budgets = sorted(int(b) for b in a.budgets.split(","))
    assert budgets[-1] == a.T
    solvers = a.solvers.split(",")
    qspecs = ["fp32"] + [s for s in a.specs.split(",") if s and s != "fp32"]
    wsamples = [w for w in a.wsample.split(",") if w and w != "none"]
    assert all(w in ("dither", "gdither") for w in wsamples), a.wsample
    msuf = "" if a.master == "fp32" else f"_m{a.master}"
    specs = list(qspecs)                                 # arm names: fp32, RTN specs, then <spec>@<arm><k>[_m<master>]
    for s_ in qspecs[1:]:
        for w in wsamples:
            for k in range(a.dseeds):
                specs.append(f"{s_}@{w}{k}{msuf}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(a.state or (CACHE / "state" / (out.stem + ".state.pt")))
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if out.exists():                                  # finished before (e.g. requeued at completion)
        try:
            prev = json.load(open(out))
            if prev["meta"]["specs"] == specs and prev["meta"]["n"] == a.n:
                print(f"{out} already complete ({len(prev['records'])} records); nothing to do")
                return
        except Exception:
            pass
    t0 = time.time()
    X, Y = load_imagenet_val(a.n, a.data)
    if a.limit:
        X, Y = X[:a.limit], Y[:a.limit]
    N = len(Y)
    print(f"{a.model}: {N} images, specs {specs}, solvers {solvers}, T {a.T}, budgets {budgets} "
          f"({time.time()-t0:.0f}s data)", flush=True)
    models, qinfo, hooks = {}, {}, {}
    for s in qspecs:
        m = load_mdeq(a.model)
        qinfo[s] = perturb_(m, s, a.seed)
        models[s] = m
    for s in specs[len(qspecs):]:                        # E6 arms: one master model per (spec, arm), re-drawn per evaluation
        base, arm, k = parse_arm(s.split("_m")[0] if msuf else s)
        q = parse_spec(base)
        assert q["gran"] not in (None, "noise"), s
        mk = (base, arm)
        if mk not in hooks:
            m = load_mdeq(a.model)
            if a.master != "fp32":
                perturb_(m, a.master, a.seed)
            slots = [(put, get()) for _, get, put in weight_slots(m)]
            hooks[mk] = (m, slots)
        m, slots = hooks[mk]
        models[s] = m
        hooks[s] = make_hook(slots, arm, q["bits"], q["gran"], k, DEV, wseed=a.seed)
        qinfo[s] = {**q, "spec": s, "base": base, "wsample": arm, "dseed": k, "master": a.master}
    print(f"  {len(models)} arms built; weight rel err "
          + ", ".join(f"{s}:{qinfo[s].get('weight_rel_err_mean', 0):.3f}" for s in qspecs), flush=True)

    keys = [(s, sv) for s in specs for sv in solvers]
    nb = len(budgets)
    acc = {"next": 0}
    for s, sv in keys:
        acc[(s, sv)] = {"c1": np.zeros((nb, N), bool), "c5": np.zeros((nb, N), bool),
                        "res": np.zeros((nb, N), np.float32), "disp": np.zeros((nb, N), np.float32),
                        "cos": np.zeros((nb, N), np.float32), "trace": np.zeros((N, a.T + 1), np.float32)}
    ref_res = np.zeros(N, np.float32)
    if state_path.exists():
        st = torch.load(state_path, weights_only=False)
        sa = dict(st.get("args", {}))
        sa.setdefault("data", "resized")                 # state files written before --data existed
        if sa == vars(a):
            acc, ref_res = st["acc"], st["ref_res"]
            print(f"  resumed at image {acc['next']}", flush=True)

    mean, std = MEAN.to(DEV), STD.to(DEV)
    nbatch = 0
    while acc["next"] < N:
        i0 = acc["next"]
        i1 = min(N, i0 + a.batch)
        tb = time.time()
        x = ((X[i0:i1].to(DEV).float() / 255.0) - mean) / std
        y = Y[i0:i1].to(DEV)
        ref_snap = {}
        z_ref = None
        for s in specs:
            m = models[s]
            for sv in solvers:
                hook = hooks.get(s)
                if hook is not None:
                    hook(0)                              # draw 0: injection and evaluation 0
                f, z0, cut = prepare(m, x)
                if hook is not None:
                    f = _dithered(f, hook)               # draw e before evaluation e >= 1
                snaps, trace = solve(f, z0, a.T, set(budgets), sv)
                if s == "fp32" and sv == "broyden":
                    z_ref = snaps[a.T][0]
                    ref_res[i0:i1] = snaps[a.T][1].cpu().numpy()
                A = acc[(s, sv)]
                A["trace"][i0:i1] = trace.cpu().numpy()
                for bi, k in enumerate(budgets):
                    z, r = snaps[k]
                    logits = head(m, z, cut)
                    top5 = logits.topk(5, dim=1).indices
                    A["c1"][bi, i0:i1] = (top5[:, 0] == y).cpu().numpy()
                    A["c5"][bi, i0:i1] = (top5 == y[:, None]).any(1).cpu().numpy()
                    A["res"][bi, i0:i1] = r.cpu().numpy()
                    b = z.shape[0]
                    A["disp"][bi, i0:i1] = ((z - z_ref).reshape(b, -1).norm(dim=1) /
                                            z_ref.reshape(b, -1).norm(dim=1).clamp_min(1e-12)).cpu().numpy()
                    if s == "fp32":
                        ref_snap[(sv, k)] = z
                    A["cos"][bi, i0:i1] = F.cosine_similarity(z.reshape(b, -1), ref_snap[(sv, k)].reshape(b, -1),
                                                              dim=1).cpu().numpy()
                del snaps, trace
        acc["next"] = i1
        nbatch += 1
        if nbatch == 1 or nbatch % 10 == 0:
            print(f"  images {i1}/{N} ({time.time()-tb:.1f}s/batch, {time.time()-t0:.0f}s total)", flush=True)
        if nbatch % a.ckpt_every == 0 and acc["next"] < N:
            tmp = state_path.with_suffix(f".tmp{os.getpid()}")
            torch.save({"args": vars(a), "acc": acc, "ref_res": ref_res}, tmp)
            os.replace(tmp, state_path)

    # ------------------------------------------------------------------ write
    records = []
    arrays = {"y": Y.numpy(), "ref_res": ref_res, "budgets": np.array(budgets)}
    for s, sv in keys:
        A = acc[(s, sv)]
        tr = A["trace"]
        hits = {}
        for tau in TAUS:
            below = tr <= tau
            first = np.where(below.any(1), below.argmax(1), -1)
            hits[str(tau)] = {"frac_reached": float((first >= 0).mean()),
                              "median_steps_if_reached": float(np.median(first[first >= 0])) if (first >= 0).any() else None}
        with np.errstate(divide="ignore", invalid="ignore"):
            logtr = np.log10(np.clip(tr, 1e-12, 1e12))
        rec = {"model": f"mdeq_{a.model}", "spec": s, "quant": qinfo[s], "seed": a.seed, "solver": sv, "n": N,
               "base": qinfo[s].get("base", s), "wsample": qinfo[s].get("wsample", "none"),
               "dseed": qinfo[s].get("dseed"), "master": qinfo[s].get("master", "fp32"),
               "T": a.T, "budgets": budgets, "hits": hits,
               "trace_log10_median": np.nanmedian(logtr, 0).round(4).tolist(),
               "trace_log10_p90": np.nanpercentile(logtr, 90, 0).round(4).tolist(),
               "by_budget": []}
        for bi, k in enumerate(budgets):
            c1, res = A["c1"][bi], A["res"][bi]
            rec["by_budget"].append({
                "budget": k, "top1": float(c1.mean()), "top5": float(A["c5"][bi].mean()),
                "res_median": float(np.median(res)), "res_mean": float(np.mean(np.clip(res, 0, 1e6))),
                "res_p90": float(np.percentile(res, 90)),
                "frac_res_le": {str(t): float((res <= t).mean()) for t in TAUS},
                "disp_median": float(np.median(A["disp"][bi])), "disp_mean": float(np.mean(np.clip(A["disp"][bi], 0, 1e6))),
                "cos_mean": float(np.mean(A["cos"][bi])),
                "auroc_res_wrong": auroc(res, ~c1)})
        records.append(rec)
        tag = f"{s}|{sv}"
        for name in ("c1", "c5", "res", "disp", "cos"):
            arrays[f"{tag}|{name}"] = A[name] if name in ("c1", "c5") else A[name].astype(np.float32)
        arrays[f"{tag}|trace"] = A["trace"].astype(np.float16)
    meta = {"model": f"mdeq_{a.model}", "seed": a.seed, "n": N, "T": a.T, "budgets": budgets, "specs": specs,
            "qspecs": qspecs, "wsample": wsamples, "dseeds": a.dseeds, "master": a.master, "limit": a.limit,
            "solvers": solvers, "data": ("ImageNet val (evanarlian/imagenet_1k_resized_256), first n/1000 per class" if a.data == "resized"
                     else "ImageNet val original JPEGs (mlx-vision/imagenet-1k val.zip), first n/1000 per class by name"),
            "ref_res_median": float(np.median(ref_res)), "secs": time.time() - t0,
            "device": torch.cuda.get_device_name(0), "torch": torch.__version__}
    np.savez_compressed(str(out.with_suffix(".npz")), **arrays)
    json.dump({"meta": meta, "records": records}, open(out, "w"))
    if state_path.exists():
        state_path.unlink()
    for r in records:
        bb = {b["budget"]: b for b in r["by_budget"]}
        print(f"  {r['spec']:9s} {r['solver']:7s} " + " ".join(
            f"k{k}:{bb[k]['top1']*100:5.1f}/{bb[k]['res_median']:.0e}" for k in budgets), flush=True)
    print("wrote", out, f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
