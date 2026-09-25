#!/usr/bin/env python3
"""M1 on a deep equilibrium model (MDEQ-Small ImageNet): cheap-precision solver iterations, then precise ones.

X6's harness (deq_sweep.py: model, weight-norm-folded perturbations, Broyden with per-sample lowest-residual
iterate, ImageNet val cache) is reused unchanged. Per image batch:
  precise from zero   Broyden on the precise map f_P (fp32 or INT8 w8c) from z0 = 0, top-1 at every budget in --curve
  cheap from zero     Broyden on the cheap map f_C (e.g. w4c, w3c, noise) from z0 = 0, top-1 at --curve budgets
  hybrid              Broyden on f_C for t_c iterations (t_c in --tc), then Broyden on f_P warm-started from the
                      cheap iterate for k iterations (k in --kp; fresh Broyden memory), read out with P's head
The injection (downsample + stage0) is computed by each model for its own map; the hybrid's precise phase uses the
precise injection. Traffic per solver iteration = one pass over the fixed-point block's weights.
Images: --lo..--hi of X6's stratified val cache (5 per class, n=5000): dev 0-2499, held-out 2500-4999.

    python m1_deq.py --cheap w4c,w3c --polish fp32,w8c --lo 0 --hi 2500 --out ~/breadth_results/m1/deq_small_dev.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import deq_sweep as D


def enc(a):
    return "".join("1" if v else "0" for v in a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small")
    ap.add_argument("--cheap", default="w4c,w3c")
    ap.add_argument("--polish", default="fp32,w8c")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lo", type=int, default=0)
    ap.add_argument("--hi", type=int, default=2500)
    ap.add_argument("--n_cache", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--curve", default="1,2,3,4,6,8,10,12,14,16,20,26,32,40,60")
    ap.add_argument("--tc", default="4,8,12,16,26,40")
    ap.add_argument("--kp", default="1,2,3,4,6,8,12,16,26")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert torch.cuda.is_available(), "NO CUDA"
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    curve = sorted(int(x) for x in a.curve.split(","))
    tcs = sorted(int(x) for x in a.tc.split(","))
    kps = sorted(int(x) for x in a.kp.split(","))
    X, Y = D.load_imagenet_val(a.n_cache, "resized")
    X, Y = X[a.lo:a.hi], Y[a.lo:a.hi]
    N = len(Y)
    cheaps = [s for s in a.cheap.split(",") if s]
    pols = [s for s in a.polish.split(",") if s]
    models, qinfo = {}, {}
    for s in sorted(set(cheaps + pols)):
        m = D.load_mdeq(a.model).to(D.DEV).eval()
        import re
        mm = re.fullmatch(r"n(\d+)([tac])(\d+)", s)
        if mm:                                      # nested (Matryoshka) slice: top b bits of the B-bit grid (polish.q_nested)
            import polish as P
            rel = []
            with torch.no_grad():
                for _, get, put in D.weight_slots(m):
                    W = get()
                    Wq = P.q_nested(W.reshape(W.shape[0], -1).float(), int(mm.group(1)), mm.group(2),
                                    int(mm.group(3))).reshape(W.shape).to(W.dtype)
                    rel.append(float((Wq - W).norm() / W.norm().clamp_min(1e-12)))
                    put(Wq)
            qinfo[s] = {"spec": s, "weight_rel_err_mean": float(np.mean(rel)), "n_tensors": len(rel)}
        else:
            qinfo[s] = D.perturb_(m, s, a.seed)
        models[s] = m
    out_path = Path(a.out)
    state = out_path.with_suffix(".state.pt")
    acc = {"next": 0, "curve": {}, "hyb": {}}
    if state.exists():
        st = torch.load(state, weights_only=False)
        if st.get("args") == vars(a):
            acc = st["acc"]
            print(f"resumed at image {acc['next']}", flush=True)
    for s in models:
        acc["curve"].setdefault(s, np.zeros((len(curve), N), bool))
    for c in cheaps:
        for p in pols:
            acc["hyb"].setdefault((c, p), np.zeros((len(tcs), len(kps), N), bool))
    mean, std = D.MEAN.to(D.DEV), D.STD.to(D.DEV)
    t0 = time.time()
    nb = 0
    with torch.no_grad():
        while acc["next"] < N:
            i0, i1 = acc["next"], min(N, acc["next"] + a.batch)
            x = ((X[i0:i1].to(D.DEV).float() / 255.0) - mean) / std
            y = Y[i0:i1].to(D.DEV)
            prep = {s: D.prepare(models[s], x) for s in models}
            cheap_z = {}
            for s, (f, z0, cut) in prep.items():
                snaps, _ = D.solve(f, z0, max(curve + (tcs if s in cheaps else [])), set(curve) | set(tcs), "broyden")
                for bi, k in enumerate(curve):
                    acc["curve"][s][bi, i0:i1] = (D.head(models[s], snaps[k][0], cut).argmax(1) == y).cpu().numpy()
                if s in cheaps:
                    cheap_z[s] = {t: snaps[t][0].clone() for t in tcs}
                del snaps
            for c in cheaps:
                for p in pols:
                    fp, _, cut = prep[p]
                    for ti, t in enumerate(tcs):
                        snaps, _ = D.solve(fp, cheap_z[c][t], max(kps), set(kps), "broyden")
                        for ki, k in enumerate(kps):
                            acc["hyb"][(c, p)][ti, ki, i0:i1] = (D.head(models[p], snaps[k][0], cut).argmax(1) == y).cpu().numpy()
                        del snaps
            acc["next"] = i1
            nb += 1
            if nb % 5 == 0 or i1 == N:
                print(f"  images {i1}/{N} ({time.time()-t0:.0f}s)", flush=True)
                tmp = state.with_suffix(".tmp")
                torch.save({"args": vars(a), "acc": acc}, tmp)
                tmp.replace(state)
    res = {"meta": {"model": f"mdeq_{a.model}", "lo": a.lo, "hi": a.hi, "n": N, "seed": a.seed, "curve": curve,
                    "tc": tcs, "kp": kps, "qinfo": qinfo, "gpu": torch.cuda.get_device_name(0),
                    "torch": torch.__version__, "secs": time.time() - t0},
           "curve": {s: {"acc": acc["curve"][s].mean(1).tolist(), "correct": {str(k): enc(acc["curve"][s][bi])
                                                                             for bi, k in enumerate(curve)}}
                     for s in models},
           "hybrid": {f"{c}|{p}": {f"{t}+{k}": {"acc": float(acc["hyb"][(c, p)][ti, ki].mean()),
                                                "correct": enc(acc["hyb"][(c, p)][ti, ki])}
                                   for ti, t in enumerate(tcs) for ki, k in enumerate(kps)}
                      for c in cheaps for p in pols}}
    json.dump(res, open(out_path, "w"))
    if state.exists():
        state.unlink()
    for s in models:
        print(f"{s:8s} from zero: " + " ".join(f"{k}:{100*v:.1f}" for k, v in zip(curve, res['curve'][s]['acc'])))
    for key, d in res["hybrid"].items():
        best = max(d.items(), key=lambda kv: kv[1]["acc"])
        print(f"{key}: " + " ".join(f"{tk}:{100*v['acc']:.1f}" for tk, v in d.items() if tk.split('+')[1] in ('1', '4', '8', '26')) +
              f" | best {best[0]} {100*best[1]['acc']:.1f}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
