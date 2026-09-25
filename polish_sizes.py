#!/usr/bin/env python3
"""M1 memory side: how many extra bits does an exact (lossless) polish cost beside Q(W)?

For each model and compressed spec, per CastedLinear (DT: conv) weight W (the fp32 checkpoint values):
  Q(W)                    the compressed weights (polish.quant_2d, identical arithmetic)
  R_sub = W - Q(W)        float32 residual; `exact_sub` records whether Q(W) + R_sub == W bit for bit
  R_xor = bits(W) ^ bits(Q(W))   (uint32; reconstruction W = Q ^ R_xor is exact by construction:
                          the erasing / XOR idea of Gorilla-Elf-style float coders)
and the size of each under: zstd -19 on the raw bytes; zstd on the 4 byte planes (byte shuffle, Blosc
style); fpzip lossless (float-specific predictive coder); zfp reversible mode.  Reference rows: the same
coders on W itself (lossless fp32 storage without Q), and the checkpoint's own precision (how many low
mantissa bits of W are zero, e.g. a bf16-trained model saved as fp32).

    PYTHONPATH=~/m1_pylib python polish_sizes.py --tasks sudoku_mlp,sudoku_attn --specs w4t,w4c,w3g32 \
        --out ~/breadth_results/m1/sizes.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def zstd_bytes(b, level=19):
    import zstandard
    return len(zstandard.ZstdCompressor(level=level).compress(b))


def shuffled(a):
    """byte planes of a 4-byte array: all byte-0s, then byte-1s, ..."""
    u = np.ascontiguousarray(a).view(np.uint8).reshape(-1, 4)
    return np.ascontiguousarray(u.T).tobytes()


def sizes(a):
    """compressed sizes (bytes) of a float32 (or uint32) array under each coder."""
    a = np.ascontiguousarray(a)
    out = {"raw": a.nbytes, "zstd": zstd_bytes(a.tobytes()), "zstd_shuf": zstd_bytes(shuffled(a))}
    if a.dtype == np.float32:
        try:
            import fpzip
            out["fpzip"] = len(fpzip.compress(a.reshape(1, 1, *a.shape) if a.ndim == 2 else a, precision=0, order="C"))
        except Exception as e:                     # noqa: BLE001
            out["fpzip_err"] = str(e)[:80]
        try:
            import zfpy
            out["zfp_rev"] = len(zfpy.compress_numpy(a))
        except Exception as e:                     # noqa: BLE001
            out["zfp_err"] = str(e)[:80]
    return out


def main():
    import torch
    import polish as P
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="sudoku_mlp,sudoku_attn")
    ap.add_argument("--specs", default="w4t,w4c,w4g32,w3g32,w3c,w8c,w8t,mx4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = json.load(open(a.out)) if Path(a.out).exists() else {}     # resumable per task
    for task in a.tasks.split(","):
        if task in res:
            continue
        m, family, _ = P.build_fp32(task, 16, 8)
        items = P.linear_items(m, family)
        Ws = {n: mod.weight.data.reshape(mod.weight.shape[0], -1).float().contiguous() for n, mod in items}
        N = sum(W.numel() for W in Ws.values())
        allW = np.concatenate([W.numpy().ravel() for W in Ws.values()])
        low16 = (allW.view(np.uint32) & 0xFFFF)
        trow = {"n_weights": int(N), "n_linears": len(Ws),
                "frac_low16_zero": float(np.mean(low16 == 0)),
                "W_bits_per_weight": {k: 8 * v / N for k, v in sizes(allW).items() if isinstance(v, int)}}
        for spec in a.specs.split(","):
            Rs, Xs, exact, qbits = [], [], 0, 0
            for n, W in Ws.items():
                Q = P.quant_2d(W, spec).float().contiguous()
                R = (W - Q).numpy()
                exact += int(np.sum((Q.numpy() + R).view(np.uint32) == W.numpy().view(np.uint32)))
                Rs.append(R.ravel())
                Xs.append((W.numpy().view(np.uint32) ^ Q.numpy().view(np.uint32)).ravel())
                qbits += P.storage_bits(spec, tuple(W.shape))
            R, X = np.concatenate(Rs), np.concatenate(Xs)
            trow[spec] = {"q_bits_per_weight": qbits / N, "exact_sub_frac": exact / N,
                          "R_sub_bits_per_weight": {k: 8 * v / N for k, v in sizes(R).items() if isinstance(v, int)},
                          "R_xor_bits_per_weight": {k: 8 * v / N for k, v in sizes(X).items() if isinstance(v, int)}}
            best = min(v for k, v in trow[spec]["R_xor_bits_per_weight"].items() if k != "raw")
            bsub = min(v for k, v in trow[spec]["R_sub_bits_per_weight"].items() if k != "raw")
            print(f"{task} {spec}: Q {qbits/N:.2f} b/w; exact residual best: xor {best:.2f}, sub {bsub:.2f} "
                  f"(sub exact on {100*exact/N:.2f}% of weights); W alone best "
                  f"{min(v for k, v in trow['W_bits_per_weight'].items() if k != 'raw'):.2f}", flush=True)
        res[task] = trow
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
