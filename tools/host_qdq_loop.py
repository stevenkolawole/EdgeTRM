#!/usr/bin/env python3
"""Host ORT simulation of a quantized (QDQ) step graph over 8 iterations on the fault-chain test rows, to tell a
toolchain fault (host simulation solves, device does not) from a graph broken by quantization (neither solves).
Same loop as aihub_faults.host, on infer/qdq_<prec>/model.onnx; writes exact and cell accuracy per iteration.

    python host_qdq_loop.py --out OUT.json sudoku_mlp:w8a8 hrm_sudoku:w4a8 ...
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_faults as F  # noqa: E402

I = F.I


def run(task, prec, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    sess = ort.InferenceSession(str(I.idir(task) / f"qdq_{prec}" / "model.onnx"), so, providers=["CPUExecutionProvider"])
    inp, lab, (zH0, zL0, emb) = F.data(task)
    N = inp.shape[0]
    zH, zL = np.repeat(zH0, N, 0)[:, None], np.repeat(zL0, N, 0)[:, None]
    out = []
    for s in range(F.NSTEPS):
        nH, nL, pr = [], [], []
        for i in range(N):
            o = sess.run(I.OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1], "puzzle_embedding": emb})
            nH.append(o[0]); nL.append(o[1]); pr.append(o[2].argmax(-1))
        zH, zL = np.stack(nH), np.stack(nL)
        p = np.concatenate(pr)
        out.append(dict(step=s + 1, exact=float((p == lab).all(-1).mean()), cell=float((p == lab).mean())))
        print(task, prec, out[-1], flush=True)
    return dict(task=task, prec=prec, n=int(N), steps=out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("graphs", nargs="+")
    a = ap.parse_args()
    res = [run(*g.split(":"), a.threads) for g in a.graphs]
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
