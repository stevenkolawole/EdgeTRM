#!/usr/bin/env python3
"""M4: ARC host-simulation labels for the AI Hub QDQ graphs X3 has not run (w4a8, w4a16), with exactly
X3's protocol (aihub_infer.stage_host_arc: 100 canonical ARC-AGI-1 test rows, each row's own puzzle
embedding, H=1, 8 steps, token-exact over non-pad cells), written to ~/m4/arc/ (X3's directories are
read only for M4).

    ~/venvs/kvcache/bin/python m4_arc_host.py --precs w4a8,w4a16 --threads 8
"""
import argparse
import pathlib
import sys
import time

import numpy as np

HOME = pathlib.Path.home()
for extra in (pathlib.Path(__file__).resolve().parent, HOME / "x3"):
    if (extra / "aihub_h1.py").exists():
        sys.path.append(str(extra))
import aihub_h1 as A  # noqa: E402
import aihub_infer as AI  # noqa: E402

OUT = HOME / "m4" / "arc"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--precs", default="w4a8,w4a16")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--nsteps", type=int, default=8)
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    import onnxruntime as ort
    OUT.mkdir(parents=True, exist_ok=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    so.log_severity_level = 3
    inp, lab, pids = AI.arc_rows(a.count)
    m, cfg, sd = A.build("arc", 1)
    embs = sd["inner.puzzle_emb.weights"][AI.torch_index(pids)].float().numpy()
    seq = cfg["seq_len"] + A.PLEN
    zH0 = np.broadcast_to(m.inner.H_init.detach().numpy(), (1, seq, 512)).astype(np.float32)
    zL0 = np.broadcast_to(m.inner.L_init.detach().numpy(), (1, seq, 512)).astype(np.float32)
    del m
    mask = lab != 0
    for prec in a.precs.split(","):
        out = OUT / f"host_{prec}_{a.count}.npz"
        if out.exists():
            continue
        path = A.ART / "arc" / "infer" / f"qdq_{prec}" / "model.onnx"
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        t0 = time.time()
        N = len(inp)
        zH, zL = np.repeat(zH0[None], N, 0), np.repeat(zL0[None], N, 0)
        preds = []
        for s_ in range(a.nsteps):
            nzH, nzL, lg = [], [], []
            for i in range(N):
                o = sess.run(AI.OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1],
                                       "puzzle_embedding": embs[i:i + 1]})
                nzH.append(o[0]); nzL.append(o[1]); lg.append(o[2])
            zH, zL = np.stack(nzH), np.stack(nzL)
            preds.append(np.concatenate(lg).argmax(-1))
            print(f"[arc] {prec} step {s_ + 1}/{a.nsteps} ({time.time() - t0:.0f}s)", flush=True)
        preds = np.stack(preds)
        np.savez_compressed(out, preds=preds, lab=lab)
        corr = (preds == lab[None]) & mask[None]
        ex = (corr.sum(-1) == mask.sum(-1)[None]).mean(-1)
        print(f"[arc] host {prec} n={N}: token-exact per step {[round(float(e), 3) for e in ex]} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
