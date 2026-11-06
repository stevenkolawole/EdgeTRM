#!/usr/bin/env python3
"""INT8 calibration samples for build_int8_engine.py: real carries from an onnxruntime fp32 rollout of the step
graph on calibration puzzles (disjoint from the test puzzles), as e1_energy.calib_samples does on the datacenter
GPUs: for each puzzle two distinct random steps s in [0, nsup) (rng default_rng(0)), and the graph inputs
(z_H, z_L, inputs, puzzle_embedding) of the call that runs step s, i.e. the carry entering it (s = 0 is init.npz).
Carry handling mirrors edge_loop.py (init broadcast over the sequence, outputs fed back). Runs on the host.

    python make_calib_carries.py --model assets/sudoku_mlp/stepc_H3_op13_fold.onnx --init assets/sudoku_mlp/init.npz \
        --puzzles assets/sudoku_mlp/calib_puzzles_1000_1064.npz --out assets/sudoku_mlp/calib_carries.npz
"""
import argparse

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--nsup", type=int, default=16)
    ap.add_argument("--per_puzzle", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    init = np.load(a.init)
    plen = int(init["plen"]) if "plen" in init.files else 16
    puz = np.load(a.puzzles)
    x = puz["inputs"].astype(np.int32)
    S = x.shape[1] + plen
    zh0 = np.broadcast_to(init["zH0"].astype(np.float32), (1, S, init["zH0"].shape[-1])).copy()
    zl0 = np.broadcast_to(init["zL0"].astype(np.float32), (1, S, init["zL0"].shape[-1])).copy()
    emb = init["emb"].astype(np.float32).reshape(1, -1)
    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    outs = [o.name for o in sess.get_outputs()][:2]
    assert outs == ["z_H_out", "z_L_out"], outs

    rng = np.random.default_rng(0)
    pick = np.stack([rng.choice(a.nsup, a.per_puzzle, replace=False) for _ in range(len(x))])
    zh, zl, xs, rows, steps = [], [], [], [], []
    for i in range(len(x)):
        zH, zL, xi = zh0, zl0, x[i:i + 1]
        for s in range(int(pick[i].max()) + 1):
            if s in pick[i]:
                zh.append(zH[0].copy()); zl.append(zL[0].copy()); xs.append(xi[0]); rows.append(i); steps.append(s)
            zH, zL = sess.run(outs, {"z_H": zH, "z_L": zL, "inputs": xi, "puzzle_embedding": emb})
    index = puz["index"][rows] if "index" in puz.files else np.array(rows)
    np.savez_compressed(a.out, z_H=np.stack(zh), z_L=np.stack(zl), inputs=np.stack(xs).astype(np.int32),
                        puzzle_embedding=np.repeat(emb, len(rows), 0), row=index, step=np.array(steps),
                        model=a.model, puzzles=a.puzzles)
    print("%s: %d samples from %d puzzles, steps %d..%d -> %s" % (a.model, len(rows), len(x), min(steps), max(steps),
                                                                  a.out))


if __name__ == "__main__":
    main()
