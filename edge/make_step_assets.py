#!/usr/bin/env python3
"""Board assets for one aihub_h1 task from its folded step graph: init.npz, host_fp32.npz (onnxruntime fp32 chain,
aihub_infer.host_loop) and a copy of the puzzles, plus a check of the graph against the PyTorch model (both run
`--nsteps` steps from the initial carry on the first `--n_check` puzzles; per step: z_H / z_L / q_halt max abs
difference and argmax agreement; free-running chains drift apart on unsolved puzzles, so each step is also run
from the PyTorch carry entering it, "teacher-forced", which isolates the one-step error). `--ref_onnx` also runs an existing graph on the same chain (e.g. the
committed assets/sudoku_mlp graph, to show a re-export of the official checkpoint matches it). Server only (torch).

    python edge/make_step_assets.py --task sudoku_mlp_w256 --H 3 \\
        --onnx ~/aihub_artifacts/h1/sudoku_mlp_w256/stepc_H3_op13_fold.onnx \\
        --puzzles edge/assets/sudoku_mlp/puzzles_0_100.npz --out edge/assets/sudoku_mlp_w256
"""
import argparse
import json
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import aihub_h1 as A  # noqa: E402
import aihub_infer as I  # noqa: E402

OUTS = ["z_H_out", "z_L_out", "logits", "q_halt"]


def ort_chain(path, inp, init, nsteps):
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    zH0, zL0, emb = init
    zH, zL = [zH0.copy() for _ in inp], [zL0.copy() for _ in inp]
    steps = []
    for _ in range(nsteps):
        o = [sess.run(OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1], "puzzle_embedding": emb})
             for i in range(len(inp))]
        zH, zL = [r[0] for r in o], [r[1] for r in o]
        steps.append(dict(zH=np.concatenate(zH), zL=np.concatenate(zL), pred=np.concatenate([r[2] for r in o]).argmax(-1),
                          q=np.array([float(r[3].reshape(-1)[0]) for r in o])))
    return steps


def torch_chain(m, cfg, inp, nsteps):
    import torch
    inner, w, S = m.inner, A.width(cfg["task"]), cfg["seq_len"] + A.PLEN
    n = len(inp)
    x = torch.as_tensor(inp, dtype=torch.int32)
    Carry = type(inner.empty_carry(1))
    c = Carry(z_H=inner.H_init.expand(n, S, w).clone(), z_L=inner.L_init.expand(n, S, w).clone())
    steps = []
    with torch.no_grad():
        for _ in range(nsteps):
            zH_in, zL_in = c.z_H.numpy().copy(), c.z_L.numpy().copy()
            c, lg, (q, _) = inner(c, {"inputs": x, "puzzle_identifiers": torch.zeros(n, dtype=torch.int32)})
            steps.append(dict(zH=c.z_H.numpy(), zL=c.z_L.numpy(), pred=lg.argmax(-1).numpy(), q=q.numpy().reshape(-1),
                              zH_in=zH_in, zL_in=zL_in))
    return steps


def ort_teacher(path, tor, inp, emb):
    """Each step of the graph from the PyTorch carry entering that step."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    steps = []
    for t in tor:
        o = [sess.run(OUTS, {"z_H": t["zH_in"][i:i + 1], "z_L": t["zL_in"][i:i + 1], "inputs": inp[i:i + 1],
                             "puzzle_embedding": emb}) for i in range(len(inp))]
        steps.append(dict(zH=np.concatenate([r[0] for r in o]), zL=np.concatenate([r[1] for r in o]),
                          pred=np.concatenate([r[2] for r in o]).argmax(-1),
                          q=np.array([float(r[3].reshape(-1)[0]) for r in o])))
    return steps


def compare(a, b):
    return [dict(zH=float(np.abs(x["zH"] - y["zH"]).max()), zL=float(np.abs(x["zL"] - y["zL"]).max()),
                 q=float(np.abs(x["q"] - y["q"]).max()), cell_agree=float((x["pred"] == y["pred"]).mean()),
                 puzzle_agree=float((x["pred"] == y["pred"]).all(-1).mean())) for x, y in zip(a, b)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(A.TASKS))
    ap.add_argument("--H", type=int, default=3)
    ap.add_argument("--onnx", required=True, help="folded opset-13 stepc graph")
    ap.add_argument("--puzzles", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_onnx", default="")
    ap.add_argument("--n_check", type=int, default=10)
    ap.add_argument("--nsteps", type=int, default=16)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    puz = np.load(a.puzzles)
    inp, lab = puz["inputs"].astype(np.int32), puz["labels"]

    m, cfg, sd = A.build(a.task, a.H)
    cfg["task"] = a.task
    zH0 = m.inner.H_init.detach().numpy().astype(np.float32)
    zL0 = m.inner.L_init.detach().numpy().astype(np.float32)
    emb = sd["inner.puzzle_emb.weights"][:1].float().numpy()
    np.savez(out / "init.npz", zH0=zH0, zL0=zL0, emb=emb, seq_len=np.int64(cfg["seq_len"]), plen=np.int64(A.PLEN),
             vocab=np.int64(cfg["vocab_size"]))
    S = cfg["seq_len"] + A.PLEN
    init = (np.broadcast_to(zH0, (1, S, zH0.shape[0])).copy(), np.broadcast_to(zL0, (1, S, zL0.shape[0])).copy(), emb)

    k = a.n_check
    tor = torch_chain(m, cfg, inp[:k], a.nsteps)
    grf = ort_chain(a.onnx, inp[:k], init, a.nsteps)
    rep = {"task": a.task, "H": a.H, "width": A.width(a.task), "onnx": str(a.onnx),
           "onnx_MB": round(pathlib.Path(a.onnx).stat().st_size / 1e6, 2), "n_check": k, "nsteps": a.nsteps,
           "graph_vs_torch": compare(grf, tor), "graph_vs_torch_teacher": compare(ort_teacher(a.onnx, tor, inp[:k], emb), tor),
           "exact_per_step_torch": [int((s["pred"] == lab[:k]).all(-1).sum()) for s in tor],
           "exact_per_step_graph": [int((s["pred"] == lab[:k]).all(-1).sum()) for s in grf]}
    if a.ref_onnx:
        rep["graph_vs_ref_onnx"] = compare(grf, ort_chain(a.ref_onnx, inp[:k], init, a.nsteps))
    last = rep["graph_vs_torch"][-1]
    print(f"[{a.task}] {rep['onnx_MB']} MB; graph vs torch at step {a.nsteps}: zH {last['zH']:.2e} q {last['q']:.2e} "
          f"puzzles agree {last['puzzle_agree']:.2f}; exact torch {rep['exact_per_step_torch'][-1]}/{k} "
          f"graph {rep['exact_per_step_graph'][-1]}/{k}", flush=True)
    tf = rep["graph_vs_torch_teacher"]
    print(f"[{a.task}] teacher-forced, worst of {a.nsteps} steps: zH {max(s['zH'] for s in tf):.2e} "
          f"q {max(s['q'] for s in tf):.2e} cells agree {min(s['cell_agree'] for s in tf):.4f}", flush=True)
    if a.ref_onnx:
        r = rep["graph_vs_ref_onnx"][-1]
        print(f"[{a.task}] graph vs {a.ref_onnx} at step {a.nsteps}: zH {r['zH']:.2e} q {r['q']:.2e} "
              f"puzzles agree {r['puzzle_agree']:.2f}", flush=True)

    import onnxruntime as ort
    sess = ort.InferenceSession(str(a.onnx), providers=["CPUExecutionProvider"])
    host = I.host_loop(sess, a.task, inp, a.nsteps, init)
    np.savez_compressed(out / "host_fp32.npz", **host)
    rep["host_exact_per_step"] = [int(v) for v in (host["preds"] == lab[None]).all(-1).sum(1)]
    rep["host_n"] = int(len(inp))
    print(f"[{a.task}] host_fp32 ({len(inp)} puzzles) exact per step {rep['host_exact_per_step']}", flush=True)
    shutil.copy(a.onnx, out / pathlib.Path(a.onnx).name)
    shutil.copy(a.puzzles, out / pathlib.Path(a.puzzles).name)
    (out / "check_vs_torch.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
