#!/usr/bin/env python3
"""M4: labels of the AI Hub QDQ graphs (and X3's ablation variants, and M4's vendor-emulated variants)
under reference QDQ semantics (ONNX Runtime ORT_ENABLE_BASIC: float ops between Q/DQ), on X3's
evaluation inputs (Sudoku puzzles 0-99 x 16 steps, maze 0-49 x 8, ARC 100 canonical rows x 8).
X3's host labels were produced at ORT_ENABLE_ALL, whose integer-kernel fusions change mixed 8/16-bit
graphs (M3, Sep 25); both are kept as separate host backends.

    ~/venvs/kvcache/bin/python m4_onnx_labels.py --workers 2 --threads 4 [--level basic]
Outputs: ~/m4/onnx_labels/<task>__<cell>#<level>.json (+ .npz preds)
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time
import traceback

import numpy as np

HOME = pathlib.Path.home()
M4 = HOME / "m4"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import m4_onnx_sweep as S  # noqa: E402

OUT = M4 / "onnx_labels"
LAB_N = {"sudoku_mlp": 100, "sudoku_attn": 100, "maze": 50, "arc": 100}


def eval_inputs(task):
    import aihub_h1 as A
    m, cfg, sd = A.build(task, 1)
    seq = cfg["seq_len"] + A.PLEN
    zH0 = np.broadcast_to(m.inner.H_init.detach().numpy(), (1, seq, 512)).astype(np.float32)
    zL0 = np.broadcast_to(m.inner.L_init.detach().numpy(), (1, seq, 512)).astype(np.float32)
    W = sd["inner.puzzle_emb.weights"].float().numpy()
    n = LAB_N[task]
    if task == "arc":
        import aihub_infer as AI
        inp, lab, pids = AI.arc_rows(n)
        emb = W[pids][:, None]
    else:
        inp, lab = A.load_data(task, list(range(n)))
        emb = np.repeat(W[:1][None], n, 0)
    return dict(inputs=inp.astype(np.int32), emb=emb.astype(np.float32), zH0=zH0, zL0=zL0), lab


def run(args):
    task, name, graph, threads, level = args
    import release_check as R
    out = OUT / f"{task}__{name}#{level}.json"
    if out.exists():
        return f"{task}/{name}", "exists"
    try:
        t0 = time.time()
        chk, lab = eval_inputs(task)
        vendor = {"qnnfused": "qnn", "qnnfused16": "qnn16"}.get(name.split("@")[-1]) if "@" in name else None
        step = R.OnnxStep(graph, threads, vendor=vendor, opt_level=level)
        r = R.rollout(step, chk, S.TASK_CFG[task]["T"])
        p = r["preds"][-1]
        if task == "arc":
            mask = lab != 0
            ex = ((p == lab) & mask).sum(-1) == mask.sum(-1)
        else:
            ex = (p == lab).all(-1)
        rec = dict(task=task, cell=name, level=level, n=len(ex), exact=float(ex.mean()),
                   cell_acc=float((p == lab).mean()), correct=[bool(x) for x in ex], secs=round(time.time() - t0, 1))
        np.savez_compressed(out.with_suffix(".npz"), preds=r["preds"], q=r["q"], lab=lab)
        out.write_text(json.dumps(rec, indent=1))
        return f"{task}/{name}", f"exact {rec['exact']:.3f} ({rec['secs']}s)"
    except Exception as e:  # noqa: BLE001
        (OUT / f"{task}__{name}#{level}.err").write_text(traceback.format_exc())
        return f"{task}/{name}", f"ERROR {type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--level", default="basic")
    p.add_argument("--tasks", default="sudoku_mlp,sudoku_attn,maze,arc")
    p.add_argument("--only", default="", help="comma list of cell names (default: all)")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",")
    only = set(a.only.split(",")) if a.only else None
    todo = [(t, n, g, a.threads, a.level) for t, n, g in S.cells() if t in tasks and (only is None or n in only)]
    print(f"{len(todo)} cells", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, msg in pool.imap_unordered(run, todo):
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
