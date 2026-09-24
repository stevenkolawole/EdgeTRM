"""Export per-task data for the M3 activation sweeps: inputs/labels/puzzle embeddings for the evaluation
and (unlabeled) scoring splits, initial carries, and the tensor -> group map of wa_collapse.classify."""
import sys, json, numpy as np, onnx
sys.path.insert(0, "/home/ubuntu/x3")
import aihub_h1 as A, aihub_infer as I, wa_collapse as W
out = "/home/ubuntu/m3/export"
import os; os.makedirs(out, exist_ok=True)
for task in sys.argv[1].split(","):
    d = I.idir(task)
    g16 = onnx.load(str(d / "qdq_w8a16" / "model.onnx"), load_external_data=False).graph
    json.dump(W.classify(g16), open(f"{out}/groups_{task}.json", "w"))
    if task == "arc":
        inp, lab, pids = I.arc_rows(100000)
        m, cfg, sd = A.build(task, 1)
        emb = sd["inner.puzzle_emb.weights"][I.torch_index(pids)].float().numpy()
        zH0 = np.broadcast_to(m.inner.H_init.detach().numpy(), (1, cfg["seq_len"] + A.PLEN, 512)).astype(np.float32)
        zL0 = np.broadcast_to(m.inner.L_init.detach().numpy(), (1, cfg["seq_len"] + A.PLEN, 512)).astype(np.float32)
        splits = {"eval": np.arange(0, 100), "score": np.arange(100, 164), "evalbig": np.arange(164, len(inp))}
        nsteps = 8
    else:
        zH0, zL0, e = I.initial_state(task)
        ntot = 2000 if task.startswith("sudoku") else 1000
        inp, lab = A.load_data(task, list(range(ntot)))
        emb = np.repeat(e, len(inp), 0)
        splits = {"eval": np.arange(0, 100), "score": np.arange(200, 264),
                  "evalbig": np.arange(1000, 2000) if task.startswith("sudoku") else np.arange(300, 1000)}
        nsteps = 16 if task.startswith("sudoku") else 8
    blob = {"zH0": zH0, "zL0": zL0, "nsteps": nsteps}
    for k, idx in splits.items():
        blob[f"{k}_inp"] = np.asarray(inp)[idx].astype(np.int32)
        blob[f"{k}_lab"] = np.asarray(lab)[idx].astype(np.int64)
        blob[f"{k}_emb"] = np.asarray(emb)[idx].astype(np.float32)
        blob[f"{k}_idx"] = idx
    np.savez_compressed(f"{out}/data_{task}.npz", **blob)
    print(task, {k: v.shape for k, v in blob.items() if hasattr(v, "shape")}, flush=True)
