#!/usr/bin/env python3
"""On-device correctness: run the compiled H=1 carry step (`stepc`: z_H, z_L, tokens, puzzle row ->
z_H', z_L', logits, q_halt) on real Qualcomm devices through AI Hub inference jobs, chaining n
supervision steps from the host with the carry passed back (the host-loop deployment pattern), and
compare with the same loop on the host: fp32 torch-equivalent ONNX and the very QDQ graph AI Hub
compiled (ONNX Runtime CPU), on the same test puzzles.

Per (task, device, precision) chain the result records, per supervision step, the device's
predictions and q_halt, and at the end: exact and per-cell accuracy, agreement with the host fp32
predictions and with the host QDQ simulation (per cell and per puzzle), and the fidelity of the
device's final carry (cosine of z_H with the host fp32 z_H at the same step, the release check
measured on device outputs).

Inputs of `--quantize_io` models are quantized on the host with the exact input encodings of the
AI Hub QDQ graph (read from its QuantizeLinear nodes); outputs come back dequantized
(`--dequantize_outputs` is the default).

    python aihub_infer.py --stage prep  --tasks sudoku_mlp,maze --precs w8a8,w4a8,w4a16
    python aihub_infer.py --stage host  --tasks ...                  (fp32 + QDQ simulations)
    python aihub_infer.py --stage run   --tasks ... --devices "Samsung Galaxy S24|..." --precs ...
    python aihub_infer.py --stage report
"""
import argparse
import json
import pathlib
import shutil
import sys
import threading
import time
import zipfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402

NSTEPS = {"sudoku_mlp": 16, "sudoku_attn": 16, "maze": 8}


def named_outputs(o):
    """qnn_dlc targets return outputs as output_0..output_3 in graph order (z_H_out, z_L_out,
    logits, q_halt); map them back to the ONNX names."""
    names = ["z_H_out", "z_L_out", "logits", "q_halt"]
    if all(n in o for n in names):
        return o
    return {names[i]: o[f"output_{i}"] for i in range(4)}
OUTS = ["z_H_out", "z_L_out", "logits", "q_halt"]


def idir(task):
    d = A.ART / task / "infer"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_set(task):
    n = A.TASKS[task]["n_test"]
    inp, lab = A.load_data(task, list(range(n)))
    return inp.astype(np.int32), lab


def initial_state(task):
    m, cfg, sd = A.build(task, 1)
    seq = cfg["seq_len"] + A.PLEN
    zH = m.inner.H_init.detach().numpy().astype(np.float32)
    zL = m.inner.L_init.detach().numpy().astype(np.float32)
    emb = sd["inner.puzzle_emb.weights"][:1].float().numpy()
    return (np.broadcast_to(zH, (1, seq, 512)).copy(), np.broadcast_to(zL, (1, seq, 512)).copy(), emb)


# ----------------------------------------------------------------------------- prep
def stage_prep(tasks, precs):
    """Download the stepc QDQ graph per precision (kept for the host simulation) and read the
    input encodings of the quantized-I/O graph."""
    import onnx
    from onnx import numpy_helper
    import qai_hub as hub
    for task in tasks:
        for prec in precs:
            d = idir(task) / f"qdq_{prec}"
            enc_path = idir(task) / f"io_enc_{prec}.json"
            if enc_path.exists() and (d / "model.onnx").exists():
                continue
            rec = A.load_rec(task)
            q = A.find(rec, kind="quantize", graph="stepc", H=1, prec=prec)[-1]
            job = hub.get_job(q["job_id"])
            if not (d / "model.onnx").exists():
                z = job.get_target_model().download(str(idir(task) / f"qdq_{prec}_dl"))
                tmp = idir(task) / f"unz_{prec}"
                with zipfile.ZipFile(z) as f:
                    f.extractall(tmp)
                inner = next(tmp.iterdir())
                if d.exists():
                    shutil.rmtree(d)
                inner.rename(d)
                shutil.rmtree(tmp)
                pathlib.Path(z).unlink()
            m = onnx.load(str(d / "model.onnx"), load_external_data=False)
            g = m.graph
            vals = {i.name: i for i in g.initializer}
            for n in g.node:
                if n.op_type == "Constant":
                    vals[n.output[0]] = n.attribute[0].t
            inputs = {i.name for i in g.input}
            enc = {}
            for n in g.node:
                if n.op_type == "QuantizeLinear" and n.input[0] in inputs:
                    s = numpy_helper.to_array(vals[n.input[1]]).astype(float)
                    zp_t = vals[n.input[2]]
                    zp = numpy_helper.to_array(zp_t).astype(float)
                    enc[n.input[0]] = {"scale": float(s), "zero_point": float(zp),
                                       "dtype": onnx.TensorProto.DataType.Name(zp_t.data_type).lower()}
            enc_path.write_text(json.dumps({"quantize_job": q["job_id"], "inputs": enc}, indent=1))
            print(f"[{task}] {prec} input encodings {enc}", flush=True)


def quantize_input(x, e):
    info = {"uint8": (0, 255, np.uint8), "int8": (-128, 127, np.int8), "uint16": (0, 65535, np.uint16),
            "int16": (-32768, 32767, np.int16)}[e["dtype"]]
    q = np.round(x / e["scale"]) + e["zero_point"]
    return np.clip(q, info[0], info[1]).astype(info[2])


# ----------------------------------------------------------------------------- host references
def host_loop(sess, task, inp, n_steps, init):
    zH0, zL0, emb = init
    N = inp.shape[0]
    zH = np.repeat(zH0, N, 0)[:, None]
    zL = np.repeat(zL0, N, 0)[:, None]
    preds, qs = [], []
    for s in range(n_steps):
        nzH, nzL, lg, qh = [], [], [], []
        for i in range(N):
            o = sess.run(OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1], "puzzle_embedding": emb})
            nzH.append(o[0]); nzL.append(o[1]); lg.append(o[2]); qh.append(float(o[3].reshape(-1)[0]))
        zH, zL = np.stack(nzH), np.stack(nzL)
        preds.append(np.concatenate(lg).argmax(-1))
        qs.append(qh)
    return {"preds": np.stack(preds), "q": np.array(qs), "zH": zH[:, 0], "zL": zL[:, 0]}


def stage_host(tasks, precs, nsteps=None):
    """Host references. The files keep every step's predictions; the carry (zH, zL) is the one after
    the last step, so a run with `nsteps` shorter than NSTEPS is saved as host_<prec>_n<k>.npz for
    the fidelity of a device chain that stops at k steps."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 16
    for task in tasks:
        inp, lab = test_set(task)
        init = initial_state(task)
        k = nsteps or NSTEPS[task]
        for prec in ["fp32"] + precs:
            out = idir(task) / (f"host_{prec}.npz" if k == NSTEPS[task] else f"host_{prec}_n{k}.npz")
            if out.exists():
                continue
            path = A.ART / task / "stepc_H1.onnx" if prec == "fp32" else idir(task) / f"qdq_{prec}" / "model.onnx"
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            t0 = time.time()
            r = host_loop(sess, task, inp, k, init)
            np.savez_compressed(out, **r)
            ex = (r["preds"][-1] == lab).all(-1).mean()
            cell = (r["preds"][-1] == lab).mean()
            print(f"[{task}] host {prec}: exact {ex:.3f} cell {cell:.4f} ({time.time()-t0:.0f}s)", flush=True)


def stage_hostbig(task, prec, start, count, nsteps, threads):
    """The host loop on a larger, calibration-disjoint slice of the test set (indices start..start+count),
    for tighter accuracies of the exact device graphs; per-step predictions saved."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    out = idir(task) / f"hostbig_{prec}_{start}_{count}.npz"
    if out.exists():
        return
    inp, lab = A.load_data(task, list(range(start, start + count)))
    inp = inp.astype(np.int32)
    path = A.ART / task / "stepc_H1.onnx" if prec == "fp32" else idir(task) / f"qdq_{prec}" / "model.onnx"
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    t0 = time.time()
    r = host_loop(sess, task, inp, nsteps, initial_state(task))
    np.savez_compressed(out, preds=r["preds"], q=r["q"], lab=lab)
    ex = [(r["preds"][s_] == lab).all(-1).mean() for s_ in range(nsteps)]
    print(f"[{task}] hostbig {prec} n={count} exact per step {[round(float(e), 3) for e in ex]} "
          f"({time.time()-t0:.0f}s)", flush=True)


def arc_rows(count):
    """ARC-AGI-1 canonical (un-augmented) test rows, as arc_breadth.py scores its token metrics."""
    root = pathlib.Path.home() / "edgetrm_data" / "arc1concept-aug-1000"
    ident = json.loads((root / "identifiers.json").read_text())
    d = root / "test"
    inputs = np.load(d / "all__inputs.npy", mmap_mode="r")
    labels = np.load(d / "all__labels.npy", mmap_mode="r")
    pidx = np.load(d / "all__puzzle_indices.npy")
    pid = np.load(d / "all__puzzle_identifiers.npy")
    per = np.zeros(len(inputs), dtype=np.int64)
    for k in range(len(pidx) - 1):
        per[pidx[k]:pidx[k + 1]] = pid[k]
    canon = [i for i in range(len(per)) if per[i] != 0 and "|||" not in ident[int(per[i])]][:count]
    return np.asarray(inputs[canon]).astype(np.int32), np.asarray(labels[canon]).astype(np.int64), per[canon]


def stage_host_arc(precs, count, nsteps, threads):
    """Host loop of the exact ARC step graphs (fp32 and AI Hub QDQ) with each row's own puzzle
    embedding; token puzzle-exact over non-pad cells (arc_breadth's pexact_token)."""
    import onnxruntime as ort
    task = "arc"
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    inp, lab, pids = arc_rows(count)
    m, cfg, sd = A.build(task, 1)
    W = sd["inner.puzzle_emb.weights"]
    embs = W[torch_index(pids)].float().numpy()
    zH0 = np.broadcast_to(m.inner.H_init.detach().numpy(), (1, cfg["seq_len"] + A.PLEN, 512)).astype(np.float32)
    zL0 = np.broadcast_to(m.inner.L_init.detach().numpy(), (1, cfg["seq_len"] + A.PLEN, 512)).astype(np.float32)
    del m
    mask = lab != 0
    for prec in precs:
        out = idir(task) / f"host_{prec}_{count}.npz"
        if out.exists():
            continue
        path = A.ART / task / "stepc_H1.onnx" if prec == "fp32" else idir(task) / f"qdq_{prec}" / "model.onnx"
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        t0 = time.time()
        N = len(inp)
        zH, zL = np.repeat(zH0[None], N, 0), np.repeat(zL0[None], N, 0)
        preds = []
        for s_ in range(nsteps):
            nzH, nzL, lg = [], [], []
            for i in range(N):
                o = sess.run(OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1],
                                    "puzzle_embedding": embs[i:i + 1]})
                nzH.append(o[0]); nzL.append(o[1]); lg.append(o[2])
            zH, zL = np.stack(nzH), np.stack(nzL)
            preds.append(np.concatenate(lg).argmax(-1))
        preds = np.stack(preds)
        np.savez_compressed(out, preds=preds, lab=lab, zH=zH[:, 0])
        corr = (preds == lab[None]) & mask[None]
        ex = (corr.sum(-1) == mask.sum(-1)[None]).mean(-1)
        cell = corr.sum((1, 2)) / mask.sum()
        print(f"[arc] host {prec} n={N}: token-exact per step {[round(float(e), 3) for e in ex]} "
              f"cell {float(cell[-1]):.4f} ({time.time()-t0:.0f}s)", flush=True)


def torch_index(a):
    import torch
    return torch.as_tensor(np.asarray(a), dtype=torch.long)


# ----------------------------------------------------------------------------- device loop
def chain(task, device, prec, log_lock):
    import qai_hub as hub
    # The qnn_dlc target is device-agnostic: use this device's own compile if it has finished,
    # otherwise the first finished compile of the same (task, prec) for any device.
    cj = None
    while cj is None:
        rec = A.load_rec(task)
        comp = [j for j in A.find(rec, kind="compile", graph="stepc", H=1, prec=prec) if j.get("job_id")]
        if not comp:
            print(f"[{task}] {device} {prec}: no stepc compile", flush=True)
            return
        comp.sort(key=lambda j: j["device"] != device)
        states = [(j, hub.get_job(j["job_id"]).get_status()) for j in comp]
        ok = [j for j, st in states if st.code == "SUCCESS"]
        if ok:
            cj = hub.get_job(ok[0]["job_id"])
        elif all(st.code == "FAILED" for _, st in states):
            print(f"[{task}] {device} {prec}: every stepc compile FAILED ({(states[0][1].message or '')[:120]})",
                  flush=True)
            return
        else:
            time.sleep(120)
    model = cj.get_target_model()
    tshapes = cj.get_target_shapes()
    enc = json.loads((idir(task) / f"io_enc_{prec}.json").read_text())["inputs"] if prec != "fp32" else {}
    inp, lab = test_set(task)
    N = inp.shape[0]
    zH0, zL0, emb = initial_state(task)
    tag = f"{prec}_{device.replace(' ', '_')}"
    state_path = idir(task) / f"dev_{tag}.npz"
    meta_path = idir(task) / f"dev_{tag}.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"steps": [], "jobs": []}
    if state_path.exists():
        st_ = np.load(state_path)
        zH, zL = st_["zH"], st_["zL"]
        preds = list(st_["preds"]) if "preds" in st_.files else []
        qs = list(st_["q"]) if "q" in st_.files else []
    else:
        zH = np.repeat(zH0, N, 0)[:, None]
        zL = np.repeat(zL0, N, 0)[:, None]
        preds, qs = [], []
    device_obj = hub.Device(device, A.DEV_BY_NAME[device][1])
    for s in range(len(preds), NSTEPS[task]):
        feeds = {"z_H": [zH[i] for i in range(N)], "z_L": [zL[i] for i in range(N)],
                 "inputs": [inp[i:i + 1] for i in range(N)], "puzzle_embedding": [emb for _ in range(N)]}
        for name in list(feeds):
            want = tshapes[name][1]
            if want in ("uint8", "int8", "uint16", "int16"):
                e = enc[name]
                assert e["dtype"] == want, (name, e, want)
                feeds[name] = [quantize_input(x, e) for x in feeds[name]]
        feeds = {k: feeds[k] for k in tshapes}          # schema order
        for attempt in range(3):
            ij = hub.submit_inference_job(model=model, device=device_obj, inputs=feeds,
                                          name=f"x3_infer_{task}_{tag}_s{s}")
            A.add_job(task, {"kind": "inference", "graph": "stepc", "H": 1, "prec": prec, "device": device,
                             "os": A.DEV_BY_NAME[device][1], "step": s, "job_id": ij.job_id,
                             "from_compile": cj.job_id,
                             "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            meta["jobs"].append({"step": s, "job_id": ij.job_id})
            meta_path.write_text(json.dumps(meta, indent=1))
            ij.wait()
            jst = ij.get_status()
            if jst.code == "SUCCESS":
                break
            print(f"[{task}] {tag} step {s} inference FAILED: {(jst.message or '')[:200]}", flush=True)
            if attempt == 2:
                meta["failed"] = (jst.message or "")[:600]
                meta_path.write_text(json.dumps(meta, indent=1))
                return
        o = named_outputs(ij.download_output_data())
        zH = np.stack([x.astype(np.float32) for x in o["z_H_out"]])
        zL = np.stack([x.astype(np.float32) for x in o["z_L_out"]])
        lg = np.concatenate([x.astype(np.float32) for x in o["logits"]])
        preds.append(lg.argmax(-1))
        qs.append(np.array([float(x.reshape(-1)[0]) for x in o["q_halt"]]))
        np.savez_compressed(state_path, zH=zH, zL=zL, preds=np.stack(preds), q=np.stack(qs))
        ex = (preds[-1] == lab).all(-1).mean()
        meta["steps"].append({"step": s, "job_id": ij.job_id, "exact": float(ex),
                              "cell": float((preds[-1] == lab).mean())})
        meta_path.write_text(json.dumps(meta, indent=1))
        with log_lock:
            print(f"[{task}] {tag} step {s}: exact {ex:.3f} ({ij.job_id})", flush=True)
    meta["done"] = True
    meta_path.write_text(json.dumps(meta, indent=1))


def stage_run(tasks, devices, precs, max_threads):
    lock = threading.Lock()
    work = [(t, d, p) for t in tasks for d in devices for p in precs
            if not (p == "fp32" and not A.DEV_BY_NAME[d][3])]
    threads = []
    for w in work:
        th = threading.Thread(target=chain, args=(*w, lock), daemon=False)
        th.start()
        threads.append(th)
        while sum(t.is_alive() for t in threads) >= max_threads:
            time.sleep(10)
    for th in threads:
        th.join()


# ----------------------------------------------------------------------------- report
def auroc(score, label):
    """AUROC of `score` for predicting label==1 (ties count half)."""
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    s = np.concatenate([pos, neg])
    r = s.argsort().argsort().astype(float) + 1
    for v in np.unique(s):
        m = s == v
        r[m] = r[m].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def stage_report(tasks):
    rows = []
    for task in tasks:
        d = idir(task)
        _, lab = test_set(task)
        hf = np.load(d / "host_fp32.npz") if (d / "host_fp32.npz").exists() else None
        for meta_path in sorted(d.glob("dev_*.json")):
            tag = meta_path.stem[4:]
            prec = tag.split("_")[0]
            device = tag[len(prec) + 1:].replace("_", " ")
            meta = json.loads(meta_path.read_text())
            sp = d / f"dev_{tag}.npz"
            if not sp.exists():
                continue
            r = np.load(sp)
            p = r["preds"]
            done = bool(meta.get("done"))
            row = {"task": task, "device": device, "prec": prec, "steps_done": int(p.shape[0]),
                   "n_steps": meta.get("n_steps", NSTEPS[task]), "done": done, "n": int(lab.shape[0]),
                   "inference_jobs": [j["job_id"] for j in meta["jobs"]]}
            last = p[-1]
            k = p.shape[0]
            corr = (last == lab).all(-1)
            row.update(exact=float(corr.mean()), cell=float((last == lab).mean()))
            if done:
                q = r["q"][-1]
                row["q_halt_auroc_correct"] = auroc(q, corr.astype(int))
            hq = d / f"host_{prec}.npz"
            if hf is not None and done:
                hp = hf["preds"][k - 1]
                row["agree_cell_vs_host_fp32"] = float((last == hp).mean())
                row["agree_puzzle_vs_host_fp32"] = float((last == hp).all(-1).mean())
                row["host_fp32_exact"] = float((hp == lab).all(-1).mean())
                hz = d / f"host_fp32_n{k}.npz"
                ref = np.load(hz) if hz.exists() else (hf if hf["preds"].shape[0] == k else None)
                if ref is not None:
                    a, b = r["zH"].reshape(len(lab), -1), ref["zH"].reshape(len(lab), -1)
                    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
                    row["fidelity_H_vs_host_fp32"] = float(cos.mean())
            if hq.exists() and done and prec != "fp32":
                h = np.load(hq)
                hp = h["preds"][k - 1]
                row["host_qdq_exact"] = float((hp == lab).all(-1).mean())
                row["agree_cell_vs_host_qdq"] = float((last == hp).mean())
                row["agree_puzzle_vs_host_qdq"] = float((last == hp).all(-1).mean())
            row["exact_per_step"] = [float((p[s] == lab).all(-1).mean()) for s in range(p.shape[0])]
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k not in ("inference_jobs", "exact_per_step")}),
                  flush=True)
    (A.ART / "infer_report.json").write_text(json.dumps(rows, indent=1))
    print("wrote", A.ART / "infer_report.json", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["prep", "host", "hostbig", "hostarc", "run", "report"])
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--count", type=int, default=1000)
    ap.add_argument("--tasks", default="sudoku_mlp,sudoku_attn,maze")
    ap.add_argument("--precs", default="w8a8,w4a8,w8a16,w4a16")
    ap.add_argument("--devices", default="Samsung Galaxy S24|Dragonwing RB3 Gen 2 Vision Kit")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--nsteps", type=int, default=0)
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    precs = [p for p in a.precs.split(",") if p]
    if a.stage == "prep":
        stage_prep(tasks, [p for p in precs if p != "fp32"])
    elif a.stage == "host":
        stage_host(tasks, [p for p in precs if p != "fp32"], a.nsteps or None)
    elif a.stage == "hostarc":
        stage_host_arc(precs, a.count, a.nsteps or 8, a.threads)
    elif a.stage == "hostbig":
        for task in tasks:
            for prec in precs:
                stage_hostbig(task, prec, a.start, a.count, a.nsteps or NSTEPS[task], a.threads)
    elif a.stage == "run":
        stage_run(tasks, a.devices.split("|"), precs, a.threads)
    else:
        stage_report(tasks)


if __name__ == "__main__":
    main()
