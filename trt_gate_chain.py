#!/usr/bin/env python3
"""TensorRT as a runtime under the label-free gate (reports/edgetrm_exp/TRT_GATE_AND_LAUNCH.md, part G1).

Per model (Sudoku-MLP, Sudoku-attn, Maze-Hard at H=1, the gate's step graph; HRM Sudoku at H=2, its only export)
and TensorRT engine of E1 (TensorRT 10.13.3.9, batch 1):
  fp16      FP16 kernels
  int8mm    E1's INT8 run: TensorRT entropy calibration (E1's cache), INT8 required on every MatMul
  int8proj  the fix rule: the same, with the attention score matmuls (q.k^T, attn.v) in FP16
an 8-step host loop at batch 1 on the gate's puzzles (test rows 0..99 Sudoku, 0..49 Maze): after every step the
carry goes device -> host -> device and is kept. Reference: the host fp32 carry from ONNX Runtime on CPU (default
optimisation) on X3's stepc_H1.onnx, as host_fp32_n2.npz in the gate; for HRM the dynamic-batch stepc_dyn_H2.onnx.
Per step: exact and cell accuracy, phi_H / phi_L (mean over puzzles of the cosine of the flattened carry with the host
carry at the same step; cos_rows below is tools/aihub_e3_steps12.cos_rows verbatim), and written-cell agreement with
the host predictions (release_check.device_probe). Verdicts as E3_FAULT_TABLE section 3 and E3_GATE_PHI_STEP2, with
the cuts fitted on the phone chains (no re-tuning).

    python trt_gate_chain.py run   --task sudoku_mlp        host reference (cached) + the three engines
    python trt_gate_chain.py score --out trt_gate.json      combine per-task records, verdicts
Env: TG_E1CODE (E1 code dir: e1_energy, aihub_h1, breadth_sweep), TG_WORK (engines under <task>/, carries),
TG_RES (records), TG_X3 (X3's stepc_H1.onnx + host_fp32.npz), TG_ONNX (E1's dynamic exports).
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

HOME = pathlib.Path.home()
E1CODE = pathlib.Path(os.environ.get("TG_E1CODE", HOME / "e1" / "code"))
WORK = pathlib.Path(os.environ.get("TG_WORK", HOME / "trt_gate" / "work"))
RES = pathlib.Path(os.environ.get("TG_RES", HOME / "trt_gate" / "results"))
X3 = pathlib.Path(os.environ.get("TG_X3", HOME / "e1" / "x3"))
ONNXD = pathlib.Path(os.environ.get("TG_ONNX", HOME / "e1" / "onnx"))
sys.path.insert(0, str(E1CODE))

STEPS = 8
PRECS = ("fp16", "int8mm", "int8proj")
TASKS = {"sudoku_mlp": dict(H=1, data="sudoku", n=100), "sudoku_attn": dict(H=1, data="sudoku", n=100),
         "maze": dict(H=1, data="maze", n=50), "hrm_sudoku": dict(H=2, data="sudoku", n=100)}
# Fixed on the phone chains (results/devices/e3_gate_phi_step2.json, cross_account, cut chosen on acct2 at step 2
# vs the host fp32 carry; PASS = phi >= cut). Not re-tuned here.
CUTS = {"phi_H_mid": 0.7946065275925629, "phi_H_minpass": 0.8206705565659004,
        "phi_L_mid": 0.8336037900632673, "phi_L_minpass": 0.8779769526762154}
WRITTEN_PASS, RATIO_PASS = 0.5, 0.95       # release_check THRESHOLDS device_written_pass, written_pass
OUTS = ["z_H_out", "z_L_out", "logits", "q_halt"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def md5(p):
    return hashlib.md5(pathlib.Path(p).read_bytes()).hexdigest()[:12]


def cos_rows(x, y):
    x = x.reshape(x.shape[0], -1).astype(np.float64); y = y.reshape(y.shape[0], -1).astype(np.float64)
    return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)


def ref_graph(task):
    H = TASKS[task]["H"]
    return X3 / task / "stepc_H1.onnx" if task != "hrm_sudoku" else ONNXD / task / f"stepc_dyn_H{H}.onnx"


def host_ref(task, x, zh0, zl0, pe):
    """ORT CPU fp32 host loop, batch 1 per call (aihub_infer.host_loop), carries of every step kept."""
    out = WORK / task / "host_fp32_n8.npz"
    if out.exists():
        d = np.load(out)
        return {k: d[k] for k in d.files}
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
    sess = ort.InferenceSession(str(ref_graph(task)), so, providers=["CPUExecutionProvider"])
    ty = {i.name: i.type for i in sess.get_inputs()}
    cast = lambda n, a: a.astype({"tensor(float)": np.float32, "tensor(int32)": np.int32,   # noqa: E731
                                  "tensor(int64)": np.int64}[ty[n]])
    N, S = len(x), zh0.shape[1]
    zH, zL = np.repeat(zh0, N, 0)[:, None], np.repeat(zl0, N, 0)[:, None]
    zHs, zLs, preds, qs = [], [], [], []
    t0 = time.time()
    for s in range(STEPS):
        nzH, nzL, pr, qh = [], [], [], []
        for i in range(N):
            o = sess.run(OUTS, {"z_H": cast("z_H", zH[i]), "z_L": cast("z_L", zL[i]), "inputs": cast("inputs", x[i:i + 1]),
                                "puzzle_embedding": cast("puzzle_embedding", pe)})
            nzH.append(o[0].astype(np.float32)); nzL.append(o[1].astype(np.float32))
            pr.append(o[2].argmax(-1)); qh.append(float(o[3].reshape(-1)[0]))
        zH, zL = np.stack(nzH), np.stack(nzL)
        zHs.append(zH[:, 0]); zLs.append(zL[:, 0]); preds.append(np.concatenate(pr)); qs.append(qh)
        log(task, "host fp32 step", s + 1, f"{time.time() - t0:.0f}s")
    r = dict(zH=np.stack(zHs), zL=np.stack(zLs), preds=np.stack(preds).astype(np.int16), q=np.array(qs, np.float32))
    np.savez(out, **r)
    return r


def trt_chain(task, prec, x):
    """Host loop on the E1 engine at batch 1: step, carry device -> host (kept) -> device."""
    import torch
    import e1_energy as E
    torch.cuda.set_stream(torch.cuda.Stream())          # as e1_energy.make_runner (no legacy-stream syncs)
    H = TASKS[task]["H"]
    ep = WORK / task / f"dyn_H{H}_b1_{prec}.engine"
    r = E.TRTRunner(ep, task, H, 1)
    N = len(x)
    zH = np.zeros((STEPS, N) + tuple(r.zH[0].shape[1:]), np.float32)
    zL = np.zeros_like(zH)
    preds = np.zeros((STEPS, N, x.shape[1]), np.int16)
    q = np.zeros((STEPS, N), np.float32)
    xg = torch.as_tensor(x, device="cuda")
    t0 = time.time()
    for i in range(N):
        r.load(xg[i:i + 1])
        for s in range(STEPS):
            r.step()
            c = r.cur
            hH, hL = r.zH[c].float().cpu().numpy(), r.zL[c].float().cpu().numpy()     # device -> host
            r.zH[c].copy_(torch.from_numpy(hH).to(r.zH[c].dtype))                     # host -> device
            r.zL[c].copy_(torch.from_numpy(hL).to(r.zL[c].dtype))
            zH[s, i], zL[s, i] = hH[0], hL[0]
            preds[s, i] = r.logits[0].argmax(-1).cpu().numpy()
            q[s, i] = float(r.qh.float().reshape(-1)[0])
    log(task, prec, f"{N} puzzles x {STEPS} steps {time.time() - t0:.0f}s")
    return dict(zH=zH, zL=zL, preds=preds, q=q, engine=str(ep), engine_md5=md5(ep))


def per_step(c, ref, x, lab):
    out = dict(exact=[], cell=[], written=[], phi_H=[], phi_L=[], phi_H_min=[], phi_L_min=[], cos_H=[], cos_L=[])
    for s in range(STEPS):
        p, rp = c["preds"][s], ref["preds"][s]
        out["exact"].append(float((p == lab).all(-1).mean()))
        out["cell"].append(float((p == lab).mean()))
        ch = rp != x                                      # release_check.device_probe: cells fp32 writes
        out["written"].append(float((((p == rp) & ch).sum(-1) / np.maximum(ch.sum(-1), 1)).mean()))
        cH, cL = cos_rows(c["zH"][s], ref["zH"][s]), cos_rows(c["zL"][s], ref["zL"][s])
        out["phi_H"].append(float(cH.mean())); out["phi_L"].append(float(cL.mean()))
        out["phi_H_min"].append(float(cH.min())); out["phi_L_min"].append(float(cL.min()))
        out["cos_H"].append(cH.round(6).tolist()); out["cos_L"].append(cL.round(6).tolist())
    return out


def run(task):
    import torch
    import e1_energy as E
    assert torch.cuda.is_available(), "CUDA not available: abort"
    torch.backends.cuda.matmul.allow_tf32 = False
    import tensorrt as trt
    t = TASKS[task]
    (WORK / task).mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    x, lab = E.load_rows(t["data"], 0, t["n"])
    inner, pe = E.build_inner(task, t["H"])
    zh0, zl0 = E.init_carry(inner)
    S = inner.config.seq_len + inner.puzzle_emb_len
    zh0 = np.broadcast_to(zh0.numpy(), (1, S, 512)).astype(np.float32)
    zl0 = np.broadcast_to(zl0.numpy(), (1, S, 512)).astype(np.float32)
    pe = pe.numpy().astype(np.float32)
    ref = host_ref(task, x, zh0, zl0, pe)
    rec = dict(task=task, H=t["H"], n=int(len(x)), puzzle_rows=[0, int(len(x))], steps=STEPS,
               ref_graph=str(ref_graph(task)), ref_graph_md5=md5(ref_graph(task)), trt=trt.__version__,
               device=torch.cuda.get_device_name(0), host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"))
    hp = ref["preds"]
    rec["host_fp32"] = dict(exact=[float((hp[s] == lab).all(-1).mean()) for s in range(STEPS)],
                            cell=[float((hp[s] == lab).mean()) for s in range(STEPS)])
    x3h = X3 / task / "infer" / "host_fp32.npz"
    if x3h.exists():                                      # the gate's host fp32 record (X3, p4d ORT)
        d = np.load(x3h)
        k = min(STEPS, d["preds"].shape[0])
        rec["host_vs_x3_record_pred_agree"] = [float((hp[s] == d["preds"][s]).mean()) for s in range(k)]
        if d["preds"].shape[0] == STEPS:
            rec["host_vs_x3_record_cosH_step8_min"] = float(cos_rows(ref["zH"][-1], d["zH"]).min())
    rec["chains"] = {}
    for prec in PRECS:
        c = trt_chain(task, prec, x)
        np.save(WORK / task / f"carry_{prec}_zH.npy", c["zH"])            # regenerable; /data on Babel
        np.save(WORK / task / f"carry_{prec}_zL.npy", c["zL"])
        rec["chains"][prec] = dict(engine=c["engine"], engine_md5=c["engine_md5"], **per_step(c, ref, x, lab))
        np.savez_compressed(RES / f"{task}_{prec}_preds.npz", preds=c["preds"].astype(np.uint8), q=c["q"])
        ch = rec["chains"][prec]
        log(task, prec, "exact", [round(v, 2) for v in ch["exact"]], "phi_H", [round(v, 3) for v in ch["phi_H"]])
    np.savez_compressed(RES / f"{task}_host_fp32_preds.npz", preds=hp.astype(np.uint8), q=ref["q"])
    (RES / f"{task}.json").write_text(json.dumps(rec))
    log("wrote", RES / f"{task}.json")


def cls(ratio):
    return "PASS" if ratio >= RATIO_PASS else ("FAIL" if ratio < 0.5 else "WARN")


def score(out):
    rows = []
    for task in TASKS:
        p = RES / f"{task}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text())
        he, hc = rec["host_fp32"]["exact"], rec["host_fp32"]["cell"]
        for prec, c in rec["chains"].items():
            r = dict(task=task, prec=prec, exact2=c["exact"][1], exact8=c["exact"][-1], host_exact8=he[-1],
                     cell2=c["cell"][1], host_cell2=hc[1], written2=c["written"][1],
                     phi_H2=c["phi_H"][1], phi_L2=c["phi_L"][1], phi_H1=c["phi_H"][0], phi_H8=c["phi_H"][-1])
            r["depth8"] = cls(r["exact8"] / max(r["host_exact8"], 1e-9))          # E3_FAULT_TABLE s3 step-8 verdict
            r["probe_cell2"] = cls(r["cell2"] / max(r["host_cell2"], 1e-9))       # E3_FAULT_TABLE s3 proxy probe
            r["probe_written2"] = "PASS" if r["written2"] >= WRITTEN_PASS else "FLAG"   # release_check.device_probe
            for k, cut in CUTS.items():
                v = r["phi_H2"] if k.startswith("phi_H") else r["phi_L2"]
                r[f"gate_{k}"] = "PASS" if v >= cut else "FLAG"
            rows.append(r)
    res = dict(cuts=CUTS, written_pass=WRITTEN_PASS, ratio_pass=RATIO_PASS, rows=rows)
    pathlib.Path(out).write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"{r['task']:12s} {r['prec']:9s} ex2 {r['exact2']:.2f} ex8 {r['exact8']:.2f}/{r['host_exact8']:.2f} "
              f"{r['depth8']:4s} | phi_H2 {r['phi_H2']:.4f} phi_L2 {r['phi_L2']:.4f} gateH {r['gate_phi_H_mid']}/"
              f"{r['gate_phi_H_minpass']} | cell2 {r['probe_cell2']} written2 {r['written2']:.3f} {r['probe_written2']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "score"])
    ap.add_argument("--task", choices=list(TASKS))
    ap.add_argument("--out", default=str(RES / "trt_gate.json"))
    a = ap.parse_args()
    if a.stage == "run":
        run(a.task)
    else:
        score(a.out)


if __name__ == "__main__":
    main()
