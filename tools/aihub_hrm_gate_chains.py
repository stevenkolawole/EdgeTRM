#!/usr/bin/env python3
"""HRM Sudoku (official checkpoint, H = 2 as in the TensorRT gate run) as 8-step chains on the phones, with the
two-step phi gate computed on the fly (reports/edgetrm_exp/AIHUB_HRM_GATE.md). Runs on the Babel login node, main AI
Hub account, QAIRT 2.45 pinned; at most --maxactive inference jobs in flight.

    export AIHUB_ART=$HOME/aihub_h/art HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
    python aihub_hrm_gate_chains.py export          # static batch-1 fp32 step graph, host fp32 loop (ORT), data + calibration
    python aihub_hrm_gate_chains.py quantize        # hub quantize job w8a8 (calibration = test rows 900..931, carries of steps 0-7)
    python aihub_hrm_gate_chains.py prep            # download the QDQ graph, break the RMSNorm pattern -> w8a8_nofuse2
    python aihub_hrm_gate_chains.py run --steps 8 --maxactive 2 fp32:S24 w8a8_nofuse2:S24 ...
    python aihub_hrm_gate_chains.py status fp32:S24 ...

Per step a chain writes, beside its state npz, <tag>.phi.json (phi_H / phi_L = mean over the 100 puzzles of the cosine of
the flattened carry with the host fp32 carry of the same step, plus the per-puzzle minimum) and, for steps 1-2,
<tag>.carry12.npz (the device z_H / z_L). The state npz keeps only the last carry.
"""
import argparse
import json
import os
import pathlib
import shutil
import sys
import time
import zipfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import aihub_h1 as A  # noqa: E402
if os.environ.get("AIHUB_ART"):
    A.ART = pathlib.Path(os.environ["AIHUB_ART"])
import aihub_acct as X  # noqa: E402
import aihub_e3_followup as E  # noqa: E402
import aihub_infer as I  # noqa: E402
import aihub_nofuse_cpu_chains as H  # noqa: E402  (MainAcct; its atomic_npz patch is replaced below)

TASK, HC, NSTEPS, N = "hrm_sudoku", 2, 8, 100
DEV = H.DEV
OUTS = ["z_H_out", "z_L_out", "logits", "q_halt"]
E.SRC[(TASK, "w8a8_nofuse2")] = ("qdq_w8a8", "variant_CA_nofuse2.onnx")
_real_npz = H._real_npz


def work():
    d = A.ART / TASK
    d.mkdir(parents=True, exist_ok=True)
    return d


def cos_rows(x, y):
    x = x.reshape(x.shape[0], -1).astype(np.float64); y = y.reshape(y.shape[0], -1).astype(np.float64)
    return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)


_HOST = {}


def host():
    if not _HOST:
        d = np.load(work() / "host_fp32_n8.npz")
        _HOST.update({k: d[k] for k in d.files})
    return _HOST


def _write_json(p, obj):
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def _npz_phi(path, **arrays):
    """Chain.step's state writer: phi sidecar for the step just ingested, carries of steps 1-2, last carry only."""
    zH, zL, preds = arrays["zH"][:, 0], arrays["zL"][:, 0], arrays["preds"]
    s = preds.shape[0] - 1
    h = host()
    cH, cL = cos_rows(zH, h["zH"][s]), cos_rows(zL, h["zL"][s])
    side = path.with_name(path.stem + ".phi.json")
    rec = json.loads(side.read_text()) if side.exists() else {}
    agree = float((preds[-1] == h["preds"][s]).mean())
    rec[str(s + 1)] = {"phi_H": float(cH.mean()), "phi_L": float(cL.mean()), "phi_H_min": float(cH.min()),
                       "phi_L_min": float(cL.min()), "pred_agree_cell": agree,
                       "pred_agree_puzzle": float((preds[-1] == h["preds"][s]).all(-1).mean())}
    _write_json(side, rec)
    if s < 2:
        cf = path.with_name(path.stem + ".carry12.npz")
        d = {k: v for k, v in np.load(cf).items()} if cf.exists() else {}
        d[f"zH_s{s + 1}"], d[f"zL_s{s + 1}"] = zH.astype(np.float32), zL.astype(np.float32)
        _real_npz(cf, **d)
    _real_npz(path, zH=arrays["zH"], zL=arrays["zL"], preds=preds, q=arrays["q"])


X.atomic_npz = _npz_phi


class HChain(X.Chain):
    def data(self):
        if self._data is None:
            d = np.load(work() / "hrm_data.npz")
            self._data = (d["inp"], d["lab"], (d["zH0"], d["zL0"], d["emb"]))
        return self._data


# ----------------------------------------------------------------------------- export + host
def make_step(inner):
    import torch

    class Emb(torch.nn.Module):
        def __init__(self, holder):
            super().__init__()
            self.h = [holder]

        def forward(self, idx):
            return self.h[0]._pe

    class Step(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self._pe = inner, None
            self.Carry = type(inner.empty_carry(1))
            self.inner.puzzle_emb = Emb(self)
            self.register_buffer("pid", torch.zeros(1, dtype=torch.int32), persistent=False)

        def forward(self, z_H, z_L, inputs, puzzle_embedding):
            self._pe = puzzle_embedding
            nc, logits, (qh, _qc) = self.inner(self.Carry(z_H=z_H, z_L=z_L),
                                               {"inputs": inputs, "puzzle_identifiers": self.pid})
            return nc.z_H, nc.z_L, logits, qh

    return Step(inner).eval().requires_grad_(False)


def export():
    import torch
    import onnxruntime as ort
    sys.path.insert(0, str(A.TRM_SRC))
    import breadth_sweep as BS
    torch.set_num_threads(2)
    m, _ = BS.build_hrm("hrm_sudoku", HC, 16, 1)
    inner = m.inner.eval()
    S = inner.config.seq_len + inner.puzzle_emb_len
    emb = inner.puzzle_emb.weights[:1].float().clone()
    zH0, zL0 = inner.H_init.detach().float().clone(), inner.L_init.detach().float().clone()
    step = make_step(inner)
    w = work()
    q, a = BS.load_qa("sudoku", 1000)                       # test rows; eval 0..99 (the TensorRT gate's rows), calibration 900..931
    inp, lab = q[:N].astype(np.int32), a[:N].astype(np.int64)
    cal_inp = q[900:932].astype(np.int32)
    args = (zH0.expand(1, S, -1).clone(), zL0.expand(1, S, -1).clone(), torch.as_tensor(inp[:1]), emb)
    path = w / "stepc_H1.onnx"      # name expected by aihub_acct.source_path for fp32 (the graph is the HRM H=2 step)
    with torch.no_grad():
        ref = step(*args)
    torch.onnx.export(step, args, str(path), input_names=["z_H", "z_L", "inputs", "puzzle_embedding"],
                      output_names=OUTS, opset_version=17, dynamo=False)
    so = ort.SessionOptions(); so.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    got = sess.run(OUTS, {"z_H": args[0].numpy(), "z_L": args[1].numpy(), "inputs": args[2].numpy(),
                          "puzzle_embedding": emb.numpy()})
    print("export", path.stat().st_size / 1e6, "MB; ort-vs-torch max abs", [float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref)], flush=True)
    np.savez(w / "hrm_data.npz", inp=inp, lab=lab, zH0=zH0.expand(1, S, -1).numpy(), zL0=zL0.expand(1, S, -1).numpy(), emb=emb.numpy())
    # host fp32 loop (ORT CPU default optimisation, batch 1 per call), carries of every step kept
    zH, zL = np.repeat(zH0.expand(1, S, -1).numpy(), N, 0)[:, None], np.repeat(zL0.expand(1, S, -1).numpy(), N, 0)[:, None]
    zHs, zLs, preds, qs = [], [], [], []
    t0 = time.time()
    for s in range(NSTEPS):
        nH, nL, pr, qh = [], [], [], []
        for i in range(N):
            o = sess.run(OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1], "puzzle_embedding": emb.numpy()})
            nH.append(o[0]); nL.append(o[1]); pr.append(o[2].argmax(-1)); qh.append(float(o[3].reshape(-1)[0]))
        zH, zL = np.stack(nH), np.stack(nL)
        zHs.append(zH[:, 0]); zLs.append(zL[:, 0]); preds.append(np.concatenate(pr)); qs.append(qh)
        print("host step", s + 1, "exact", float((preds[-1] == lab).all(-1).mean()), "cell", float((preds[-1] == lab).mean()),
              f"{time.time() - t0:.0f}s", flush=True)
    np.savez(w / "host_fp32_n8.npz", zH=np.stack(zHs), zL=np.stack(zLs), preds=np.stack(preds).astype(np.int16), q=np.array(qs, np.float32))


def calib():
    """calibration: carries of test rows 900..931 at a random step of an 8-step torch rollout"""
    import torch
    sys.path.insert(0, str(A.TRM_SRC))
    import breadth_sweep as BS
    torch.set_num_threads(2)
    m, _ = BS.build_hrm("hrm_sudoku", HC, 16, 1)
    inner = m.inner.eval()
    S = inner.config.seq_len + inner.puzzle_emb_len
    emb = inner.puzzle_emb.weights[:1].float().clone()
    zH0, zL0 = inner.H_init.detach().float().clone(), inner.L_init.detach().float().clone()
    step = make_step(inner)
    w = work()
    q, _a = BS.load_qa("sudoku", 1000)
    cal_inp = q[900:932].astype(np.int32)
    rng = np.random.default_rng(0)
    x = torch.as_tensor(cal_inp)
    cH = zH0.expand(len(cal_inp), S, -1).clone(); cL = zL0.expand(len(cal_inp), S, -1).clone()
    states = []
    with torch.no_grad():
        for s in range(NSTEPS):
            states.append((cH.clone(), cL.clone()))
            nh, nl = [], []
            for b in range(0, len(x), 8):
                o = step(cH[b:b + 8], cL[b:b + 8], x[b:b + 8], emb.expand(len(x[b:b + 8]), -1))
                nh.append(o[0]); nl.append(o[1])
            cH, cL = torch.cat(nh), torch.cat(nl)
    pick = [int(v) for v in rng.integers(0, NSTEPS, len(cal_inp))]
    np.savez_compressed(w / "calib.npz",
                        z_H=np.stack([states[s][0][i].numpy() for i, s in enumerate(pick)])[:, None].astype(np.float32),
                        z_L=np.stack([states[s][1][i].numpy() for i, s in enumerate(pick)])[:, None].astype(np.float32),
                        inputs=cal_inp[:, None], puzzle_embedding=np.repeat(emb.numpy()[None], len(cal_inp), 0).astype(np.float32))
    print("calibration steps", pick, flush=True)


def quantize():
    import qai_hub as hub
    C = hub.Client()
    d = np.load(work() / "calib.npz")
    names = ["z_H", "z_L", "inputs", "puzzle_embedding"]
    calib = {k: [d[k][i] for i in range(d[k].shape[0])] for k in names}
    j = C.submit_quantize_job(model=str(work() / "stepc_H1.onnx"), calibration_data=calib,
                              weights_dtype=hub.QuantizeDtype.INT8, activations_dtype=hub.QuantizeDtype.INT8,
                              name="hrm_sudoku_stepc_H2_w8a8")
    (work() / "quantize_w8a8.json").write_text(json.dumps({"job_id": j.job_id}))
    print("quantize job", j.job_id, flush=True)


def prep():
    import onnx
    import qai_hub as hub
    C = hub.Client()
    jid = json.loads((work() / "quantize_w8a8.json").read_text())["job_id"]
    job = C.get_job(jid)
    print("quantize status", job.wait().code if hasattr(job, "wait") else job.get_status().code, flush=True)
    base = I.idir(TASK)
    dd = base / "qdq_w8a8"
    if not (dd / "model.onnx").exists():
        z = job.get_target_model().download(str(base / "qdq_dl"))
        tmp = base / "unz"
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
        inner = next(tmp.iterdir())
        if dd.exists():
            shutil.rmtree(dd)
        inner.rename(dd)
        shutil.rmtree(tmp)
        pathlib.Path(z).unlink()
    out = dd / "variant_CA_nofuse2.onnx"
    m = onnx.load(str(dd / "model.onnx"), load_external_data=False)
    n = E.break_rmsnorm(m)
    onnx.save(m, str(out))
    print("rewrote", n, "ReduceMean nodes ->", out, flush=True)


# ----------------------------------------------------------------------------- chains
def run(specs, steps, maxactive):
    acct = H.MainAcct()
    X.OS.update({d.name: d.os for d in acct.C.get_devices() if d.name not in X.OS})
    chains = []
    for s in specs:
        prec, dev = s.split(":")
        c = HChain(acct, TASK, prec, DEV[dev], "qnn_dlc", steps)
        chains.append((s, c))
    print(f"[hrm] {len(chains)} chains, {steps} steps, maxactive {maxactive}", flush=True)
    while True:
        live = [c for _, c in chains if not c.done]
        started = [c for c in live if c.meta.get("current") or c.meta.get("compile_job") or c.steps_done]
        allowed = {id(c) for c in started[:maxactive]}
        for c in live:
            if len(allowed) < maxactive:
                allowed.add(id(c))
        for s, c in chains:
            if c.done:
                H.slim(c)
                continue
            if id(c) not in allowed:
                continue
            try:
                c.step({})
            except Exception as exc:  # noqa: BLE001
                print(f"[{s}] ERROR {type(exc).__name__}: {str(exc)[:400]}", flush=True)
            errs = c.meta.get("errors") or []
            if errs and not c.meta.get("failed") and "MODEL_GRAPH_ERROR" in errs[-1]["error"]:
                c.meta["failed"] = "deterministic compose error: " + errs[-1]["error"][:300]
                c.save()
            comp = c.meta.get("compile_job")
            if comp and "converter" not in c.meta and s.split(":")[0] != "fp32":
                rec = [j for j in acct.find(job_id=comp) if j.get("kind") in ("compile", "compile_orphan")]
                if rec:
                    if "converter" not in rec[0]:
                        rec[0].update(E.annotate(acct, rec[0]))
                        acct.save()
                    c.meta["converter"] = {k: rec[0].get(k) for k in ("converter", "float_fallback_ops", "dlc_dtype_counts", "dlc_rms_norm_ops")}
                    c.save()
        act = [(s, c) for s, c in chains if not c.done]
        print(f"[hrm] {time.strftime('%H:%M:%S', time.gmtime())} active {len(act)}: " + ", ".join(f"{s} s{c.steps_done}" for s, c in act), flush=True)
        if not act:
            break
        time.sleep(90)


def status(specs):
    acct = H.MainAcct()
    for s in specs:
        prec, dev = s.split(":")
        c = HChain(acct, TASK, prec, DEV[dev], "qnn_dlc", 0)
        ex = " ".join(f"{x['exact'] * 100:.0f}" for x in c.meta["steps"])
        print(f"{s:22s} compile {c.meta.get('compile_job')} qairt {c.meta.get('qairt_sdk_version')} guard "
              f"{(c.meta.get('guard') or {}).get('verdict')} exact [{ex}] "
              f"{'done' if c.meta.get('done') else ('FAILED ' + str(c.meta.get('failed'))[:100] if c.meta.get('failed') else 'running')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "calib", "quantize", "prep", "run", "status"])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--maxactive", type=int, default=2)
    ap.add_argument("chains", nargs="*")
    a = ap.parse_intermixed_args()
    {"export": export, "calib": calib, "quantize": quantize, "prep": prep}[a.cmd]() if a.cmd in ("export", "calib", "quantize", "prep") else \
        (run(a.chains, a.steps, a.maxactive) if a.cmd == "run" else status(a.chains))
