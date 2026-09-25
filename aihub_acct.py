#!/usr/bin/env python3
"""On-device chains on additional AI Hub accounts (profiles acct2 / acct3 in ~/.qai_hub; tokens are never
read, printed or logged here: qai_hub.Client(profile=...) loads them).

Why: the main account's compile queue is FIFO and ~900 deep (X3.md, Sep 25 05:02 UTC); each extra account
has its own empty queue. Models do not carry across accounts, so every graph is re-uploaded.

Plans (task, precision, device, runtime, supervision steps):
  acct2  the two diagnostics: W4A16 via precompiled_qnn_onnx (1 step, compared with the host QDQ graph at
         step 1) and the B2 fix w8a8rms16 (all activations 8 bit, every RMSNorm tensor uint16), 8 steps, S24
  acct3  M4's probe: Sudoku-MLP / Sudoku-attn fp32 (runs fp16 on HTP), w8a8, w8a16; Maze fp32, w8a8;
         the reference-QDQ override w8a8sqvar16 (x*x + variance path at 16 bit; wa_collapse B2ab);
         on the Galaxy S24 and the Snapdragon 8 Elite QRD, 8 steps, 100 test puzzles (Maze 50)

Chain state: ~/aihub_artifacts/h1/<task>/infer/dev_<prec>_<device>_<acct>.{json,npz} (same format as
aihub_sched.Chain, so device_acc.py / aihub_infer.py report read them); job records with the account in
~/aihub_artifacts/h1/acct_<acct>/jobs.json.

    python aihub_acct.py            (loops until every chain is done or failed; resumable)
"""
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402
import aihub_infer as I  # noqa: E402
from aihub_sched import named_outputs  # noqa: E402

S24, E8 = "Samsung Galaxy S24", "Snapdragon 8 Elite QRD"
OS = {S24: "14", E8: "15"}
PLANS = {
    "acct2": [("sudoku_mlp", "w4a16", S24, "precompiled_qnn_onnx", 1),
              ("sudoku_mlp", "w8a8rms16", S24, "qnn_dlc", 8),
              # device-legal rebuilds (embedding table encoding follows its 8-bit Gather output)
              ("sudoku_mlp", "w8a8rms16t", S24, "qnn_dlc", 8)],
    "acct3": [(t, p, d, "qnn_dlc", 8) for d in (S24, E8)
              for t, p in (("sudoku_mlp", "fp32"), ("sudoku_mlp", "w8a8"), ("sudoku_mlp", "w8a16"),
                           ("sudoku_mlp", "w8a8sqvar16t"), ("sudoku_attn", "fp32"), ("sudoku_attn", "w8a8"),
                           ("sudoku_attn", "w8a16"), ("maze", "fp32"), ("maze", "w8a8"))],
}


def source_path(task, prec):
    d = I.idir(task)
    if prec == "fp32":
        return A.ART / task / "stepc_H1.onnx"
    # an upload directory must hold exactly model.onnx + model.data (qdq_* dirs also hold ablation variants)
    base, onnx_file = {"w8a8rms16": ("qdq_w8a16", "variant_B2_rms16.onnx"),
                       "w8a8sqvar16": ("qdq_w8a16", "variant_B2ab_sq_var16.onnx"),
                       "w8a8rms16t": ("qdq_w8a16", "variant_B2t_rms16.onnx"),
                       "w8a8sqvar16t": ("qdq_w8a16", "variant_B2abt_sq_var16.onnx")}.get(prec, (f"qdq_{prec}", "model.onnx"))
    up = d / f"upload_{prec}"
    if not (up / "model.onnx").exists():
        up.mkdir(exist_ok=True)
        for f in up.iterdir():
            f.unlink()
        os.link(d / base / "model.data", up / "model.data")
        os.link(d / base / onnx_file, up / "model.onnx")
    if not (d / f"io_enc_{prec}.json").exists():
        write_io_enc(up / "model.onnx", d / f"io_enc_{prec}.json")
    return up


def write_io_enc(model_path, out):
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(model_path), load_external_data=False).graph
    vals = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            vals[n.output[0]] = n.attribute[0].t
    ins, enc = {i.name for i in g.input}, {}
    for n in g.node:
        if n.op_type == "QuantizeLinear" and n.input[0] in ins:
            zt = vals[n.input[2]]
            enc[n.input[0]] = {"scale": float(numpy_helper.to_array(vals[n.input[1]])),
                               "zero_point": float(numpy_helper.to_array(zt)),
                               "dtype": onnx.TensorProto.DataType.Name(zt.data_type).lower()}
    out.write_text(json.dumps({"source": str(model_path), "inputs": enc}, indent=1))


class Acct:
    def __init__(self, name):
        import qai_hub as hub
        self.hub, self.name = hub, name
        self.C = hub.Client(profile=name)
        self.dir = A.ART / f"acct_{name}"
        self.dir.mkdir(exist_ok=True)
        self.path = self.dir / "jobs.json"
        self.rec = json.loads(self.path.read_text()) if self.path.exists() else {"account": name, "jobs": []}

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.rec, indent=1))
        tmp.replace(self.path)

    def add(self, **kw):
        kw.update(account=self.name, submitted=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.rec["jobs"].append(kw)
        self.save()

    def find(self, **kw):
        return [j for j in self.rec["jobs"] if all(j.get(k) == v for k, v in kw.items())]

    def status(self):
        out, off = {}, 0
        while off < 3000:
            s = self.C.get_job_summaries(offset=off, limit=100)
            if not s:
                break
            for x in s:
                out[x.job_id] = x.status.code
            off += len(s)
        return out

    def model(self, task, prec):
        m = self.find(kind="upload", task=task, prec=prec)
        if m:
            return self.C.get_model(m[-1]["model_id"])
        mdl = self.C.upload_model(str(source_path(task, prec)), name=f"x3_{self.name}_{task}_stepc_H1_{prec}")
        self.add(kind="upload", task=task, prec=prec, model_id=mdl.model_id)
        print(f"[{self.name}] uploaded {task} {prec} -> {mdl.model_id}", flush=True)
        return mdl

    def compile(self, task, prec, device, runtime):
        """qnn_dlc targets are device-agnostic: one compile per (task, prec); precompiled_qnn_onnx is
        per device."""
        key = dict(kind="compile", task=task, prec=prec, runtime=runtime)
        if runtime != "qnn_dlc":
            key["device"] = device
        c = self.find(**key)
        if c:
            return c[-1]
        opts = f"--target_runtime {runtime}" + ("" if prec == "fp32" else " --quantize_io")
        cj = self.C.submit_compile_job(model=self.model(task, prec), device=self.hub.Device(device, OS[device]),
                                       options=opts, name=f"x3_{self.name}_{task}_stepc_{prec}_{runtime}")
        self.add(kind="compile", task=task, prec=prec, runtime=runtime, device=device, options=opts,
                 job_id=cj.job_id)
        print(f"[{self.name}] compile {task} {prec} {runtime} ({device}) -> {cj.job_id}", flush=True)
        return self.find(**key)[-1]


class Chain:
    def __init__(self, acct, task, prec, device, runtime, nsteps):
        self.a, self.task, self.prec, self.device, self.runtime, self.nsteps = acct, task, prec, device, runtime, nsteps
        self.tag = f"{prec}_{device.replace(' ', '_')}_{acct.name}"
        d = I.idir(task)
        self.meta_path, self.state_path = d / f"dev_{self.tag}.json", d / f"dev_{self.tag}.npz"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else \
            {"steps": [], "jobs": [], "n_steps": nsteps, "account": acct.name, "runtime": runtime}
        self._data = None

    def save(self):
        self.meta_path.write_text(json.dumps(self.meta, indent=1))

    @property
    def done(self):
        return self.meta.get("done") or self.meta.get("failed")

    def data(self):
        if self._data is None:
            inp, lab = I.test_set(self.task)
            self._data = (inp, lab, I.initial_state(self.task))
        return self._data

    def step(self, bs):
        cur = self.meta.get("current")
        if cur:
            code = bs.get(cur["job_id"]) or self.a.C.get_job(cur["job_id"]).get_status().code
            if code not in ("SUCCESS", "FAILED"):
                return
            job = self.a.C.get_job(cur["job_id"])
            self.meta.pop("current")
            if code == "FAILED":
                msg = (job.get_status().message or "")[:600]
                self.meta.setdefault("errors", []).append({"step": cur["step"], "job_id": cur["job_id"], "error": msg})
                if len(self.meta["errors"]) >= 3:
                    self.meta["failed"] = msg
                self.save()
                print(f"[chain {self.task} {self.tag}] step {cur['step']} FAILED: {msg[:160]}", flush=True)
                return
            o = named_outputs(job.download_output_data())
            inp, lab, _ = self.data()
            zH = np.stack([x.astype(np.float32) for x in o["z_H_out"]])
            zL = np.stack([x.astype(np.float32) for x in o["z_L_out"]])
            lg = np.concatenate([x.astype(np.float32) for x in o["logits"]])
            q = np.array([float(x.reshape(-1)[0]) for x in o["q_halt"]])
            preds, qs = ([], [])
            if self.state_path.exists():
                st = np.load(self.state_path)
                preds, qs = list(st["preds"]), list(st["q"])
            assert len(preds) == cur["step"], (self.tag, len(preds), cur)     # one state row per device step
            preds.append(lg.argmax(-1))
            qs.append(q)
            np.savez_compressed(self.state_path, zH=zH, zL=zL, preds=np.stack(preds), q=np.stack(qs))
            ex = float((preds[-1] == lab).all(-1).mean())
            self.meta["steps"].append({"step": cur["step"], "job_id": cur["job_id"], "exact": ex,
                                       "cell": float((preds[-1] == lab).mean())})
            if len(preds) >= self.nsteps:
                self.meta["done"] = True
            self.save()
            print(f"[chain {self.task} {self.tag}] step {cur['step']}: exact {ex:.3f}"
                  + ("  DONE" if self.meta.get("done") else ""), flush=True)
            return
        if self.done:
            return
        comp = self.a.compile(self.task, self.prec, self.device, self.runtime)
        code = bs.get(comp["job_id"]) or self.a.C.get_job(comp["job_id"]).get_status().code
        if code == "FAILED":
            msg = (self.a.C.get_job(comp["job_id"]).get_status().message or "")[:600]
            self.meta["failed"] = "compile failed: " + msg
            self.save()
            print(f"[chain {self.task} {self.tag}] compile FAILED: {msg[:160]}", flush=True)
            return
        if code != "SUCCESS":
            return
        cj = self.a.C.get_job(comp["job_id"])
        tshapes = cj.get_target_shapes()
        inp, lab, (zH0, zL0, emb) = self.data()
        N = inp.shape[0]
        if self.state_path.exists():
            st = np.load(self.state_path)
            zH, zL, s = st["zH"], st["zL"], st["preds"].shape[0]
        else:
            zH, zL, s = np.repeat(zH0, N, 0)[:, None], np.repeat(zL0, N, 0)[:, None], 0
        enc = json.loads((I.idir(self.task) / f"io_enc_{self.prec}.json").read_text())["inputs"] \
            if self.prec != "fp32" else {}
        feeds = {"z_H": [zH[i] for i in range(N)], "z_L": [zL[i] for i in range(N)],
                 "inputs": [inp[i:i + 1] for i in range(N)], "puzzle_embedding": [emb for _ in range(N)]}
        for name in list(feeds):
            want = tshapes[name][1]
            if want in ("uint8", "int8", "uint16", "int16"):
                assert enc[name]["dtype"] == want, (name, enc[name], want)
                feeds[name] = [I.quantize_input(x, enc[name]) for x in feeds[name]]
        feeds = {k: feeds[k] for k in tshapes}
        ij = self.a.C.submit_inference_job(model=cj.get_target_model(),
                                           device=self.a.hub.Device(self.device, OS[self.device]),
                                           inputs=feeds, name=f"x3_{self.a.name}_infer_{self.task}_{self.tag}_s{s}")
        self.a.add(kind="inference", task=self.task, prec=self.prec, device=self.device, runtime=self.runtime,
                   step=s, job_id=ij.job_id, from_compile=comp["job_id"])
        self.meta["current"] = {"step": s, "job_id": ij.job_id, "from_compile": comp["job_id"]}
        self.meta["jobs"].append({"step": s, "job_id": ij.job_id})
        self.save()
        print(f"[chain {self.task} {self.tag}] submitted step {s} -> {ij.job_id}", flush=True)


def main():
    accts = {n: Acct(n) for n in PLANS}
    chains = []
    for n, plan in PLANS.items():
        for spec in dict.fromkeys(plan):
            chains.append(Chain(accts[n], *spec))
    print(f"[acct] {len(chains)} chains on {list(accts)}", flush=True)
    while True:
        for a in accts.values():
            bs = a.status()
            for c in chains:
                if c.a is a:
                    try:
                        c.step(bs)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[chain {c.task} {c.tag}] ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        active = [c for c in chains if not c.done]
        print(f"[acct] {time.strftime('%H:%M:%S', time.gmtime())} active chains {len(active)}", flush=True)
        if not active:
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
