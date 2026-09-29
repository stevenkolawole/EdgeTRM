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
         M3's override sets m3g40 / m3g60 / m3adT40 (Sudoku-MLP);
         on the Galaxy S24 and the Snapdragon 8 Elite QRD, 8 steps, 100 test puzzles (Maze 50)

Chain state: ~/aihub_artifacts/h1/<task>/infer/dev_<prec>_<device>_<acct>.{json,npz} (same format as
aihub_sched.Chain, so device_acc.py / aihub_infer.py report read them); job records with the account in
~/aihub_artifacts/h1/acct_<acct>/jobs.json.

    python aihub_acct.py            (loops until every chain is done or failed; resumable)

Toolchain pin (Sep 29, DIAG_ACCT_SPLIT.md): every compile is submitted with `--qairt_version QAIRT` (2.45). The hub
moved its default converter from QAIRT 2.45 to 2.50 between Sep 25 and Sep 26, and 2.50 compiles the `z_L_out` carry
of the quantized graphs with a zero-point-0 encoding (range [0, 1.3] instead of the graph's [-4.6, 4.7]), clipping the
carry on every step. Compile records are keyed on (task, prec, runtime, qairt); the version read back from the target
model's QAIRT_SDK_VERSION metadata is stored in jobs.json and in every dev_*.json. Before the first inference job of a
chain, guard_outputs() compares the DLC's output encodings with the source graph's output Q/DQ pairs: an integer
output whose encoding is not the graph's is a FAULT (the chain is refused); a float output where the graph has an
integer Q/DQ (what 2.45 does with the qdq_* graphs) is lossless and is recorded as "float_carry".
"""
import fcntl
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
QAIRT = "2.45"                       # --qairt_version takes major.minor; 2.45 = QAIRT 2.45.0.260326154327 on the hub
QNN_RUNTIMES = ("qnn_dlc", "precompiled_qnn_onnx", "qnn_context_binary", "qnn_lib_aarch64_android")
CARRY = ("z_H_out", "z_L_out")       # graph outputs fed back as the next step's inputs
PLANS = {
    "acct2": [("sudoku_mlp", "w4a16", S24, "precompiled_qnn_onnx", 1),
              ("sudoku_mlp", "w8a8rms16", S24, "qnn_dlc", 8),
              # device-legal rebuilds (embedding table encoding follows its 8-bit Gather output)
              ("sudoku_mlp", "w8a8rms16u", S24, "qnn_dlc", 8),
              # Maze bisection Ma_rope16 (RoPE Mul/Add, Neg/Concat, cos/sin at 16 bit): accuracy chain on the
              # 50 mazes, against the host run of the same graph (audit item 111)
              ("maze", "w8a8rope16", S24, "qnn_dlc", 8), ("maze", "w8a8rope16", E8, "qnn_dlc", 8)],
    "acct3": [(t, p, d, "qnn_dlc", 8) for d in (S24, E8)
              for t, p in (("sudoku_mlp", "fp32"), ("sudoku_mlp", "w8a8"), ("sudoku_mlp", "w8a16"),
                           ("sudoku_mlp", "w8a8sqvar16u"), ("sudoku_attn", "fp32"), ("sudoku_attn", "w8a8"),
                           ("sudoku_attn", "w8a16"), ("maze", "fp32"), ("maze", "w8a8"))]
             # M3 sensitivity-guided override sets (device-legal; request in X3.md, Sep 25 ~11:40), priority order
             + [("sudoku_mlp", p, d, "qnn_dlc", 8) for p in ("m3g40", "m3g60", "m3adT40") for d in (S24, E8)],
}


def source_path(task, prec):
    d = I.idir(task)
    if prec == "fp32":
        return A.ART / task / "stepc_H1.onnx"
    # an upload directory must hold exactly model.onnx + model.data (qdq_* dirs also hold ablation variants)
    base, onnx_file = {"w8a8rms16": ("qdq_w8a16", "variant_B2_rms16.onnx"),
                       "w8a8sqvar16": ("qdq_w8a16", "variant_B2ab_sq_var16.onnx"),
                       "w8a8rms16t": ("qdq_w8a16", "variant_B2t_rms16.onnx"),
                       "w8a8sqvar16t": ("qdq_w8a16", "variant_B2abt_sq_var16.onnx"),
                       "w8a8rms16u": ("qdq_w8a16", "variant_B2u_rms16.onnx"),
                       "w8a8sqvar16u": ("qdq_w8a16", "variant_B2abu_sq_var16.onnx"),
                       "w8a8rope16": ("qdq_w8a16", "variant_Ma_rope16.onnx"),
                       "m3g40": ("qdq_w8a16", "variant_m3_greedy_40_legal.onnx"),
                       "m3g60": ("qdq_w8a16", "variant_m3_greedy_60_legal.onnx"),
                       "m3adT40": ("qdq_w8a16", "variant_m3_a_dT_40_legal.onnx")}.get(prec, (f"qdq_{prec}", "model.onnx"))
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


def graph_out_enc(model_path):
    """{output name: {scale, zero_point, dtype} | None} from the Q/DQ pair that produces each graph output
    (None = float output, e.g. the fp32 graph). Graph output order = DLC output_0.. order."""
    import onnx
    from onnx import numpy_helper
    g = onnx.load(str(model_path), load_external_data=False).graph
    vals = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            vals[n.output[0]] = n.attribute[0].t
    prod = {o: n for n in g.node for o in n.output}
    enc = {}
    for o in g.output:
        n, e = prod.get(o.name), None
        if n is not None and n.op_type == "DequantizeLinear" and n.input[1] in vals and n.input[2] in vals:
            zt = vals[n.input[2]]
            e = {"scale": float(numpy_helper.to_array(vals[n.input[1]])),
                 "zero_point": float(numpy_helper.to_array(zt)),
                 "dtype": onnx.TensorProto.DataType.Name(zt.data_type).lower()}
        enc[o.name] = e
    return enc


def out_enc(task, prec):
    """cached source-graph output encodings, ~/aihub_artifacts/h1/<task>/infer/out_enc_<prec>.json."""
    d = I.idir(task)
    f = d / f"out_enc_{prec}.json"
    if not f.exists():
        src = source_path(task, prec)
        src = src / "model.onnx" if src.is_dir() else src
        f.write_text(json.dumps({"source": str(src), "outputs": graph_out_enc(src)}, indent=1))
    return json.loads(f.read_text())["outputs"]


def guard_outputs(acct, comp, task, prec):
    """Compare the compiled target's output specs with the source graph's output Q/DQ encodings.
    Per output: match | float_carry (DLC float where the graph quantizes: lossless, the 2.45 artifact) |
    fault (integer output with an encoding that is not the graph's: the 2.50 z_L_out clipping) |
    quantized_float_output (DLC quantizes an output the graph leaves float). verdict = worst status over CARRY."""
    tm = acct.C.get_job(comp["job_id"]).get_target_model()
    spec = tm.output_spec
    tensors = [t for v in (spec.values() if isinstance(spec, dict) else [spec]) for t in v]
    genc = out_enc(task, prec)
    rows = {}
    for t, name in zip(tensors, genc):
        g = genc[name]
        dtype = str(t.dtype)
        dlc = {"name": t.name, "dtype": dtype, "shape": [int(x) for x in t.shape],
               "scale": None if t.scale is None else float(t.scale),
               "zero_point": None if t.zero_point is None else float(t.zero_point)}
        if g is None:
            st = "match" if dtype.startswith("float") else "quantized_float_output"
        elif dtype.startswith("float"):
            st = "float_carry" if name in CARRY else "float_output"
        else:
            bits_ok = dtype.replace("u", "") == g["dtype"].replace("u", "")
            zp = dlc["zero_point"]
            zp_ok = zp == g["zero_point"] or (bits_ok and {dtype, g["dtype"]} == {"int8", "uint8"}
                                              and abs(zp - g["zero_point"]) == 128)
            sc_ok = dlc["scale"] is not None and abs(dlc["scale"] - g["scale"]) <= 1e-4 * abs(g["scale"])
            st = "match" if bits_ok and zp_ok and sc_ok else "fault"
        rows[name] = {"graph": g, "dlc": dlc, "status": st}
    order = ["fault", "quantized_float_output", "float_carry", "float_output", "match"]
    carry = [rows[n]["status"] for n in CARRY if n in rows] or ["match"]
    verdict = min(carry, key=order.index)
    return {"compile_job": comp["job_id"], "verdict": verdict, "outputs": rows,
            "n_dlc_outputs": len(tensors), "n_graph_outputs": len(genc)}


class Acct:
    def __init__(self, name):
        import qai_hub as hub
        self.hub, self.name = hub, name
        self.C = hub.Client(profile=name)
        self.dir = A.ART / f"acct_{name}"
        self.dir.mkdir(exist_ok=True)
        self.path = self.dir / "jobs.json"
        self.rec = json.loads(self.path.read_text()) if self.path.exists() else {"account": name, "jobs": []}

    # jobs.json is shared by every driver that holds this account (aihub_breadth + aihub_ctl run at once since Sep 29):
    # records are merged by (kind, job/model id) under a file lock instead of overwritten from one process's memory.
    @staticmethod
    def _key(j):
        return (j.get("kind"), j.get("job_id") or j.get("model_id"))

    def _merge(self):
        if not self.path.exists():
            return
        try:
            disk = json.loads(self.path.read_text())["jobs"]
        except (ValueError, KeyError):
            return
        mine = {self._key(j): j for j in self.rec["jobs"]}
        for j in disk:
            k = self._key(j)
            if k not in mine:
                self.rec["jobs"].append(j)
                mine[k] = j
            else:
                for f, v in j.items():               # fields the other driver added (qairt_sdk_version, ...)
                    mine[k].setdefault(f, v)

    def save(self):
        with open(self.dir / "jobs.lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            self._merge()
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.rec, indent=1))
            tmp.replace(self.path)

    def add(self, **kw):
        kw.update(account=self.name, submitted=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self.rec["jobs"].append(kw)
        self.save()

    def find(self, **kw):
        self._merge()
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

    def compile(self, task, prec, device, runtime, qairt=QAIRT, prefer=None):
        """qnn_dlc targets are device-agnostic: one compile per (task, prec, qairt); precompiled_qnn_onnx is
        per device. QNN runtimes are pinned with --qairt_version; records without a `qairt` key (submitted
        before Sep 29 on whatever the hub's default was) never match a pinned lookup. `prefer` = the compile
        job a chain already passed the guard on: it stays on that DLC even if another driver's compile of the
        same cell became the canonical record."""
        q = qairt if runtime in QNN_RUNTIMES else None
        if prefer:
            c = [j for j in self.find(job_id=prefer) if j.get("kind") in ("compile", "compile_orphan")
                 and j.get("task") == task and j.get("prec") == prec and j.get("qairt") == q]
            if c:
                return c[0]
        key = dict(kind="compile", task=task, prec=prec, runtime=runtime, qairt=q)
        if runtime != "qnn_dlc":
            key["device"] = device
        c = self.find(**key)
        if c:
            return min(c, key=lambda j: j.get("submitted", ""))    # earliest: the same choice in every driver
        opts = (f"--target_runtime {runtime}" + ("" if prec == "fp32" else " --quantize_io")
                + (f" --qairt_version {q}" if q else ""))
        cj = self.C.submit_compile_job(model=self.model(task, prec), device=self.hub.Device(device, OS[device]),
                                       options=opts, name=f"x3_{self.name}_{task}_stepc_{prec}_{runtime}"
                                       + (f"_qairt{q}" if q else ""))
        self.add(kind="compile", task=task, prec=prec, runtime=runtime, device=device, options=opts,
                 job_id=cj.job_id, qairt=q)
        print(f"[{self.name}] compile {task} {prec} {runtime} ({device}, qairt {q}) -> {cj.job_id}", flush=True)
        return self.find(**key)[-1]

    def verify(self, comp):
        """read QAIRT_SDK_VERSION off the compiled target model into the record; FAULT if it is not the pin."""
        if comp.get("qairt_sdk_version"):
            return comp["qairt_sdk_version"]
        tm = self.C.get_job(comp["job_id"]).get_target_model()
        md = {str(k).split(".")[-1]: v for k, v in tm.metadata.items()}
        v = md.get("QAIRT_SDK_VERSION")
        comp["qairt_sdk_version"] = v
        comp["target_model_id"] = tm.model_id
        self.save()
        if comp.get("qairt") and not (v or "").startswith(comp["qairt"] + "."):
            raise RuntimeError(f"FAULT compile {comp['job_id']} pinned qairt {comp['qairt']} but QAIRT_SDK_VERSION={v}")
        return v

    def check(self, comp, task, prec):
        """verify + guard for one compile record: (sdk version, guard dict). Raises RuntimeError on a FAULT."""
        v = self.verify(comp)
        g = guard_outputs(self, comp, task, prec)
        if g["verdict"] == "fault":
            bad = {n: r for n, r in g["outputs"].items() if r["status"] == "fault"}
            raise RuntimeError(f"FAULT compile {comp['job_id']} (QAIRT {v}) output encoding != graph: "
                               + json.dumps(bad))
        return v, g


def atomic_json(path, obj):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    """a killed driver must not leave a truncated .npz (a zip) behind: write beside, then rename."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


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
        atomic_json(self.meta_path, self.meta)

    @property
    def done(self):
        return self.meta.get("done") or self.meta.get("failed")

    def data(self):
        if self._data is None:
            inp, lab = I.test_set(self.task)
            self._data = (inp, lab, I.initial_state(self.task))
        return self._data

    @property
    def steps_done(self):
        return len(self.meta["steps"])

    def inflight(self, bs):
        cur = self.meta.get("current")
        return 1 if cur and bs.get(cur["job_id"], "PENDING") not in ("SUCCESS", "FAILED") else 0

    def step(self, bs, free=None):
        """advance one chain; returns 1 when it submitted a device job. free=None: no slot limit."""
        cur = self.meta.get("current")
        if cur:
            code = bs.get(cur["job_id"]) or self.a.C.get_job(cur["job_id"]).get_status().code
            if code not in ("SUCCESS", "FAILED"):
                return 0
            job = self.a.C.get_job(cur["job_id"])
            if code == "FAILED":
                self.meta.pop("current")
                msg = (job.get_status().message or "")[:600]
                self.meta.setdefault("errors", []).append({"step": cur["step"], "job_id": cur["job_id"], "error": msg})
                if len(self.meta["errors"]) >= 3:
                    self.meta["failed"] = msg
                self.save()
                print(f"[chain {self.task} {self.tag}] step {cur['step']} FAILED: {msg[:160]}", flush=True)
                return 0
            o = named_outputs(job.download_output_data())
            inp, lab, _ = self.data()
            zH = np.stack([x.astype(np.float32) for x in o["z_H_out"]])
            zL = np.stack([x.astype(np.float32) for x in o["z_L_out"]])
            lg = np.concatenate([x.astype(np.float32) for x in o["logits"]])
            q = np.array([float(x.reshape(-1)[0]) for x in o["q_halt"]])
            preds, qs, zHs, zLs, kept = [], [], [], [], []
            if self.state_path.exists():
                st = np.load(self.state_path)
                preds, qs = list(st["preds"]), list(st["q"])
                if "zH_steps" in st.files:
                    zHs, zLs, kept = list(st["zH_steps"]), list(st["zL_steps"]), list(st["carry_steps"])
            assert len(preds) == cur["step"], (self.tag, len(preds), cur)     # one state row per device step
            preds.append(lg.argmax(-1))
            qs.append(q)
            # Sep 29 (E3_GATE_PHI_STEP2): keep the carry of EVERY step (zH_steps / zL_steps (k, N, 1, seq, 512), carry_steps =
            # the record steps kept; a chain resumed from an older record keeps them from this step on), not only the last.
            zHs.append(zH); zLs.append(zL); kept.append(cur["step"])
            atomic_npz(self.state_path, zH=zH, zL=zL, preds=np.stack(preds), q=np.stack(qs),
                       zH_steps=np.stack(zHs), zL_steps=np.stack(zLs), carry_steps=np.array(kept))
            self.meta.pop("current")                 # only once the step is ingested: an exception above leaves it for a retry
            ex = float((preds[-1] == lab).all(-1).mean())
            self.meta["steps"].append({"step": cur["step"], "job_id": cur["job_id"], "exact": ex,
                                       "cell": float((preds[-1] == lab).mean())})
            if len(preds) >= self.nsteps:
                self.meta["done"] = True
            self.save()
            print(f"[chain {self.task} {self.tag}] step {cur['step']}: exact {ex:.3f}"
                  + ("  DONE" if self.meta.get("done") else ""), flush=True)
            return 0
        if self.done or (free is not None and free <= 0):
            return 0
        comp = self.a.compile(self.task, self.prec, self.device, self.runtime, prefer=self.meta.get("compile_job"))
        code = bs.get(comp["job_id"]) or self.a.C.get_job(comp["job_id"]).get_status().code
        if code == "FAILED":
            msg = (self.a.C.get_job(comp["job_id"]).get_status().message or "")[:600]
            self.meta["failed"] = "compile failed: " + msg
            self.save()
            print(f"[chain {self.task} {self.tag}] compile FAILED: {msg[:160]}", flush=True)
            return 0
        if code != "SUCCESS":
            return 0
        if (self.meta.get("guard") or {}).get("compile_job") != comp["job_id"]:
            # pre-inference guard: pinned version read back, output encodings against the source graph
            try:
                ver, g = self.a.check(comp, self.task, self.prec)
            except RuntimeError as exc:
                self.meta.update(compile_job=comp["job_id"], qairt=comp.get("qairt"),
                                 qairt_sdk_version=comp.get("qairt_sdk_version"), failed=str(exc)[:600],
                                 guard={"compile_job": comp["job_id"], "verdict": "fault", "error": str(exc)[:600]})
                self.save()
                print(f"FAULT [chain {self.task} {self.tag}] {str(exc)[:400]}", flush=True)
                return 0
            self.meta.update(compile_job=comp["job_id"], qairt=comp.get("qairt"), qairt_sdk_version=ver, guard=g)
            self.save()
            if g["verdict"] != "match":
                print(f"WARN [chain {self.task} {self.tag}] guard {g['verdict']} on {comp['job_id']} (QAIRT {ver}): "
                      + ", ".join(f"{n} {r['dlc']['dtype']} vs graph {r['graph']}" for n, r in g["outputs"].items()
                                  if n in CARRY), flush=True)
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
                e = dict(enc[name])
                if e["dtype"] != want:  # TFLite compiles signed int8 where the QNN encodings say uint8: same scale, zero point shifted by 128
                    assert {e["dtype"], want} == {"uint8", "int8"}, (name, e, want)
                    e = {"scale": e["scale"], "zero_point": e["zero_point"] + (-128 if want == "int8" else 128), "dtype": want}
                feeds[name] = [I.quantize_input(x, e) for x in feeds[name]]
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
        return 1


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
