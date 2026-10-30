#!/usr/bin/env python3
"""Maze / Sudoku-MLP chains on the MAIN AI Hub account from the Babel login node (reports/edgetrm_exp/AIHUB_V75_CPU_PLAN.md).

Two chain kinds, both resumable (state beside the records, rerun the same command):
  qnn:<task>:<prec>:<dev>   qnn_dlc on the NPU, QAIRT pinned at aihub_acct.QAIRT (2.45), compile annotated with the
                            converter path, float-fallback ops and DLC dtype counts (aihub_e3_followup.annotate);
                            prec w8a8_nofuse2 = the RMSNorm-broken all-integer graph (aihub_e3_followup.build)
  cpu:<task>:<graph>:<dev>  tflite with `--compute_unit cpu` (aihub_tflite.TChain; XNNPACK, confirmed in the job log)
  onnx:<task>:<graph>:<dev> onnx runtime target (ORT, QDQ graph as is, float I/O) with `--compute_unit cpu`
  <dev> in S23 S24 S25.

At most --maxactive chains are in flight (default 2), so at most two AI Hub inference jobs at a time. The main account is
reached with qai_hub.Client() (default profile); no config file is read or printed here. Per-step carries are not kept
(the shared $HOME is nearly full): the state npz holds the last carry only and is cut down to preds+q when a chain ends.

    export AIHUB_ART=$HOME/aihub_h/art HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1
    python aihub_nofuse_cpu_chains.py build
    python aihub_nofuse_cpu_chains.py run --steps 8 --maxactive 2 qnn:maze:w8a8_nofuse2:S23 qnn:maze:w8a8_nofuse2:S24
    python aihub_nofuse_cpu_chains.py run --steps 2 cpu:maze:fp32:S25 cpu:maze:w8a8_nofuse2:S25 ...
    python aihub_nofuse_cpu_chains.py status
"""
import argparse
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402
if os.environ.get("AIHUB_ART"):
    A.ART = pathlib.Path(os.environ["AIHUB_ART"])
import aihub_acct as X  # noqa: E402
import aihub_e3_followup as E  # noqa: E402
import aihub_tflite as T  # noqa: E402
import aihub_v79_classa as V  # noqa: E402
import aihub_infer as I  # noqa: E402

DEV = {"S23": "Samsung Galaxy S23", "S24": "Samsung Galaxy S24", "S25": "Samsung Galaxy S25"}
KEEP_ONLY_LAST = ("zH_steps", "zL_steps", "carry_steps")
_real_npz = X.atomic_npz


def _npz_last_carry(path, **arrays):
    _real_npz(path, **{k: v for k, v in arrays.items() if k not in KEEP_ONLY_LAST})


X.atomic_npz = _npz_last_carry
V.LOGS = pathlib.Path(os.environ.get("AIHUB_LOGS", str(pathlib.Path.home() / "aihub_h" / "logs" / "joblogs")))


class MainAcct(X.Acct):
    def __init__(self, name="main"):
        import qai_hub as hub
        self.hub, self.name = hub, name
        self.C = hub.Client()
        self.dir = A.ART / f"acct_{name}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "jobs.json"
        self.rec = json.loads(self.path.read_text()) if self.path.exists() else {"account": name, "jobs": []}


def _onnx_compile(self, task, prec, device, runtime, qairt=X.QAIRT, prefer=None):
    """Acct.compile for the onnx runtime target: the QDQ graph goes through as is (float I/O, no --quantize_io)."""
    key = dict(kind="compile", task=task, prec=prec, runtime="onnx", device=device)
    c = self.find(**key)
    if c:
        return min(c, key=lambda j: j.get("submitted", ""))
    opts = "--target_runtime onnx"
    cj = self.C.submit_compile_job(model=self.model(task, prec), device=self.hub.Device(device, X.OS[device]),
                                   options=opts, name=f"h_{task}_stepc_{prec}_onnx")
    self.add(kind="compile", task=task, prec=prec, runtime="onnx", device=device, options=opts, job_id=cj.job_id, qairt=None)
    print(f"[{self.name}] compile {task} {prec} onnx ({device}) -> {cj.job_id}", flush=True)
    return self.find(**key)[-1]


MainAcct.onnx_compile = _onnx_compile


class OnnxChain(T.TChain):
    """TChain on the onnx runtime target (ORT on the phone CPU); records are tagged <graph>_onnxcpu_..."""

    def __init__(self, acct, task, graph, device, nsteps):
        super().__init__(acct, task, graph, "cpu", device, nsteps)
        self.runtime = "onnx"
        self.tag = f"{graph}_onnxcpu_{device.replace(' ', '_')}_{acct.name}"
        d = I.idir(task)
        self.meta_path, self.state_path = d / f"dev_{self.tag}.json", d / f"dev_{self.tag}.npz"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else \
            {"steps": [], "jobs": [], "n_steps": nsteps, "account": acct.name, "runtime": "onnx",
             "delegate": "cpu", "options": self.options, "graph": graph}

    def step(self, bs, free=None):
        real = self.a.compile
        self.a.compile = lambda task, prec, dev, rt, **kw: self.a.onnx_compile(task, prec, dev, rt, **kw)
        try:
            return super().step(bs, free)
        finally:
            self.a.compile = real


def make_chain(acct, spec, steps):
    kind, task, prec, dev = spec.split(":")
    d = DEV[dev]
    if kind == "qnn":
        return X.Chain(acct, task, prec, d, "qnn_dlc", steps)
    if kind == "onnx":
        return OnnxChain(acct, task, prec, d, steps)
    return T.TChain(acct, task, prec, "cpu", d, steps)


def slim(c):
    """chain finished: drop the large carry arrays from the state npz, keep preds and q (the records are copied off)."""
    if c.state_path.exists() and (c.meta.get("done") or c.meta.get("failed")):
        st = np.load(c.state_path)
        if "zH" in st.files:
            _real_npz(c.state_path, preds=st["preds"], q=st["q"])


def run(specs, steps, maxactive):
    acct = MainAcct()
    X.OS.update({d.name: d.os for d in acct.C.get_devices() if d.name not in X.OS})
    chains = []
    for s in specs:
        c = make_chain(acct, s, steps)
        if c.meta.get("done") and c.steps_done < steps:
            c.meta.pop("done")
            c.meta["n_steps"] = steps
            c.save()
        chains.append((s, c))
    print(f"[h] {len(chains)} chains, {steps} steps, maxactive {maxactive}", flush=True)
    while True:
        live = [c for _, c in chains if not c.done]
        started = [c for c in live if c.meta.get("current") or c.meta.get("compile_job") or c.steps_done]
        allowed = set(id(c) for c in started[:maxactive])
        for c in live:
            if len(allowed) < maxactive and id(c) not in allowed:
                allowed.add(id(c))
        for s, c in chains:
            if c.done:
                slim(c)
                continue
            if id(c) not in allowed:
                continue
            try:
                c.step({})
            except Exception as exc:  # noqa: BLE001
                print(f"[{s}] ERROR {type(exc).__name__}: {str(exc)[:400]}", flush=True)
            errs = c.meta.get("errors") or []
            if errs and not c.meta.get("failed") and "MODEL_GRAPH_ERROR" in errs[-1]["error"]:
                c.meta["failed"] = "deterministic compose error, no retry: " + errs[-1]["error"][:300]
                c.save()
            comp = c.meta.get("compile_job")
            if comp and s.startswith("qnn") and "converter" not in c.meta:
                rec = [j for j in acct.find(job_id=comp) if j.get("kind") in ("compile", "compile_orphan")]
                if rec:
                    if "converter" not in rec[0]:
                        rec[0].update(E.annotate(acct, rec[0]))
                        for f in glob.glob(str(V.LOGS / comp / "*.dlc")) + glob.glob(str(V.LOGS / comp / "model")):
                            os.remove(f)
                        acct.save()
                    c.meta["converter"] = {k: rec[0].get(k) for k in ("converter", "float_fallback_ops", "dlc_dtype_counts", "dlc_rms_norm_ops")}
                    c.save()
        act = [(s, c) for s, c in chains if not c.done]
        print(f"[h] {time.strftime('%H:%M:%S', time.gmtime())} active {len(act)}: "
              + ", ".join(f"{s} s{c.steps_done}" for s, c in act), flush=True)
        if not act:
            break
        time.sleep(90)


def status(specs):
    acct = MainAcct()
    for s in specs:
        c = make_chain(acct, s, 0)
        ex = " ".join(f"{x['exact'] * 100:.0f}" for x in c.meta["steps"])
        ce = " ".join(f"{x['cell'] * 100:.2f}" for x in c.meta["steps"])
        print(f"{s:36s} compile {c.meta.get('compile_job')} qairt {c.meta.get('qairt_sdk_version')} guard "
              f"{(c.meta.get('guard') or {}).get('verdict')} exact [{ex}] cell [{ce}] "
              f"{'done' if c.meta.get('done') else ('FAILED ' + str(c.meta.get('failed'))[:80] if c.meta.get('failed') else 'running')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "run", "status"])
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--maxactive", type=int, default=2)
    ap.add_argument("chains", nargs="*")
    a = ap.parse_intermixed_args()
    if a.cmd == "build":
        E.build(["maze:w8a8_nofuse2", "sudoku_mlp:w8a8_nofuse2"])
    elif a.cmd == "run":
        run(a.chains, a.steps, a.maxactive)
    else:
        status(a.chains)
