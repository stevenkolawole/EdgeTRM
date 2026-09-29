#!/usr/bin/env python3
"""TFLite rows for the non-Qualcomm devices (Pixel 9 / Pixel 10 Tensor, Galaxy A53 Exynos): E3_TFLITE.md.

Why (fault table class B): every breadth chain on the three TFLite devices was submitted without inference options, so
the hub picked `compute unit=all` = the GPUv2 (OpenCL) delegate for 408/410 (fp32) or 548/550 (w8a8) nodes with the
Gather (2-D indices / v2) and the 2x2 Pad on XNNPACK. "Node number 410" is the GPU delegate's own super-node (the
original graph has 410 nodes): the fp32 chain dies in TfLiteGpuDelegate Invoke ("Given object is not valid") and the
w8a8 chain segfaults in tflite::gpu::QuantizeOutputs() (runtime logs in ~/x3/joblogs/<job>/). Both faults are the GPU
delegate's, not an op's. The fix is the delegate choice: `--compute_unit cpu` (XNNPACK), `gpu`, `npu` per chain.

Prec tags name the delegate: <graph>_<delegate>, e.g. fp32_cpu, w8a8_nofuse2_cpu, w8a8_nofuse2_gpu, fp32_npu; the
graph part resolves through aihub_e3_followup.source_path (w8a8_nofuse2 = the RMSNorm-broken QDQ graph), and the
records land in the breadth layout: ~/aihub_artifacts/h1/sudoku_mlp/infer/dev_<tag>_<device>_<acct>.{json,npz}.
The compile records are the breadth ones (tflite compiles are per device; --quantize_io for the QDQ graphs, no QAIRT
pin: the TFLite converter is not versioned by --qairt_version). Inference jobs carry `options` in jobs.json.

    python aihub_tflite.py run --steps 2 --only sudoku_mlp:fp32_cpu:P9@acct2 ...    (probe; a chain stops at its
                                                                                     first failure: the faults are
                                                                                     deterministic)
    python aihub_tflite.py run --steps 8 --only ...                                 (extend passing probes)
    python aihub_tflite.py status
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_acct as X  # noqa: E402
import aihub_e3_followup as E  # noqa: E402,F401  (patches X.source_path for the nofuse graphs)
import aihub_infer as I  # noqa: E402
from aihub_breadth import device_os  # noqa: E402

P9, P10, A53 = "Google Pixel 9", "Google Pixel 10", "Samsung Galaxy A53 5G"
SHORT = {P9: "P9", P10: "P10", A53: "A53"}
DELEGATES = ("cpu", "gpu", "npu")
# (acct, task, graph, delegate, device); the order is the probe order
CHAINS = [("acct2", "sudoku_mlp", "fp32", "cpu", P9),
          ("acct2", "sudoku_mlp", "w8a8_nofuse2", "cpu", P9),
          ("acct2", "sudoku_mlp", "w8a8", "cpu", P9),            # the fused graph on XNNPACK int8: was the segfault XNNPACK's?
          ("acct2", "sudoku_mlp", "w8a8_nofuse2", "gpu", P9),
          ("acct2", "sudoku_mlp", "fp32", "npu", P10),           # does the hub expose a Tensor NPU path at all?
          ("acct2", "sudoku_mlp", "fp32", "cpu", P10),
          ("acct2", "sudoku_mlp", "w8a8_nofuse2", "cpu", P10),
          ("acct3", "sudoku_mlp", "fp32", "cpu", A53),
          ("acct3", "sudoku_mlp", "w8a8_nofuse2", "cpu", A53),
          ("acct2", "sudoku_mlp", "fp32", "gpu", P9),            # the class-B failure itself, re-submitted explicitly (not planned)
          ("acct2", "sudoku_mlp", "w8a8_nofuse2", "npu", P10)]


def tag(acct, task, graph, delegate, dev):
    return f"{task}:{graph}_{delegate}:{SHORT.get(dev, dev)}@{acct}"


class TChain(X.Chain):
    """aihub_acct.Chain whose inference jobs carry `--compute_unit <delegate>` and whose record tag names it."""

    def __init__(self, acct, task, graph, delegate, device, nsteps):
        assert delegate in DELEGATES, delegate
        super().__init__(acct, task, graph, device, "tflite", nsteps)
        self.delegate, self.options = delegate, f"--compute_unit {delegate}"
        self.tag = f"{graph}_{delegate}_{device.replace(' ', '_')}_{acct.name}"
        d = I.idir(task)
        self.meta_path, self.state_path = d / f"dev_{self.tag}.json", d / f"dev_{self.tag}.npz"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else \
            {"steps": [], "jobs": [], "n_steps": nsteps, "account": acct.name, "runtime": "tflite",
             "delegate": delegate, "options": self.options, "graph": graph}

    def step(self, bs, free=None):
        before = (self.meta.get("current") or {}).get("job_id")
        real = self.a.C.submit_inference_job

        def with_options(*args, **kw):
            kw["options"] = self.options
            return real(*args, **kw)

        self.a.C.submit_inference_job = with_options
        try:
            used = super().step(bs, free)
        finally:
            self.a.C.submit_inference_job = real
        cur = (self.meta.get("current") or {}).get("job_id")
        if used and cur and cur != before:                       # annotate the jobs.json record with the delegate
            for j in self.a.find(kind="inference", job_id=cur):
                j.update(options=self.options, delegate=self.delegate, prec_tag=f"{self.prec}_{self.delegate}")
            self.a.save()
        # deterministic runtime faults: stop at the first failure instead of the base class's three attempts
        errs = self.meta.get("errors") or []
        if errs and not self.meta.get("failed"):
            self.meta["failed"] = "stopped after one failure (deterministic delegate fault): " + errs[-1]["error"][:300]
            self.save()
            print(f"[chain {self.task} {self.tag}] stopped after one failure", flush=True)
        return used


def accounts(names):
    accts = {n: X.Acct(n) for n in names}
    for a in accts.values():
        X.OS.update(device_os(a.C))
    return accts


def run(only, steps):
    sel = [c for c in CHAINS if not only or tag(*c) in only]
    unknown = [o for o in only if o not in {tag(*c) for c in CHAINS}]
    assert not unknown, f"unknown chains: {unknown}"
    accts = accounts(sorted({c[0] for c in sel}))
    chains = []
    for acct, task, graph, delegate, dev in sel:
        if dev not in X.OS:
            print(f"[tflite] DROPPED {tag(acct, task, graph, delegate, dev)}: not hosted on {acct}", flush=True)
            continue
        c = TChain(accts[acct], task, graph, delegate, dev, steps)
        if c.meta.get("done") and c.steps_done < steps:
            c.meta.pop("done")
            c.meta["n_steps"] = steps
            c.save()
        chains.append(c)
    print(f"[tflite] {len(chains)} chains, {steps} steps", flush=True)
    while True:
        for acct in accts.values():
            bs = acct.status()
            for c in [c for c in chains if c.a is acct]:
                try:
                    c.step(bs)
                except Exception as exc:  # noqa: BLE001
                    print(f"[chain {c.task} {c.tag}] ERROR {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        active = [c for c in chains if not c.done]
        print(f"[tflite] {time.strftime('%H:%M:%S', time.gmtime())} active {len(active)}: "
              + ", ".join(f"{c.tag} s{c.steps_done}" for c in active), flush=True)
        if not active:
            break
        time.sleep(60)


def status():
    for acct, task, graph, delegate, dev in CHAINS:
        f = I.idir(task) / f"dev_{graph}_{delegate}_{dev.replace(' ', '_')}_{acct}.json"
        if not f.exists():
            print(f"{tag(acct, task, graph, delegate, dev):45s} (no record)")
            continue
        m = json.loads(f.read_text())
        ex = " ".join(f"{s['exact'] * 100:.0f}" for s in m["steps"])
        ce = " ".join(f"{s['cell'] * 100:.1f}" for s in m["steps"])
        st = "done" if m.get("done") else ("FAILED " + m["failed"][:90] if m.get("failed") else "running")
        print(f"{tag(acct, task, graph, delegate, dev):45s} compile {m.get('compile_job')} guard "
              f"{(m.get('guard') or {}).get('verdict')} exact [{ex}] cell [{ce}] {st}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "status"])
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    run(a.only, a.steps) if a.cmd == "run" else status()
