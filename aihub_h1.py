#!/usr/bin/env python3
"""X3 device study on Qualcomm AI Hub: the configuration we recommend (H=1), the
quantization-granularity compile matrix, and an on-device correctness check.

Every device latency before this script measured the official TRM step (H=3,
`aihub_tasks.py` / `aihub_export.py`). The paper recommends the shallowest
saturating depth, H=1 with n supervision steps. Accuracy depends on H x n only
(H=1 n=8 and H=2 n=4 are the same computation; `results/breadth/*_all.json`),
so the device question is the step latency at H=1 and how it scales.

Graphs exported per task (sudoku_mlp, sudoku_attn, maze, arc):
  step   one supervision step, logits only: identical interface to the H=3
         export, so H=1 and H=3 rows compare like for like
  stepc  one step that also returns the new carry (z_H, z_L) and the halting
         logit: the deployable step, used for on-device correctness
  f      one application of the shared block (L_level): the unit the step
         repeats H*(L+1) times

Precisions (AI Hub quantize jobs = AIMET QDQ; per-output-channel symmetric
weights, per-tensor asymmetric activations -- verified by reading the QDQ):
  fp32 (runs as fp16 on HTP), w8a8, w4a8, w8a16, w4a16
Granularity variants (built here by rewriting the weight Q/DQ of the AI Hub
QDQ graph; activations untouched): w4a8_t, w8a8_t (per-tensor), w4a8_g32,
w4a8_g128 (block-wise along the input dimension, ONNX opset-21 block_size).

Stages (all resumable; every job id lands in ~/aihub_artifacts/h1/<task>/jobs.json):
    python aihub_h1.py --stage export   --tasks sudoku_mlp,sudoku_attn,maze,arc
    python aihub_h1.py --stage calib    --tasks ...
    python aihub_h1.py --stage quantize --tasks ... --graphs step,stepc --precs w8a8,w4a8,w8a16,w4a16
    python aihub_h1.py --stage granvar  --tasks ...           (build per-tensor / block-wise QDQ)
    python aihub_h1.py --stage compile  --tasks ... --graphs step --precs fp32,w8a8,w4a8 --devices all
    python aihub_h1.py --stage profile  --tasks ...
    python aihub_h1.py --stage status   --tasks ...  (writes ~/aihub_artifacts/h1/status.json)

Run on p4d with ~/venvs/kvcache/bin/python (torch, onnx, onnxruntime, qai_hub).
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HOME = pathlib.Path.home()
HERE = pathlib.Path(__file__).resolve().parent
TRM_SRC = HOME / "edgetrm_src" / "TinyRecursiveModels"
CKPT = HOME / "edgetrm_ckpt"
ART = HOME / "aihub_artifacts" / "h1"

TASKS = {
    "sudoku_mlp": dict(ckpt=CKPT / "sudoku_extreme" / "step_39060_sudoku_epoch_60k", seq_len=81, vocab=11,
                       L_cycles=6, mlp_t=True, pos="none", data="sudoku", n_cal=32, n_test=100),
    "sudoku_attn": dict(ckpt=CKPT / "sudoku_extreme" / "step_39060_sudoku_60k_epoch_attn_type", seq_len=81,
                        vocab=11, L_cycles=6, mlp_t=False, pos="rope", data="sudoku", n_cal=32, n_test=100),
    "maze": dict(ckpt=CKPT / "maze_hard" / "model.pt", seq_len=900, vocab=6, L_cycles=4, mlp_t=False,
                 pos="rope", data="maze", n_cal=16, n_test=50),
    "arc": dict(ckpt=CKPT / "arc_v1_public" / "step_518071", seq_len=900, vocab=12, L_cycles=4, mlp_t=False,
                pos="rope", data="arc", n_cal=16, n_test=0),
}
COMMON = dict(H_layers=0, L_layers=2, hidden_size=512, expansion=4, num_heads=8, halt_max_steps=16,
              halt_exploration_prob=0.1, forward_dtype="float32", puzzle_emb_ndim=512, puzzle_emb_len=16,
              no_ACT_continue=True)
PLEN = COMMON["puzzle_emb_len"]

# One device per chipset the account reaches (get_devices(), Sep 24 2026): 21 with a QNN/HTP
# path, 14 without (Tensor, Exynos, pre-v68 Snapdragon), which compile to TF-Lite.
# (name, os, has_qnn, htp_fp16)
DEVICES = [
    ("Samsung Galaxy S21", "12", True, False),          # SD 888, v68 (integer-only HTP)
    ("Samsung Galaxy A73 5G", "12", True, False),       # SD 778G, v68
    ("Dragonwing RB3 Gen 2 Vision Kit", "1.6", True, False),  # QCS6490, v68
    ("SA8295P ADP", "14", True, True),                  # automotive, v68
    ("Samsung Galaxy S22 5G", "12", True, True),        # 8 Gen 1, v69
    ("Samsung Galaxy S23", "13", True, True),           # 8 Gen 2, v73
    ("Snapdragon 7 Gen 4 QRD", "15", True, False),      # v73, no fp16
    ("Dragonwing Q-6690 MTP", "15", True, False),       # QCM6690, v73, no fp16
    ("QCS8550 (Proxy)", "12", True, True),              # v73
    ("Dragonwing IQ-9075 EVK", "1.9", True, True),      # QCS9075, v73
    ("SA8775P ADP", "14", True, True),                  # automotive, v73
    ("Snapdragon X Elite CRD", "11", True, True),       # laptop, v73
    ("Snapdragon X Plus 8-Core CRD", "11", True, True), # laptop, v73
    ("Samsung Galaxy S24", "14", True, True),           # 8 Gen 3, v75
    ("SA7255P ADP", "14", True, True),                  # automotive, v75
    ("Arduino VENTUNO Q", "24.04", True, True),         # QCS8275, v75
    ("Samsung Galaxy S25", "15", True, True),           # 8 Elite for Galaxy, v79
    ("Snapdragon 8 Elite QRD", "15", True, True),       # v79
    ("Samsung Galaxy S26", "16", True, True),           # 8 Elite Gen 5 for Galaxy, v81
    ("Snapdragon 8 Elite Gen 5 QRD", "16", True, True), # v81
    ("Snapdragon X2 Elite CRD", "11", True, True),      # laptop, v81
    ("Google Pixel 3", "10", False, False),             # SD 845, v65 (no QNN on AI Hub)
    ("Google Pixel 3a", "10", False, False),            # SD 670, v65
    ("Google Pixel 4", "11", False, False),             # SD 855, v66
    ("Google Pixel 5", "12", False, False),             # SD 765G, v66
    ("Samsung Galaxy Tab S7", "11", False, False),      # SD 865+, v66
    ("Xiaomi Redmi Note 10 5G", "11", False, False),    # SD 678... (sm6150-ac), v66
    ("Google Pixel 6", "13", False, False),             # Tensor G1
    ("Google Pixel 7", "14", False, False),             # Tensor G2
    ("Google Pixel 8", "14", False, False),             # Tensor G3
    ("Google Pixel 9", "15", False, False),             # Tensor G4
    ("Google Pixel 10", "16", False, False),            # Tensor G5
    ("Samsung Galaxy Note 20 (Intl)", "11", False, False),  # Exynos 990
    ("Samsung Galaxy A53 5G", "12", False, False),      # Exynos 1280
    ("Samsung Galaxy A14 5G", "13", False, False),      # Exynos 1330
]
DEV_BY_NAME = {d[0]: d for d in DEVICES}
CORE = ["Samsung Galaxy S21", "Dragonwing RB3 Gen 2 Vision Kit", "Samsung Galaxy S24", "Snapdragon X Elite CRD",
        "SA8775P ADP", "Samsung Galaxy S26"]
QDTYPE = {"w8a8": ("INT8", "INT8"), "w4a8": ("INT4", "INT8"), "w8a16": ("INT8", "INT16"),
          "w4a16": ("INT4", "INT16"),
          # AI Hub's own fix for sensitive layers: Lite Mixed Precision keeps the most sensitive
          # 10% of layers at INT16 activations (quantize option --lite_mp)
          "w8a8mp": ("INT8", "INT8", "--lite_mp percentage=10;override_qtype=int16"),
          "w4a8mp": ("INT4", "INT8", "--lite_mp percentage=10;override_qtype=int16")}


# ----------------------------------------------------------------------------- model
def load_sd(path):
    import torch
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return {k.replace("_orig_mod.model.", "").replace("model.", ""): v for k, v in sd.items()}


def build(task, H):
    if str(TRM_SRC) not in sys.path:
        sys.path.insert(0, str(TRM_SRC))
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    t = TASKS[task]
    sd = load_sd(t["ckpt"])
    vocab = sd["inner.embed_tokens.embedding_weight"].shape[0]
    cfg = dict(batch_size=1, seq_len=t["seq_len"], vocab_size=vocab,
               num_puzzle_identifiers=sd["inner.puzzle_emb.weights"].shape[0], H_cycles=H,
               L_cycles=t["L_cycles"], mlp_t=t["mlp_t"], pos_encodings=t["pos"], **COMMON)
    m = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss[:5], unexp[:5])
    return m, cfg, sd


def wrappers():
    import torch

    class Step(torch.nn.Module):
        """One supervision step; the puzzle-embedding row is a graph input (streamed from flash)."""

        def __init__(self, inner, with_carry):
            super().__init__()
            self.inner, self.with_carry, self._value = inner, with_carry, None
            outer = self

            class Emb(torch.nn.Module):
                def forward(self, idx):
                    return outer._value
            self.inner.puzzle_emb = Emb()

        def forward(self, z_H, z_L, inputs, puzzle_embedding):
            self._value = puzzle_embedding
            carry = type(self.inner.empty_carry(1))(z_H=z_H, z_L=z_L)
            batch = {"inputs": inputs, "puzzle_identifiers": torch.zeros(1, dtype=torch.int32)}
            new_carry, logits, (q_halt, _q_cont) = self.inner(carry, batch)
            if self.with_carry:
                return new_carry.z_H, new_carry.z_L, logits, q_halt
            return logits

    class F(torch.nn.Module):
        """One application of the shared block: out = L_level(hidden, injection)."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, hidden, injection):
            cs = self.inner.rotary_emb() if hasattr(self.inner, "rotary_emb") else None
            return self.inner.L_level(hidden, injection, cos_sin=cs)

    return Step, F


def graph_io(graph, task):
    t = TASKS[task]
    seq = t["seq_len"] + PLEN
    if graph in ("step", "stepc"):
        return {"z_H": (1, seq, 512), "z_L": (1, seq, 512), "inputs": (1, t["seq_len"]),
                "puzzle_embedding": (1, 512)}
    return {"hidden": (1, seq, 512), "injection": (1, seq, 512)}


def onnx_name(graph, H):
    return f"{graph}_H{H}.onnx" if graph != "f" else "f.onnx"


def stage_export(tasks, H):
    import torch
    import onnxruntime as ort
    Step, F = wrappers()
    for task in tasks:
        art = ART / task
        art.mkdir(parents=True, exist_ok=True)
        t = TASKS[task]
        for graph in ("step", "stepc", "f"):
            path = art / onnx_name(graph, H)
            if path.exists():
                print(f"[{task}] {path.name} exists", flush=True)
                continue
            m, cfg, sd = build(task, H)
            io = graph_io(graph, task)
            rng = np.random.default_rng(0)
            if graph == "f":
                w = F(m.inner).eval()
                args = (torch.randn(io["hidden"]), torch.randn(io["injection"]))
                names, outs = ["hidden", "injection"], ["out"]
            else:
                w = Step(m.inner, graph == "stepc").eval()
                args = (torch.randn(io["z_H"]), torch.randn(io["z_L"]),
                        torch.as_tensor(rng.integers(1, cfg["vocab_size"], io["inputs"]), dtype=torch.int32),
                        sd["inner.puzzle_emb.weights"][:1].float().clone())
                names = ["z_H", "z_L", "inputs", "puzzle_embedding"]
                outs = ["z_H_out", "z_L_out", "logits", "q_halt"] if graph == "stepc" else ["logits"]
            with torch.no_grad():
                ref = w(*args)
            ref = ref if isinstance(ref, (tuple, list)) else (ref,)
            torch.onnx.export(w, args, str(path), input_names=names, output_names=outs, opset_version=17,
                              dynamo=False)
            got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(
                outs, {n: a.numpy() for n, a in zip(names, args)})
            errs = [float(np.abs(g - r.numpy()).max()) for g, r in zip(got, ref)]
            print(f"[{task}] {path.name} {path.stat().st_size/1e6:.1f} MB outs "
                  f"{[tuple(r.shape) for r in ref]} ort-vs-torch max abs {['%.1e' % e for e in errs]}", flush=True)
            assert max(errs) < 5e-2, errs
            del m


# ----------------------------------------------------------------------------- data
def load_data(task, idx):
    """Real puzzles (inputs, labels) for the given test-set indices (sudoku, maze)."""
    sys.path.insert(0, str(HERE))
    import breadth_sweep as B
    inp, lab = B.load_qa(TASKS[task]["data"], max(idx) + 1)
    return inp[idx], lab[idx]


def host_rollout(task, H, inp, nsteps, collect_f=False):
    """fp32 host run of the H-step model from the initial carry, batch=all puzzles. Returns
    per-step (z_H, z_L) entering each step, final logits, per-step q_halt, and (optionally)
    the (hidden, injection) pairs every L_level call sees."""
    import torch
    m, cfg, sd = build(task, H)
    inner = m.inner
    n = inp.shape[0]
    emb = sd["inner.puzzle_emb.weights"][:1].float()
    x = torch.as_tensor(inp, dtype=torch.int32)
    zH = inner.H_init.expand(n, PLEN + cfg["seq_len"], 512).clone()
    zL = inner.L_init.expand(n, PLEN + cfg["seq_len"], 512).clone()
    Carry = type(inner.empty_carry(1))
    fpairs = []
    if collect_f:
        orig = inner.L_level.forward

        def hook(hidden, injection, **kw):
            fpairs.append((hidden[:2].clone(), injection[:2].clone()))
            return orig(hidden, injection, **kw)
        inner.L_level.forward = hook
    states, qs, logits = [], [], None
    bs = 16
    with torch.no_grad():
        for s in range(nsteps):
            states.append((zH.clone(), zL.clone()))
            nzH, nzL, lg, qh = [], [], [], []
            for b in range(0, n, bs):
                sl = slice(b, b + bs)
                c, out, (q, _) = inner(Carry(z_H=zH[sl], z_L=zL[sl]),
                                       {"inputs": x[sl], "puzzle_identifiers": torch.zeros(x[sl].shape[0],
                                                                                           dtype=torch.int32)})
                nzH.append(c.z_H); nzL.append(c.z_L); lg.append(out); qh.append(q)
            zH, zL, logits = torch.cat(nzH), torch.cat(nzL), torch.cat(lg)
            qs.append(torch.cat(qh))
    return dict(states=states, final=(zH, zL), logits=logits, q=torch.stack(qs, 1), emb=emb, fpairs=fpairs)


def stage_calib(tasks, H, nsteps_cal=8):
    """Calibration sets from real carries (calibration puzzles are test indices 900-..., disjoint
    from the correctness sample 0..n_test-1), and the correctness sample itself."""
    import torch
    for task in tasks:
        art = ART / task
        t = TASKS[task]
        out = art / f"calib_H{H}.npz"
        if out.exists():
            print(f"[{task}] {out.name} exists", flush=True)
            continue
        rng = np.random.default_rng(0)
        if t["data"] == "arc":
            # ARC: real puzzle-embedding rows and random grid tokens; carries from the rollout.
            m, cfg, sd = build(task, H)
            W = sd["inner.puzzle_emb.weights"]
            ids = rng.choice(W.shape[0], t["n_cal"], replace=False)
            inp = rng.integers(1, cfg["vocab_size"], (t["n_cal"], t["seq_len"])).astype(np.int64)
            del m
        else:
            idx = list(range(900, 900 + t["n_cal"]))
            inp, _ = load_data(task, idx)
        r = host_rollout(task, H, inp, nsteps_cal, collect_f=True)
        # one calibration sample per puzzle, at a step spread over the rollout
        steps = [int(s) for s in rng.integers(0, nsteps_cal, len(inp))]
        zH = np.stack([r["states"][s][0][i].numpy() for i, s in enumerate(steps)])[:, None]
        zL = np.stack([r["states"][s][1][i].numpy() for i, s in enumerate(steps)])[:, None]
        if t["data"] == "arc":
            emb = W[ids].float().numpy()[:, None]
        else:
            emb = np.repeat(r["emb"].numpy()[None], len(inp), 0)
        fp = r["fpairs"]
        pick = rng.choice(len(fp), min(len(inp), len(fp)), replace=False)
        fh = np.stack([fp[k][0][0].numpy() for k in pick])[:, None]
        fi = np.stack([fp[k][1][0].numpy() for k in pick])[:, None]
        np.savez_compressed(out, z_H=zH.astype(np.float32), z_L=zL.astype(np.float32),
                            inputs=inp.astype(np.int32)[:, None], puzzle_embedding=emb.astype(np.float32),
                            hidden=fh.astype(np.float32), injection=fi.astype(np.float32))
        print(f"[{task}] wrote {out} ({out.stat().st_size/1e6:.1f} MB), steps {steps}", flush=True)


def calib_dict(task, H, graph):
    d = np.load(ART / task / f"calib_H{H}.npz")
    keys = graph_io(graph, task).keys()
    return {k: [d[k][i] for i in range(d[k].shape[0])] for k in keys}


# ----------------------------------------------------------------------------- job records
def rec_path(task):
    return ART / task / "jobs.json"


def load_rec(task):
    p = rec_path(task)
    return json.loads(p.read_text()) if p.exists() else {"task": task, "jobs": []}


def _locked(task):
    import fcntl
    lock = open(ART / task / "jobs.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def add_job(task, entry):
    """Append one job record under an exclusive lock (several stages run concurrently)."""
    lock = _locked(task)
    try:
        rec = load_rec(task)
        rec["jobs"].append(entry)
        _write(task, rec)
    finally:
        lock.close()


def update_job(task, job_id, **fields):
    lock = _locked(task)
    try:
        rec = load_rec(task)
        for j in rec["jobs"]:
            if j.get("job_id") == job_id:
                j.update(fields)
        _write(task, rec)
    finally:
        lock.close()


def _write(task, rec):
    p = rec_path(task)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=1))
    tmp.replace(p)


def find(rec, **kw):
    return [j for j in rec["jobs"] if all(j.get(k) == v for k, v in kw.items())]


def stage_quantize(tasks, H, graphs, precs):
    import qai_hub as hub
    for task in tasks:
        rec = load_rec(task)
        for graph in graphs:
            for prec in precs:
                if find(rec, kind="quantize", graph=graph, H=H, prec=prec):
                    continue
                wd, ad = QDTYPE[prec][:2]
                opts = QDTYPE[prec][2] if len(QDTYPE[prec]) > 2 else ""
                path = ART / task / onnx_name(graph, H)
                qj = hub.submit_quantize_job(model=str(path), calibration_data=calib_dict(task, H, graph),
                                             weights_dtype=getattr(hub.QuantizeDtype, wd),
                                             activations_dtype=getattr(hub.QuantizeDtype, ad),
                                             name=f"x3_{task}_{graph}_H{H}_{prec}", options=opts)
                add_job(task, {"kind": "quantize", "graph": graph, "H": H, "prec": prec, "job_id": qj.job_id,
                               "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                print(f"[{task}] quantize {graph} H{H} {prec} -> {qj.job_id}", flush=True)


def source_model(task, rec, graph, H, prec):
    """The model to compile: the fp32 ONNX, an AI Hub quantized QDQ, or a granularity variant."""
    import qai_hub as hub
    if prec == "fp32":
        return uploaded_fp32(task, graph, H)
    if prec in QDTYPE:
        q = find(rec, kind="quantize", graph=graph, H=H, prec=prec)
        if not q:
            return None
        job = hub.get_job(q[-1]["job_id"])
        st = job.get_status()
        if st.code != "SUCCESS":
            return "WAIT" if st.code not in ("FAILED",) else None
        return job.get_target_model()
    v = find(rec, kind="granvar", graph=graph, H=H, prec=prec)
    if not v:
        return None
    return hub.get_model(v[-1]["model_id"])


def resolve_devices(spec):
    if spec == "all":
        return DEVICES
    if spec == "qnn":
        return [d for d in DEVICES if d[2]]
    if spec == "core":
        return [DEV_BY_NAME[n] for n in CORE]
    return [DEV_BY_NAME[n] for n in spec.split("|")]


def compile_opts(dev, prec, graph):
    """AI Hub dropped the qnn_context_binary target after the Sep-10 H=3 sweep; the QNN runtime is
    now reached through qnn_dlc (NPU by default). H=3 is re-profiled through the same path."""
    name, os_, has_qnn, fp16 = dev
    if not has_qnn:
        return "--target_runtime tflite", "tflite"
    if prec == "fp32":
        return "--target_runtime qnn_dlc", "qnn_dlc"
    # Integer-only HTPs reject any float tensor, I/O included; quantize the I/O everywhere so the
    # rows are comparable (the H=3 table did the same).
    return "--target_runtime qnn_dlc --quantize_io", "qnn_dlc_qio"


def uploaded_fp32(task, graph, H):
    """Upload each fp32 ONNX once and reuse the model id (a path would be re-uploaded per job)."""
    import qai_hub as hub
    import fcntl
    p = ART / "uploaded.json"
    lock = open(ART / "uploaded.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        d = json.loads(p.read_text()) if p.exists() else {}
        key = f"{task}/{onnx_name(graph, H)}"
        if key not in d:
            m = hub.upload_model(str(ART / task / onnx_name(graph, H)), name=f"x3_{task}_{graph}_H{H}_fp32")
            d[key] = m.model_id
            p.write_text(json.dumps(d, indent=1))
        return hub.get_model(d[key])
    finally:
        lock.close()


def stage_compile(tasks, H, graphs, precs, devspec, wait=True):
    import qai_hub as hub
    devs = resolve_devices(devspec)
    pending = True
    while pending:
        pending = False
        for task in tasks:
            rec = load_rec(task)
            for graph in graphs:
                for prec in precs:
                    todo = [d for d in devs if not [j for j in find(rec, kind="compile", graph=graph, H=H, prec=prec,
                                                                      device=d[0]) if j.get("job_id")]]
                    if not todo:
                        continue
                    if prec != "fp32" and all(not d[2] for d in todo):
                        # TF-Lite cannot hold INT4 or these QDQ graphs; the H=3 sweep documented it.
                        todo = todo
                    model = source_model(task, rec, graph, H, prec)
                    if model == "WAIT":
                        pending = True
                        continue
                    if model is None:
                        print(f"[{task}] {graph} H{H} {prec}: no source model", flush=True)
                        continue
                    for d in todo:
                        opts, runtime = compile_opts(d, prec, graph)
                        try:
                            cj = hub.submit_compile_job(model=model, device=hub.Device(d[0], d[1]), options=opts,
                                                        name=f"x3_{task}_{graph}_H{H}_{prec}_{d[0]}")
                            jid, err = cj.job_id, None
                        except Exception as exc:  # noqa: BLE001
                            jid, err = None, str(exc)[:300]
                        add_job(task, {"kind": "compile", "graph": graph, "H": H, "prec": prec, "device": d[0],
                                       "os": d[1], "runtime": runtime, "options": opts, "job_id": jid,
                                       "submit_error": err,
                                       "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                        print(f"[{task}] compile {graph} H{H} {prec:9s} {d[0]:32s} -> {jid or 'SUBMIT-ERR ' + err[:80]}",
                              flush=True)
        if pending and wait:
            print("[compile] waiting for quantize jobs ...", flush=True)
            time.sleep(60)
        elif not wait:
            break


def bulk_status(max_jobs=6000):
    """{job_id: (type, code)} for the account's recent jobs, 100 per API call (get_job is one call
    per job, which made a pass over ~2000 records take a quarter of an hour)."""
    import qai_hub as hub
    out, off = {}, 0
    while off < max_jobs:
        s = hub.get_job_summaries(offset=off, limit=100)
        if not s:
            break
        for x in s:
            out[x.job_id] = (x.job_type.name, x.status.code)
        off += len(s)
    return out


def stage_profile(tasks, loop=False):
    """Profile every successful compile whose (task, graph, H, prec, device) has no profile yet."""
    import qai_hub as hub
    while True:
        waiting = 0
        bs = bulk_status()
        for task in tasks:
            rec = load_rec(task)
            have = {(j["graph"], j["H"], j["prec"], j["device"]) for j in rec["jobs"]
                    if j["kind"] == "profile" and j.get("job_id")}
            for j in list(rec["jobs"]):
                if j["kind"] != "compile" or not j.get("job_id") or j.get("final"):
                    continue
                code = bs.get(j["job_id"], (None, None))[1]
                if code is None:
                    code = hub.get_job(j["job_id"]).get_status().code
                if code == "FAILED":
                    st = hub.get_job(j["job_id"]).get_status()
                    j["final"], j["error"] = "FAILED", (st.message or "")[:600]
                    update_job(task, j["job_id"], final="FAILED", error=j["error"])
                    print(f"[{task}] compile FAILED {j['graph']} {j['prec']} {j['device']}: {j['error'][:120]}", flush=True)
                    continue
                if code != "SUCCESS":
                    waiting += 1
                    continue
                j["final"] = "SUCCESS"
                update_job(task, j["job_id"], final="SUCCESS")
                if (j["graph"], j["H"], j["prec"], j["device"]) in have:
                    continue          # already profiled through a fanned-out DLC
                job = hub.get_job(j["job_id"])
                try:
                    pj = hub.submit_profile_job(model=job.get_target_model(), device=hub.Device(j["device"], j["os"]),
                                                name=f"x3_prof_{task}_{j['graph']}_H{j['H']}_{j['prec']}_{j['device']}")
                    pid, err = pj.job_id, None
                except Exception as exc:  # noqa: BLE001
                    pid, err = None, str(exc)[:300]
                add_job(task, {"kind": "profile", "graph": j["graph"], "H": j["H"], "prec": j["prec"],
                               "device": j["device"], "os": j["os"], "runtime": j["runtime"], "job_id": pid,
                               "submit_error": err, "from_compile": j["job_id"],
                               "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                have.add((j["graph"], j["H"], j["prec"], j["device"]))
                print(f"[{task}] profile {j['graph']} H{j['H']} {j['prec']:9s} {j['device']:32s} -> {pid or err[:80]}",
                      flush=True)
        if not loop or waiting == 0:
            break
        print(f"[profile] {waiting} compiles still running; sleeping", flush=True)
        time.sleep(90)


def stage_fanout(tasks, graphs=None):
    """qnn_dlc and tflite targets are device-agnostic, so one successful compile per (task, graph,
    H, prec, runtime class) is profiled on every device of that class. AI Hub runs only a handful
    of compiles per account at a time, so per-device compiles would take a day; the per-device
    compiles already queued are skipped by stage_profile once a fanned-out profile exists."""
    import qai_hub as hub
    bs = bulk_status()
    for task in tasks:
        rec = load_rec(task)
        have = {(j["graph"], j["H"], j["prec"], j["device"]) for j in rec["jobs"]
                if j["kind"] == "profile" and j.get("job_id")}
        src = {}
        for j in rec["jobs"]:
            if j["kind"] != "compile" or not j.get("job_id"):
                continue
            if graphs and j["graph"] not in graphs:
                continue
            code = bs.get(j["job_id"], (None, None))[1]
            if code == "SUCCESS":
                cls = "tflite" if j["runtime"] == "tflite" else "qnn"
                src.setdefault((j["graph"], j["H"], j["prec"], cls), j)
        for (graph, H, prec, cls), j in sorted(src.items(), key=lambda kv: str(kv[0])):
            model = None
            devs = [d for d in DEVICES if (d[2] if cls == "qnn" else not d[2])]
            wanted = {(c["device"]) for c in rec["jobs"] if c["kind"] == "compile" and c["graph"] == graph
                      and c["H"] == H and c["prec"] == prec}
            if graph == "step":           # the device table wants every device of the class
                wanted |= {d[0] for d in devs}
            for d in devs:
                if d[0] not in wanted or (graph, H, prec, d[0]) in have:
                    continue
                if model is None:
                    model = hub.get_job(j["job_id"]).get_target_model()
                try:
                    pj = hub.submit_profile_job(model=model, device=hub.Device(d[0], d[1]),
                                                name=f"x3_prof_{task}_{graph}_H{H}_{prec}_{d[0]}")
                    pid, err = pj.job_id, None
                except Exception as exc:  # noqa: BLE001
                    pid, err = None, str(exc)[:300]
                add_job(task, {"kind": "profile", "graph": graph, "H": H, "prec": prec, "device": d[0], "os": d[1],
                               "runtime": j["runtime"], "job_id": pid, "submit_error": err,
                               "from_compile": j["job_id"], "fanout_from_device": j["device"],
                               "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                have.add((graph, H, prec, d[0]))
                print(f"[{task}] fanout {graph} H{H} {prec:9s} {d[0]:32s} <- {j['job_id']} -> {pid or err[:80]}",
                      flush=True)


def summarize_profile(p):
    s = p["execution_summary"]
    units = {}
    for l in p.get("execution_detail", []):
        units[l.get("compute_unit", "?")] = units.get(l.get("compute_unit", "?"), 0) + 1
    row = {"ms": s["estimated_inference_time"] / 1000,
           "load_peak_mb": s.get("first_load_peak_memory", 0) / 2 ** 20,
           "infer_peak_mb": s.get("inference_memory_peak_range", [0, 0])[1] / 2 ** 20,
           "peak_mb": s.get("estimated_inference_peak_memory", 0) / 2 ** 20, "units": units,
           "all_times_ms": [t / 1000 for t in s.get("all_inference_times", [])][:200]}
    for k in ("compute_unit_breakdown", "runtime", "runtime_version", "qnn_version", "framework_version"):
        if k in s:
            row[k] = s[k]
    rc = p.get("runtime_config") or p.get("tool_versions") or {}
    if rc:
        row["tool"] = rc
    return row


def runtime_log_errors(job, keep=6):
    """The on-device runtime log states why a profile failed (MODEL_GRAPH_ERROR alone does not)."""
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as d:
            paths = job.download_job_logs(d)
            lines = []
            for p in (paths if isinstance(paths, list) else [paths]):
                for l in open(p, errors="replace"):
                    if "<E>" in l or "error]" in l:
                        lines.append(l.split("] ", 3)[-1].strip()[:300])
            return lines[:keep]
    except Exception as exc:  # noqa: BLE001
        return [f"log download failed: {str(exc)[:120]}"]


def stage_status(tasks, dump_dir=None):
    import qai_hub as hub
    rows = []
    bs = bulk_status()
    for task in tasks:
        rec = load_rec(task)
        for j in rec["jobs"]:
            if j["kind"] not in ("quantize", "compile", "profile", "inference"):
                continue
            row = {"task": task, **{k: j.get(k) for k in ("kind", "graph", "H", "prec", "device", "runtime", "job_id",
                                                          "submit_error", "from_compile")}}
            if not j.get("job_id"):
                row["status"] = "SUBMIT_ERROR"
                rows.append(row)
                continue
            if j.get("status_final") in ("SUCCESS", "FAILED") and "summary" in j:
                row.update(j["summary"]); row["status"] = j["status_final"]
                rows.append(row)
                continue
            code = bs.get(j["job_id"], (None, None))[1]
            if code is not None and code not in ("SUCCESS", "FAILED"):
                row["status"] = code
                rows.append(row)
                continue
            job = hub.get_job(j["job_id"])
            st = job.get_status()
            row["status"] = st.code
            summary = {}
            if st.code == "FAILED":
                summary["error"] = (st.message or "")[:600]
                if j["kind"] in ("profile", "inference"):
                    summary["log_errors"] = runtime_log_errors(job)
            elif st.code == "SUCCESS" and j["kind"] == "profile":
                p = job.download_profile()
                summary.update(summarize_profile(p))
                if dump_dir:
                    d = pathlib.Path(dump_dir) / task
                    d.mkdir(parents=True, exist_ok=True)
                    (d / f"{j['job_id']}.json").write_text(json.dumps(p))
            if st.code in ("SUCCESS", "FAILED"):
                update_job(task, j["job_id"], status_final=st.code, summary=summary)
            row.update(summary)
            rows.append(row)
    out = ART / "status.json"
    old = json.loads(out.read_text()) if out.exists() else []
    keep = [r for r in old if r["task"] not in tasks]
    out.write_text(json.dumps(keep + rows, indent=1))
    by = {}
    for r in rows:
        by[(r["kind"], r["status"])] = by.get((r["kind"], r["status"]), 0) + 1
    print("status", by, "->", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["export", "calib", "quantize", "compile", "profile", "fanout", "status"])
    ap.add_argument("--tasks", default="sudoku_mlp,sudoku_attn,maze,arc")
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--graphs", default="step")
    ap.add_argument("--precs", default="fp32,w8a8,w4a8")
    ap.add_argument("--devices", default="all")
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--no_wait", action="store_true")
    ap.add_argument("--dump", default="")
    a = ap.parse_args()
    tasks = [t for t in a.tasks.split(",") if t]
    graphs = [g for g in a.graphs.split(",") if g]
    precs = [p for p in a.precs.split(",") if p]
    if a.stage == "export":
        stage_export(tasks, a.H)
    elif a.stage == "calib":
        stage_calib(tasks, a.H)
    elif a.stage == "quantize":
        stage_quantize(tasks, a.H, graphs, precs)
    elif a.stage == "compile":
        stage_compile(tasks, a.H, graphs, precs, a.devices, wait=not a.no_wait)
    elif a.stage == "profile":
        stage_profile(tasks, loop=a.loop)
    elif a.stage == "fanout":
        stage_fanout(tasks, graphs if a.graphs != "step" or True else None)
    else:
        stage_status(tasks, a.dump or None)


if __name__ == "__main__":
    main()
