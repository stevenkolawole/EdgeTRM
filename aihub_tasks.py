#!/usr/bin/env python3
"""On-device measurements for the Maze and Sudoku inner steps, matching what
was done for ARC in aihub_export.py: export one recursion step to ONNX,
check it against torch, then on Qualcomm AI Hub compile FP32 (QNN) for the
phones, quantize INT8 and INT4, compile with quantized I/O for all three
devices, and profile everything. Job ids go to ~/aihub_artifacts/<task>/.

    python3 aihub_tasks.py --task maze --stage export
    python3 aihub_tasks.py --task maze --stage submit
    python3 aihub_tasks.py --task sudoku_mlp --stage export ...

Run on p4d with ~/venvs/kvcache/bin/python (torch, onnx, qai_hub).
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

HOME = pathlib.Path.home()
TRM_SRC = HOME / "edgetrm_src" / "TinyRecursiveModels"
CKPT = HOME / "edgetrm_ckpt"
ART_ROOT = HOME / "aihub_artifacts"
PHONES = ["Samsung Galaxy S24", "Samsung Galaxy S22 5G"]
RB3 = "Dragonwing RB3 Gen 2 Vision Kit"
QIO = "--target_runtime qnn_context_binary --quantize_io"
QNN = "--target_runtime qnn_context_binary"

# Every field the TRM config needs, per checkpoint. Cycle counts are the ones
# the paper evaluates with (Maze: its all_config.yaml; Sudoku: the community
# reproduction's H=3, L=6, the same the accuracy harness uses).
TASKS = {
    "maze": dict(ckpt=CKPT / "maze_hard" / "model.pt", seq_len=900, vocab=6,
                 H_cycles=3, L_cycles=4, mlp_t=False, pos_encodings="rope"),
    # The MLP-mixing Sudoku checkpoint has no positional encoding (its token
    # mixer is position-aware by construction); the attention one uses RoPE.
    "sudoku_mlp": dict(ckpt=CKPT / "sudoku_extreme" / "step_39060_sudoku_epoch_60k",
                       seq_len=81, vocab=11, H_cycles=3, L_cycles=6, mlp_t=True, pos_encodings="none"),
    "sudoku_attn": dict(ckpt=CKPT / "sudoku_extreme" / "step_39060_sudoku_60k_epoch_attn_type",
                        seq_len=81, vocab=11, H_cycles=3, L_cycles=6, mlp_t=False, pos_encodings="rope"),
}
COMMON = dict(H_layers=0, L_layers=2, hidden_size=512, expansion=4, num_heads=8,
              halt_max_steps=16, halt_exploration_prob=0.1,
              forward_dtype="float32", puzzle_emb_ndim=512, puzzle_emb_len=16,
              no_ACT_continue=True)


def load_sd(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return {k.replace("_orig_mod.model.", "").replace("model.", ""): v for k, v in sd.items()}


def build(task):
    sys.path.insert(0, str(TRM_SRC))
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
    t = TASKS[task]
    sd = load_sd(t["ckpt"])
    cfg = dict(batch_size=1, seq_len=t["seq_len"], vocab_size=t["vocab"],
               num_puzzle_identifiers=sd["inner.puzzle_emb.weights"].shape[0],
               H_cycles=t["H_cycles"], L_cycles=t["L_cycles"], mlp_t=t["mlp_t"],
               pos_encodings=t["pos_encodings"], **COMMON)
    m = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss[:5], unexp[:5])
    return m, cfg


class InnerStep(torch.nn.Module):
    """One recursion step, fixed shape, puzzle embedding row as a graph input
    (the streamed-from-flash configuration the paper deploys)."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._value = None
        outer = self

        class Emb(torch.nn.Module):
            def forward(self, idx):
                return outer._value
        self.inner.puzzle_emb = Emb()

    def forward(self, z_H, z_L, inputs, puzzle_embedding):
        self._value = puzzle_embedding
        carry = type(self.inner.empty_carry(1))(z_H=z_H, z_L=z_L)
        batch = {"inputs": inputs, "puzzle_identifiers": torch.zeros(1, dtype=torch.int32)}
        out = self.inner(carry, batch)
        for o in (out if isinstance(out, (tuple, list)) else [out]):
            if torch.is_tensor(o) and o.dim() == 3:
                return o
        return out[-1]


def stage_export(task, art):
    m, cfg = build(task)
    seq = cfg["seq_len"] + COMMON["puzzle_emb_len"]
    z_H = torch.zeros(1, seq, 512)
    z_L = torch.zeros(1, seq, 512)
    inputs = torch.zeros(1, cfg["seq_len"], dtype=torch.int32)
    emb = torch.zeros(1, 512)
    w = InnerStep(m.inner).eval()
    with torch.no_grad():
        ref = w(z_H, z_L, inputs, emb)
    onnx_path = art / "inner_step.onnx"
    torch.onnx.export(w, (z_H, z_L, inputs, emb), str(onnx_path),
                      input_names=["z_H", "z_L", "inputs", "puzzle_embedding"],
                      output_names=["logits"], opset_version=17, dynamo=False)
    (art / "shapes.json").write_text(json.dumps({"seq": seq, "seq_in": cfg["seq_len"],
                                                 "vocab": cfg["vocab_size"]}))
    print(f"[{task}] exported {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB), out {tuple(ref.shape)}")
    import onnxruntime as ort
    got = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"]).run(
        ["logits"], {"z_H": z_H.numpy(), "z_L": z_L.numpy(), "inputs": inputs.numpy(),
                     "puzzle_embedding": emb.numpy()})[0]
    err = float(np.abs(got - ref.numpy()).max())
    print(f"[{task}] onnxruntime max abs diff vs torch {err:.3e}")
    # Logits are O(10); the MLP-mixing step accumulates over 97 positions and
    # lands at 7e-3, the attention steps at 1e-4. Both are fp32 noise.
    assert err < 2e-2


def stage_submit(task, art, n_cal=8):
    import qai_hub as hub
    onnx_path = art / "inner_step.onnx"
    sh = json.loads((art / "shapes.json").read_text())
    rec = {"task": task, "submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "jobs": []}

    def save():
        (art / "job_ids.json").write_text(json.dumps(rec, indent=2))

    # FP32 on the phones, QNN target (the RB3 NPU is fixed-point only; its
    # FP32 row would run on the GPU, which the ARC run already documented).
    compiles = []
    for d in PHONES:
        cj = hub.submit_compile_job(model=str(onnx_path), device=hub.Device(d), options=QNN,
                                    name=f"{task}_fp32_qnn_{d}")
        rec["jobs"].append({"kind": "compile", "precision": "fp32", "runtime": "qnn_context_binary",
                            "device": d, "job_id": cj.job_id})
        compiles.append(("fp32", d, cj))
        print(f"[{task}] compile fp32 {d} -> {cj.job_id}", flush=True)
    save()
    # Quantize INT8 / INT4 with unit-scale calibration (carry states are RMS-normed).
    rng = np.random.default_rng(0)
    calib = {
        "z_H": [rng.normal(0, 1, (1, sh["seq"], 512)).astype(np.float32) for _ in range(n_cal)],
        "z_L": [rng.normal(0, 1, (1, sh["seq"], 512)).astype(np.float32) for _ in range(n_cal)],
        "inputs": [rng.integers(0, sh["vocab"], (1, sh["seq_in"])).astype(np.int32) for _ in range(n_cal)],
        "puzzle_embedding": [rng.normal(0, 1, (1, 512)).astype(np.float32) for _ in range(n_cal)],
    }
    for prec in ("int8", "int4"):
        qj = hub.submit_quantize_job(model=str(onnx_path), calibration_data=calib,
                                     weights_dtype=getattr(hub.QuantizeDtype, prec.upper()),
                                     activations_dtype=hub.QuantizeDtype.INT8,
                                     name=f"{task}_quant_{prec}")
        rec["jobs"].append({"kind": "quantize", "precision": prec, "device": "-", "job_id": qj.job_id})
        print(f"[{task}] quantize {prec} -> {qj.job_id}", flush=True)
        save()
        qj.wait()
        if qj.get_status().code != "SUCCESS":
            print(f"[{task}] quantize {prec} FAILED: {(qj.get_status().message or '')[:200]}")
            continue
        qm = qj.get_target_model()
        for d in PHONES + [RB3]:
            cj = hub.submit_compile_job(model=qm, device=hub.Device(d), options=QIO,
                                        name=f"{task}_{prec}_qnn_qio_{d}")
            rec["jobs"].append({"kind": "compile", "precision": prec, "runtime": "qnn_context_binary_qio",
                                "device": d, "job_id": cj.job_id})
            compiles.append((prec, d, cj))
            print(f"[{task}] compile {prec} {d} -> {cj.job_id}", flush=True)
        save()
    for prec, d, cj in compiles:
        cj.wait()
        st = cj.get_status()
        if st.code != "SUCCESS":
            print(f"[{task}] compile {prec} {d} FAILED: {(st.message or '')[:200]}", flush=True)
            continue
        pj = hub.submit_profile_job(model=cj.get_target_model(), device=hub.Device(d),
                                    name=f"{task}_profile_{prec}_{d}")
        rec["jobs"].append({"kind": "profile", "precision": prec, "device": d,
                            "job_id": pj.job_id, "from_compile": cj.job_id})
        print(f"[{task}] profile {prec} {d} -> {pj.job_id}", flush=True)
        save()


def stage_status(task, art):
    import qai_hub as hub
    rec = json.loads((art / "job_ids.json").read_text())
    for j in rec["jobs"]:
        job = hub.get_job(j["job_id"])
        st = job.get_status()
        line = f"{task:12s} {j['kind']:8s} {j['precision']:5s} {j['device'][:22]:22s} {st.code:9s}"
        if j["kind"] == "profile" and st.code == "SUCCESS":
            p = job.download_profile()
            s = p["execution_summary"]
            units = {}
            for l in p.get("execution_detail", []):
                units[l.get("compute_unit", "?")] = units.get(l.get("compute_unit", "?"), 0) + 1
            line += (f" {s['estimated_inference_time']/1000:.2f} ms load {s['first_load_peak_memory']/2**20:.1f} MB"
                     f" infer+ {s['inference_memory_peak_range'][1]/2**20:.1f} MB"
                     f" peak {s['estimated_inference_peak_memory']/2**20:.1f} MB units={units}")
        elif st.code == "FAILED":
            line += " " + (st.message or "")[:120]
        print(line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TASKS))
    ap.add_argument("--stage", required=True, choices=["export", "submit", "status"])
    a = ap.parse_args()
    art = ART_ROOT / a.task
    art.mkdir(parents=True, exist_ok=True)
    {"export": stage_export, "submit": stage_submit, "status": stage_status}[a.stage](a.task, art)


if __name__ == "__main__":
    main()
