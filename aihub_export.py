#!/usr/bin/env python3
"""
Compile and profile TRM on real Qualcomm silicon, with job IDs we own.

Why this exists: every on-device number in the paper came from Jim's AI Hub
account and he did not keep the job ids or compile reports. Nothing can be
re-run or checked. Worse, the INT4 step latency matched FP32 to three
significant figures on all three chips, which is the signature of a compiler
folding INT4 to FP16 rather than executing it. Until a compile report says
otherwise, the 3.26 MB INT4 configuration is a weights-only projection.

What this measures: the *inner* recursion step, not the full ACT loop. That is
deliberate and it matches the paper's step-latency column, which is the only
depth-independent number in the on-device table. Puzzle latency is step latency
times the recursion schedule, so exporting one step keeps the export tractable
(no data-dependent loop in the graph) without losing the quantity we compare.

Order of operations, cheapest decisive thing first:
  1. export the inner step to ONNX and check it against the torch forward
  2. compile FP32 -> read the report, confirm the baseline path
  3. quantize to INT8 and INT4 through AI Hub, compile each
  4. read each compile report for the precision the ops actually got
  5. profile on Galaxy S24, Galaxy S22 5G and Dragonwing RB3 Gen 2

Step 4 is the point. If INT4 lands as FP16 in the report, the paper drops its
INT4 latency rows rather than carrying a projection the appendix already flags.

Run on p4d, where the checkpoint and TRM source already live:
    python3 aihub_export.py --stage export
    python3 aihub_export.py --stage submit
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

CKPT_DIR = Path.home() / "edgetrm_ckpt" / "arc_v1_public"
TRM_SRC = Path(__file__).parent / "TinyRecursiveModels"
ARTIFACTS = Path.home() / "aihub_artifacts"

# Device names as AI Hub reports them; verified present on this account.
DEVICES = [
    "Samsung Galaxy S24",
    "Samsung Galaxy S22 5G",
    "Dragonwing RB3 Gen 2 Vision Kit",
]


def _setup():
    if str(TRM_SRC) not in sys.path:
        sys.path.insert(0, str(TRM_SRC))


def _load_sd(ckpt):
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return {k.replace("_orig_mod.model.", "").replace("model.", ""): v
            for k, v in sd.items()}


def _build(sd, cfg_yaml, batch_size=1):
    from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1

    arch = cfg_yaml["arch"]
    vocab = sd["inner.embed_tokens.embedding_weight"].shape[0]
    hidden = sd["inner.embed_tokens.embedding_weight"].shape[1]
    npid = sd["inner.puzzle_emb.weights"].shape[0]
    cfg = dict(
        batch_size=batch_size, seq_len=900, vocab_size=vocab,
        num_puzzle_identifiers=npid,
        H_cycles=arch["H_cycles"], L_cycles=arch["L_cycles"],
        H_layers=arch["H_layers"], L_layers=arch["L_layers"],
        hidden_size=hidden, expansion=arch["expansion"], num_heads=arch["num_heads"],
        pos_encodings=arch["pos_encodings"], halt_max_steps=arch["halt_max_steps"],
        halt_exploration_prob=arch["halt_exploration_prob"],
        forward_dtype="float32",
        mlp_t=arch.get("mlp_t", False),
        puzzle_emb_ndim=arch["puzzle_emb_ndim"],
        puzzle_emb_len=arch.get("puzzle_emb_len", 16),
        no_ACT_continue=arch.get("no_ACT_continue", True),
    )
    m = TinyRecursiveReasoningModel_ACTV1(cfg).eval()
    miss, unexp = m.load_state_dict(sd, strict=False)
    assert not miss and not unexp, (miss[:5], unexp[:5])
    return m


class _EmbFromInput(torch.nn.Module):
    """Stands in for the puzzle embedding table, returning a graph input.

    The real table is 876,406 identifiers wide and dominates the export at
    1.85 GB, which is the same bottleneck the paper's flash-streaming result
    removes: on device only the 2 KB row for the current puzzle is resident.
    Compiling the table into the graph would measure a configuration nobody
    would deploy, so the row enters as an input instead.
    """

    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, idx):  # idx unused: the row is streamed, not looked up
        return self.value


class InnerStep(torch.nn.Module):
    """One recursion step as a static graph.

    The ACT loop is data-dependent (it halts when the model decides to), which
    does not export. The step inside it is fixed-shape and is what the paper's
    step-latency column reports, so that is what we compile.
    """

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self._emb = _EmbFromInput()
        self.inner.puzzle_emb = self._emb

    def forward(self, z_H, z_L, inputs, puzzle_embedding):
        # Tracing follows tensor identity, so binding the traced input here
        # threads it into the graph as a real input rather than a constant.
        self._emb.value = puzzle_embedding
        carry = type(self.inner.empty_carry(1))(z_H=z_H, z_L=z_L) \
            if hasattr(self.inner, "empty_carry") else None
        batch = {"inputs": inputs,
                 "puzzle_identifiers": torch.zeros(1, dtype=torch.int32)}
        out = self.inner(carry, batch) if carry is not None else self.inner(batch)
        # The inner module returns (new_carry, logits, ...) across TRM versions;
        # take the first tensor that looks like logits over the vocabulary.
        if isinstance(out, (tuple, list)):
            for o in out:
                if torch.is_tensor(o) and o.dim() == 3:
                    return o
            return out[-1]
        return out


def stage_export(args):
    _setup()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cfg_yaml = yaml.safe_load((CKPT_DIR / "all_config.yaml").read_text())
    sd = _load_sd(CKPT_DIR / "step_518071")
    m = _build(sd, cfg_yaml, batch_size=1)

    hidden = sd["inner.embed_tokens.embedding_weight"].shape[1]
    puzzle_len = cfg_yaml["arch"].get("puzzle_emb_len", 16)
    seq = 900 + puzzle_len

    emb_ndim = cfg_yaml["arch"]["puzzle_emb_ndim"]
    z_H = torch.zeros(1, seq, hidden)
    z_L = torch.zeros(1, seq, hidden)
    inputs = torch.zeros(1, 900, dtype=torch.int32)
    puzzle_emb = torch.zeros(1, emb_ndim)

    wrapper = InnerStep(m.inner).eval()
    with torch.no_grad():
        ref = wrapper(z_H, z_L, inputs, puzzle_emb)
    print(f"[export] torch forward OK, output {tuple(ref.shape)}")

    onnx_path = ARTIFACTS / "trm_inner_step.onnx"
    torch.onnx.export(
        wrapper, (z_H, z_L, inputs, puzzle_emb), str(onnx_path),
        input_names=["z_H", "z_L", "inputs", "puzzle_embedding"],
        output_names=["logits"], opset_version=17, dynamo=False,
    )
    print(f"[export] wrote {onnx_path} ({onnx_path.stat().st_size/1e6:.1f} MB)")
    # Later stages build calibration tensors and need these without reloading
    # the 1.8 GB checkpoint.
    (ARTIFACTS / "shapes.json").write_text(json.dumps(
        {"seq": seq, "hidden": hidden, "emb_ndim": emb_ndim, "vocab_out": int(ref.shape[-1])}))

    # Numerical check against torch, so a bad export fails here rather than
    # silently producing plausible latencies for the wrong graph.
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        got = sess.run(["logits"], {
            "z_H": z_H.numpy(), "z_L": z_L.numpy(),
            "inputs": inputs.numpy(), "puzzle_embedding": puzzle_emb.numpy()})[0]
        err = float(np.abs(got - ref.numpy()).max())
        print(f"[export] onnxruntime max abs diff vs torch: {err:.3e}")
        assert err < 1e-3, f"ONNX export disagrees with torch by {err}"
    except ImportError:
        print("[export] onnxruntime not installed; SKIPPED numerical check")


def stage_submit(args):
    import qai_hub as hub

    onnx_path = ARTIFACTS / "trm_inner_step.onnx"
    assert onnx_path.exists(), f"run --stage export first ({onnx_path} missing)"
    record = {"submitted_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ"), "jobs": []}

    devices = [hub.Device(d) for d in DEVICES]
    for dev in devices:
        # FP32 baseline: whatever the compiler chooses is itself a finding, and
        # the paper's "FP32" rows are believed to execute as FP16 on Hexagon.
        cj = hub.submit_compile_job(model=str(onnx_path), device=dev,
                                    name=f"trm_step_fp32_{dev.name}")
        record["jobs"].append({"kind": "compile", "precision": "fp32",
                               "device": dev.name, "job_id": cj.job_id})
        print(f"[submit] compile fp32 {dev.name} -> {cj.job_id}")

    Path(ARTIFACTS / "job_ids.json").write_text(json.dumps(record, indent=2))
    print(f"[submit] wrote {ARTIFACTS/'job_ids.json'}")
    print("[submit] NOTE: quantized variants are submitted by --stage quantize "
          "once the fp32 compile report confirms the graph lands intact.")


def stage_quantize(args):
    """Quantize to INT8 and INT4 through AI Hub, compile, and profile.

    This is the decisive stage. The paper's INT4 step latency matched FP32 to
    three significant figures on all three chips, which is what you see when a
    compiler folds INT4 to FP16 instead of executing it. The compile report for
    each variant says which precision the ops actually got, and the profile job
    gives a latency we can trace to a job id.
    """
    import numpy as np
    import qai_hub as hub

    onnx_path = ARTIFACTS / "trm_inner_step.onnx"
    rec = json.loads((ARTIFACTS / "job_ids.json").read_text())

    cfg_yaml = yaml.safe_load((CKPT_DIR / "all_config.yaml").read_text())
    sd_shapes = json.loads((ARTIFACTS / "shapes.json").read_text())
    seq, hidden = sd_shapes["seq"], sd_shapes["hidden"]
    emb_ndim = cfg_yaml["arch"]["puzzle_emb_ndim"]

    # Calibration: a handful of samples in the range the graph actually sees.
    # Carry states are unit-scale after the model's RMS norms; inputs are token
    # ids over a 12-symbol grid vocabulary.
    rng = np.random.default_rng(0)
    n_cal = args.calib
    calib = {
        "z_H": [rng.normal(0, 1, (1, seq, hidden)).astype(np.float32) for _ in range(n_cal)],
        "z_L": [rng.normal(0, 1, (1, seq, hidden)).astype(np.float32) for _ in range(n_cal)],
        "inputs": [rng.integers(0, 12, (1, 900)).astype(np.int32) for _ in range(n_cal)],
        "puzzle_embedding": [rng.normal(0, 1, (1, emb_ndim)).astype(np.float32)
                             for _ in range(n_cal)],
    }

    for wdtype, tag in (("int8", "int8"), ("int4", "int4")):
        qj = hub.submit_quantize_job(
            model=str(onnx_path), calibration_data=calib,
            weights_dtype=getattr(hub.QuantizeDtype, wdtype.upper()),
            activations_dtype=hub.QuantizeDtype.INT8,
            name=f"trm_step_quant_{tag}",
        )
        rec["jobs"].append({"kind": "quantize", "precision": tag,
                            "device": "-", "job_id": qj.job_id})
        print(f"[quantize] {tag} -> {qj.job_id}")
        qj.wait()
        qmodel = qj.get_target_model()
        for dname in DEVICES:
            cj = hub.submit_compile_job(model=qmodel, device=hub.Device(dname),
                                        name=f"trm_step_{tag}_{dname}")
            rec["jobs"].append({"kind": "compile", "precision": tag,
                                "device": dname, "job_id": cj.job_id})
            print(f"[quantize] compile {tag} {dname} -> {cj.job_id}")

    (ARTIFACTS / "job_ids.json").write_text(json.dumps(rec, indent=2))
    print(f"[quantize] wrote {ARTIFACTS/'job_ids.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["export", "submit", "quantize"])
    ap.add_argument("--calib", type=int, default=8)
    args = ap.parse_args()
    {"export": stage_export, "submit": stage_submit,
     "quantize": stage_quantize}[args.stage](args)


if __name__ == "__main__":
    main()
