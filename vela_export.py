#!/usr/bin/env python3
"""MCU-tier measurement: one recursion step as an INT8 TFLite graph, compiled
with Arm's Vela for the Ethos-U55 on a Cortex-M55 system (the Corstone-300
reference design). Vela reports the exact SRAM and flash the compiled graph
needs and a cycle estimate, which is the number the paper's 4-8 MB claims
have been standing in for analytically.

    ~/venvs/mcu/bin/python vela_export.py --task maze
    ~/venvs/mcu/bin/python vela_export.py --task sudoku_mlp

Steps: build the torch inner step (aihub_tasks.build), PT2E static INT8
quantization with unit-scale calibration, litert-torch conversion to
.tflite, then `vela` with the Ethos-U55-256 high-end embedded system config
and shared-SRAM memory mode. Output under ~/mcu/<task>/.
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_tasks as T  # noqa: E402

OUT_ROOT = pathlib.Path.home() / "mcu"


class Step(torch.nn.Module):
    """Same inner step as aihub_tasks.InnerStep, but the single puzzle row
    (these checkpoints have one identifier) is folded in as a constant, so
    the graph has three inputs: z_H, z_L, inputs."""

    def __init__(self, inner, emb_row):
        super().__init__()
        self.inner = inner
        self.register_buffer("emb_row", emb_row)
        outer = self

        class Emb(torch.nn.Module):
            def forward(self, idx):
                return outer.emb_row
        self.inner.puzzle_emb = Emb()

    def forward(self, z_H, z_L, inputs):
        carry = type(self.inner.empty_carry(1))(z_H=z_H, z_L=z_L)
        batch = {"inputs": inputs, "puzzle_identifiers": torch.zeros(1, dtype=torch.int32)}
        out = self.inner(carry, batch)
        for o in (out if isinstance(out, (tuple, list)) else [out]):
            if torch.is_tensor(o) and o.dim() == 3:
                return o
        return out[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(T.TASKS))
    ap.add_argument("--n_cal", type=int, default=8)
    ap.add_argument("--accel", default="ethos-u55-256")
    ap.add_argument("--fp32_only", action="store_true", help="skip quantization (debug)")
    ap.add_argument("--all_ops", action="store_true", help="quantize every op, not just matrix ops")
    a = ap.parse_args()
    out = OUT_ROOT / a.task
    out.mkdir(parents=True, exist_ok=True)

    m, cfg = T.build(a.task)
    sd = T.load_sd(T.TASKS[a.task]["ckpt"])
    emb_row = sd["inner.puzzle_emb.weights"][:1].clone().float()
    seq = cfg["seq_len"] + T.COMMON["puzzle_emb_len"]
    step = Step(m.inner, emb_row).eval()
    z_H = torch.zeros(1, seq, 512)
    z_L = torch.zeros(1, seq, 512)
    inputs = torch.zeros(1, cfg["seq_len"], dtype=torch.int32)
    with torch.no_grad():
        ref = step(z_H, z_L, inputs)
    print(f"[{a.task}] torch step OK, logits {tuple(ref.shape)}", flush=True)

    import litert_torch
    if a.fp32_only:
        edge = litert_torch.convert(step, (z_H, z_L, inputs))
        path = out / "inner_step_fp32.tflite"
    else:
        from litert_torch.quantize import pt2e_quantizer as pq
        from litert_torch.quantize import quant_config as qc
        qcfg = pq.get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False)
        quantizer = pq.PT2EQuantizer()
        if a.all_ops:
            quantizer.set_global(qcfg)
        else:
            # Matrix ops only (the phones' QNN path quantizes the same set):
            # a global config also annotates the scalar embed-scale multiply,
            # whose folded 0-d quantized constant the litert converter rejects.
            for op in (torch.ops.aten.linear.default, torch.ops.aten.matmul.default,
                       torch.ops.aten.bmm.default, torch.ops.aten.mm.default):
                quantizer.set_operator_type(op, qcfg)
        try:  # torch < 2.8
            from torch.export import export_for_training as _export
        except ImportError:  # torch >= 2.8 exports the training IR by default
            from torch.export import export as _export
        try:  # torch >= 2.8 moved PT2E into torchao
            from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e
        except ImportError:
            from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
        exported = _export(step, (z_H, z_L, inputs)).module()
        prepared = prepare_pt2e(exported, quantizer)
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for _ in range(a.n_cal):
                prepared(torch.tensor(rng.normal(0, 1, (1, seq, 512)), dtype=torch.float32),
                         torch.tensor(rng.normal(0, 1, (1, seq, 512)), dtype=torch.float32),
                         torch.tensor(rng.integers(0, cfg["vocab_size"], (1, cfg["seq_len"])), dtype=torch.int32))
        # fold_quantize=True stores the weights as int8 constants. With it off
        # (the first export) the flatbuffer kept every weight in float32 next
        # to a runtime quantize op, so the "INT8" graphs were 25-35 MB and the
        # float attention scores dominated the arena.
        converted = convert_pt2e(prepared, fold_quantize=True)
        edge = litert_torch.convert(converted, (z_H, z_L, inputs),
                                    quant_config=qc.QuantConfig(pt2e_quantizer=quantizer))
        path = out / "inner_step_int8.tflite"
    edge.export(str(path))
    print(f"[{a.task}] wrote {path} ({path.stat().st_size/1e6:.2f} MB)", flush=True)

    # Weight-dtype audit: every constant above 64 KB, by dtype. A real INT8
    # graph has no large float32 constants.
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path))
    by = {}
    for d in it.get_tensor_details():
        n = int(np.prod(d["shape"])) if len(d["shape"]) else 1
        nb = n * np.dtype(d["dtype"]).itemsize
        if nb >= 65536:
            by.setdefault(d["dtype"].__name__, []).append(nb)
    for k, v in sorted(by.items()):
        print(f"[{a.task}] tensors >=64KB {k}: {len(v)} tensors, {sum(v)/1e6:.1f} MB", flush=True)

    # Numerical check of the TFLite graph against torch.
    got = edge(z_H, z_L, inputs)
    got = got if torch.is_tensor(got) else torch.as_tensor(np.asarray(got))
    err = float((got.float() - ref).abs().max())
    print(f"[{a.task}] tflite max abs diff vs torch {err:.3e}", flush=True)

    # Vela: Ethos-U55 on a Cortex-M55 system, shared SRAM (Corstone-300 layout).
    vela = pathlib.Path(sys.executable).parent / "vela"
    cmd = [str(vela), str(path), "--accelerator-config", a.accel,
           "--system-config", "Ethos_U55_High_End_Embedded", "--memory-mode", "Shared_Sram",
           "--output-dir", str(out / "vela"), "--verbose-performance"]
    print("[vela]", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "vela_stdout.txt").write_text(r.stdout + "\n" + r.stderr)
    tail = "\n".join(r.stdout.splitlines()[-40:])
    print(tail)
    print(f"[vela] rc={r.returncode}; full log in {out/'vela_stdout.txt'}")


if __name__ == "__main__":
    main()
