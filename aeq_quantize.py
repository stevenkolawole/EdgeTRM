#!/usr/bin/env python3
"""MCU-tier graph, take two: export the FP32 inner step to TFLite, quantize it
post-training with the AI Edge Quantizer (static INT8 weights and
activations, per-channel), audit the weight dtypes, then run Vela and the
TFLite Micro arena measurement on the result.

The first export (vela_export.py, PT2E with fold_quantize=False) kept every
weight in float32 beside a runtime quantize op: 25-35 MB "INT8" graphs whose
float attention scores set the arena. This path quantizes the finished graph.

    ~/venvs/mcu/bin/python aeq_quantize.py --task sudoku_mlp
"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_tasks as T  # noqa: E402
from vela_export import Step  # noqa: E402

OUT_ROOT = pathlib.Path.home() / "mcu"
TFLM_PY = pathlib.Path.home() / "venvs" / "tflm313" / "bin" / "python"


def audit(path, tag):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=str(path))
    by = {}
    for d in it.get_tensor_details():
        n = int(np.prod(d["shape"])) if len(d["shape"]) else 1
        nb = n * np.dtype(d["dtype"]).itemsize
        if nb >= 65536:
            by.setdefault(d["dtype"].__name__, []).append(nb)
    for k, v in sorted(by.items()):
        print(f"[{tag}] tensors >=64KB {k}: {len(v)} tensors, {sum(v)/1e6:.1f} MB", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(T.TASKS))
    ap.add_argument("--n_cal", type=int, default=8)
    ap.add_argument("--accel", default="ethos-u55-256")
    ap.add_argument("--wbits", type=int, default=8, choices=(4, 8))
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

    import litert_torch
    fp32_path = out / "inner_step_fp32.tflite"
    edge = litert_torch.convert(step, (z_H, z_L, inputs))
    edge.export(str(fp32_path))
    print(f"[{a.task}] fp32 tflite {fp32_path.stat().st_size/1e6:.2f} MB", flush=True)
    audit(fp32_path, a.task + " fp32")

    # Post-training static INT8 with the AI Edge Quantizer.
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import quantizer, recipe
    it = Interpreter(model_path=str(fp32_path))
    sig = it.get_signature_list()
    key = list(sig)[0]
    names = sig[key]["inputs"]
    print(f"[{a.task}] signature {key} inputs {names}", flush=True)
    rng = np.random.default_rng(0)
    cal = []
    for _ in range(a.n_cal):
        arrs = [rng.normal(0, 1, (1, seq, 512)).astype(np.float32),
                rng.normal(0, 1, (1, seq, 512)).astype(np.float32),
                rng.integers(0, cfg["vocab_size"], (1, cfg["seq_len"])).astype(np.int32)]
        # match by name order as exported (args_0.., or z_H/z_L/inputs)
        cal.append({n: v for n, v in zip(sorted(names), arrs)} if all(n.startswith("args_") for n in names)
                   else {n: v for n, v in zip(names, arrs)})
    qt = quantizer.Quantizer(str(fp32_path))
    if a.wbits == 8:
        qt.load_quantization_recipe(recipe.static_wi8_ai8())
        int8_path = out / "inner_step_int8.tflite"
        if int8_path.exists() and not (out / "inner_step_int8_pt2e_unfolded.tflite").exists():
            int8_path.rename(out / "inner_step_int8_pt2e_unfolded.tflite")  # keep the first export for the record
    else:
        # 4-bit per-channel weights, 8-bit activations: the calibrated-INT4
        # deployment configuration as a static graph (TFLM has int4 kernels).
        from ai_edge_quantizer import qtyping
        qt.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                             activation_num_bits=8, weight_num_bits=a.wbits,
                             weight_granularity=qtyping.QuantGranularity.CHANNELWISE)
        int8_path = out / f"inner_step_w{a.wbits}a8.tflite"
    calib = qt.calibrate({key: cal})
    res = qt.quantize(calib)
    if hasattr(res, "export_model"):
        res.export_model(str(int8_path), overwrite=True)
    else:
        int8_path.write_bytes(res.quantized_model)
    print(f"[{a.task}] int8 tflite {int8_path.stat().st_size/1e6:.2f} MB", flush=True)
    audit(int8_path, a.task + " int8")

    # Numerical check against torch (random calibration: expect a loose match).
    it8 = Interpreter(model_path=str(int8_path))
    it8.allocate_tensors()
    dets = it8.get_input_details()
    feed = {"z_H": z_H.numpy(), "z_L": z_L.numpy(), "inputs": inputs.numpy()}
    order = [z_H.numpy(), z_L.numpy(), inputs.numpy()]
    for i, d in enumerate(sorted(dets, key=lambda d: d["name"])):
        arr = order[i]
        if d["dtype"] == np.int8:
            s, zp = d["quantization"]
            arr = np.clip(np.round(arr / s + zp), -128, 127).astype(np.int8)
        it8.set_tensor(d["index"], arr.astype(d["dtype"]) if d["dtype"] != np.int8 else arr)
    it8.invoke()
    od = it8.get_output_details()[0]
    got = it8.get_tensor(od["index"]).astype(np.float32)
    if od["dtype"] == np.int8:
        s, zp = od["quantization"]
        got = (got - zp) * s
    print(f"[{a.task}] int8 tflite max abs logit diff vs torch {float(np.abs(got - ref.numpy()).max()):.3e}", flush=True)

    vela = pathlib.Path(sys.executable).parent / "vela"
    cmd = [str(vela), str(int8_path), "--accelerator-config", a.accel,
           "--system-config", "Ethos_U55_High_End_Embedded", "--memory-mode", "Shared_Sram",
           "--output-dir", str(out / "vela"), "--verbose-performance"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    (out / "vela_stdout.txt").write_text(r.stdout + "\n" + r.stderr)
    for line in r.stdout.splitlines():
        if any(k in line for k in ("Total SRAM", "Total Off-chip", "CPU operators", "NPU operators", "Inferences", "Batch Inference time")):
            print("[vela]", line.strip(), flush=True)
    print(f"[vela] rc={r.returncode}", flush=True)

    r = subprocess.run([str(TFLM_PY), str(pathlib.Path(__file__).resolve().parent / "tflm_arena.py"), str(int8_path)],
                       capture_output=True, text=True)
    for line in (r.stdout + r.stderr).splitlines():
        if "Arena" in line or "arena" in line or "tflm" in line:
            print(line.strip(), flush=True)


if __name__ == "__main__":
    main()
