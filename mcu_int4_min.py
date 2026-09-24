#!/usr/bin/env python3
"""Minimal reproduction for Vela 5.2 on TFLite INT4 weights: one Linear(512, 1536)
on a [1, 97, 512] input (the TRM SwiGLU up-projection shape), quantized with the
AI Edge Quantizer as W8A8, W4A8 per-channel and W4A8 per-tensor, then compiled
with Vela (default regor core, and --debug-force-legacy-core) for U55-256.

    ~/venvs/mcu/bin/python mcu_int4_min.py --out ~/mcu_h1/vela_int4/min
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    import litert_torch
    from ai_edge_quantizer import qtyping, quantizer, recipe
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(512, 1536, bias=False)).eval()
    x = torch.randn(1, 97, 512)
    fp32 = out / "lin_fp32.tflite"
    litert_torch.convert(m, (x,)).export(str(fp32))
    rng = np.random.default_rng(0)
    cal = [{"args_0": rng.normal(0, 1, (1, 97, 512)).astype(np.float32)} for _ in range(8)]
    res = []
    for tag, gran in (("w8a8", None), ("w4a8_c", "CHANNELWISE"), ("w4a8_t", "TENSORWISE")):
        qt = quantizer.Quantizer(str(fp32))
        qt.load_quantization_recipe(recipe.static_wi8_ai8())
        if gran:
            qt.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                                 activation_num_bits=8, weight_num_bits=4,
                                 weight_granularity=getattr(qtyping.QuantGranularity, gran))
        g = out / f"lin_{tag}.tflite"
        qt.quantize(qt.calibrate({"serving_default": cal})).export_model(str(g), overwrite=True)
        for core in ("regor", "legacy"):
            cmd = [str(pathlib.Path(sys.executable).parent / "vela"), str(g), "--accelerator-config", "ethos-u55-256",
                   "--system-config", "Ethos_U55_High_End_Embedded", "--memory-mode", "Shared_Sram",
                   "--output-dir", str(out / f"vela_{tag}_{core}"), "--show-cpu-operations"]
            if core == "legacy":
                cmd.append("--debug-force-legacy-core")
            r = subprocess.run(cmd, capture_output=True, text=True)
            lines = [l.strip() for l in (r.stdout + r.stderr).splitlines()
                     if any(k in l for k in ("CPU operators", "NPU operators", "Weight", "weight", "None value",
                                             "Placing on CPU", "Error", "error", "cycles"))][:12]
            rec = dict(graph=tag, core=core, rc=r.returncode, lines=lines)
            res.append(rec)
            print(json.dumps(rec), flush=True)
    (out / "min.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
