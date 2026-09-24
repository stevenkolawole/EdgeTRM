#!/usr/bin/env python3
"""Does Vela put INT4 weights on the Ethos-U? The TFLite route cannot: Vela 5.2's
SUPPORTED_OPS.md lists "Weight tensors must be 8-bit precision" for
FULLY_CONNECTED / CONV_2D on U55, U65 and U85, and regor crashes (SIGFPE)
instead of falling back to the CPU when it meets an INT4 FULLY_CONNECTED. The
TOSA route is documented: "PRO-INT extensions: EXT-INT4: 4-bit integer weights",
TOSA CONV2D DataTypes (IFM, Weights, OFM) include (i8_t,i4_t,i32_t).

This builds the TRM step's SwiGLU shapes as a TOSA 1.0 graph by hand (97 tokens,
1x1 CONV2D 512->1536, RESCALE, 1x1 CONV2D 1536->512, RESCALE), once with INT8
and once with INT4 weights, and compiles both with Vela for U55-256, U65-256
and U85-256. Serializer: tosa-tools 2026.2.1 (the one ExecuTorch's Arm backend
pins), run from a scratch venv:

    ~/mcu_h1/venv_tosa/bin/python mcu_tosa_int4.py --out ~/mcu_h1/tosa_int4
    ~/mcu_h1/venv_tosa/bin/python mcu_tosa_int4.py --out ~/mcu_h1/tosa_int4 --real_w ~/mcu_h1/tosa_int4/real_w.npz
    (then vela is run from ~/venvs/mcu by the same script via --vela)
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np


def build(path, wdtype_name, seq=97, d=512, inter=1536, seed=0, real=None):
    """real=None: random weights, x -> CONV2D(512->1536) -> RESCALE -> CONV2D(1536->512) -> RESCALE.
    real=(w_up [3072,512], w_down [512,1536]) integer weights from our quantized graphs: two chains,
    x[97,512] -> CONV2D gate_up (512->3072) -> RESCALE and h[97,1536] -> CONV2D down (1536->512) -> RESCALE,
    i.e. one block's full SwiGLU weight set (the SiLU-gate multiply between them is not in the graph)."""
    import tosa_serializer as ts
    rng = np.random.default_rng(seed)
    fb = ts.TosaSerializer("", targetMajor=1, targetMinor=0, targetPatch=0, targetDraft=False)
    WD = getattr(ts.DType, wdtype_name)
    lim = 7 if wdtype_name == "INT4" else 127

    def conv(inp, cin, cout, name, w=None):
        if w is None:
            w = rng.integers(-lim, lim + 1, (cout, cin)).astype(np.int8)
        w = np.asarray(w, np.int8).reshape(cout, 1, 1, cin)
        assert np.abs(w).max() <= lim
        wt = fb.addConst([cout, 1, 1, cin], WD, w, name=name + "_w")
        b = fb.addConst([cout], ts.DType.INT32, np.zeros(cout, np.int32), name=name + "_b")
        izp = fb.addConst([1], ts.DType.INT8, [0], name=name + "_izp")
        wzp = fb.addConst([1], WD, [0], name=name + "_wzp")
        acc = fb.addIntermediate([1, 1, seq, cout], ts.DType.INT32)
        attr = ts.TosaSerializerAttribute()
        attr.Conv2dAttribute(pad=[0, 0, 0, 0], stride=[1, 1], dilation=[1, 1], local_bound=False,
                             acc_type=ts.DType.INT32)
        fb.addOperator(ts.Op.CONV2D, [inp, wt.name, b.name, izp.name, wzp.name], [acc.name], attr)
        return acc

    def rescale(acc, c, name, final=False):
        mult = fb.addConst([1], ts.DType.INT32, [1 << 30], name=name + "_mul")
        shift = fb.addConst([1], ts.DType.INT8, [40], name=name + "_shift")
        izp = fb.addConst([1], ts.DType.INT32, [0], name=name + "_rizp")
        ozp = fb.addConst([1], ts.DType.INT8, [0], name=name + "_rozp")
        o = fb.addOutput([1, 1, seq, c], ts.DType.INT8) if final else fb.addIntermediate([1, 1, seq, c], ts.DType.INT8)
        attr = ts.TosaSerializerAttribute()
        attr.RescaleAttribute(scale32=True, rounding_mode=ts.RoundingMode.SINGLE_ROUND, per_channel=False,
                              input_unsigned=False, output_unsigned=False)
        fb.addOperator(ts.Op.RESCALE, [acc.name, mult.name, shift.name, izp.name, ozp.name], [o.name], attr)
        return o

    if real is None:
        fb.addInputTensor(ts.TosaSerializerTensor("x", [1, 1, seq, d], ts.DType.INT8))
        h = rescale(conv("x", d, inter, "up"), inter, "r1")
        rescale(conv(h.name, inter, d, "down"), d, "r2", final=True)
    else:
        w_up, w_down = real
        fb.addInputTensor(ts.TosaSerializerTensor("x", [1, 1, seq, w_up.shape[1]], ts.DType.INT8))
        fb.addInputTensor(ts.TosaSerializerTensor("h", [1, 1, seq, w_down.shape[1]], ts.DType.INT8))
        rescale(conv("x", w_up.shape[1], w_up.shape[0], "up", w_up), w_up.shape[0], "r1", final=True)
        rescale(conv("h", w_down.shape[1], w_down.shape[0], "down", w_down), w_down.shape[0], "r2", final=True)
    pathlib.Path(path).write_bytes(fb.serialize())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vela", default=str(pathlib.Path.home() / "venvs/mcu/bin/vela"))
    ap.add_argument("--real_w", help="npz with int8_up/int8_down/int4_up/int4_down integer weights")
    a = ap.parse_args()
    out = pathlib.Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    configs = [("ethos-u55-256", "Ethos_U55_High_End_Embedded", "Shared_Sram"),
               ("ethos-u65-256", "Ethos_U65_High_End", "Dedicated_Sram"),
               ("ethos-u85-256", "Ethos_U85_SYS_Flash_High", "Shared_Sram")]
    res = []
    real = np.load(a.real_w) if a.real_w else None
    for wd in ("INT8", "INT4"):
        tag = ("real_" if real is not None else "") + wd.lower()
        g = out / f"ffn_{tag}.tosa"
        rec = dict(weights=wd, source="real (sudoku_attn H=1 block, AEQ per-channel)" if real is not None
                   else "random", graph=str(g))
        try:
            build(g, wd, real=None if real is None else (real[f"{wd.lower()}_up"], real[f"{wd.lower()}_down"]))
            rec["serialized_bytes"] = g.stat().st_size
        except Exception as e:  # noqa: BLE001
            rec["build_error"] = f"{type(e).__name__}: {e}"
            res.append(rec)
            continue
        for accel, sysc, mem in configs:
            od = out / f"vela_{tag}_{accel}"
            cmd = [a.vela, str(g), "--accelerator-config", accel, "--system-config", sysc, "--memory-mode", mem,
                   "--output-dir", str(od)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            vr = dict(accelerator=accel, system_config=sysc, memory_mode=mem, rc=r.returncode)
            if r.returncode != 0:
                vr["error"] = (r.stderr or r.stdout)[-1200:]
            else:
                import csv
                for p in od.glob("*_summary_*.csv"):
                    row = list(csv.DictReader(open(p)))[-1]
                    for k in ("sram_memory_used", "off_chip_flash_memory_used", "on_chip_flash_memory_used",
                              "dram_memory_used", "total_original_weights", "total_npu_encoded_weights",
                              "cycles_total", "cycles_npu", "inference_time", "nn_macs",
                              "off_chip_flash_weight_read_bytes", "dram_weight_read_bytes",
                              "sram_weight_read_bytes"):
                        if k in row:
                            vr[k] = float(row[k])
                vr["stdout_tail"] = r.stdout[-600:]
            rec.setdefault("vela", []).append(vr)
            print(wd, accel, "rc", r.returncode, {k: vr.get(k) for k in ("total_original_weights",
                  "total_npu_encoded_weights", "cycles_total", "error")}, flush=True)
        res.append(rec)
    (out / ("tosa_int4_real.json" if real is not None else "tosa_int4.json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
