#!/usr/bin/env python3
"""MCU tier at the recommended depth: the Ethos-U / TFLite Micro evidence for
the H=1 inner step (and an H=3 re-run through the same pipeline), with the
weight-granularity matrix the AI Hub study covers on phones.

Differences from the H=3 run (vela_export.py / aeq_quantize.py, Sep 10):
  * H_cycles override (aihub_tasks.TASKS is patched in-process, nothing else
    changes; the Step wrapper still folds the puzzle-embedding row in).
  * Calibration on REAL carries (~/aihub_artifacts/h1/<task>/calib_H<H>.npz,
    written by aihub_h1.py --stage calib), not unit-normal noise. The last
    N_HELDOUT samples are held out for the numerical check.
  * W4A8 is built as "static INT8 everywhere, INT4 weights on
    FULLY_CONNECTED". The H=3 recipe (add_static_config(ALL_SUPPORTED,
    act 8, weight 4)) silently left every non-FC op in float32: AEQ's
    recipe manager skips an ALL_SUPPORTED entry for any op whose policy lacks
    the config (static_wi4_ai8 covers FULLY_CONNECTED and CONV_2D only), so
    that graph carried 170 QUANTIZE/DEQUANTIZE pairs around float
    MUL/ADD/RSQRT/LOGISTIC. It is kept here as variant w4a8_allops.

Variants (weights / activations; granularity applies to the weights):
  int8        static W8A8, per-channel weights               (recipe.static_wi8_ai8)
  w4a8_c      W8A8 base, FULLY_CONNECTED weights INT4 per-channel
  w4a8_t      W8A8 base, FULLY_CONNECTED weights INT4 per-tensor
  w4a8_b32    W8A8 base, FULLY_CONNECTED weights INT4 block-wise 32   (policy check; forced retry)
  w4a8_b128   W8A8 base, FULLY_CONNECTED weights INT4 block-wise 128  (policy check; forced retry)
  w8a16       static W8A16 (16x8), per-channel weights       (recipe.static_wi8_ai16)
  w4a16_c     W8A16 base, FULLY_CONNECTED weights INT4 per-channel
  w4a8_allops the H=3 aeq_quantize --wbits 4 recipe, for the Vela root cause

For every variant: quantizer, TFLite interpreter (vs torch on real carries),
Vela (Ethos-U55-256, U55 High-End Embedded, Shared_Sram: the Corstone-300 FVP
configuration) and TFLite Micro (arena, and one invoke compared with the
TFLite interpreter) acceptance, with the error text when one refuses.

    ~/venvs/mcu/bin/python mcu_h1.py --stage graphs --tasks sudoku_mlp,sudoku_attn,maze --H 1
    ~/venvs/mcu/bin/python mcu_h1.py --stage sweep  --tasks ... --H 1      (13 Vela configs)
    ~/venvs/mcu/bin/python mcu_h1.py --stage fvpprep --tasks sudoku_mlp,sudoku_attn --H 1
    ~/venvs/mcu/bin/python mcu_h1.py --stage collect --H 1,3

Outputs under ~/mcu_h1/<task>/H<H>/. Every stage is resumable (a finished
record is never recomputed without --force).
"""
import argparse
import csv
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

HOME = pathlib.Path.home()
HERE = pathlib.Path(__file__).resolve().parent
OUT_ROOT = HOME / "mcu_h1"
CALIB = HOME / "aihub_artifacts" / "h1"
TFLM_PY = HOME / "venvs" / "tflm313" / "bin" / "python"
TASKS3 = ["sudoku_mlp", "sudoku_attn", "maze"]
N_HELDOUT = 4
THREADS = int(os.environ.get("OMP_NUM_THREADS", "16"))

# The Corstone-300 FVP configuration (and the H=3 aeq_quantize default).
DEFAULT_VELA = ("ethos-u55-256", "Ethos_U55_High_End_Embedded", "Shared_Sram", None)
# The eval kit's U65 default: Y256, Ethos_U65_High_End, Dedicated_Sram, 384 KiB cache
# (scripts/cmake/configuration_options/npu_opts.cmake, DEFAULT_NPU_CACHE_SIZE 393216).
FVP_U65_VELA = ("ethos-u65-256", "Ethos_U65_High_End", "Dedicated_Sram", 393216)

VARIANTS = {
    "int8": dict(act=8, w=8, fc=None),
    "w4a8_c": dict(act=8, w=8, fc=(4, "CHANNELWISE")),
    "w4a8_t": dict(act=8, w=8, fc=(4, "TENSORWISE")),
    "w4a8_b32": dict(act=8, w=8, fc=(4, "BLOCKWISE_32")),
    "w4a8_b128": dict(act=8, w=8, fc=(4, "BLOCKWISE_128")),
    "w8a16": dict(act=16, w=8, fc=None),
    "w4a16_c": dict(act=16, w=8, fc=(4, "CHANNELWISE")),
    "w4a8_allops": dict(act=8, w=4, fc=None),
}


def tail(s, n=1500):
    s = s or ""
    return s[-n:]


def wjson(path, obj):
    path = pathlib.Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def rjson(path):
    path = pathlib.Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def outdir(task, H):
    return OUT_ROOT / task / f"H{H}"


# ----------------------------------------------------------------------------- torch side
def build_step(task, H):
    sys.path.insert(0, str(HERE))
    import aihub_tasks as T
    from vela_export import Step
    T.TASKS[task] = dict(T.TASKS[task], H_cycles=H)
    m, cfg = T.build(task)
    assert m.inner.config.H_cycles == H, m.inner.config.H_cycles
    sd = T.load_sd(T.TASKS[task]["ckpt"])
    emb_row = sd["inner.puzzle_emb.weights"][:1].clone().float()
    return Step(m.inner, emb_row).eval(), cfg


def load_calib(task, H):
    d = np.load(CALIB / task / f"calib_H{H}.npz")
    return (d["z_H"].astype(np.float32), d["z_L"].astype(np.float32), d["inputs"].astype(np.int32))


def split(n):
    held = list(range(n - N_HELDOUT, n))
    return list(range(0, n - N_HELDOUT)), held, [0] + held


def stage_export(task, H, force=False):
    """fp32 TFLite export and the torch reference logits on the evaluation samples."""
    import torch
    torch.set_num_threads(THREADS)
    out = outdir(task, H)
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / "inner_step_fp32.tflite"
    ref = out / "ref.npz"
    if fp32.exists() and ref.exists() and not force:
        return
    step, cfg = build_step(task, H)
    zH, zL, inp = load_calib(task, H)
    cal_idx, held, eval_idx = split(len(zH))
    t0 = time.time()
    with torch.no_grad():
        logits = np.stack([step(torch.from_numpy(zH[i]), torch.from_numpy(zL[i]),
                                torch.from_numpy(inp[i])).numpy() for i in eval_idx])
    print(f"[{task} H{H}] torch ref on {len(eval_idx)} samples {time.time()-t0:.1f}s logits {logits.shape}", flush=True)
    np.savez(ref, eval_idx=np.array(eval_idx), heldout=np.array(held), z_H=zH[eval_idx], z_L=zL[eval_idx],
             inputs=inp[eval_idx], logits=logits)
    import litert_torch
    t0 = time.time()
    edge = litert_torch.convert(step, (torch.from_numpy(zH[0]), torch.from_numpy(zL[0]), torch.from_numpy(inp[0])))
    edge.export(str(fp32))
    print(f"[{task} H{H}] fp32 tflite {fp32.stat().st_size/1e6:.2f} MB ({time.time()-t0:.0f}s)", flush=True)


# ----------------------------------------------------------------------------- graph audit
def audit(path):
    """Op histogram (with the tensor types each op touches), constant bytes per
    tensor type, and the weight quantization layout, read from the flatbuffer."""
    from ai_edge_litert.tools import flatbuffer_utils as fu
    m = fu.read_model(str(path))
    sg = m.subgraphs[0]
    tname = {v: k for k, v in vars(fu.schema_fb.TensorType).items() if not k.startswith("_")}
    ops, float_ops = {}, {}
    for op in sg.operators:
        name = fu.opcode_to_name(m, op.opcodeIndex)
        types = sorted({tname.get(sg.tensors[i].type, str(sg.tensors[i].type))
                        for i in list(op.inputs) + list(op.outputs) if i >= 0})
        key = f"{name}[{','.join(types)}]"
        ops[key] = ops.get(key, 0) + 1
        if "FLOAT32" in types and name not in ("QUANTIZE", "DEQUANTIZE", "EMBEDDING_LOOKUP"):
            float_ops[name] = float_ops.get(name, 0) + 1
    const_bytes, wq = {}, {}
    for t in sg.tensors:
        buf = m.buffers[t.buffer] if t.buffer < len(m.buffers) else None
        nb = len(buf.data) if (buf is not None and buf.data is not None) else 0
        if nb == 0 and buf is not None and getattr(buf, "size", 0):
            nb = int(buf.size)
        if nb == 0:
            continue
        ty = tname.get(t.type, str(t.type))
        const_bytes[ty] = const_bytes.get(ty, 0) + nb
        q = t.quantization
        if q is not None and nb >= 4096:
            nsc = len(q.scale) if q.scale is not None else 0
            det = q.detailsType if q.details is not None else 0
            lay = ("blockwise" if det else "per-channel" if nsc > 1 else "per-tensor" if nsc == 1 else "none")
            k = f"{ty}:{lay}"
            wq[k] = wq.get(k, 0) + 1
    return dict(n_ops=len(sg.operators), ops=ops, float_compute_ops=float_ops,
                const_bytes_by_type=const_bytes, weight_quant_layout=wq,
                file_bytes=pathlib.Path(path).stat().st_size)


# ----------------------------------------------------------------------------- quantization (child)
def child_quant(task, H, variant, force_blockwise=False):
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import qtyping, quantizer, recipe
    out = outdir(task, H)
    vdir = out / "variants"
    vdir.mkdir(exist_ok=True)
    tag = variant + ("_forced" if force_blockwise else "")
    graph = out / f"inner_step_{tag}.tflite"
    rec = dict(task=task, H=H, variant=tag, stage="quantizer", ok=False)
    v = VARIANTS[variant]
    fp32 = out / "inner_step_fp32.tflite"
    try:
        it = Interpreter(model_path=str(fp32))
        key = list(it.get_signature_list())[0]
        names = it.get_signature_list()[key]["inputs"]
        assert names == ["args_0", "args_1", "args_2"], names
        zH, zL, inp = load_calib(task, H)
        cal_idx, _, _ = split(len(zH))
        cal = [{"args_0": zH[i], "args_1": zL[i], "args_2": inp[i]} for i in cal_idx]
        rec["n_calibration"] = len(cal)
        qt = quantizer.Quantizer(str(fp32))
        G = qtyping.QuantGranularity
        if variant == "w4a8_allops":
            qt.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                                 activation_num_bits=8, weight_num_bits=4, weight_granularity=G.CHANNELWISE)
        else:
            qt.load_quantization_recipe(recipe.static_wi8_ai8() if v["act"] == 8 else recipe.static_wi8_ai16())
            if v["fc"] is not None:
                bits, gran = v["fc"]
                if not force_blockwise:
                    qt.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                                         activation_num_bits=v["act"], weight_num_bits=bits,
                                         weight_granularity=getattr(G, gran))
                else:
                    # skip_checks bypasses the policy table (default_policy.py: static_wi4_ai8 allows
                    # CHANNELWISE and TENSORWISE only) to see what the rest of the stack does.
                    cfg = qtyping.OpQuantizationConfig(
                        activation_tensor_config=qtyping.TensorQuantizationConfig(
                            num_bits=v["act"], symmetric=v["act"] == 16),
                        weight_tensor_config=qtyping.TensorQuantizationConfig(
                            num_bits=bits, symmetric=True, granularity=getattr(G, gran)),
                        compute_precision=qtyping.ComputePrecision.INTEGER, explicit_dequantize=False,
                        skip_checks=True)
                    qt.update_quantization_recipe(regex=".*",
                                                  operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                                                  op_config=cfg)
        rec["recipe"] = json.loads(json.dumps(qt.get_quantization_recipe(), default=str))
        t0 = time.time()
        calib = qt.calibrate({key: cal}, num_threads=THREADS)
        res = qt.quantize(calib)
        res.export_model(str(graph), overwrite=True)
        rec["seconds"] = round(time.time() - t0, 1)
        rec["graph"] = str(graph)
        rec["audit"] = audit(graph)
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = tail(traceback.format_exc(), 3000)
    wjson(vdir / f"{tag}.quant.json", rec)
    print(f"[{task} H{H} {tag}] quantizer ok={rec['ok']} {rec.get('error','')[:300]}", flush=True)


# ----------------------------------------------------------------------------- TFLite interpreter check (child)
def _q(x, det):
    if det["dtype"] in (np.int8, np.int16):
        s, zp = det["quantization"]
        info = np.iinfo(det["dtype"])
        return np.clip(np.round(x / s + zp), info.min, info.max).astype(det["dtype"])
    return x.astype(det["dtype"])


def _dq(y, det):
    if det["dtype"] in (np.int8, np.int16):
        s, zp = det["quantization"]
        return (y.astype(np.float32) - zp) * s
    return y.astype(np.float32)


def child_interp(graph, refnpz, outjson, rawnpz):
    """Run the graph on the evaluation samples; compare with torch fp32 logits. Also
    saves, for sample eval_idx[1] (the first held-out one), the exact input arrays
    in model-input order and the raw output, for the TFLM comparison."""
    from ai_edge_litert import interpreter as I
    rec = dict(stage="interpreter", graph=str(graph), ok=False)
    try:
        d = np.load(refnpz)
        try:  # default op resolver (XNNPACK delegate applied)
            it = I.Interpreter(model_path=str(graph), num_threads=THREADS)
            it.allocate_tensors()
            rec["resolver"] = "AUTO (XNNPACK)"
        except Exception as e:  # noqa: BLE001
            # The XNNPACK delegate refuses some graphs (block-wise INT4 FULLY_CONNECTED with INT8
            # activations); the builtin kernels are the other thing a phone/desktop TFLite runs.
            rec["default_resolver_error"] = tail(f"{type(e).__name__}: {e}", 600)
            it = I.Interpreter(model_path=str(graph), num_threads=THREADS,
                               experimental_op_resolver_type=I.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
            it.allocate_tensors()
            rec["resolver"] = "BUILTIN_WITHOUT_DEFAULT_DELEGATES"
        ins = it.get_input_details()
        od = it.get_output_details()[0]
        rec["io"] = dict(inputs=[(x["name"], str(np.dtype(x["dtype"])), list(map(int, x["shape"])),
                                  list(map(float, x["quantization"]))) for x in ins],
                         output=(od["name"], str(np.dtype(od["dtype"])), list(map(int, od["shape"])),
                                 list(map(float, od["quantization"]))))
        per = []
        for k, idx in enumerate(d["eval_idx"]):
            src = {"args_0": d["z_H"][k], "args_1": d["z_L"][k], "args_2": d["inputs"][k]}
            feed = []
            for det in ins:
                m = re.search(r"args_(\d)", det["name"])
                x = _q(src[f"args_{m.group(1)}"], det)
                it.set_tensor(det["index"], x)
                feed.append(x)
            t0 = time.time()
            it.invoke()
            dt = time.time() - t0
            raw = it.get_tensor(od["index"]).copy()
            got = _dq(raw, od)
            ref = d["logits"][k]
            agree = float((got.argmax(-1) == ref.argmax(-1)).mean())
            per.append(dict(sample=int(idx), heldout=bool(idx in set(d["heldout"].tolist())),
                            max_abs=float(np.abs(got - ref).max()), mean_abs=float(np.abs(got - ref).mean()),
                            ref_absmax=float(np.abs(ref).max()), argmax_agree=agree, invoke_s=round(dt, 2)))
            if k == 1:
                s_, zp_ = (od["quantization"] if od["dtype"] in (np.int8, np.int16) else (1.0, 0))
                np.savez(rawnpz, **{f"in{j}": a for j, a in enumerate(feed)}, out_raw=raw, sample=int(idx),
                         torch_logits=ref, out_scale=float(s_), out_zp=int(zp_))
        held = [p for p in per if p["heldout"]]
        rec.update(per_sample=per,
                   heldout_max_abs=max(p["max_abs"] for p in held),
                   heldout_mean_abs=float(np.mean([p["mean_abs"] for p in held])),
                   heldout_argmax_agree=float(np.mean([p["argmax_agree"] for p in held])),
                   calib_sample_max_abs=per[0]["max_abs"], calib_sample_argmax_agree=per[0]["argmax_agree"],
                   ok=True)
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = tail(traceback.format_exc(), 3000)
    wjson(outjson, rec)


# ----------------------------------------------------------------------------- TFLM (child, tflm313 venv)
def child_tflm(graph, rawnpz, outjson, invoke, refnpz=None):
    from tflite_micro.python.tflite_micro import runtime
    rec = dict(stage="tflm", graph=str(graph), ok=False)
    try:
        it = runtime.Interpreter.from_file(str(graph), arena_size=1024 * 1024 * 1024,
                                           intrepreter_config=runtime.InterpreterConfig.kAllocationRecording)
        rec["loaded"] = True
        d = None
        if invoke and rawnpz and pathlib.Path(rawnpz).exists():
            d = dict(np.load(rawnpz))
        elif invoke and refnpz and pathlib.Path(refnpz).exists():
            # No TFLite-interpreter output to compare with (the interpreter refused the graph):
            # quantize the first held-out sample with TFLM's own input parameters, compare with torch.
            r = np.load(refnpz)
            src = [r["z_H"][1], r["z_L"][1], r["inputs"][1]]
            d = {"sample": int(r["eval_idx"][1]), "torch_logits": r["logits"][1]}
            for j in range(3):
                det = it.get_input_details(j)
                qp = det["quantization_parameters"]
                x = src[j]
                if det["dtype"] in (np.int8, np.int16):
                    info = np.iinfo(det["dtype"])
                    x = np.clip(np.round(x / float(qp["scales"][0]) + int(qp["zero_points"][0])),
                                info.min, info.max)
                d[f"in{j}"] = x.astype(det["dtype"])
            oq = it.get_output_details(0)["quantization_parameters"]
            d["out_scale"] = float(oq["scales"][0]) if len(oq["scales"]) else 1.0
            d["out_zp"] = int(oq["zero_points"][0]) if len(oq["zero_points"]) else 0
            rec["inputs_from"] = "ref.npz (TFLM input quantization)"
        if d is not None:
            j = 0
            while f"in{j}" in d:
                it.set_input(d[f"in{j}"], j)
                j += 1
            t0 = time.time()
            it.invoke()
            rec["invoke_s"] = round(time.time() - t0, 1)
            out = it.get_output(0)
            rec["invoke_ok"] = True
            if "out_raw" in d:
                ref = d["out_raw"]
                rec["vs_tflite_max_abs_raw"] = int(np.abs(out.astype(np.int64) - ref.astype(np.int64)).max())
                rec["vs_tflite_frac_equal"] = float((out == ref).mean())
            rec["sample"] = int(d["sample"])
            if "torch_logits" in d:
                got = (out.astype(np.float32) - float(d["out_zp"])) * float(d["out_scale"]) \
                    if out.dtype in (np.int8, np.int16) else out.astype(np.float32)
                tl = d["torch_logits"]
                rec["vs_torch_max_abs"] = float(np.abs(got - tl).max())
                rec["vs_torch_argmax_agree"] = float((got.argmax(-1) == tl.argmax(-1)).mean())
                rec["output_distinct_values"] = int(len(np.unique(out)))
        sys.stdout.flush()
        it.print_allocations()  # to stderr; the parent parses it
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = tail(traceback.format_exc(), 3000)
    wjson(outjson, rec)


def _tflm_child(graph, rawnpz, outjson, invoke, refnpz, timeout):
    cmd = [str(TFLM_PY), str(pathlib.Path(__file__).resolve()), "--child", "tflm", "--graph", str(graph),
           "--raw", str(rawnpz), "--outjson", str(outjson)] + (["--invoke"] if invoke else []) \
        + (["--ref", str(refnpz)] if refnpz else [])
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = "timeout", str(e.stdout or ""), str(e.stderr or "")
    rec = rjson(outjson) or dict(stage="tflm", graph=str(graph), ok=False)
    rec["rc"] = rc
    txt = (so or "") + "\n" + (se or "")
    for k in ("total", "head", "tail"):
        m = re.search(rf"Arena allocation {k} (\d+) bytes", txt)
        if m:
            rec[f"arena_{k}_bytes"] = int(m.group(1))
    if rc != 0:
        rec["ok"] = False
        rec.setdefault("error", "")
        rec["error"] = (rec["error"] + " | " if rec["error"] else "") + f"rc={rc}; " + tail(se or so, 1500)
    errs = [l for l in txt.splitlines() if re.search(r"(?i)(error|failed|didn't find|not supported|was not true)", l)]
    if errs:
        rec["tflm_messages"] = errs[:20]
    return rec


def run_tflm(graph, rawnpz, outjson, invoke=True, timeout=3600, refnpz=None):
    """Two TFLM children: allocation only (the arena, recorded exactly as tflm_arena.py does), then
    one invoke on a real sample (numerics). An invoke that aborts does not lose the arena."""
    outjson = pathlib.Path(outjson)
    rec = _tflm_child(graph, rawnpz, outjson, False, None, timeout)
    if invoke and rec.get("ok"):
        inv = _tflm_child(graph, rawnpz, outjson.with_suffix(".invoke.json"), True, refnpz, timeout)
        rec["invoke"] = {k: v for k, v in inv.items() if not k.startswith("arena_") and k != "traceback"}
        for k in ("invoke_ok", "invoke_s", "inputs_from", "sample", "vs_tflite_frac_equal", "vs_tflite_max_abs_raw",
                  "vs_torch_max_abs", "vs_torch_argmax_agree", "output_distinct_values"):
            if k in inv:
                rec[k] = inv[k]
        if not inv.get("ok"):
            rec["invoke_ok"] = False
            rec["invoke_error"] = inv.get("error")
    wjson(outjson, rec)
    return rec


# ----------------------------------------------------------------------------- Vela
def vela_bin():
    return HOME / "venvs" / "mcu" / "bin" / "vela"


def run_vela(graph, accel, sysconf, mem, odir, arena_cache=None, keep_tflite=False, extra=(), timeout=7200):
    odir = pathlib.Path(odir)
    odir.mkdir(parents=True, exist_ok=True)
    cmd = [str(vela_bin()), str(graph), "--accelerator-config", accel, "--system-config", sysconf,
           "--memory-mode", mem, "--output-dir", str(odir)] + list(extra)
    if arena_cache:
        cmd += ["--arena-cache-size", str(arena_cache)]
    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, so, se = r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        rc, so, se = "timeout", str(e.stdout or ""), str(e.stderr or "")
    (odir / "vela_stdout.txt").write_text(f"$ {' '.join(cmd)}\n{so}\n--- stderr\n{se}")
    rec = dict(rc=rc, seconds=round(time.time() - t0, 1), accelerator=accel, system_config=sysconf,
               memory_mode=mem, arena_cache_size_arg=arena_cache, cmd=" ".join(cmd))
    if rc != 0:
        rec["ok"] = False
        rec["error"] = tail(se or so, 1500) or (f"killed by signal {-rc}" if isinstance(rc, int) and rc < 0 else "")
        if isinstance(rc, int) and rc < 0:
            import signal as _s
            rec["signal"] = _s.Signals(-rc).name
        return rec
    rec["ok"] = True
    for p in odir.glob("*_summary_*.csv"):
        rows = list(csv.DictReader(open(p)))
        if rows:
            for k, v in rows[-1].items():
                try:
                    rec[k] = float(v)
                except (TypeError, ValueError):
                    rec[k] = v
    for line in so.splitlines():
        m = re.match(r"\s*(CPU|NPU) operators = (\d+) \(([\d.]+)%\)", line)
        if m:
            rec[f"{m.group(1).lower()}_operators"] = int(m.group(2))
    cpu_ops = [l.strip() for l in so.splitlines() if "cpu" in l.lower() and "operator" in l.lower()][:40]
    rec["cpu_op_lines"] = cpu_ops
    if "inference_time" in rec:
        rec["inference_ms"] = rec["inference_time"] * 1e3  # the CSV column is in seconds
    if not keep_tflite:
        for p in odir.glob("*_vela.tflite"):
            p.unlink()
    return rec


# ----------------------------------------------------------------------------- stage: graphs
def child_cmd(*args):
    return [sys.executable, str(pathlib.Path(__file__).resolve())] + [str(a) for a in args]


def stage_graphs(task, H, variants, force=False, tflm_invoke=True):
    out = outdir(task, H)
    stage_export(task, H, force=force)
    vdir = out / "variants"
    vdir.mkdir(exist_ok=True)
    # the fp32 graph's own numerical check (sanity: input mapping, converter)
    fj = vdir / "fp32.interp.json"
    if not fj.exists() or force:
        subprocess.run(child_cmd("--child", "interp", "--graph", out / "inner_step_fp32.tflite", "--ref",
                                 out / "ref.npz", "--outjson", fj, "--raw", vdir / "fp32.raw.npz"))
    f = rjson(fj)
    print(f"[{task} H{H}] fp32 tflite vs torch held-out max abs {f.get('heldout_max_abs')} "
          f"argmax {f.get('heldout_argmax_agree')} {f.get('error','')}", flush=True)
    todo = []
    for v in variants:
        todo.append((v, False))
        if VARIANTS[v]["fc"] and VARIANTS[v]["fc"][1].startswith("BLOCKWISE"):
            todo.append((v, True))
    for v, forced in todo:
        tag = v + ("_forced" if forced else "")
        qj = vdir / f"{tag}.quant.json"
        if forced:
            base = rjson(vdir / f"{v}.quant.json")
            if base and base.get("ok"):
                continue  # the policy accepted it; no forced retry needed
        if not qj.exists() or force:
            r = subprocess.run(child_cmd("--child", "quant", "--task", task, "--H", H, "--variant", v)
                               + (["--force_blockwise"] if forced else []), capture_output=True, text=True)
            if not qj.exists():
                wjson(qj, dict(task=task, H=H, variant=tag, stage="quantizer", ok=False,
                               error=f"child rc={r.returncode}: " + tail(r.stderr, 1500)))
            (vdir / f"{tag}.quant.log").write_text(r.stdout[-20000:] + "\n--- stderr\n" + r.stderr[-20000:])
        q = rjson(qj)
        if not q.get("ok"):
            continue
        graph = pathlib.Path(q["graph"])
        ij = vdir / f"{tag}.interp.json"
        if not ij.exists() or force:
            r = subprocess.run(child_cmd("--child", "interp", "--graph", graph, "--ref", out / "ref.npz",
                                         "--outjson", ij, "--raw", vdir / f"{tag}.raw.npz"),
                               capture_output=True, text=True)
            if not ij.exists():
                wjson(ij, dict(stage="interpreter", ok=False,
                               error=f"child rc={r.returncode} (signal {-r.returncode if r.returncode < 0 else ''}): "
                                     + tail(r.stderr, 1500)))
        i = rjson(ij)
        print(f"[{task} H{H} {tag}] interpreter ok={i.get('ok')} held-out max_abs={i.get('heldout_max_abs')} "
              f"argmax={i.get('heldout_argmax_agree')} {str(i.get('error',''))[:200]}", flush=True)
        vj = vdir / f"{tag}.vela.json"
        if not vj.exists() or force:
            accel, sysc, mem, ac = DEFAULT_VELA
            keep = task.startswith("sudoku")
            rec = run_vela(graph, accel, sysc, mem, out / "vela_default" / tag, ac, keep_tflite=keep)
            wjson(vj, rec)
        vr = rjson(vj)
        print(f"[{task} H{H} {tag}] vela rc={vr['rc']} sram={vr.get('sram_memory_used')} "
              f"flash={vr.get('off_chip_flash_memory_used')} cycles={vr.get('cycles_total')} "
              f"cpu_ops={vr.get('cpu_operators')} {str(vr.get('error',''))[:200]}", flush=True)
        tj = vdir / f"{tag}.tflm.json"
        if not tj.exists() or force:
            run_tflm(graph, vdir / f"{tag}.raw.npz", tj, invoke=tflm_invoke, refnpz=out / "ref.npz")
        t = rjson(tj)
        print(f"[{task} H{H} {tag}] tflm ok={t.get('ok')} arena={t.get('arena_total_bytes')} "
              f"invoke_eq={t.get('vs_tflite_frac_equal')} {str(t.get('error',''))[:200]}", flush=True)


# ----------------------------------------------------------------------------- stage: sweep
def stage_sweep(tasks, Hs, variants, workers=6):
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, str(HERE))
    from vela_breadth import CONFIGS
    jobs = []
    for H in Hs:
        for task in tasks:
            vdir = outdir(task, H) / "variants"
            for vj in sorted(vdir.glob("*.vela.json")):
                tag = vj.name[:-len(".vela.json")]
                if tag not in variants and tag.replace("_forced", "") not in variants:
                    continue
                if not rjson(vj).get("ok"):
                    continue
                q = rjson(vdir / f"{tag}.quant.json")
                for accel, sysconf, mem in CONFIGS:
                    rj = outdir(task, H) / "vela_sweep" / f"{tag}_{accel}_{sysconf}_{mem}.json"
                    if not rj.exists():
                        jobs.append((task, H, tag, q["graph"], accel, sysconf, mem, rj))
    print(f"[sweep] {len(jobs)} Vela runs to do", flush=True)

    def one(j):
        task, H, tag, graph, accel, sysconf, mem, rj = j
        rj.parent.mkdir(parents=True, exist_ok=True)
        rec = run_vela(graph, accel, sysconf, mem, rj.with_suffix(""))
        rec.update(task=task, H=H, variant=tag)
        wjson(rj, rec)
        print(f"[sweep] {task} H{H} {tag} {accel} {sysconf} {mem} rc={rec['rc']} ({rec['seconds']}s) "
              f"cycles={rec.get('cycles_total')} sram={rec.get('sram_memory_used')}", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(one, jobs))


# ----------------------------------------------------------------------------- stage: fvpprep
def stage_fvpprep(tasks, H, variants):
    """Vela-compile the graphs the Corstone-300 FVP runs (kept on disk)."""
    for task in tasks:
        out = outdir(task, H)
        for tag in variants:
            q = rjson(out / "variants" / f"{tag}.quant.json")
            if not q or not q.get("ok"):
                print(f"[fvpprep] {task} H{H} {tag}: no graph", flush=True)
                continue
            for name, (accel, sysc, mem, ac) in (("u55_256", DEFAULT_VELA), ("u65_256", FVP_U65_VELA)):
                odir = out / "fvp" / f"{tag}_{name}"
                rj = odir / "vela.json"
                if rj.exists():
                    continue
                rec = run_vela(q["graph"], accel, sysc, mem, odir, ac, keep_tflite=True)
                rec.update(task=task, H=H, variant=tag, fvp_target=name)
                wjson(rj, rec)
                print(f"[fvpprep] {task} H{H} {tag} {name} rc={rec['rc']} cycles={rec.get('cycles_total')} "
                      f"{str(rec.get('error',''))[:200]}", flush=True)


# ----------------------------------------------------------------------------- stage: collect
def stage_collect(tasks, Hs, dest):
    dest = pathlib.Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    matrix, sweep, tflm = [], [], []
    for H in Hs:
        for task in tasks:
            vdir = outdir(task, H) / "variants"
            if not vdir.exists():
                continue
            fp = rjson(vdir / "fp32.interp.json") or {}
            matrix.append(dict(task=task, H=H, variant="fp32", interpreter=fp))
            for qj in sorted(vdir.glob("*.quant.json")):
                tag = qj.name[:-len(".quant.json")]
                q = rjson(qj)
                q.pop("recipe", None)
                i = rjson(vdir / f"{tag}.interp.json")
                v = rjson(vdir / f"{tag}.vela.json")
                t = rjson(vdir / f"{tag}.tflm.json")
                for r in (i, t):
                    if r:
                        r.pop("traceback", None)
                matrix.append(dict(task=task, H=H, variant=tag, quantizer=q, interpreter=i, vela_default=v, tflm=t))
                if t:
                    tflm.append(dict(task=task, H=H, variant=tag, **{k: t.get(k) for k in (
                        "ok", "arena_total_bytes", "arena_head_bytes", "arena_tail_bytes", "invoke_ok",
                        "vs_tflite_frac_equal", "vs_tflite_max_abs_raw", "vs_torch_max_abs", "vs_torch_argmax_agree",
                        "output_distinct_values", "sample", "invoke_s", "invoke_error", "error", "tflm_messages")}))
            for rj in sorted((outdir(task, H) / "vela_sweep").glob("*.json")):
                sweep.append(rjson(rj))
    wjson(dest / "mcu_h1_matrix.json", matrix)
    wjson(dest / "vela_h1_sweep.json", sweep)
    wjson(dest / "tflm_h1.json", tflm)
    print(f"[collect] matrix {len(matrix)} sweep {len(sweep)} tflm {len(tflm)} -> {dest}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["graphs", "sweep", "fvpprep", "collect"])
    ap.add_argument("--child", choices=["quant", "interp", "tflm"])
    ap.add_argument("--tasks", default=",".join(TASKS3))
    ap.add_argument("--task")
    ap.add_argument("--H", default="1")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--variant")
    ap.add_argument("--force_blockwise", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_tflm_invoke", action="store_true")
    ap.add_argument("--graph")
    ap.add_argument("--ref")
    ap.add_argument("--raw")
    ap.add_argument("--outjson")
    ap.add_argument("--invoke", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--dest", default=str(OUT_ROOT / "collected"))
    a = ap.parse_args()
    if a.child == "quant":
        return child_quant(a.task, int(a.H), a.variant, a.force_blockwise)
    if a.child == "interp":
        return child_interp(a.graph, a.ref, a.outjson, a.raw)
    if a.child == "tflm":
        return child_tflm(a.graph, a.raw, a.outjson, a.invoke, a.ref)
    tasks = a.tasks.split(",")
    Hs = [int(h) for h in a.H.split(",")]
    variants = a.variants.split(",")
    if a.stage == "graphs":
        for H in Hs:
            for task in tasks:
                stage_graphs(task, H, variants, force=a.force, tflm_invoke=not a.no_tflm_invoke)
    elif a.stage == "sweep":
        stage_sweep(tasks, Hs, variants, a.workers)
    elif a.stage == "fvpprep":
        for H in Hs:
            stage_fvpprep(tasks, H, variants)
    elif a.stage == "collect":
        stage_collect(tasks, Hs, a.dest)


if __name__ == "__main__":
    main()
