#!/usr/bin/env python3
"""M4: the TFLite / TFLite-Micro backend of the release-check study.

The MCU helper (`mcu_h1.py`) exported a logits-only step, which cannot be chained. This script exports
the carry step (z_H, z_L, inputs, puzzle_embedding) -> (z_H', z_L', logits, q_halt) at H=1 with
litert_torch, quantizes it with the AI Edge Quantizer recipes of `mcu_h1.py` (same calibration carries:
~/aihub_artifacts/h1/<task>/calib_H1.npz minus the last 4), and runs the host loop on the TFLite
interpreter to get a correctness label per (task, variant, runtime):
  runtime auto  = default op resolver (XNNPACK delegate where it applies: what a phone CPU runs)
  runtime ref   = builtin reference kernels (the kernels TFLite Micro shares)

    ~/venvs/mcu/bin/python m4_tflite.py --stage export --tasks sudoku_mlp,sudoku_attn,maze
    ~/venvs/mcu/bin/python m4_tflite.py --stage quant  --tasks ... --variants int8,w8a16,...
    ~/venvs/mcu/bin/python m4_tflite.py --stage label  --tasks ... --variants ... --runtime auto --n 100
Outputs: ~/m4/tflite/<task>/stepc_<variant>.tflite, label_<variant>_<runtime>_<n>.json (+ .npz).
"""
import argparse
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for extra in (HERE, pathlib.Path.home() / "x3", pathlib.Path.home() / "m4"):
    if (extra / "aihub_h1.py").exists() and str(extra) not in sys.path:
        sys.path.append(str(extra))
import aihub_h1 as A  # noqa: E402

OUT = pathlib.Path.home() / "m4" / "tflite"
THREADS = int(os.environ.get("OMP_NUM_THREADS", "8"))
NSTEPS = {"sudoku_mlp": 16, "sudoku_attn": 16, "maze": 8}
VARIANTS = {
    "int8": dict(act=8, fc=None),
    "w4a8_c": dict(act=8, fc=(4, "CHANNELWISE")),
    "w4a8_t": dict(act=8, fc=(4, "TENSORWISE")),
    "w8a16": dict(act=16, fc=None),
    "w4a16_c": dict(act=16, fc=(4, "CHANNELWISE")),
    "w4a8_allops": dict(act=8, fc="allops"),
    "w8a8_fc": dict(act=8, fc="fc_only"),        # only FULLY_CONNECTED quantized (weights + its I/O)
}


def odir(task):
    d = OUT / task
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_export(task):
    import torch
    fp32 = odir(task) / "stepc_fp32.tflite"
    if fp32.exists():
        return
    Step, _ = A.wrappers()
    m, cfg, sd = A.build(task, 1)
    w = Step(m.inner, True).eval()
    c = np.load(A.ART / task / "calib_H1.npz")
    args = (torch.from_numpy(c["z_H"][0]), torch.from_numpy(c["z_L"][0]), torch.from_numpy(c["inputs"][0]),
            torch.from_numpy(c["puzzle_embedding"][0]))
    with torch.no_grad():
        ref = w(*args)
    import litert_torch
    t0 = time.time()
    edge = litert_torch.convert(w, args)
    edge.export(str(fp32))
    got = edge(*args)
    errs = [float(np.abs(np.asarray(g) - r.numpy()).max()) for g, r in zip(got, ref)]
    print(f"[{task}] stepc fp32 tflite {fp32.stat().st_size / 1e6:.1f} MB ({time.time() - t0:.0f}s), "
          f"litert-vs-torch max abs {errs}", flush=True)


def stage_quant(task, variant):
    from ai_edge_quantizer import qtyping, quantizer, recipe
    from ai_edge_litert.interpreter import Interpreter
    fp32 = odir(task) / "stepc_fp32.tflite"
    graph = odir(task) / f"stepc_{variant}.tflite"
    rec_path = odir(task) / f"quant_{variant}.json"
    if graph.exists():
        return
    rec = dict(task=task, variant=variant, ok=False)
    try:
        it = Interpreter(model_path=str(fp32))
        key = list(it.get_signature_list())[0]
        names = it.get_signature_list()[key]["inputs"]
        c = np.load(A.ART / task / "calib_H1.npz")
        n = c["z_H"].shape[0] - 4
        src = [c["z_H"], c["z_L"], c["inputs"], c["puzzle_embedding"]]
        cal = [{nm: src[k][i].astype(np.float32 if k != 2 else np.int32) for k, nm in enumerate(sorted(names))}
               for i in range(n)]
        v = VARIANTS[variant]
        G = qtyping.QuantGranularity
        qt = quantizer.Quantizer(str(fp32))
        if v["fc"] == "allops":
            qt.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                                 activation_num_bits=8, weight_num_bits=4, weight_granularity=G.CHANNELWISE)
        elif v["fc"] == "fc_only":
            qt.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                                 activation_num_bits=8, weight_num_bits=8, weight_granularity=G.CHANNELWISE)
        else:
            qt.load_quantization_recipe(recipe.static_wi8_ai8() if v["act"] == 8 else recipe.static_wi8_ai16())
            if v["fc"] is not None:
                bits, gran = v["fc"]
                qt.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                                     activation_num_bits=v["act"], weight_num_bits=bits,
                                     weight_granularity=getattr(G, gran))
        t0 = time.time()
        calib = qt.calibrate({key: cal}, num_threads=THREADS)
        res = qt.quantize(calib)
        res.export_model(str(graph), overwrite=True)
        rec.update(ok=True, seconds=round(time.time() - t0, 1), n_calibration=n, bytes=graph.stat().st_size)
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2500:])
    rec_path.write_text(json.dumps(rec, indent=1))
    print(f"[{task}] quant {variant}: ok={rec['ok']} {rec.get('error', '')[:200]}", flush=True)


def stage_label(task, variant, runtime, n, start=0):
    """Host loop on the TFLite interpreter over evaluation puzzles start..start+n (labels used)."""
    import release_check as R
    out = odir(task) / f"label_{variant}_{runtime}_{start}_{n}.json"
    if out.exists():
        return
    graph = odir(task) / f"stepc_{variant}.tflite"
    if not graph.exists():
        print(f"[{task}] {variant}: no graph", flush=True)
        return
    inp, lab = A.load_data(task, list(range(start, start + n)))
    m, cfg, sd = A.build(task, 1)
    seq = cfg["seq_len"] + A.PLEN
    zH0 = np.broadcast_to(m.inner.H_init.detach().numpy(), (1, seq, 512)).astype(np.float32)
    zL0 = np.broadcast_to(m.inner.L_init.detach().numpy(), (1, seq, 512)).astype(np.float32)
    emb = sd["inner.puzzle_emb.weights"][:1].float().numpy()
    del m
    chk = dict(inputs=inp.astype(np.int32), emb=np.repeat(emb[None], n, 0), zH0=zH0, zL0=zL0)
    T = NSTEPS[task]
    t0 = time.time()
    rec = dict(task=task, variant=variant, runtime=runtime, start=start, n=n, steps=T, ok=False)
    try:
        step = R.TfliteStep(graph, THREADS, resolver=runtime)
        r = R.rollout(step, chk, T)
        corr = (r["preds"] == lab[None]).all(-1)
        rec.update(ok=True, exact=float(corr[-1].mean()), cell=float((r["preds"][-1] == lab).mean()),
                   exact_per_step=[float(x) for x in corr.mean(1)], correct=[bool(x) for x in corr[-1]],
                   distinct_logit_argmax=int(len(np.unique(r["preds"][-1]))),
                   secs=round(time.time() - t0, 1))
        np.savez_compressed(out.with_suffix(".npz"), preds=r["preds"], q=r["q"], lab=lab)
    except Exception as e:  # noqa: BLE001
        rec.update(error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc()[-2500:])
    out.write_text(json.dumps(rec, indent=1))
    print(f"[{task}] label {variant} {runtime} n={n}: exact {rec.get('exact')} cell {rec.get('cell')} "
          f"({rec.get('secs')}s) {rec.get('error', '')[:200]}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, choices=["export", "quant", "label"])
    p.add_argument("--tasks", default="sudoku_mlp,sudoku_attn")
    p.add_argument("--variants", default="int8,w8a16,w4a8_c,w4a8_t,w4a16_c,w4a8_allops,w8a8_fc")
    p.add_argument("--runtime", default="auto")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--start", type=int, default=0)
    a = p.parse_args()
    for task in a.tasks.split(","):
        if a.stage == "export":
            stage_export(task)
        for v in a.variants.split(","):
            if a.stage == "quant":
                stage_quant(task, v)
            elif a.stage == "label":
                stage_label(task, v if v else "fp32", a.runtime, a.n, a.start)


if __name__ == "__main__":
    main()
