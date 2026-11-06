#!/usr/bin/env python3
"""Calibrated TensorRT INT8 engine for a static batch-1 step graph (Jetson AGX Xavier, TensorRT 8.5).

Entropy calibration (IInt8EntropyCalibrator2, batch 1) on the real carries of make_calib_carries.py, the October
datacenter recipe (e1_energy.make_calibrator). Device buffers are torch CUDA tensors (no pycuda on the board).
  --variant int8     INT8 + FP16 flags: TensorRT picks INT8 where it times faster, FP16 elsewhere (E1's "int8")
  --variant int8f32  INT8 flag only: INT8 where faster, FP32 elsewhere (E1's "int8f32")
The calibration cache (--cache) is written on the first build and read by later ones; it does not depend on the
variant. Writes the .plan, <plan>.layers.json (engine inspector, DETAILED) and <plan>.build.json (summary:
layer counts by input/output dtype, build seconds, calibration seconds per batch); TensorRT's log at INFO goes to
<plan>.build.log.

    python3 build_int8_engine.py --onnx assets/sudoku_mlp/stepc_H3_op13_fold.onnx \
        --calib assets/sudoku_mlp/calib_carries.npz --variant int8 \
        --cache engines/sudoku_mlp/stepc_H3_op13_fold_int8.calib.cache \
        --out engines/sudoku_mlp/stepc_H3_op13_fold_int8.plan
"""
import argparse
import json
import os
import re
import time

import numpy as np
import tensorrt as trt
import torch


class FileLogger(trt.ILogger):
    def __init__(self, path, verbose=False):
        trt.ILogger.__init__(self)
        self.f = open(path, "w")
        self.max = int(trt.ILogger.Severity.VERBOSE if verbose else trt.ILogger.Severity.INFO)

    def log(self, sev, msg):
        if int(sev) <= self.max:
            self.f.write("[%s] %s\n" % (str(sev).split(".")[-1], msg))
            self.f.flush()
        if int(sev) <= int(trt.ILogger.Severity.ERROR):
            print("TRT", sev, msg, flush=True)


class Calib(trt.IInt8EntropyCalibrator2):
    def __init__(self, data, cache_path):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.n = len(data["z_H"])
        self.dev = dict((k, torch.as_tensor(np.ascontiguousarray(v)).cuda()) for k, v in data.items())
        self.cache_path = cache_path
        self.i = 0
        self.cur = None
        self.t = []

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        self.t.append(time.time())
        if self.i >= self.n:
            return None
        self.cur = dict((k, self.dev[k][self.i:self.i + 1].contiguous()) for k in names)
        torch.cuda.synchronize()
        self.i += 1
        if self.i % 16 == 0:
            print("calib batch %d/%d  %.1f s/batch" % (self.i, self.n, (self.t[-1] - self.t[0]) / max(1, self.i - 1)),
                  flush=True)
        return [int(self.cur[k].data_ptr()) for k in names]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_path, "wb") as f:
            f.write(bytes(cache))


PAT = re.compile(r"(Int8|INT8|FP16|Half|FP32|Float|Int32|INT32|Int64|INT64|Bool|BOOL)")
NORM = {"INT8": "Int8", "Half": "FP16", "Float": "FP32", "INT32": "Int32", "INT64": "Int64", "BOOL": "Bool"}


def dts(lst):
    got = set()
    for o in lst:
        m = PAT.search(str(o.get("Format/Datatype", "")) if isinstance(o, dict) else str(o))
        got.add(NORM.get(m.group(1), m.group(1)) if m else "?")
    return ",".join(sorted(got)) or "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib", required=True, help="npz from make_calib_carries.py")
    ap.add_argument("--variant", choices=["int8", "int8f32"], default="int8")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workspace_gb", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true", help="TensorRT VERBOSE log (large)")
    a = ap.parse_args()

    t0 = time.time()
    logger = FileLogger(a.out + ".build.log", a.verbose)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(net, logger)
    with open(a.onnx, "rb") as f:
        ok = parser.parse(f.read())
    assert ok, [str(parser.get_error(i)) for i in range(parser.num_errors)]
    names = [net.get_input(i).name for i in range(net.num_inputs)]
    raw = np.load(a.calib)
    data = dict((k, raw[k]) for k in names)
    for i in range(net.num_inputs):
        t = net.get_input(i)
        assert tuple(t.shape) == (1,) + data[t.name].shape[1:], (t.name, t.shape, data[t.name].shape)
    cache_existed = os.path.exists(a.cache)

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(a.workspace_gb * (1 << 30)))
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    cfg.set_flag(trt.BuilderFlag.INT8)
    if a.variant == "int8":
        cfg.set_flag(trt.BuilderFlag.FP16)
    calib = Calib(data, a.cache)
    cfg.int8_calibrator = calib
    ser = builder.build_serialized_network(net, cfg)
    assert ser is not None, "build failed; see %s.build.log" % a.out
    with open(a.out, "wb") as f:
        f.write(ser)
    build_s = time.time() - t0

    rt = trt.Runtime(logger)
    eng = rt.deserialize_cuda_engine(ser)
    info = json.loads(eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON))
    with open(a.out + ".layers.json", "w") as f:
        json.dump(info, f, indent=1)
    by_io, by_type_out = {}, {}
    for L in info.get("Layers", []):
        if isinstance(L, str):
            continue
        key = "in:%s|out:%s" % (dts(L.get("Inputs", [])), dts(L.get("Outputs", [])))
        by_io[key] = by_io.get(key, 0) + 1
        lt = "%s|out:%s" % (L.get("LayerType", "?"), dts(L.get("Outputs", [])))
        by_type_out[lt] = by_type_out.get(lt, 0) + 1
    ct = calib.t
    summ = dict(onnx=a.onnx, calib=a.calib, n_calib=calib.n, variant=a.variant, cache=a.cache,
                cache_reused=cache_existed, tensorrt=trt.__version__, build_s=round(build_s, 1),
                calib_s_per_batch=round((ct[-1] - ct[0]) / max(1, len(ct) - 1), 2) if len(ct) > 1 else None,
                plan_bytes=len(ser), n_layers=sum(by_io.values()), by_io_dtype=by_io, by_type_out=by_type_out)
    with open(a.out + ".build.json", "w") as f:
        json.dump(summ, f, indent=1)
    print("%s: %s build %.0f s, %d layers, int8-out layers %d, plan %.1f MB" % (
        a.out, a.variant, build_s, summ["n_layers"], sum(v for k, v in by_io.items() if "out:" in k and "Int8" in
                                                         k.split("out:")[1]), len(ser) / 1e6), flush=True)


if __name__ == "__main__":
    main()
