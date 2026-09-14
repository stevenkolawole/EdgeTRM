#!/usr/bin/env python3
"""x86 server-CPU row: one recursive inner step per task through onnxruntime (CPUExecutionProvider)
at FP32, static INT8 (per-channel weights, uint8 activations, QOperator format) and 4-bit weight-only
(MatMulNBits, block 32), at several thread counts. Latency is the median of timed runs after warm-up.

    ~/venvs/kvcache/bin/python cpu_bench.py --out ~/breadth/results/cpu_onnxruntime.json
"""
import argparse
import json
import os
import platform
import shutil
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

ART = Path.home() / "aihub_artifacts"
TASKS = {"maze": ART / "maze" / "inner_step.onnx", "sudoku_attn": ART / "sudoku_attn" / "inner_step.onnx",
         "sudoku_mlp": ART / "sudoku_mlp" / "inner_step.onnx", "arc": ART / "trm_inner_step.onnx"}


def feeds_for(sess):
    feeds = {}
    for inp in sess.get_inputs():
        shape = [d if isinstance(d, int) else 1 for d in inp.shape]
        if "int" in inp.type:   # token ids: the maze vocabulary has 6 symbols, sudoku 11, ARC 12 -> stay below 6
            feeds[inp.name] = np.random.randint(0, 6, size=shape).astype(np.int32 if "int32" in inp.type else np.int64)
        else:
            feeds[inp.name] = np.random.randn(*shape).astype(np.float32)
    return feeds


def time_model(path, threads, warm=10, runs=50):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    feeds = feeds_for(sess)
    for _ in range(warm):
        sess.run(None, feeds)
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, feeds)
        ts.append((time.perf_counter() - t) * 1000)
    ts.sort()
    return {"ms_median": statistics.median(ts), "ms_p10": ts[len(ts) // 10], "ms_p90": ts[9 * len(ts) // 10], "runs": runs}


class RandomReader:
    """calibration reader for static quantization (latency does not depend on the values)."""
    def __init__(self, sess, n=8):
        self.sess, self.n, self.i = sess, n, 0

    def get_next(self):
        if self.i >= self.n:
            return None
        self.i += 1
        return feeds_for(self.sess)


def make_int8(src, dst):
    from onnxruntime.quantization import QuantFormat, QuantType, quantize_static, quant_pre_process
    pre = dst.with_suffix(".pre.onnx")
    quant_pre_process(str(src), str(pre))
    sess = ort.InferenceSession(str(pre), providers=["CPUExecutionProvider"])
    quantize_static(str(pre), str(dst), RandomReader(sess), quant_format=QuantFormat.QOperator,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    pre.unlink(missing_ok=True)


def make_int4(src, dst):
    import onnx
    try:                                   # onnxruntime >= 1.19 (nbits) / older (4bits)
        from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer as Q4
        kw = {"block_size": 32, "is_symmetric": True, "bits": 4}
    except ImportError:
        from onnxruntime.quantization.matmul_4bits_quantizer import MatMul4BitsQuantizer as Q4
        kw = {"block_size": 32, "is_symmetric": True}
    model = onnx.load(str(src))
    q = Q4(model, **kw)
    q.process()
    q.model.save_model_to_file(str(dst), True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="1,4,16")
    ap.add_argument("--tasks", default="sudoku_mlp,sudoku_attn,maze,arc")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    threads = [int(t) for t in a.threads.split(",")]
    cpu = ""
    try:
        cpu = [l for l in open("/proc/cpuinfo") if "model name" in l][0].split(":", 1)[1].strip()
    except Exception:
        pass
    meta = {"cpu": cpu, "cores": os.cpu_count(), "onnxruntime": ort.__version__, "platform": platform.platform(),
            "provider": "CPUExecutionProvider", "note": "other jobs share this host; see intra-op thread column"}
    rows = []
    tmp = Path(tempfile.mkdtemp(prefix="cpu_bench_"))
    for task in a.tasks.split(","):
        src = TASKS[task]
        variants = {"fp32": src}
        try:
            make_int8(src, tmp / f"{task}_int8.onnx")
            variants["int8_static"] = tmp / f"{task}_int8.onnx"
        except Exception as exc:  # noqa: BLE001
            print(f"[{task}] int8 static failed: {str(exc)[:160]}", flush=True)
        try:
            make_int4(src, tmp / f"{task}_int4.onnx")
            variants["int4_matmulnbits"] = tmp / f"{task}_int4.onnx"
        except Exception as exc:  # noqa: BLE001
            print(f"[{task}] int4 MatMulNBits failed: {str(exc)[:160]}", flush=True)
        for prec, path in variants.items():
            size_mb = os.path.getsize(path) / 2 ** 20
            for th in threads:
                try:
                    r = time_model(path, th)
                except Exception as exc:  # noqa: BLE001
                    print(f"[{task}] {prec} threads={th} failed: {str(exc)[:160]}", flush=True)
                    continue
                rows.append({"task": task, "precision": prec, "threads": th, "file_mb": size_mb, **r})
                print(f"  {task:12s} {prec:16s} threads={th:2d}  {r['ms_median']:8.2f} ms  (p10 {r['ms_p10']:.2f} p90 {r['ms_p90']:.2f})  {size_mb:.1f} MB",
                      flush=True)
                json.dump({"meta": meta, "rows": rows}, open(a.out, "w"), indent=1)
    shutil.rmtree(tmp, ignore_errors=True)
    print("wrote", a.out, len(rows))


if __name__ == "__main__":
    main()
