#!/usr/bin/env python3
"""Launch and host round-trip share of a looped reasoner on one GPU (TRT_GATE_AND_LAUNCH.md, part G2).

Models: Sudoku-MLP and Maze-Hard, the H=1 carry step (the gate's and the phones' step graph), batch 1 and 32,
TensorRT FP16 engines (TensorRT as installed for E1, builder optimisation level 3, TF32 off). Per k in 1, 2, 4, 8, 16:
  unrolled  k steps exported as one ONNX graph (the step module called k times) -> one engine, one launch per loop
  device    k launches of the 1-step engine, the carry fed back on the device (ping-pong buffers, no host sync)
  host      k launches of the 1-step engine, after each step z_H, z_L, logits, q_halt copied to pinned host memory,
            the stream synchronised, and the carry copied back (the phones' host loop)
A loop ends with a stream synchronisation. Reported: ms per loop, median over 5 repeats of >= 200 loops after
warm-up; SM / memory clock, power and clock-event reasons read after every repeat (clocks not pinned).

    python loop_launch_overhead.py --task sudoku_mlp --out results.json
Env: TG_E1CODE (e1_energy), TG_WORK (exports + engines).
"""
import argparse
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time

import numpy as np

HOME = pathlib.Path.home()
E1CODE = pathlib.Path(os.environ.get("TG_E1CODE", HOME / "e1" / "code"))
WORK = pathlib.Path(os.environ.get("TG_WORK", HOME / "trt_gate" / "work_launch"))
sys.path.insert(0, str(E1CODE))
H = 1
KS = (1, 2, 4, 8, 16)
BATCHES = (1, 32)
INS = ["z_H", "z_L", "inputs", "puzzle_embedding"]
OUTS = ["z_H_out", "z_L_out", "logits", "q_halt"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def export(task, k):
    """k-step graph: the E1 step module (make_step) called k times; dynamic batch; outputs of the last step."""
    import torch
    import e1_energy as E
    path = WORK / task / f"unroll_H{H}_k{k}.onnx"
    if path.exists():
        return path
    inner, pe = E.build_inner(task, H)
    zh0, zl0 = E.init_carry(inner)
    step = E.make_step(inner)
    S = inner.config.seq_len + inner.puzzle_emb_len

    class Unroll(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.step = step

        def forward(self, z_H, z_L, inputs, puzzle_embedding):
            for _ in range(k):
                z_H, z_L, logits, qh, _qc = self.step(z_H, z_L, inputs, puzzle_embedding)
            return z_H, z_L, logits, qh

    x, _ = E.load_rows(E.TASKS[task]["data"], 0, 2)
    args = (zh0.expand(2, S, -1).contiguous(), zl0.expand(2, S, -1).contiguous(), torch.as_tensor(x),
            pe.expand(2, -1).contiguous())
    tmp = path.with_suffix(".tmp.onnx")
    with torch.no_grad():
        torch.onnx.export(Unroll().eval(), args, str(tmp), input_names=INS, output_names=OUTS, opset_version=17,
                          dynamo=False, dynamic_axes={n: {0: "B"} for n in INS + OUTS})
    tmp.rename(path)
    log(task, "exported k", k, f"{path.stat().st_size / 1e6:.1f} MB")
    return path


def build(task, k, B):
    import tensorrt as trt
    ep = WORK / task / f"unroll_H{H}_k{k}_b{B}_fp16.engine"
    meta = ep.with_suffix(".json")
    if ep.exists() and meta.exists():
        return ep, json.loads(meta.read_text())
    src = export(task, k)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, logger)
    assert parser.parse_from_file(str(src)), [parser.get_error(i) for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    cfg.clear_flag(trt.BuilderFlag.TF32)
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.builder_optimization_level = 3
    tc_path = WORK / task / "timing.cache"
    tcache = cfg.create_timing_cache(tc_path.read_bytes() if tc_path.exists() else b"")
    cfg.set_timing_cache(tcache, ignore_mismatch=False)
    prof = builder.create_optimization_profile()
    for i in range(net.num_inputs):
        inp = net.get_input(i)
        shp = (B,) + tuple(inp.shape)[1:]
        prof.set_shape(inp.name, shp, shp, shp)
    cfg.add_optimization_profile(prof)
    t0 = time.time()
    ser = builder.build_serialized_network(net, cfg)
    assert ser is not None, f"build failed {task} k{k} b{B}"
    ep.write_bytes(bytes(ser))
    tc_path.write_bytes(bytes(tcache.serialize()))
    m = dict(build_s=round(time.time() - t0, 1), size_MB=round(len(bytes(ser)) / 1e6, 2), n_layers=net.num_layers,
             trt=trt.__version__, onnx_MB=round(src.stat().st_size / 1e6, 1))
    meta.write_text(json.dumps(m))
    log(task, f"built k{k} b{B}", m)
    return ep, m


class Engine:
    def __init__(self, path, B, S, x, zh0, zl0, pe, stream):
        import torch
        import tensorrt as trt
        self.torch = torch
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(pathlib.Path(path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        tdt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64}
        shapes = {"z_H": (B, S, 512), "z_L": (B, S, 512), "inputs": (B, x.shape[1]), "puzzle_embedding": (B, 512)}
        for n in INS:
            if -1 in tuple(self.engine.get_tensor_shape(n)):
                self.ctx.set_input_shape(n, shapes[n])
        self.buf = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            self.buf[n] = torch.zeros(tuple(self.ctx.get_tensor_shape(n)), dtype=tdt[self.engine.get_tensor_dtype(n)],
                                      device="cuda")
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
        self.buf["inputs"].copy_(torch.as_tensor(x[:B]).to(self.buf["inputs"].dtype))
        self.buf["puzzle_embedding"].copy_(pe.expand(B, -1).to(self.buf["puzzle_embedding"].dtype))
        self.zh0 = zh0.expand(B, S, -1).to(self.buf["z_H"].dtype).cuda()
        self.zl0 = zl0.expand(B, S, -1).to(self.buf["z_L"].dtype).cuda()
        self.stream = stream
        self.sh = stream.cuda_stream
        self.reset()
        # device mode: ping-pong pair (input <-> output) for the carry
        self.pair = [(self.buf["z_H"], self.buf["z_L"]), (self.buf["z_H_out"], self.buf["z_L_out"])]
        # host mode: pinned host copies of every output
        self.host = {n: torch.empty_like(self.buf[n], device="cpu").pin_memory() for n in OUTS}

    def reset(self):
        self.buf["z_H"].copy_(self.zh0)
        self.buf["z_L"].copy_(self.zl0)
        self.torch.cuda.synchronize()

    def bind(self, src, dst):
        c = self.ctx
        c.set_tensor_address("z_H", src[0].data_ptr()); c.set_tensor_address("z_L", src[1].data_ptr())
        c.set_tensor_address("z_H_out", dst[0].data_ptr()); c.set_tensor_address("z_L_out", dst[1].data_ptr())

    def loop_unrolled(self, k):
        assert self.ctx.execute_async_v3(self.sh)
        self.stream.synchronize()

    def loop_device(self, k):
        for s in range(k):
            self.bind(self.pair[s % 2], self.pair[1 - s % 2])
            assert self.ctx.execute_async_v3(self.sh)
        self.stream.synchronize()

    def loop_host(self, k):
        torch = self.torch
        b, h = self.buf, self.host
        self.bind(self.pair[0], self.pair[1])
        with torch.cuda.stream(self.stream):
            for _ in range(k):
                assert self.ctx.execute_async_v3(self.sh)
                for n in OUTS:
                    h[n].copy_(b[n], non_blocking=True)
                self.stream.synchronize()                 # the host holds this step's outputs
                b["z_H"].copy_(h["z_H_out"], non_blocking=True)
                b["z_L"].copy_(h["z_L_out"], non_blocking=True)
        self.stream.synchronize()


def clocks(uuid):
    q = "uuid,clocks.sm,clocks.mem,clocks.max.sm,power.draw,temperature.gpu,clocks_event_reasons.active"
    try:
        o = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                           text=True, timeout=20).stdout.strip().splitlines()
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}
    rows = [[v.strip() for v in r.split(",")] for r in o]
    row = next((r for r in rows if uuid and uuid in r[0]), rows[0] if len(rows) == 1 else None)
    if row is None:
        return {"error": "no matching GPU row", "rows": len(rows)}
    return dict(sm_MHz=row[1], mem_MHz=row[2], max_sm_MHz=row[3], power_W=row[4], temp_C=row[5], reasons=row[6])


def timeit(fn, k, iters, reps, warm, uuid):
    for _ in range(warm):
        fn(k)
    per, clk = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn(k)
        per.append((time.perf_counter() - t0) / iters * 1e3)
        clk.append(clocks(uuid))
    return dict(ms_per_loop=statistics.median(per), reps_ms=[round(v, 5) for v in per], clocks=clk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sudoku_mlp", "maze"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm", type=int, default=30)
    a = ap.parse_args()
    import torch
    import e1_energy as E
    assert torch.cuda.is_available(), "CUDA not available: abort"
    torch.backends.cuda.matmul.allow_tf32 = False
    import tensorrt as trt
    (WORK / a.task).mkdir(parents=True, exist_ok=True)
    out = pathlib.Path(a.out)
    rec = json.loads(out.read_text()) if out.exists() else {}
    props = torch.cuda.get_device_properties(0)
    uuid = str(getattr(props, "uuid", ""))
    drv = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    rec["setup"] = dict(task=a.task, H=H, device=torch.cuda.get_device_name(0), torch_uuid=uuid, nvidia_smi=drv,
                        trt=trt.__version__, torch=torch.__version__, cuda=torch.version.cuda, host=os.uname().nodename,
                        job=os.environ.get("SLURM_JOB_ID"), iters=a.iters, reps=a.reps, warm=a.warm,
                        precision="TensorRT FP16 (TF32 off), builder optimisation level 3, I/O dtypes as exported")
    inner, pe = E.build_inner(a.task, H)
    zh0, zl0 = E.init_carry(inner)
    x, _ = E.load_rows(E.TASKS[a.task]["data"], 0, max(BATCHES))
    pe = pe.cuda()
    S = inner.config.seq_len + inner.puzzle_emb_len
    stream = torch.cuda.Stream()
    rec.setdefault("rows", {})
    for B in BATCHES:
        one_path, one_meta = build(a.task, 1, B)
        one = Engine(one_path, B, S, x, zh0, zl0, pe, stream)
        for k in KS:
            key = f"b{B}_k{k}"
            if key in rec["rows"] and "host" in rec["rows"][key]:
                continue
            ep, meta = build(a.task, k, B)
            unr = Engine(ep, B, S, x, zh0, zl0, pe, stream)
            row = dict(B=B, k=k, unrolled_engine=meta, one_step_engine=one_meta)
            # agreement: final carry of the unrolled engine vs k device-fed launches from the same initial carry
            unr.reset(); unr.loop_unrolled(k)
            one.reset(); one.loop_device(k)
            fin = one.pair[k % 2][0].float().flatten(1)
            row["unrolled_vs_device_cosH_min"] = float(torch.nn.functional.cosine_similarity(
                unr.buf["z_H_out"].float().flatten(1), fin, dim=-1).min())
            one.reset()
            row["unrolled"] = timeit(unr.loop_unrolled, k, a.iters, a.reps, a.warm, uuid)
            one.reset()
            row["device"] = timeit(one.loop_device, k, a.iters, a.reps, a.warm, uuid)
            one.reset()
            row["host"] = timeit(one.loop_host, k, a.iters, a.reps, a.warm, uuid)
            u, d, h = (row[m]["ms_per_loop"] for m in ("unrolled", "device", "host"))
            row["launch_share_of_device_loop"] = (d - u) / d
            row["roundtrip_share_of_host_loop"] = (h - d) / h
            row["overhead_share_of_host_loop"] = (h - u) / h
            rec["rows"][key] = row
            out.write_text(json.dumps(rec, indent=1))
            log(a.task, key, f"unrolled {u:.3f} device {d:.3f} host {h:.3f} ms/loop",
                f"cos {row['unrolled_vs_device_cosH_min']:.4f}", row["host"]["clocks"][-1].get("sm_MHz"))
            del unr
    log("done", out)


if __name__ == "__main__":
    main()
