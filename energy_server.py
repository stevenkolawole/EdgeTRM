#!/usr/bin/env python3
"""Measured energy per recursion step on a server GPU (NVIDIA L40S on Babel) from the board's own
energy counter (NVML nvmlDeviceGetTotalEnergyConsumption, millijoules since driver load), with
idle subtraction; and, where the node exposes it to users, CPU package energy from RAPL.

This is not an edge device. It is the one real energy sensor we can reach without buying or renting
hardware, and it gives the float reference point (per step, per answer, per solved instance) that the
edge numbers will be compared with once a Jetson / Pixel / MCU measurement exists (X3.md decisions).

Protocol per (task, H, precision, batch): 5 repetitions of [idle 10 s][active 20 s of back-to-back
steps][idle 10 s]; energy from the counter difference over the active window (first 2 s dropped),
idle power from the bracketing windows; E_total = counter delta / steps, E_incr = E_total -
P_idle x t_step. Batch 1 is the edge-like latency mode; batch 256 (Sudoku) / 32 (Maze) the throughput
mode, reported per puzzle.

    python energy_server.py --tasks sudoku_mlp,sudoku_attn,maze_attn --out ~/breadth_results/x3/energy_l40s.json
"""
import argparse
import glob
import json
import os
import statistics
import time

import numpy as np
import torch

import breadth_sweep as B


class _CtypesNVML:
    """Minimal NVML through ctypes (pynvml may be absent from the env)."""

    def __init__(self):
        import ctypes
        self.c = ctypes
        self.lib = ctypes.CDLL("libnvidia-ml.so.1")
        assert self.lib.nvmlInit_v2() == 0

    def handle(self, i):
        h = self.c.c_void_p()
        assert self.lib.nvmlDeviceGetHandleByIndex_v2(i, self.c.byref(h)) == 0
        return h

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        v = self.c.c_ulonglong()
        assert self.lib.nvmlDeviceGetTotalEnergyConsumption(h, self.c.byref(v)) == 0
        return v.value

    def nvmlDeviceGetPowerUsage(self, h):
        v = self.c.c_uint()
        assert self.lib.nvmlDeviceGetPowerUsage(h, self.c.byref(v)) == 0
        return v.value

    def nvmlSystemGetDriverVersion(self):
        buf = self.c.create_string_buffer(96)
        self.lib.nvmlSystemGetDriverVersion(buf, 96)
        return buf.value.decode()


def nvml():
    # CUDA_VISIBLE_DEVICES under SLURM maps our GPU to CUDA index 0, but NVML enumerates every GPU
    # on the node; pick the NVML device whose PCI bus id matches torch's device 0.
    lib = _CtypesNVML()
    bus = torch.cuda.get_device_properties(0)
    want = getattr(bus, "pci_bus_id", None)
    idx = 0
    if want is not None:
        n = lib.c.c_uint()
        lib.lib.nvmlDeviceGetCount_v2(lib.c.byref(n))
        for i in range(n.value):
            h = lib.handle(i)
            class Pci(lib.c.Structure):
                _fields_ = [("busIdLegacy", lib.c.c_char * 16), ("domain", lib.c.c_uint), ("bus", lib.c.c_uint),
                            ("device", lib.c.c_uint), ("pciDeviceId", lib.c.c_uint),
                            ("pciSubSystemId", lib.c.c_uint), ("busId", lib.c.c_char * 32)]
            p = Pci()
            if lib.lib.nvmlDeviceGetPciInfo_v3(h, lib.c.byref(p)) == 0 and p.bus == want:
                idx = i
                break
    return lib, lib.handle(idx)


def rapl_paths():
    out = []
    for p in sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")):
        try:
            int(open(p).read())
            out.append(p)
        except OSError:
            pass
    return out


def energy_mj(pyn, h):
    return pyn.nvmlDeviceGetTotalEnergyConsumption(h)   # mJ


def power_w(pyn, h):
    return pyn.nvmlDeviceGetPowerUsage(h) / 1000.0


def idle_window(pyn, h, secs):
    e0, t0 = energy_mj(pyn, h), time.perf_counter()
    time.sleep(secs)
    return (energy_mj(pyn, h) - e0) / 1000.0 / (time.perf_counter() - t0)   # W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default="sudoku_mlp,sudoku_attn,maze_attn")
    ap.add_argument("--H", type=int, default=1)
    ap.add_argument("--dtypes", default="float32,float16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--idle_s", type=float, default=10)
    ap.add_argument("--active_s", type=float, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    pyn, h = nvml()
    res = {"gpu": torch.cuda.get_device_name(0), "driver": pyn.nvmlSystemGetDriverVersion(),
           "host": os.uname().nodename, "rapl_readable": rapl_paths(), "torch": torch.__version__,
           "protocol": vars(a), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": []}
    for task in a.tasks.split(","):
        spec = B.TRM_TASKS[task]
        inp, lab = B.load_qa(spec["data"], 256)
        for dt in a.dtypes.split(","):
            for batch in (1, spec["batch"] if task != "maze_attn" else 32):
                m, meta = B.build(task, a.H, 16, batch)
                dtype = getattr(torch, dt)
                inner = m.inner.to(B.DEV).to(dtype)
                inner.forward_dtype = dtype
                for mod in inner.modules():          # embeddings cast to forward_dtype fixed at build time
                    if hasattr(mod, "cast_to"):
                        mod.cast_to = dtype
                x = torch.as_tensor(inp[:batch], device=B.DEV)
                Carry = type(inner.empty_carry(1))
                zH = inner.H_init.to(dtype).expand(batch, x.shape[1] + 16, 512).contiguous()
                zL = inner.L_init.to(dtype).expand(batch, x.shape[1] + 16, 512).contiguous()
                bt = {"inputs": x, "puzzle_identifiers": torch.zeros(batch, dtype=torch.int32, device=B.DEV)}
                with torch.no_grad():
                    c = Carry(z_H=zH, z_L=zL)
                    for _ in range(5):
                        c, _, _ = inner(c, bt)
                    torch.cuda.synchronize()
                    reps = []
                    for r in range(a.reps):
                        p_idle_pre = idle_window(pyn, h, a.idle_s)
                        t0 = time.perf_counter()
                        n = 0
                        e_start = t_start = None
                        while True:
                            c, _, _ = inner(c, bt)
                            n += 1
                            if n % 8 == 0:
                                torch.cuda.synchronize()
                                now = time.perf_counter()
                                if e_start is None and now - t0 > 2.0:
                                    e_start, t_start, n_start = energy_mj(pyn, h), now, n
                                if now - t0 > a.active_s + 2.0:
                                    break
                        e_end, t_end = energy_mj(pyn, h), time.perf_counter()
                        steps = n - n_start
                        t_step = (t_end - t_start) / steps
                        e_step = (e_end - e_start) / steps                   # mJ per step (whole batch)
                        p_act = (e_end - e_start) / 1000.0 / (t_end - t_start)
                        p_idle_post = idle_window(pyn, h, a.idle_s)
                        p_idle = (p_idle_pre + p_idle_post) / 2
                        reps.append({"steps": steps, "t_step_ms": t_step * 1e3, "p_active_w": p_act,
                                     "p_idle_w": p_idle, "e_total_mj_per_step": e_step,
                                     "e_incr_mj_per_step": e_step - p_idle * t_step * 1e3,
                                     "e_total_mj_per_puzzle_step": e_step / batch,
                                     "e_incr_mj_per_puzzle_step": (e_step - p_idle * t_step * 1e3) / batch})
                        print(f"{task} {dt} b{batch} rep{r}: {t_step*1e3:.2f} ms/step P {p_act:.1f} W idle "
                              f"{p_idle:.1f} W -> {e_step/batch:.3f} mJ/puzzle-step total, "
                              f"{(e_step - p_idle*t_step*1e3)/batch:.3f} incr", flush=True)
                med = {k: statistics.median(r[k] for r in reps) for k in reps[0]}
                res["runs"].append({"task": task, "H": a.H, "L": meta["L_cycles"], "dtype": dt, "batch": batch,
                                    "reps": reps, "median": med})
                json.dump(res, open(a.out, "w"), indent=1)
                del m, inner
                torch.cuda.empty_cache()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
