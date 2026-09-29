#!/usr/bin/env python3
"""Board runner for the edge measurements (MLSys): one supervision-step graph (X3's `stepc` export: z_H, z_L,
inputs, puzzle_embedding -> z_H_out, z_L_out, logits, q_halt[, q_cont]) looped from the host with the carry fed
back, the frozen M5 exit rule applied per puzzle, an optional finishing graph for j steps after the exit, per-step
wall latency, and energy from INA3221 sysfs rails integrated over timed windows with an idle window subtracted.

Python 3.6 (JetPack 4.6) and 3.11 (Raspberry Pi OS); stdlib + numpy at import time. Backends:
  --backend ort   onnxruntime CPU EP on the .onnx (Pi 4; XNNPACK INT8 for QDQ graphs); --threads N
  --backend trt   TensorRT 8.x engine (.plan from build_engines.sh); buffers through pycuda if installed, else
                  ctypes on libcudart (no extra package). TensorRT >= 8.5 (Orin) is handled by the tensor API.

Exit rules (E1 / M5 R1-R3, chosen on fp32 and frozen): sudoku_mlp two consecutive halts (patience 2), sudoku_attn
native q_halt > 0, maze at least 2 steps then native, hrm_sudoku q_halt > q_continue. Cap NSUP = 16 steps.

Passes, in order, all on the same process (ENERGY_PROTOCOL.md: accuracy from the device run itself, idle window
in the same session at the same mode, windows >= 60 s, median of 5):
  check    fixed NSUP steps on every puzzle: exact per step vs labels, cell agreement per step, final z_H cosine
           and q_halt against the host reference (--ref host_fp32.npz from aihub_infer.stage_host), offline exit
           statistics from the q_halt trajectory, per-step latency (the BRINGUP A.7 gate)
  record   every puzzle once in the requested mode (exit / fixed depth, finishing): steps, correctness, per-step ms
  idle     --nidle windows of --win_s seconds with the process alive and no work (power sampler on)
  energy   --nwin windows of >= --win_s seconds looping the workload; energy per rail, per step, per solved
The JSON record (--out) keeps E1's energy.json layout (meta / idle / windows[cfg] / windows[cfg].first_pass)
so e1_report.py-style read-outs extend to boards, plus "check" and "record" blocks for the board-only detail.

    python3 edge_loop.py --backend ort --task sudoku_mlp --model assets/sudoku_mlp/stepc_H1.onnx \
        --init assets/sudoku_mlp/init.npz --puzzles assets/sudoku_mlp/puzzles_0_100.npz \
        --ref assets/sudoku_mlp/host_fp32.npz --board pi-a --precision fp32 --mode exit \
        --threads 4 --no_power --out results/pi-a_ort_fp32_exit.json
    python3 edge_loop.py --backend trt --task sudoku_mlp --model engines/sudoku_mlp/stepc_H1_fp16.plan \
        --finish engines/sudoku_mlp/stepc_H1_fp32.plan --finish_steps 2 --precision fp16 --board nano-a ...
"""
import argparse
import glob
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time

import numpy as np

NSUP = 16
RULES = {"sudoku_mlp": "patience2", "sudoku_attn": "native", "maze": "min2native", "hrm_sudoku": "hrm_own"}
INS = ["z_H", "z_L", "inputs", "puzzle_embedding"]
OUTS = ["z_H_out", "z_L_out", "logits", "q_halt"]
# Jetson Nano 4 GB dev kit (BRINGUP.md A.4): on-board INA3221 at I2C 6-0040, mW
NANO_POWER_DIR = "/sys/bus/i2c/drivers/ina3221x/6-0040/iio:device0"
NANO_POWER_FILES = "in_power0_input,in_power1_input,in_power2_input"
NANO_RAIL_NAMES = "POM_5V_IN,GPU,CPU"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a)
    sys.stdout.flush()


def dump(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=False)
    os.rename(tmp, path)


def now_utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ----------------------------------------------------------------------------- exit rules (as e1_energy.py)
def rule_fires(rule, s, qh, qc, prev_halt):
    """s = 1-based step just completed; qh/qc floats; prev_halt = did the head fire at step s-1."""
    halt = (qh > qc) if rule == "hrm_own" else (qh > 0)
    if rule == "native" or rule == "hrm_own":
        return halt, halt
    if rule == "patience2":
        return halt and prev_halt, halt
    if rule == "min2native":
        return halt and s >= 2, halt
    raise ValueError(rule)


def offline_exit(rule, qh, qc, nsup=NSUP):
    """qh, qc: (n, nsup) arrays -> exit step (1-based) per puzzle, capped at nsup."""
    n = qh.shape[0]
    out = np.full(n, nsup)
    for i in range(n):
        prev = False
        for s in range(1, nsup + 1):
            fire, prev = rule_fires(rule, s, float(qh[i, s - 1]), float(qc[i, s - 1]), prev)
            if fire:
                out[i] = s
                break
    return out


# ----------------------------------------------------------------------------- runners (common interface)
class Runner(object):
    """load(x, zh0, zl0, emb) sets the batch of tokens (B, seq) and resets the carry; step() runs one graph call;
    afterwards .qh / .qc are host floats (B,) when need_q, .logits() returns the host (B, seq, vocab) array,
    get_carry()/set_carry() move the (z_H, z_L) host arrays (finishing hand-over)."""
    backend = "?"
    version = "?"

    def close(self):
        pass


def np_dtype_of(s):
    return {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(int32)": np.int32,
            "tensor(int64)": np.int64, "tensor(bool)": np.bool_}[s]


class OrtRunner(Runner):
    backend = "ort"

    def __init__(self, path, threads=None, B=1, provider="CPUExecutionProvider"):
        import onnxruntime as ort
        self.version = ort.__version__
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        provs = [provider] if provider in ort.get_available_providers() else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, so, providers=provs)
        self.providers = self.sess.get_providers()
        self.in_dt = dict((i.name, np_dtype_of(i.type)) for i in self.sess.get_inputs())
        self.in_shape = dict((i.name, list(i.shape)) for i in self.sess.get_inputs())
        outs = [o.name for o in self.sess.get_outputs()]
        self.has_qc = "q_cont" in outs
        self.out_names = OUTS + (["q_cont"] if self.has_qc else [])
        self.B = B
        self.qh = self.qc = None
        self._logits = None

    def load(self, x, zh0, zl0, emb):
        B, S = self.B, zh0.shape[0]
        self.zH = np.ascontiguousarray(np.broadcast_to(zh0, (B, S, zh0.shape[-1])).astype(self.in_dt["z_H"]))
        self.zL = np.ascontiguousarray(np.broadcast_to(zl0, (B, S, zl0.shape[-1])).astype(self.in_dt["z_L"]))
        self.x = np.ascontiguousarray(x.astype(self.in_dt["inputs"]))
        self.emb = np.ascontiguousarray(np.broadcast_to(emb, (B, emb.shape[-1])).astype(self.in_dt["puzzle_embedding"]))

    def step(self, need_q=True):
        o = self.sess.run(self.out_names, {"z_H": self.zH, "z_L": self.zL, "inputs": self.x,
                                           "puzzle_embedding": self.emb})
        self.zH, self.zL, self._logits = o[0], o[1], o[2]
        self.qh = o[3].reshape(-1).astype(np.float64)
        self.qc = o[4].reshape(-1).astype(np.float64) if self.has_qc else np.zeros(self.B)

    def logits(self):
        return self._logits

    def get_carry(self):
        return self.zH.astype(np.float32), self.zL.astype(np.float32)

    def set_carry(self, zH, zL):
        self.zH = np.ascontiguousarray(zH.astype(self.in_dt["z_H"]))
        self.zL = np.ascontiguousarray(zL.astype(self.in_dt["z_L"]))


class CudaMem(object):
    """Device buffers, copies and one stream: pycuda if importable, else ctypes on libcudart (JetPack ships it)."""

    def __init__(self):
        self.keep = []
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            self.cuda = cuda
            self.mode = "pycuda"
            self.stream = cuda.Stream()
            self.handle = int(self.stream.handle)
            return
        except ImportError:
            pass
        import ctypes
        self.ct = ctypes
        lib = None
        for cand in [os.environ.get("CUDART", "")] + sorted(glob.glob("/usr/local/cuda*/lib64/libcudart.so*")) + \
                ["libcudart.so"]:
            if not cand:
                continue
            try:
                lib = ctypes.CDLL(cand)
                break
            except OSError:
                continue
        if lib is None:
            raise RuntimeError("neither pycuda nor libcudart found; pip3 install pycuda or set CUDART=/path/libcudart.so")
        self.lib = lib
        self.mode = "cudart:" + (lib._name or "?")
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMemcpyAsync.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p]
        lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        n = ctypes.c_int(0)
        self.check(lib.cudaGetDeviceCount(ctypes.byref(n)), "cudaGetDeviceCount")
        assert n.value > 0, "no CUDA device (cudaGetDeviceCount = 0): abort"
        s = ctypes.c_void_p()
        self.check(lib.cudaStreamCreate(ctypes.byref(s)), "cudaStreamCreate")
        self.stream = s
        self.handle = s.value or 0

    def check(self, err, what):
        if err != 0:
            raise RuntimeError("%s failed: CUDA error %d" % (what, err))

    def alloc(self, nbytes):
        if self.mode == "pycuda":
            a = self.cuda.mem_alloc(int(nbytes))
            self.keep.append(a)
            return int(a)
        p = self.ct.c_void_p()
        self.check(self.lib.cudaMalloc(self.ct.byref(p), int(nbytes)), "cudaMalloc")
        return p.value

    def htod(self, ptr, arr):
        arr = np.ascontiguousarray(arr)
        if self.mode == "pycuda":
            self.cuda.memcpy_htod_async(ptr, arr, self.stream)
        else:
            self.check(self.lib.cudaMemcpyAsync(ptr, arr.ctypes.data, arr.nbytes, 1, self.stream), "H2D")

    def dtoh(self, arr, ptr):
        if self.mode == "pycuda":
            self.cuda.memcpy_dtoh_async(arr, ptr, self.stream)
        else:
            self.check(self.lib.cudaMemcpyAsync(arr.ctypes.data, ptr, arr.nbytes, 2, self.stream), "D2H")

    def sync(self):
        if self.mode == "pycuda":
            self.stream.synchronize()
        else:
            self.check(self.lib.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")


class TrtRunner(Runner):
    """TensorRT engine of the step graph. The carry ping-pongs between two device buffer pairs (no copy per step);
    q_halt (and q_cont) come back to the host after every step when need_q; logits only on request."""
    backend = "trt"

    def __init__(self, plan, B=1, cuda=None):
        import tensorrt as trt
        self.trt = trt
        self.version = trt.__version__
        self.cu = cuda or CudaMem()
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(plan, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        assert self.engine is not None, "engine failed to deserialize: " + plan
        self.ctx = self.engine.create_execution_context()
        self.B = B
        self.v3 = hasattr(self.engine, "num_io_tensors") and hasattr(self.ctx, "execute_async_v3")
        names, shapes, dtypes, is_in = [], {}, {}, {}
        if self.v3:
            for i in range(self.engine.num_io_tensors):
                n = self.engine.get_tensor_name(i)
                names.append(n)
                is_in[n] = self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
                shp = tuple(self.engine.get_tensor_shape(n))
                if is_in[n] and -1 in shp:
                    self.ctx.set_input_shape(n, (B,) + shp[1:])
                dtypes[n] = trt.nptype(self.engine.get_tensor_dtype(n))
            for n in names:
                shapes[n] = tuple(self.ctx.get_tensor_shape(n))
        else:
            for i in range(self.engine.num_bindings):
                n = self.engine.get_binding_name(i)
                names.append(n)
                is_in[n] = self.engine.binding_is_input(i)
                shp = tuple(self.engine.get_binding_shape(i))
                if is_in[n] and -1 in shp:
                    self.ctx.set_binding_shape(i, (B,) + shp[1:])
                dtypes[n] = trt.nptype(self.engine.get_binding_dtype(i))
            for i, n in enumerate(names):
                shapes[n] = tuple(self.ctx.get_binding_shape(i))
        self.names, self.shapes, self.dtypes, self.is_in = names, shapes, dtypes, is_in
        self.has_qc = "q_cont" in names
        for n in INS + OUTS:
            assert n in names, "engine lacks tensor %s (have %s)" % (n, names)
        assert shapes["z_H"][0] == B, "engine batch %s != --batch %d" % (shapes["z_H"], B)
        # device buffers: one per tensor, plus a second pair for the carry outputs (ping-pong)
        self.dev = {}
        for n in names:
            self.dev[n] = self.cu.alloc(int(np.prod(shapes[n])) * np.dtype(dtypes[n]).itemsize)
        self.zH_dev = [self.dev["z_H"], self.dev["z_H_out"]]
        self.zL_dev = [self.dev["z_L"], self.dev["z_L_out"]]
        assert shapes["z_H"] == shapes["z_H_out"] and shapes["z_L"] == shapes["z_L_out"], (shapes,)
        self.cur = 0
        self.h_qh = np.zeros(shapes["q_halt"], dtypes["q_halt"])
        self.h_qc = np.zeros(shapes["q_cont"], dtypes["q_cont"]) if self.has_qc else None
        self.h_logits = np.zeros(shapes["logits"], dtypes["logits"])
        self.h_zH = np.zeros(shapes["z_H"], dtypes["z_H"])
        self.h_zL = np.zeros(shapes["z_L"], dtypes["z_L"])
        self.qh = self.qc = None
        self._logits_fresh = False
        self.mem_mode = self.cu.mode

    def _bind(self):
        c, o = self.cur, 1 - self.cur
        addr = dict(self.dev)
        addr["z_H"], addr["z_L"] = self.zH_dev[c], self.zL_dev[c]
        addr["z_H_out"], addr["z_L_out"] = self.zH_dev[o], self.zL_dev[o]
        return addr

    def load(self, x, zh0, zl0, emb):
        B = self.B
        S = self.shapes["z_H"][1]
        self.cur = 0
        self.cu.htod(self.zH_dev[0], np.broadcast_to(zh0, (B, S, zh0.shape[-1])).astype(self.dtypes["z_H"]))
        self.cu.htod(self.zL_dev[0], np.broadcast_to(zl0, (B, S, zl0.shape[-1])).astype(self.dtypes["z_L"]))
        self.cu.htod(self.dev["inputs"], x.astype(self.dtypes["inputs"]).reshape(self.shapes["inputs"]))
        self.cu.htod(self.dev["puzzle_embedding"],
                     np.broadcast_to(emb, (B, emb.shape[-1])).astype(self.dtypes["puzzle_embedding"]))
        self.cu.sync()

    def step(self, need_q=True):
        addr = self._bind()
        if self.v3:
            for n, p in addr.items():
                self.ctx.set_tensor_address(n, p)
            ok = self.ctx.execute_async_v3(self.cu.handle)
        else:
            ok = self.ctx.execute_async_v2([addr[n] for n in self.names], self.cu.handle)
        assert ok, "TensorRT execute failed"
        self.cur = 1 - self.cur
        self._logits_fresh = False
        if need_q:
            self.cu.dtoh(self.h_qh, self.dev["q_halt"])
            if self.has_qc:
                self.cu.dtoh(self.h_qc, self.dev["q_cont"])
            self.cu.sync()
            self.qh = self.h_qh.reshape(-1).astype(np.float64)
            self.qc = self.h_qc.reshape(-1).astype(np.float64) if self.has_qc else np.zeros(self.B)
        else:
            self.cu.sync()

    def logits(self):
        if not self._logits_fresh:
            self.cu.dtoh(self.h_logits, self.dev["logits"])
            self.cu.sync()
            self._logits_fresh = True
        return self.h_logits

    def get_carry(self):
        self.cu.dtoh(self.h_zH, self.zH_dev[self.cur])
        self.cu.dtoh(self.h_zL, self.zL_dev[self.cur])
        self.cu.sync()
        return self.h_zH.astype(np.float32), self.h_zL.astype(np.float32)

    def set_carry(self, zH, zL):
        self.cu.htod(self.zH_dev[self.cur], zH.astype(self.dtypes["z_H"]))
        self.cu.htod(self.zL_dev[self.cur], zL.astype(self.dtypes["z_L"]))
        self.cu.sync()


def make_runner(backend, path, B, threads, cuda=None):
    if backend == "ort":
        return OrtRunner(path, threads, B)
    if backend == "trt":
        return TrtRunner(path, B, cuda)
    raise ValueError(backend)


# ----------------------------------------------------------------------------- board state: power, thermal, clocks
class PowerSampler(threading.Thread):
    """Reads sysfs power files (mW) at ~hz in a background thread while .on is set; take() returns and clears the
    rows [(t_perf, mW_rail0, mW_rail1, ...)]. Files are re-read with pread(0) so the INA3221 driver refreshes."""

    def __init__(self, files, names, hz=50.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.files, self.names, self.hz = list(files), list(names), float(hz)
        self.fds = [os.open(f, os.O_RDONLY) for f in self.files]
        self.on = threading.Event()
        self.stop_ = threading.Event()
        self.lock = threading.Lock()
        self.rows = []
        self.errors = 0

    def read_once(self):
        vals = []
        for fd in self.fds:
            raw = os.pread(fd, 64, 0).strip()
            if not raw:
                raise ValueError("empty read")          # the whole sample is dropped (errors += 1)
            vals.append(float(raw))
        return vals

    def run(self):
        period = 1.0 / self.hz
        while not self.stop_.is_set():
            t = time.perf_counter()
            if self.on.is_set():
                try:
                    vals = self.read_once()
                    with self.lock:
                        self.rows.append([t] + vals)
                except Exception:
                    self.errors += 1
            dt = period - (time.perf_counter() - t)
            if dt > 0:
                time.sleep(dt)

    def take(self):
        with self.lock:
            r, self.rows = self.rows, []
        return r

    def close(self):
        self.stop_.set()
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass


def integrate(rows, names, t0, t1, total_rail=0):
    """Trapezoid integral of each rail's power (mW -> W) over [t0, t1] on the sample timestamps; the samples
    just outside the window are clamped to its edges. Returns E_J per rail, mean/max W, n_samples, hz."""
    if not rows:
        return {"n_samples": 0, "rails": {}, "E_J": None, "P_W": None}
    a = np.array(rows, dtype=np.float64)
    a = a[np.isfinite(a).all(1)]
    if len(a) == 0:
        return {"n_samples": 0, "rails": {}, "E_J": None, "P_W": None}
    t = a[:, 0]
    sel = (t >= t0) & (t <= t1)
    if sel.sum() < 2:
        return {"n_samples": int(sel.sum()), "rails": {}, "E_J": None, "P_W": None}
    a = a[sel]
    t = a[:, 0]
    # extend to the window edges with the boundary samples held constant
    t_ext = np.concatenate([[t0], t, [t1]])
    out = {"n_samples": int(len(t)), "hz": float(len(t) / max(t1 - t0, 1e-9)), "rails": {}}
    for k, name in enumerate(names):
        p = a[:, 1 + k] / 1e3
        p_ext = np.concatenate([[p[0]], p, [p[-1]]])
        e = float(np.sum(0.5 * (p_ext[1:] + p_ext[:-1]) * np.diff(t_ext)))
        out["rails"][name] = {"E_J": e, "P_W_mean": e / (t1 - t0), "P_W_max": float(p.max()), "P_W_min": float(p.min())}
    tot = names[total_rail]
    out["E_J"] = out["rails"][tot]["E_J"]
    out["P_W"] = out["rails"][tot]["P_W_mean"]
    out["total_rail"] = tot
    return out


def read_thermal():
    z = {}
    for d in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*")):
        try:
            with open(os.path.join(d, "type")) as f:
                typ = f.read().strip()
            with open(os.path.join(d, "temp")) as f:
                z[os.path.basename(d) + ":" + typ] = int(f.read().strip()) / 1000.0
        except (OSError, ValueError):
            pass
    vc = run_cmd(["vcgencmd", "measure_temp"])
    if vc:
        z["vcgencmd"] = vc
    thr = run_cmd(["vcgencmd", "get_throttled"])
    if thr:
        z["throttled"] = thr
    return z


def run_cmd(cmd, sudo=False, timeout=5):
    """stdout of a command, '' if absent or failing; tried as is, then with sudo -n (no prompt) if sudo."""
    for c in ([cmd, ["sudo", "-n"] + cmd] if sudo else [cmd]):
        try:
            p = subprocess.run(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            if p.returncode == 0:
                return p.stdout.decode(errors="replace").strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
    return ""


def read_clocks():
    c = {}
    for f in ("scaling_governor", "scaling_cur_freq", "scaling_max_freq"):
        vals = []
        for p in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/" + f)):
            try:
                with open(p) as fh:
                    vals.append(fh.read().strip())
            except OSError:
                pass
        if vals:
            c["cpu_" + f] = vals
    for p in sorted(glob.glob("/sys/class/devfreq/*/cur_freq")) + sorted(glob.glob("/sys/devices/*gpu*/devfreq/*/cur_freq")):
        try:
            with open(p) as fh:
                c[p] = fh.read().strip()
        except OSError:
            pass
    for name, cmd in (("nvpmodel", ["nvpmodel", "-q"]), ("jetson_clocks", ["jetson_clocks", "--show"])):
        out = run_cmd(cmd, sudo=True)
        if out:
            c[name] = out
    for name, cmd in (("vcgencmd_clock_arm", ["vcgencmd", "measure_clock", "arm"]),
                      ("vcgencmd_throttled", ["vcgencmd", "get_throttled"])):
        out = run_cmd(cmd)
        if out:
            c[name] = out
    for p in ("/etc/nv_tegra_release", "/etc/os-release"):
        if os.path.exists(p):
            with open(p, errors="replace") as fh:
                c[os.path.basename(p)] = fh.read().strip().splitlines()[:3]
    return c


def throttled_flag(clocks_or_thermal):
    """Pi: vcgencmd get_throttled bit 2 (0x4) = currently throttled, bit 1 (0x2) = ARM frequency capped now."""
    s = clocks_or_thermal.get("throttled") or clocks_or_thermal.get("vcgencmd_throttled") or ""
    try:
        v = int(s.split("=")[-1], 16)
        return bool(v & 0x6)
    except ValueError:
        return None


# ----------------------------------------------------------------------------- the host loop
class Loop(object):
    def __init__(self, a, runner, finisher, init, x, lab):
        self.a, self.runner, self.finisher = a, runner, finisher
        self.zh0, self.zl0, self.emb = init
        self.x, self.lab = x, lab
        self.n = len(x)
        self.rule = a.rule
        self.k = a.steps_per_call
        self.B = a.batch

    def run_puzzle(self, i, mode, depth, per_step=True):
        """One puzzle (batch 1) in `mode`: returns steps taken, correct (bool), per-call ms, q_halt per call,
        correctness before finishing, finishing ms."""
        r = self.runner
        r.load(self.x[i:i + 1], self.zh0, self.zl0, self.emb)
        ms, qhs, prev = [], [], False
        steps, cap = 0, depth
        need_q = mode == "exit"
        while steps < cap:
            t0 = time.perf_counter()
            r.step(need_q=need_q)
            ms.append((time.perf_counter() - t0) * 1e3)
            steps += self.k
            if need_q:
                qh, qc = float(r.qh[0]), float(r.qc[0])
                qhs.append(qh)
                fire, prev = rule_fires(self.rule, steps, qh, qc, prev)
                if fire:
                    break
        ok_pre = None
        fms = []
        if self.finisher is not None and self.a.finish_steps > 0:
            ok_pre = bool((r.logits()[0].argmax(-1) == self.lab[i]).all())
            zH, zL = r.get_carry()
            f = self.finisher
            f.load(self.x[i:i + 1], self.zh0, self.zl0, self.emb)
            f.set_carry(zH, zL)
            for _ in range(self.a.finish_steps):
                t0 = time.perf_counter()
                f.step(need_q=False)
                fms.append((time.perf_counter() - t0) * 1e3)
            steps += self.a.finish_steps * self.a.finish_steps_per_call
            ok = bool((f.logits()[0].argmax(-1) == self.lab[i]).all())
        else:
            ok = bool((r.logits()[0].argmax(-1) == self.lab[i]).all())
        return steps, ok, ms, qhs, ok_pre, fms

    def run_batch(self, idx, depth):
        """Fixed depth on a batch (B > 1 needs a dynamic-batch graph / engine built at that batch)."""
        r = self.runner
        r.load(self.x[idx], self.zh0, self.zl0, self.emb)
        ms = []
        steps = 0
        while steps < depth:
            t0 = time.perf_counter()
            r.step(need_q=False)
            ms.append((time.perf_counter() - t0) * 1e3)
            steps += self.k
        okv = (r.logits().argmax(-1) == self.lab[idx]).all(-1)
        return steps, okv, ms


def check_pass(loop, ref, nsup):
    """Fixed nsup steps on every puzzle, per-step predictions kept: the correctness gate against labels and the
    host reference (aihub_infer.host_loop's host_fp32.npz: preds (steps, N, seq), q (steps, N), zH (N, S, 512))."""
    a, r = loop.a, loop.runner
    n, seq = loop.n, loop.x.shape[1]
    calls = int(np.ceil(nsup / float(loop.k)))
    preds = np.zeros((n, calls, seq), np.uint8)
    qh = np.zeros((n, calls), np.float64)
    qc = np.zeros((n, calls), np.float64)
    ms = np.zeros((n, calls), np.float64)
    zH_final = np.zeros((n, loop.zh0.shape[0], loop.zh0.shape[-1]), np.float32)
    t_all = time.perf_counter()
    for i in range(n):
        r.load(loop.x[i:i + 1], loop.zh0, loop.zl0, loop.emb)
        for c in range(calls):
            t0 = time.perf_counter()
            r.step(need_q=True)
            ms[i, c] = (time.perf_counter() - t0) * 1e3
            preds[i, c] = r.logits()[0].argmax(-1)
            qh[i, c], qc[i, c] = float(r.qh[0]), float(r.qc[0])
        zH_final[i] = r.get_carry()[0][0]
        if (i + 1) % 10 == 0 or i == n - 1:
            log("check %d/%d exact@last %.3f mean step %.2f ms" % (
                i + 1, n, (preds[:i + 1, -1] == loop.lab[:i + 1]).all(-1).mean(), ms[:i + 1].mean()))
    lab = loop.lab[:, None, :]
    ex = (preds == lab).all(-1)
    cell = (preds == lab).mean(-1)
    ex_step = offline_exit(loop.rule, qh, qc, calls) * loop.k
    at_exit = ex[np.arange(n), np.minimum(ex_step // loop.k, calls) - 1]
    out = {"n": int(n), "nsup": int(calls * loop.k), "steps_per_call": loop.k,
           "exact_per_step": [round(float(v), 4) for v in ex.mean(0)],
           "cell_per_step": [round(float(v), 4) for v in cell.mean(0)],
           "exit_rule": loop.rule, "exit_exact": round(float(at_exit.mean()), 4),
           "exit_mean_steps": round(float(ex_step.mean()), 3),
           "exact_count_per_step": [int(v) for v in ex.sum(0)],
           "step_ms": lat_summary(ms), "wall_s": round(time.perf_counter() - t_all, 1),
           "per_puzzle": [{"i": int(i), "exact_per_step": [bool(v) for v in ex[i]],
                           "ms": [round(float(v), 3) for v in ms[i]],
                           "q_halt": [round(float(v), 5) for v in qh[i]]} for i in range(n)]}
    if ref is not None:
        rp = ref["preds"]                                     # (steps, N, seq)
        m = min(n, rp.shape[1])
        ks = min(rp.shape[0], calls)
        agree = np.array([(preds[:m, k] == rp[k, :m]).mean() for k in range(ks)])
        out["ref"] = {"file": a.ref, "n_compared": int(m), "steps_compared": int(ks),
                      "cell_agreement_per_step": [round(float(v), 5) for v in agree],
                      "puzzle_agreement_per_step": [round(float((preds[:m, k] == rp[k, :m]).all(-1).mean()), 4)
                                                    for k in range(ks)],
                      "puzzle_agreement_final": round(float((preds[:m, ks - 1] == rp[ks - 1, :m]).all(-1).mean()), 4),
                      "ref_exact_per_step": [round(float((rp[k, :m] == loop.lab[:m]).all(-1).mean()), 4) for k in range(ks)],
                      "ref_exact_count_per_step": [int((rp[k, :m] == loop.lab[:m]).all(-1).sum()) for k in range(ks)],
                      "exact_count_match": [int(v) for v in ex[:m, :ks].sum(0)] ==
                                           [int((rp[k, :m] == loop.lab[:m]).all(-1).sum()) for k in range(ks)]}
        if "q" in ref:
            rq = np.asarray(ref["q"])[:ks, :m]
            out["ref"]["q_halt_max_abs_diff"] = float(np.abs(qh[:m, :ks] - rq.T).max())
            out["ref"]["q_halt_sign_agreement"] = float(((qh[:m, :ks] > 0) == (rq.T > 0)).mean())
        if "zH" in ref and ks == calls:
            rz = np.asarray(ref["zH"])[:m].reshape(m, -1)
            z = zH_final[:m].reshape(m, -1)
            cos = (z * rz).sum(-1) / np.maximum(np.linalg.norm(z, axis=-1) * np.linalg.norm(rz, axis=-1), 1e-12)
            out["ref"]["zH_cos_final_mean"] = round(float(cos.mean()), 6)
            out["ref"]["zH_cos_final_min"] = round(float(cos.min()), 6)
            out["ref"]["zH_final_max_abs_diff"] = float(np.abs(z - rz).max())
    return out, {"preds": preds, "qh": qh, "qc": qc, "ms": ms, "zH_final": zH_final}


def lat_summary(ms):
    v = np.asarray(ms, dtype=np.float64).reshape(-1)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {}
    s = {"n": int(len(v)), "mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()),
         "max": float(v.max()), "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}
    m2 = np.asarray(ms, dtype=np.float64)
    if m2.ndim == 2:
        s["per_step_index_median"] = [float(np.median(m2[:, c])) for c in range(m2.shape[1])]
    return {k: (round(x, 4) if isinstance(x, float) else x) for k, x in s.items()}


def record_pass(loop, mode, depth):
    """Every puzzle once in the run mode: per puzzle steps, correctness, per-step ms (E1's first_pass + detail)."""
    per, first = [], {}
    t_all = time.perf_counter()
    if loop.B == 1:
        for i in range(loop.n):
            steps, ok, ms, qhs, ok_pre, fms = loop.run_puzzle(i, mode, depth)
            d = {"i": int(i), "steps": int(steps), "ok": bool(ok), "ms": [round(float(v), 3) for v in ms],
                 "q_halt": [round(float(v), 5) for v in qhs]}
            if ok_pre is not None:
                d["ok_before_finish"] = bool(ok_pre)
                d["finish_ms"] = [round(float(v), 3) for v in fms]
            per.append(d)
            first[i] = (steps, ok)
            if (i + 1) % 10 == 0 or i == loop.n - 1:
                log("record %d/%d exact %.3f mean steps %.2f step %.2f ms" % (
                    i + 1, loop.n, np.mean([p["ok"] for p in per]), np.mean([p["steps"] for p in per]),
                    np.mean([v for p in per for v in p["ms"]])))
    else:
        for s0 in range(0, loop.n, loop.B):
            idx = np.arange(s0, s0 + loop.B) % loop.n
            steps, okv, ms = loop.run_batch(idx, depth)
            for j, i in enumerate(idx.tolist()):
                if i not in first:
                    first[i] = (steps, bool(okv[j]))
                    per.append({"i": int(i), "steps": int(steps), "ok": bool(okv[j]),
                                "ms": [round(float(v), 3) for v in ms] if j == 0 else []})
    allms = [v for p in per for v in p["ms"]]
    fp = {"puzzles_covered": len(first), "exact": float(np.mean([v[1] for v in first.values()])),
          "mean_steps": float(np.mean([v[0] for v in first.values()])), "completion_untimed_s": 0.0,
          "wall_s": round(time.perf_counter() - t_all, 1),
          "per_puzzle": dict((str(i), [int(v[0]), bool(v[1])]) for i, v in sorted(first.items()))}
    return per, fp, lat_summary(allms)


def window(loop, mode, depth, win_s, sampler, names, total_rail, idle_P):
    """One timed window of >= win_s looping the workload; energy from the sampler's rows over the window."""
    n, B = loop.n, loop.B
    ptr = [0]

    def unit():
        if B == 1:
            i = ptr[0] % n
            ptr[0] += 1
            steps, ok, ms, qhs, ok_pre, fms = loop.run_puzzle(i, mode, depth)
            return steps, 1, int(ok), len(ms) + len(fms)
        idx = np.arange(ptr[0], ptr[0] + B) % n
        ptr[0] += B
        steps, okv, ms = loop.run_batch(idx, depth)
        return steps * B, B, int(okv.sum()), len(ms)

    th0 = read_thermal()
    if sampler:
        sampler.take()
        sampler.on.set()
    steps = puz = solved = units = calls = 0
    t0 = time.perf_counter()
    while True:
        s_, p_, c_, k_ = unit()
        steps += s_
        puz += p_
        solved += c_
        calls += k_
        units += 1
        if time.perf_counter() - t0 >= win_s:
            break
    t1 = time.perf_counter()
    if sampler:
        sampler.on.clear()
    th1 = read_thermal()
    T = t1 - t0
    w = {"T_s": T, "steps": steps, "puzzles": puz, "solved": solved, "units": units, "graph_calls": calls,
         "ms_per_step": 1e3 * T / steps, "ms_per_puzzle": 1e3 * T / puz, "ms_per_solved": 1e3 * T / max(solved, 1),
         "ms_per_call": 1e3 * T / max(calls, 1), "acc_window": solved / float(puz), "at": now_utc(),
         "loadavg": list(os.getloadavg()), "thermal_start": th0, "thermal_end": th1,
         "throttled": throttled_flag(th0) or throttled_flag(th1)}
    if sampler:
        e = integrate(sampler.take(), names, t0, t1, total_rail)
        w["rails"] = e["rails"]
        w["n_samples"] = e["n_samples"]
        w["sample_hz"] = e.get("hz")
        E = e["E_J"]
        if E is not None:
            Ed = E - idle_P * T if idle_P is not None else None
            w.update({"E_J": E, "P_W": E / T, "E_dyn_J": Ed, "idle_P_W": idle_P,
                      "mJ_per_step": 1e3 * E / steps, "J_per_puzzle": E / puz, "J_per_solved": E / max(solved, 1)})
            if Ed is not None:
                w.update({"mJ_per_step_dyn": 1e3 * Ed / steps, "J_per_solved_dyn": Ed / max(solved, 1)})
    w["clean"] = not w["throttled"]
    return w


def idle_window(win_s, sampler, names, total_rail, label):
    th0 = read_thermal()
    sampler.take()
    sampler.on.set()
    t0 = time.perf_counter()
    time.sleep(win_s)
    t1 = time.perf_counter()
    sampler.on.clear()
    e = integrate(sampler.take(), names, t0, t1, total_rail)
    w = {"label": label, "T_s": t1 - t0, "E_J": e["E_J"], "P_W": e["P_W"], "rails": e["rails"],
         "n_samples": e["n_samples"], "sample_hz": e.get("hz"), "at": now_utc(), "loadavg": list(os.getloadavg()),
         "thermal_start": th0, "thermal_end": read_thermal(), "clean": True}
    return w


def cfg_name(a):
    s = "%s_%s_b%d_%s" % (a.backend, a.precision, a.batch, a.mode)
    if a.mode == "fixed" and a.depth != NSUP:
        s += str(a.depth)
    if a.steps_per_call != 1:
        s += "_k%d" % a.steps_per_call
    if a.finish and a.finish_steps > 0:
        s += "_finish%d%s" % (a.finish_steps, ("_" + a.finish_precision) if a.finish_precision else "")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["ort", "trt"])
    ap.add_argument("--task", required=True, choices=sorted(RULES))
    ap.add_argument("--model", required=True, help=".onnx (ort) or .plan (trt) of the step graph")
    ap.add_argument("--init", required=True, help="init.npz: zH0, zL0 (512,), emb (1, 512)")
    ap.add_argument("--puzzles", required=True, help="npz from make_puzzles.py: inputs, labels")
    ap.add_argument("--n", type=int, default=None, help="first n puzzles of the file (default all)")
    ap.add_argument("--ref", default=None, help="host_fp32.npz (aihub_infer.stage_host) for the check pass")
    ap.add_argument("--board", default=socket.gethostname())
    ap.add_argument("--precision", default="fp32", help="label of the engine/graph precision (fp32, fp16, int8...)")
    ap.add_argument("--mode", choices=["exit", "fixed"], default="exit")
    ap.add_argument("--depth", type=int, default=NSUP, help="steps in fixed mode (also the cap in exit mode)")
    ap.add_argument("--rule", default=None, help="override the task's frozen exit rule")
    ap.add_argument("--steps_per_call", type=int, default=1, help="k for a k-unrolled graph/engine")
    ap.add_argument("--finish", default=None, help="second graph/engine run for --finish_steps after the exit")
    ap.add_argument("--finish_steps", type=int, default=0)
    ap.add_argument("--finish_precision", default="")
    ap.add_argument("--finish_steps_per_call", type=int, default=1)
    ap.add_argument("--batch", type=int, default=1, help="fixed mode only for B > 1 (dynamic-batch graph)")
    ap.add_argument("--threads", type=int, default=None, help="ort intra-op threads")
    ap.add_argument("--no_check", action="store_true", help="skip the fixed-depth correctness pass")
    ap.add_argument("--check_n", type=int, default=None, help="puzzles in the check pass (default all)")
    ap.add_argument("--warm", type=int, default=3, help="warm-up puzzles before timing")
    ap.add_argument("--win_s", type=float, default=60.0)
    ap.add_argument("--nwin", type=int, default=5, help="energy windows (0 = latency only)")
    ap.add_argument("--nidle", type=int, default=1, help="idle windows of win_s before the run")
    ap.add_argument("--nidle_end", type=int, default=0)
    ap.add_argument("--power_dir", default=NANO_POWER_DIR)
    ap.add_argument("--power_files", default=NANO_POWER_FILES, help="comma list of mW files in --power_dir")
    ap.add_argument("--rail_names", default=NANO_RAIL_NAMES)
    ap.add_argument("--total_rail", type=int, default=0, help="index of the rail used as the headline energy")
    ap.add_argument("--power_hz", type=float, default=50.0)
    ap.add_argument("--no_power", action="store_true", help="no power monitor (Pi 4, Nano 2 GB): latency only")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_npz", default=None, help="also save the check pass predictions/latencies here")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    a.rule = a.rule or RULES[a.task]
    if a.batch > 1 and a.mode == "exit":
        raise SystemExit("--batch > 1 needs --mode fixed")

    init = np.load(a.init)
    plen = int(init["plen"]) if "plen" in init.files else 16
    puz = np.load(a.puzzles)
    x = puz["inputs"].astype(np.int32)
    lab = puz["labels"].astype(np.int64)
    if a.n:
        x, lab = x[:a.n], lab[:a.n]
    S = x.shape[1] + plen
    zh0 = np.broadcast_to(init["zH0"].astype(np.float32), (S, init["zH0"].shape[-1])).copy()
    zl0 = np.broadcast_to(init["zL0"].astype(np.float32), (S, init["zL0"].shape[-1])).copy()
    emb = init["emb"].astype(np.float32).reshape(1, -1)
    ref = None
    if a.ref:
        ref = dict(np.load(a.ref))

    # power sampler
    files = [os.path.join(a.power_dir, f) for f in a.power_files.split(",") if f]
    names = a.rail_names.split(",")[:len(files)] or ["rail%d" % i for i in range(len(files))]
    sampler = None
    if not a.no_power:
        missing = [f for f in files if not os.path.exists(f)]
        if missing:
            log("power files missing, running without energy:", missing)
        else:
            sampler = PowerSampler(files, names, a.power_hz)
            sampler.start()
            log("power sampler on %s at %.0f Hz, first read %s mW" % (a.power_dir, a.power_hz, sampler.read_once()))

    t_load = time.perf_counter()
    cuda = CudaMem() if a.backend == "trt" else None
    runner = make_runner(a.backend, a.model, a.batch, a.threads, cuda)
    finisher = make_runner(a.backend, a.finish, a.batch, a.threads, cuda) if a.finish else None
    t_load = time.perf_counter() - t_load
    loop = Loop(a, runner, finisher, (zh0, zl0, emb), x, lab)
    name = cfg_name(a)
    rec = {"meta": {
        "board": a.board, "host": socket.gethostname(), "platform": "edge: " + a.board + " " + platform.machine(),
        "backend": a.backend, "runtime_version": runner.version, "python": platform.python_version(),
        "numpy": np.__version__, "task": a.task, "model": os.path.abspath(a.model),
        "model_size_MB": round(os.path.getsize(a.model) / 1e6, 2), "precision": a.precision, "mode": a.mode,
        "depth": a.depth, "nsup": NSUP, "rule": a.rule, "steps_per_call": a.steps_per_call, "B": a.batch,
        "finish": os.path.abspath(a.finish) if a.finish else None, "finish_steps": a.finish_steps,
        "finish_precision": a.finish_precision, "threads": a.threads,
        "providers": getattr(runner, "providers", None), "cuda_mem": getattr(runner, "mem_mode", None),
        "init": os.path.abspath(a.init), "puzzles": os.path.abspath(a.puzzles), "n_puzzles": int(len(x)),
        "puzzle_index": [int(v) for v in puz["index"][:len(x)]] if "index" in puz.files else None,
        "ref": os.path.abspath(a.ref) if a.ref else None,
        "power": {"dir": a.power_dir, "files": files, "rails": names, "total_rail": names[a.total_rail] if names else None,
                  "hz": a.power_hz, "active": sampler is not None,
                  "note": "INA3221 sysfs mW, trapezoid over sample timestamps; idle window same length, same session"},
        "win_s": a.win_s, "nwin": a.nwin, "nidle": a.nidle, "clocks": read_clocks(), "thermal_start": read_thermal(),
        "load_s": round(t_load, 2), "at": now_utc(), "argv": sys.argv, "note": a.note,
        "record_layout": "E1 energy.json: meta / idle[] / windows[cfg]{windows[], median_clean, first_pass}"},
        "idle": [], "windows": {}, "check": None, "record": None}
    dump(a.out, rec)

    # warm-up
    for i in range(min(a.warm, len(x))):
        loop.run_puzzle(i, a.mode, a.depth)

    # check pass (fixed NSUP steps, all puzzles, vs labels and the host reference)
    if not a.no_check:
        nchk = min(a.check_n or len(x), len(x))
        sub = Loop(a, runner, None, (zh0, zl0, emb), x[:nchk], lab[:nchk])
        chk, arrays = check_pass(sub, ref, NSUP)
        rec["check"] = chk
        log("check: exact per step", chk["exact_count_per_step"], "of", nchk, "| exit", chk["exit_rule"],
            chk["exit_exact"], chk["exit_mean_steps"], "| step ms median", chk["step_ms"].get("median"))
        if "ref" in chk:
            log("check vs ref: cell agreement", chk["ref"]["cell_agreement_per_step"][-1], "puzzle agreement",
                chk["ref"]["puzzle_agreement_final"], "exact counts match", chk["ref"]["exact_count_match"],
                "zH cos min", chk["ref"].get("zH_cos_final_min"), "q_halt maxdiff", chk["ref"].get("q_halt_max_abs_diff"))
        if a.save_npz:
            np.savez_compressed(a.save_npz, **arrays)
        dump(a.out, rec)

    # record pass in the run mode
    per, fp, lat = record_pass(loop, a.mode, a.depth)
    rec["record"] = {"cfg": name, "per_puzzle": per, "step_ms": lat, "exact": fp["exact"], "mean_steps": fp["mean_steps"],
                     "exact_before_finish": (float(np.mean([p["ok_before_finish"] for p in per]))
                                             if per and "ok_before_finish" in per[0] else None)}
    log("record: exact %.4f mean steps %.3f step ms median %s" % (fp["exact"], fp["mean_steps"], lat.get("median")))
    import re
    mH = re.search(r"_H(\d+)", os.path.basename(a.model))
    w_entry = {"rt": a.backend, "prec": a.precision, "H": int(mH.group(1)) if mH else None, "B": a.batch, "mode": a.mode, "engine": os.path.basename(a.model),
               "windows": [], "median_clean": {}, "n_clean": 0, "first_pass": fp}
    rec["windows"][name] = w_entry
    dump(a.out, rec)

    # idle windows, energy windows
    idle_P = None
    if sampler and a.nwin > 0:
        for k in range(a.nidle):
            w = idle_window(a.win_s, sampler, names, a.total_rail, "start")
            rec["idle"].append(w)
            log("idle start %.3f W (%d samples, %.1f Hz)" % (w["P_W"] or -1, w["n_samples"], w["sample_hz"] or 0))
            dump(a.out, rec)
        ps = [w["P_W"] for w in rec["idle"] if w["P_W"] is not None]
        idle_P = statistics.median(ps) if ps else None
    if a.nwin > 0:
        for k in range(a.nwin):
            w = window(loop, a.mode, a.depth, a.win_s, sampler, names, a.total_rail, idle_P)
            w_entry["windows"].append(w)
            log("%s w%d %.1f s %d steps %.3f ms/step acc %.3f%s%s" % (
                name, k, w["T_s"], w["steps"], w["ms_per_step"], w["acc_window"],
                (" %.3f W %.2f mJ/step %.3f J/solved" % (w["P_W"], w["mJ_per_step"], w["J_per_solved"])) if "E_J" in w else "",
                "" if w["clean"] else " THROTTLED"))
            good = [g for g in w_entry["windows"] if g["clean"]]
            keys = [kk for kk in ("P_W", "mJ_per_step", "mJ_per_step_dyn", "J_per_puzzle", "J_per_solved", "J_per_solved_dyn",
                                  "ms_per_step", "ms_per_puzzle", "ms_per_solved", "acc_window") if good and kk in good[0]]
            w_entry["median_clean"] = dict((kk, statistics.median([g[kk] for g in good])) for kk in keys)
            w_entry["n_clean"] = len(good)
            dump(a.out, rec)
        if sampler:
            for k in range(a.nidle_end):
                rec["idle"].append(idle_window(a.win_s, sampler, names, a.total_rail, "end"))
    rec["meta"]["thermal_end"] = read_thermal()
    rec["meta"]["clocks_end"] = read_clocks()
    rec["meta"]["finished_at"] = now_utc()
    if sampler:
        rec["meta"]["power"]["sampler_errors"] = sampler.errors
        sampler.close()
    dump(a.out, rec)
    log("wrote", a.out)


if __name__ == "__main__":
    main()
