#!/usr/bin/env python3
"""E9C: a LIVE loop scheduler serving arriving puzzles on one GPU (MLSys), measured with the E1 energy protocol.

E9 (sched_sim.py, records only) showed that allocating LOOPS across puzzles with the halting head as the release signal
beats fixed-depth FIFO under a shared loop budget and under deadlines (20 Sudoku cells, deadline 16 loops, load 4:
fixed-FIFO 11.3 % of arrivals answered correctly in time, round-robin anytime 39.3, head-gated 43.1, head-gated + 8-bit
finishing 45.4). E9B showed the head is the right stop signal. This script runs the same thing LIVE: a runtime on one
device (Babel L40S; smoke on a p4d A100), puzzles arriving as a Poisson stream, a deadline per puzzle, the scheduler
deciding every loop which puzzles are in the batch, energy from the NVML counter with idle subtraction.

Serving model (continuous batching OVER LOOPS): every live puzzle holds a carry (z_H, z_L) in a slot pool (--pool 512
resident puzzles; an arrival that finds the pool full is a miss and is counted). Each TICK the scheduler picks at most
--cap 32 live puzzles and runs ONE forward pass of the step module on that sub-batch (gather -> step -> scatter); the
puzzles it did not pick wait with their carry intact. A puzzle whose head has fired and is being finished with the 8-bit
master runs in a second sub-batch of the same tick, with the master weights (two weight sets cannot share a GEMM). The
eager step module, the RTN cheap weights and the w8c master are those of e1_dither.py (fp32 compute, TF32 off).

Clock and load. tau = the median wall time of one tick at the full batch (--cap slots, cheap weights), measured in the
job; one "loop" of time = tau. The device's loop rate is cap / tau puzzle-loops per second, so fixed depth 16 can serve
cap / (16 tau) puzzles per second: that is load 1. Load lambda = Poisson arrivals at lambda x cap / (16 tau) per second.
A puzzle's deadline is delta x tau seconds after the runtime admits it (delta = 16, 32 loops): the request is seen at
the first tick after its Poisson arrival (the sub-tick admission delay is recorded), so a fixed-depth puzzle that finds
the device idle can still finish exactly at a zero-slack deadline of 16, as in E9's event-driven simulation. Every
policy at a given (load, delta) sees the SAME Poisson stream and the same puzzle ids (seeded), so the policies are
paired. Latency in loops = latency in seconds / tau (tau is the FULL-batch tick; at light load a tick is shorter, so a
puzzle's own loop count, also reported, exceeds its latency in loops).

Policies (all serve the same compressed model: RTN cheap weights, --spec per task from the --master w8c copy):
  fifo    fixed depth 16, first come first served over the cap slots; an answer exists only after loop 16; a puzzle that
          cannot finish its 16 loops before its deadline even if started now is dropped (E9's fifo16 admission rule)
  rr      round-robin anytime: each tick the cap live puzzles with the fewest loops so far (processor sharing); a puzzle
          leaves at loop 16; at its deadline its last completed state is its answer
  head    as rr, and a puzzle leaves when its head rule fires (M5 fp32-chosen rule per model, cap 16), freeing its share
  ctl     head + 8-bit finishing: a fired (or capped) puzzle runs --jfin 2 loops with the w8c master weights, then leaves
  ctl_dither  ctl with the per-step re-draw on the cheap loops (stochastic rounding of the cheap weights from the w8c
          master before every cheap forward, one draw per tick shared by the sub-batch): the E_COMBINED winner
Hit = the puzzle's answer, taken at its departure or at its deadline (last in-time state), is correct AND was produced
no later than the deadline. Metrics per (policy, load, delta) and per 60 s window: hit fraction over the puzzles finalised
in the window (departed, expired, dropped), mean and p95 latency of the hits in loops and ms (arrival -> answer; an
anytime puzzle answered at its deadline counts delta), throughput (puzzles finalised per s, hits per s), J per hit and
idle-subtracted J per hit, ticks, mean batch size. Protocol as e1_dither / e1_grid: NVML total-energy counter, 5 idle
windows x 30 s at the start and 2 at the end, 10 s warm-up per configuration, --nwin 5 windows of --win_s 60 s, clean-card
check before and after every window, 10 Hz clock / power samples; medians and 95 % t-intervals over the windows.

Cost caveat (stated before running): on this GPU an 8-bit finishing loop costs the SAME kernels as a cheap loop (fake
quantised fp32 weights in both), so the finishing loops are charged one tick each here, not r = 8 / b_eff as in E9's cost
model; the dither re-draw is a real per-tick cost here (the E1 dither row measured its share).

PREDICTIONS (written before any live number existed; also in reports/edgetrm_exp/E9C_LIVE.md):
  P1 the simulated ordering holds live at every load and both deadlines: fifo < rr < head < ctl (hit fraction), with
     fifo collapsing with load (drop-on-admission) and head/ctl degrading least;
  P2 the live gap ctl - fifo at load 4, deadline 16, is within a third of the simulated gap of the matching cell
     (sudoku_mlp w4c: 64.6 - 14.2 = 50 points; sudoku_attn w3c: 38.5 - 14.5 = 24; pooled 20 Sudoku cells 45.4 - 11.3 =
     34), i.e. live gap in [2/3, 4/3] x simulated;
  P3 joules per correct answer fall from fifo to rr to head to ctl at every load, because the loops a fixed-depth policy
     spends after the head would have released a puzzle are the energy; at load <= 0.5 the card idles most of the time
     and the TOTAL J per hit is dominated by the idle floor (dyn J per hit shows the ordering more clearly);
  P4 ctl_dither: hit fraction >= ctl on Sudoku-MLP and Sudoku-attn (E_COMBINED: combined > controller on Sudoku), equal
     within the interval on HRM and Maze, at a higher J per hit (the re-draw is a per-tick cost, ~7-15 % on Sudoku).

Output: $SL_RES/<task>/sched_live.json (default $HOME/e9/results/<task>/; resumable per configuration).
    python sched_live.py --task sudoku_mlp                       # full grid: 5 policies x 5 loads x 2 deadlines
    python sched_live.py --task sudoku_mlp --smoke               # tiny: 3 s windows, loads 1,4, deadline 16
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import e1_energy as E  # noqa: E402
import e1_dither as D  # noqa: E402
import e1_grid as G  # noqa: E402

RES = pathlib.Path(os.environ.get("SL_RES", pathlib.Path.home() / "e9" / "results"))
CELLS = D.CELLS                                        # sudoku_mlp w4c, sudoku_attn w3c, hrm_sudoku w3c, maze w4c
POLICIES = ["fifo", "rr", "head", "ctl", "ctl_dither"]
LOADS = [0.25, 0.5, 1.0, 2.0, 4.0]
DELTAS = [16, 32]
NSUP = E.NSUP
log = E.log


# ----------------------------------------------------------------------------- slot pool (carries of every live puzzle)
class Pool:
    """P resident carries; run(idx) advances the puzzles in idx with one forward pass on the gathered sub-batch."""

    def __init__(self, task, H, P, cap, spec, master_spec, seed, labg):
        import torch
        self.torch = torch
        inner, pe = E.build_inner(task, H)
        zh0, zl0 = E.init_carry(inner)
        self.step_mod = E.make_step(inner, "fp32").cuda()
        self.S = inner.config.seq_len + inner.puzzle_emb_len
        seq = inner.config.seq_len
        self.P, self.cap = P, cap
        self.lin = D._linears(self.step_mod)
        assert self.lin, "no CastedLinear found"
        W32 = [mod.weight.data.float().clone() for mod in self.lin]
        mb, mg = D.parse_spec(master_spec)
        self.master32 = [D.round_grid(W, mb, mg) for W in W32]
        self.bits, self.gran = D.parse_spec(spec)
        self.cheap_w = [D.round_grid(M, self.bits, self.gran) for M in self.master32]
        self.nparam = sum(W.numel() for W in W32)
        self.gen = torch.Generator(device="cuda").manual_seed(int(seed))
        self.n_redraw = 0
        self.cur = None
        with torch.inference_mode():
            self.zh0 = zh0.cuda()
            self.zl0 = zl0.cuda()
            self.pe = pe.cuda().expand(cap, -1).contiguous()
            self.zH = self.zh0.expand(P, self.S, -1).clone()
            self.zL = self.zl0.expand(P, self.S, -1).clone()
            self.x = torch.zeros(P, seq, dtype=torch.int32, device="cuda")
            self.pid = torch.zeros(P, dtype=torch.long, device="cuda")
            self.ok = torch.zeros(P, dtype=torch.bool, device="cuda")
            self.qh = torch.zeros(P, device="cuda")
            self.qc = torch.zeros(P, device="cuda")
        self.labg = labg
        self.set_cheap()

    def _set(self, ws, tag):
        if self.cur == tag:
            return
        for mod, W in zip(self.lin, ws):
            mod.weight.data = W
        self.cur = tag

    def set_cheap(self):
        self._set(self.cheap_w, "cheap")

    def set_master(self):
        self._set(self.master32, "master")

    def redraw(self):
        torch = self.torch
        for mod, M in zip(self.lin, self.master32):
            U = torch.rand(M.shape, generator=self.gen, device=M.device, dtype=M.dtype)
            mod.weight.data = D.round_grid(M, self.bits, self.gran, U)
        self.cur = "dither"
        self.n_redraw += 1

    def reset(self, slots, pids, xrows):
        torch = self.torch
        with torch.inference_mode():
            idx = torch.as_tensor(slots, dtype=torch.long, device="cuda")
            self.x[idx] = xrows.to(self.x.dtype)
            self.pid[idx] = torch.as_tensor(pids, dtype=torch.long, device="cuda")
            self.zH[idx] = self.zh0
            self.zL[idx] = self.zl0
            self.ok[idx] = False
            self.qh[idx] = 0.0
            self.qc[idx] = 0.0

    def run(self, idx):
        """idx: cuda long tensor of slots (<= cap). Returns (ok, qh, qc) host arrays for those slots."""
        torch = self.torch
        k = idx.numel()
        with torch.inference_mode():
            o = self.step_mod(self.zH[idx], self.zL[idx], self.x[idx], self.pe[:k])
            self.zH[idx] = o[0]
            self.zL[idx] = o[1]
            a = o[2].argmax(-1)
            self.ok[idx] = (a == self.labg[self.pid[idx]]).all(-1)
            self.qh[idx] = o[3].float().reshape(-1)
            if o[4] is not None:
                self.qc[idx] = o[4].float().reshape(-1)
            out = torch.stack([self.ok[idx].float(), self.qh[idx], self.qc[idx]]).cpu().numpy()
        return out[0] > 0.5, out[1], out[2]


# ----------------------------------------------------------------------------- the live scheduler
class Puz:
    __slots__ = ("pid", "slot", "t_arr", "dl", "k", "kf", "phase", "prev", "last_ok", "t_last")

    def __init__(self, pid, slot, t_arr, dl):
        self.pid, self.slot, self.t_arr, self.dl = pid, slot, t_arr, dl
        self.k = self.kf = 0
        self.phase = "cheap"
        self.prev = False
        self.last_ok = False
        self.t_last = None


class Live:
    """One configuration: policy, load, deadline. run_until(t) advances the stream and returns the window's counts."""

    def __init__(self, pool, policy, rule, load, delta, tau, cap, xg, n, rng, jfin, clock):
        self.pool, self.policy, self.rule = pool, policy, rule
        self.load, self.delta, self.tau, self.cap = load, delta, tau, cap
        self.xg, self.n, self.rng, self.jfin, self.clk = xg, n, rng, jfin, clock
        self.rate = load * cap / (NSUP * tau)                       # arrivals per second; load 1 = fixed-16 at full batch
        self.dl_s = delta * tau
        self.free = list(range(pool.P))
        self.live = []
        self.next_arr = self.clk() + rng.exponential(1.0 / self.rate)
        self.torch = pool.torch
        self.reset_counts()

    def reset_counts(self):
        self.c = {"arrivals": 0, "final": 0, "hit": 0, "dropped_pool": 0, "dropped_fifo": 0, "expired": 0,
                  "departed": 0, "departed_late": 0, "ticks": 0, "slot_loops": 0, "fin_loops": 0, "idle_ticks": 0,
                  "lat_hit": [], "lat_done": [], "loops_final": [], "loops_hit": [], "adm_delay": []}

    def _finalize(self, p, t, reason, ok=None):
        c = self.c
        if reason == "done":                                       # departure at t with answer correctness ok
            in_time = t <= p.dl + 1e-9
            if in_time:
                p.last_ok = ok
                c["departed"] += 1
                c["lat_done"].append(t - p.t_arr)
            else:
                c["departed_late"] += 1
            hit = (ok and in_time) if self.policy == "fifo" else p.last_ok
            lat = (t - p.t_arr) if in_time else self.dl_s
        elif reason == "expired":
            hit = False if self.policy == "fifo" else p.last_ok
            lat = self.dl_s
            c["expired"] += 1
        else:                                                      # dropped (pool full / fifo cannot finish)
            hit = False
            lat = None
            c["dropped_" + reason] += 1
        c["final"] += 1
        c["loops_final"].append(p.k + p.kf)
        if hit:
            c["hit"] += 1
            c["lat_hit"].append(lat)
            c["loops_hit"].append(p.k + p.kf)
        if p.slot is not None:
            self.free.append(p.slot)
        self.live.remove(p)

    def _admit(self, now):
        c = self.c
        while self.next_arr <= now:
            t_pois = self.next_arr
            self.next_arr = t_pois + self.rng.exponential(1.0 / self.rate)
            pid = int(self.rng.integers(0, self.n))
            c["arrivals"] += 1
            c["adm_delay"].append(now - t_pois)
            p = Puz(pid, None, now, now + self.dl_s)                  # the runtime sees the request at this tick
            self.live.append(p)
            if not self.free:
                self._finalize(p, now, "pool")
                continue
            p.slot = self.free.pop()
            self.pool.reset([p.slot], [pid], self.xg[pid:pid + 1])

    def tick(self):
        torch = self.torch
        now = self.clk()
        self._admit(now)
        c = self.c
        for p in list(self.live):
            if p.dl <= now:
                self._finalize(p, now, "expired")
            elif self.policy == "fifo" and p.k == 0 and now + NSUP * self.tau > p.dl:
                self._finalize(p, now, "fifo")
        if self.policy == "fifo":
            members = sorted(self.live, key=lambda p: p.t_arr)[:self.cap]
        else:
            members = sorted(self.live, key=lambda p: (p.k + p.kf, p.t_arr))[:self.cap]
        if not members:
            c["idle_ticks"] += 1
            wait = self.next_arr - self.clk()
            if wait > 0:
                time.sleep(min(wait, 0.25))
            return
        cheap = [p for p in members if p.phase == "cheap"]
        fin = [p for p in members if p.phase == "fin"]
        res = {}
        if cheap:
            if self.policy == "ctl_dither":
                self.pool.redraw()
            else:
                self.pool.set_cheap()
            idx = torch.as_tensor([p.slot for p in cheap], dtype=torch.long, device="cuda")
            ok, qh, qc = self.pool.run(idx)
            for j, p in enumerate(cheap):
                res[p] = (bool(ok[j]), float(qh[j]), float(qc[j]))
        if fin:
            self.pool.set_master()
            idx = torch.as_tensor([p.slot for p in fin], dtype=torch.long, device="cuda")
            ok, qh, qc = self.pool.run(idx)
            for j, p in enumerate(fin):
                res[p] = (bool(ok[j]), float(qh[j]), float(qc[j]))
        t_end = self.clk()
        c["ticks"] += 1
        c["slot_loops"] += len(members)
        c["fin_loops"] += len(fin)
        for p in members:
            ok, qh, qc = res[p]
            in_time = t_end <= p.dl + 1e-9
            if p.phase == "cheap":
                p.k += 1
                if in_time:
                    p.last_ok, p.t_last = ok, t_end
                halt = (qh > qc) if self.rule == "hrm_own" else (qh > 0)
                if self.rule == "patience2":
                    fire = halt and p.prev
                elif self.rule == "min2native":
                    fire = halt and p.k >= 2
                else:
                    fire = halt
                p.prev = halt
                capped = p.k >= NSUP
                if self.policy in ("fifo", "rr"):
                    if capped:
                        self._finalize(p, t_end, "done", ok)
                elif self.policy == "head":
                    if fire or capped:
                        self._finalize(p, t_end, "done", ok)
                else:
                    if fire or capped:
                        if self.jfin > 0:
                            p.phase = "fin"
                        else:
                            self._finalize(p, t_end, "done", ok)
            else:
                p.kf += 1
                if in_time:
                    p.last_ok, p.t_last = ok, t_end
                if p.kf >= self.jfin:
                    self._finalize(p, t_end, "done", ok)

    def run_until(self, t_stop):
        while self.clk() < t_stop:
            self.tick()

    def window(self):
        """Window counts -> metrics; resets the counters."""
        c = self.c
        tau = self.tau

        def lat(v):
            if not v:
                return None
            a = np.array(v)
            return {"mean_ms": 1e3 * float(a.mean()), "p95_ms": 1e3 * float(np.percentile(a, 95)),
                    "mean_loops": float(a.mean() / tau), "p95_loops": float(np.percentile(a, 95) / tau), "n": len(v)}
        m = {k: v for k, v in c.items() if not isinstance(v, list)}
        m["hit_frac"] = c["hit"] / max(c["final"], 1)
        m["lat_hit"] = lat(c["lat_hit"])
        m["lat_done"] = lat(c["lat_done"])
        m["loops_per_final"] = float(np.mean(c["loops_final"])) if c["loops_final"] else None
        m["loops_per_hit"] = float(np.mean(c["loops_hit"])) if c["loops_hit"] else None
        m["adm_delay_ms_mean"] = 1e3 * float(np.mean(c["adm_delay"])) if c["adm_delay"] else None
        m["batch_mean"] = c["slot_loops"] / max(c["ticks"], 1)
        m["live_now"] = len(self.live)
        self.reset_counts()
        return m


def measure_tau(pool, cap, xg, n, rng, iters=50):
    """Median wall time per tick at the full batch (cheap weights), at batch 1, and of one re-draw."""
    torch = pool.torch
    out = {}
    for B in (cap, 1):
        slots = list(range(B))
        pids = rng.integers(0, n, B)
        pool.reset(slots, pids, xg[torch.as_tensor(pids, device="cuda")])
        idx = torch.arange(B, device="cuda")
        pool.set_cheap()
        for _ in range(10):
            pool.run(idx)
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t = time.perf_counter()
            pool.run(idx)
            ts.append(time.perf_counter() - t)
        out[f"tick_s_b{B}"] = float(statistics.median(ts))
    ts = []
    for _ in range(10):
        torch.cuda.synchronize()
        t = time.perf_counter()
        pool.redraw()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    out["redraw_s"] = float(statistics.median(ts))
    pool.set_cheap()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(E.TASKS))
    ap.add_argument("--spec", default=None)
    ap.add_argument("--master", default="w8c")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--loads", default=",".join(str(v) for v in LOADS))
    ap.add_argument("--deltas", default=",".join(str(v) for v in DELTAS))
    ap.add_argument("--cap", type=int, default=32, help="batch cap: puzzles per forward pass")
    ap.add_argument("--pool", type=int, default=512, help="resident carries (live puzzles); arrivals beyond it are misses")
    ap.add_argument("--jfin", type=int, default=2, help="8-bit finishing loops after the head fires (ctl policies)")
    ap.add_argument("--win_s", type=float, default=60.0)
    ap.add_argument("--nwin", type=int, default=5)
    ap.add_argument("--warm_s", type=float, default=10.0)
    ap.add_argument("--idle_s", type=float, default=30.0)
    ap.add_argument("--nidle", type=int, default=5)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--n_limit", type=int, default=None, help="use puzzle rows 0..n_limit-1")
    ap.add_argument("--only", default=None)
    ap.add_argument("--out", default="sched_live.json")
    ap.add_argument("--smoke", action="store_true", help="3 s x 2 windows, loads 1,4, delta 16, 1 idle window of 3 s")
    a = ap.parse_args()
    if a.smoke:
        a.win_s, a.nwin, a.warm_s, a.idle_s, a.nidle = 3.0, 2, 3.0, 3.0, 1
        a.loads, a.deltas = "1,4", "16"
        a.out = "sched_live_smoke.json"
    import torch
    import pynvml as nv
    assert torch.cuda.is_available(), "CUDA not available: abort"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    task = a.task
    t = E.TASKS[task]
    H, rule = t["H"], t["rule"]
    spec = a.spec or CELLS[task]
    policies = a.policies.split(",")
    loads = [float(v) for v in a.loads.split(",")]
    deltas = [int(v) for v in a.deltas.split(",")]
    n = min(a.n_limit or t["n"], t["n"])
    nv.nvmlInit()
    uuid_t = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    h = next(nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())
             if (lambda u: u.decode() if isinstance(u, bytes) else u)(nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(i))) == uuid_t)
    samp = E.Sampler(nv, h)
    samp.start()
    outd = RES / task
    outd.mkdir(parents=True, exist_ok=True)
    outp = outd / a.out
    rec = json.loads(outp.read_text()) if outp.exists() else {"configs": {}, "idle": []}
    rec["meta"] = {"gpu": torch.cuda.get_device_name(0), "uuid": uuid_t, "host": __import__("socket").gethostname(),
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "driver": nv.nvmlSystemGetDriverVersion(),
                   "torch": torch.__version__, "H": H, "nsup": NSUP, "rule": rule, "n": n, "spec": spec,
                   "master": a.master, "cap": a.cap, "pool": a.pool, "jfin": a.jfin, "win_s": a.win_s, "nwin": a.nwin,
                   "warm_s": a.warm_s, "seed": a.seed, "policies": policies, "loads": loads, "deltas": deltas,
                   "smoke": a.smoke,
                   "clock": "tau = median tick at the full batch (cheap weights); load 1 = cap/(16 tau) arrivals per s "
                            "(fixed depth 16 at full batch exactly busy); deadline = delta x tau s after arrival; "
                            "latency in loops = s / tau",
                   "cost_caveat": "an 8-bit finishing loop and a dither re-draw are charged their real tick time on this "
                                  "GPU (same fp32 kernels for cheap and master weights), not E9's r = 8/b_eff",
                   "hit": "answer (at departure, or the last in-time state at the deadline) correct and produced by the "
                          "deadline; fifo answers only at loop 16; dropped arrivals (pool full, fifo cannot finish) are misses",
                   "platform": "datacenter GPU (not edge), one exclusive card; clocks not pinned"}

    def clean():
        return not [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h) if p.pid != os.getpid()]

    def idle(k, label):
        for _ in range(k):
            torch.cuda.synchronize()
            samp.take()
            samp.on.set()
            e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            time.sleep(a.idle_s)
            e1, t1 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            samp.on.clear()
            w = {"label": label, "P_W": (e1 - e0) / 1e3 / (t1 - t0), "clean": clean(), "T_s": t1 - t0,
                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **samp.take()}
            rec["idle"].append(w)
            log("idle", label, round(w["P_W"], 2), "W", w.get("sm_MHz_mean"))
        E.dump(outp, rec)

    x, lab = E.load_rows(t["data"], 0, n)
    xg = torch.as_tensor(x, device="cuda")
    labg = torch.as_tensor(lab, device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.set_stream(stream)
    pool = Pool(task, H, a.pool, a.cap, spec, a.master, a.seed, labg)
    log(task, "pool", a.pool, "cap", a.cap, "spec", spec, "master", a.master, "params", pool.nparam)
    idle(a.nidle, "start")
    idle_P = statistics.median([w["P_W"] for w in rec["idle"] if w["clean"]])
    tau_info = measure_tau(pool, a.cap, xg, n, np.random.default_rng(a.seed))
    tau = tau_info[f"tick_s_b{a.cap}"]
    rec["meta"]["tau"] = tau_info
    rec["meta"]["tau_s"] = tau
    rec["meta"]["loop_rate_per_s"] = a.cap / tau
    log(task, "tau", {k: round(v * 1e3, 3) for k, v in tau_info.items()}, "ms; loop rate", round(a.cap / tau, 1), "/s")
    E.dump(outp, rec)
    clock = time.perf_counter
    only = set(a.only.split(",")) if a.only else None
    cfgs = [(pol, load, delta) for delta in deltas for load in loads for pol in policies]
    for pol, load, delta in cfgs:
        name = f"{pol}_l{load:g}_d{delta}"
        if only and name not in only:
            continue
        if name in rec["configs"] and rec["configs"][name].get("n_clean", 0) >= a.nwin:
            continue
        rng = np.random.default_rng([a.seed, int(round(load * 1000)), delta])          # same arrivals for every policy
        S = Live(pool, pol, rule, load, delta, tau, a.cap, xg, n, rng, a.jfin, clock)
        S.run_until(clock() + a.warm_s)
        S.window()
        wins = []
        for k in range(a.nwin):
            ok0 = clean()
            samp.take()
            samp.on.set()
            torch.cuda.synchronize()
            e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), clock()
            S.run_until(t0 + a.win_s)
            torch.cuda.synchronize()
            e1, t1 = nv.nvmlDeviceGetTotalEnergyConsumption(h), clock()
            samp.on.clear()
            m = S.window()
            E_, T_ = (e1 - e0) / 1e3, t1 - t0
            w = {"E_J": E_, "T_s": T_, "P_W": E_ / T_, "E_dyn_J": E_ - idle_P * T_, **m,
                 "final_per_s": m["final"] / T_, "hit_per_s": m["hit"] / T_, "arrivals_per_s": m["arrivals"] / T_,
                 "J_per_hit": E_ / max(m["hit"], 1), "J_per_hit_dyn": (E_ - idle_P * T_) / max(m["hit"], 1),
                 "mJ_per_slot_loop": 1e3 * E_ / max(m["slot_loops"], 1), "clean": bool(ok0 and clean()),
                 "loadavg": os.getloadavg(), **samp.take()}
            wins.append(w)
            lh = w["lat_hit"] or {}
            log(task, name, f"w{k}", f"{w['P_W']:.1f} W", f"hit {w['hit_frac']:.3f}", f"({m['hit']}/{m['final']}, arr {m['arrivals']})",
                f"lat {lh.get('mean_loops', float('nan')):.1f}/{lh.get('p95_loops', float('nan')):.1f} loops",
                f"{w['J_per_hit']:.3f} J/hit", f"batch {w['batch_mean']:.1f}", f"live {m['live_now']}",
                f"{w.get('sm_MHz_mean')} MHz", "clean" if w["clean"] else "DIRTY")
        good = [w for w in wins if w["clean"]]

        def med(key, sub=None):
            v = [(w[key][sub] if sub else w[key]) for w in good if (w[key] is not None and (not sub or w[key].get(sub) is not None))]
            return {"median": statistics.median(v), "ci95": G.ci95(v)} if v else None
        summ = {"n_clean": len(good), "hit_frac": med("hit_frac"), "J_per_hit": med("J_per_hit"),
                "J_per_hit_dyn": med("J_per_hit_dyn"), "final_per_s": med("final_per_s"), "hit_per_s": med("hit_per_s"),
                "lat_hit_mean_loops": med("lat_hit", "mean_loops"), "lat_hit_p95_loops": med("lat_hit", "p95_loops"),
                "lat_hit_mean_ms": med("lat_hit", "mean_ms"), "lat_hit_p95_ms": med("lat_hit", "p95_ms"),
                "loops_per_final": med("loops_per_final"), "loops_per_hit": med("loops_per_hit"),
                "adm_delay_ms_mean": med("adm_delay_ms_mean"), "batch_mean": med("batch_mean"), "P_W": med("P_W"),
                "arrivals_per_s": med("arrivals_per_s")}
        rec["configs"][name] = {"policy": pol, "load": load, "delta": delta, "rate_per_s": S.rate, "deadline_s": S.dl_s,
                                "idle_P_W": idle_P, "tau_s": tau, "n_redraws": pool.n_redraw, "windows": wins,
                                "summary": summ, "n_clean": len(good)}
        E.dump(outp, rec)
        log(task, name, "hit", summ["hit_frac"] and round(summ["hit_frac"]["median"], 4), "J/hit",
            summ["J_per_hit"] and round(summ["J_per_hit"]["median"], 4), "dyn",
            summ["J_per_hit_dyn"] and round(summ["J_per_hit_dyn"]["median"], 4))
        del S
    idle(2, "end")
    samp.stop_.set()
    log("done", outp)


if __name__ == "__main__":
    main()
