#!/usr/bin/env python3
"""The controller on a phone NPU (MLSys 2027, "the controller on the chip"): exit + 8-bit finishing, end to end on
the device, accuracy read from the device outputs. Runs on p4d from ~/x3 (aihub_acct.Acct handles the accounts:
qai_hub.Client(profile=...) loads the tokens; nothing here reads, prints or copies them).

Method, per puzzle (ICLR Section 4, E+P; rules frozen in EdgeTRM/m5_ctl_ho.py rules_for):
  1. run the CHEAP compiled step graph (w8a8; Maze w8a8rope16, the RoPE-at-16-bit fix) in the host loop, carry fed back
     from the host, until the halting head fires; the head is read at every H-th device step (H = 3 cycles per
     supervision step for all three models, the granularity the GPU rules were selected at: M5 samples q at h-1::h);
     Sudoku-MLP patience 2 (two consecutive supervision steps with q_halt > 0), Sudoku-attention native (q_halt > 0),
     Maze minimum 2 supervision steps then q_halt > 0. Cap CAP = 16 device steps (5 head reads).
  2. from the exit state run FIN = 6 device steps of the PRECISE graph (w8a16 for Sudoku; Maze fp32, which the HTP runs
     in fp16 since INT4 is broken there and no Maze w8a16 graph is compiled on these accounts) = j = 2 supervision
     steps; the prediction after every finishing cycle is recorded, so j = 1 (3 cycles) is readable too. The dev-row
     E+P choices for the GPU cells are per cell (jE in {0, 1, 4}, jF = 4 mostly; M5.md R20/R21, m5_ctl_ho.json) and
     none was made for an INT8-activation cheap cell, so j = 2 is the pre-registered value. Puzzles whose head never
     fires within the cap also get the finishing steps (the P branch of the controller with jF = j); the exit-state
     prediction is kept separately so exit-alone, finish-fired-only and finish-all are all readable.
One chain per (account, device, task): at each device step at most two inference jobs, the cheap graph for the
puzzles still running and the precise graph for the puzzles that owe finishing steps; at most SLOTS = 2 device jobs in
flight per account (chains are served in plan order, so the first cells complete first). Each chain also holds one
profile job per graph on its device (reused from the account's records when one exists) for the per-step latency.

Devices / cells (acct2: Galaxy S24; acct3: Galaxy S25, the 8 Elite silicon, since the 8 Elite QRDs left the hosted
fleet; every graph already compiled on the account except Sudoku-attn w8a16 on acct2, which compiles first):
  sudoku_mlp   cheap w8a8          precise w8a16   100 puzzles
  maze         cheap w8a8rope16    precise fp32     50 puzzles
  sudoku_attn  cheap w8a8          precise w8a16   100 puzzles (its device w8a8 graph works: S24 0.50, 8 Elite 0.47,
                                                    S25 0.23 at 8 cycles, non-zero everywhere)
Records: ~/aihub_artifacts/h1/<task>/infer/ctl_<cheap>-<precise>_<device>_<acct>.{json,npz} (the ctl_ prefix keeps
them apart from the breadth dev_* records, which device_acc.py globs). npz: zH, zL (current carry), status (0 cheap,
1 finishing, 2 done), cheap_steps, fin_steps, fired_at (-1 = never), run (patience counter), cheap_preds [N, CAP, L],
cheap_q [N, CAP], fin_preds [N, FIN, L], fin_q [N, FIN], exit_pred, final_pred [N, L], lab. json: per device step the
job ids and puzzle counts, errors, profile results (ms per step per graph), and at the end the summary.

Predictions, written before launch (Sep 28):
  - Exit alone saves about 2x device steps on Sudoku-MLP (S24 / 8 Elite: the head fires for ~60% of the puzzles by
    cycle 8 in the breadth records; mean cheap steps ~9-11 of 16) and 5-8x on Maze on the S24 (fires for 82% of the
    mazes at cycle 3; mean ~3 of 16), with accuracy within 1 point of the fixed 8-step chain on the same puzzles
    (breadth records: S24 w8a8 0.61, 8 Elite 0.57; S24 Maze w8a8rope16 0.84-0.86).
  - w8a16 / fp16 finishing lifts collapsed cells: the device Maze w8a8rope16 scores 0 at every cycle on the S25 and
    on the 8 Elite QRD, and its head fires for only 8-18% of the mazes, so most mazes run to the cap and then finish
    from a collapsed state on the same device. The ICLR return law says the precise graph brings them back when the
    loop still settles: predicted Maze S25 / 8 Elite final exact >= 0.6 after 6 fp16 cycles (device fp32 reaches 0.80
    at cycle 3 from the initial state), against 0 without finishing. Sudoku-MLP on the S25 (w8a8 0.31, w8a16 0.42 at
    8 cycles, head firing for <10%) is the second degraded cell: finishing should add >= 5 points over exit alone.
  - Sudoku-attn native rule: the device head fires for every puzzle at cycle 1 (host fp32 too: the first H1 cycle);
    reading it every 3 cycles avoids that, and the S24 / 8 Elite cells should land within 2 points of fixed 8 at
    ~10 cheap cycles.

Sep 29 (DIAG_ACCT_SPLIT.md): compiles are pinned to QAIRT aihub_acct.QAIRT (2.45) and every graph passes
aihub_acct.guard_outputs before its first job (verdict per graph in the json under "guard"; a FAULT fails the chain).
Profile jobs are reused only when they ran on the pinned compile (from_compile).

    python aihub_ctl.py                     (loops until every chain is done or failed; resumable, gate-first)
    python aihub_ctl.py --smoke             (one chain, S24 Sudoku-MLP, 2 device steps + 3 finishing, acct2)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_acct as X  # noqa: E402
import aihub_infer as I  # noqa: E402
from aihub_breadth import device_os  # noqa: E402
from aihub_sched import named_outputs  # noqa: E402

CAP, FIN, SLOTS = 16, 6, 2
HCYC = {"sudoku_mlp": 3, "sudoku_attn": 3, "maze": 3}                 # device cycles per supervision step
RULES = {"sudoku_mlp": ("pat", 2), "sudoku_attn": ("native", 0), "maze": ("min", 2)}
S24, S25, E8 = "Samsung Galaxy S24", "Samsung Galaxy S25", "Snapdragon 8 Elite QRD"
CELLS = [("sudoku_mlp", "w8a8", "w8a16"), ("maze", "w8a8rope16", "fp32"), ("sudoku_attn", "w8a8", "w8a16")]
# Sep 29 00:40: neither "Snapdragon 8 Elite QRD" nor "Snapdragon 8 Elite Gen 5 QRD" is hosted any more (both accounts
# list only the X Elite / X2 Elite CRDs), so the 8 Elite silicon is covered by the Galaxy S25, run on acct3 (its qnn_dlc
# compiles are device-agnostic) so the two phones advance in parallel. Devices an account does not list are dropped.
PLAN = {"acct2": [(S24, *c) for c in CELLS], "acct3": [(S25, *c) for c in CELLS]}


def profile_row(p):
    """estimated per-step latency (ms) and compute-unit placement of a downloaded profile."""
    s = p["execution_summary"]
    units = {}
    for l in p.get("execution_detail", []):
        units[l.get("compute_unit", "?")] = units.get(l.get("compute_unit", "?"), 0) + 1
    row = {"ms": s["estimated_inference_time"] / 1000, "units": units,
           "peak_mb": s.get("estimated_inference_peak_memory", 0) / 2 ** 20,
           "all_times_ms": [t / 1000 for t in s.get("all_inference_times", [])][:200]}
    for k in ("compute_unit_breakdown", "runtime", "runtime_version", "qnn_version"):
        if k in s:
            row[k] = s[k]
    return row


def fires(task, q, k, run):
    """head read after device step k (0-based) for puzzles at cycle k+1; returns (fire mask, new run counters)."""
    base = q > 0
    kind, p = RULES[task]
    sup = (k + 1) // HCYC[task]                                          # supervision steps completed
    if kind == "native":
        return base, run
    if kind == "pat":
        run = np.where(base, run + 1, 0)
        return run >= p, run
    if kind == "min":
        return base & (sup >= p), run
    raise ValueError(kind)


class CtlChain:
    def __init__(self, acct, device, task, cheap, precise, cap=CAP, fin=FIN):
        self.a, self.device, self.task, self.cheap, self.precise = acct, device, task, cheap, precise
        self.cap, self.fin = cap, fin
        self.tag = f"{cheap}-{precise}_{device.replace(' ', '_')}_{acct.name}"
        d = I.idir(task)
        self.meta_path, self.state_path = d / f"ctl_{self.tag}.json", d / f"ctl_{self.tag}.npz"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else \
            {"task": task, "device": device, "account": acct.name, "cheap": cheap, "precise": precise, "cap": cap,
             "fin": fin, "H": HCYC[task], "rule": list(RULES[task]), "steps": [], "errors": [], "profile": {}}
        self._data, self._st, self._comp, self._tshapes, self._enc = None, None, {}, {}, {}

    # ------------------------------------------------------------------ state
    def save(self):
        self.meta_path.write_text(json.dumps(self.meta, indent=1))

    @property
    def done(self):
        return self.meta.get("done") or self.meta.get("failed")

    def data(self):
        if self._data is None:
            inp, lab = I.test_set(self.task)
            self._data = (inp, lab, I.initial_state(self.task))
        return self._data

    def state(self):
        if self._st is None:
            inp, lab, (zH0, zL0, _) = self.data()
            N, L = lab.shape
            if self.state_path.exists():
                self._st = {k: v for k, v in np.load(self.state_path).items()}
            else:
                self._st = {"zH": np.repeat(zH0, N, 0)[:, None], "zL": np.repeat(zL0, N, 0)[:, None],
                            "status": np.zeros(N, np.int8), "cheap_steps": np.zeros(N, np.int32),
                            "fin_steps": np.zeros(N, np.int32), "fired_at": np.full(N, -1, np.int32),
                            "run": np.zeros(N, np.int32), "cheap_preds": np.full((N, self.cap, L), -1, np.int32),
                            "cheap_q": np.full((N, self.cap), np.nan, np.float32),
                            "fin_preds": np.full((N, self.fin, L), -1, np.int32),
                            "fin_q": np.full((N, self.fin), np.nan, np.float32),
                            "exit_pred": np.full((N, L), -1, np.int32), "final_pred": np.full((N, L), -1, np.int32),
                            "lab": lab.astype(np.int32)}
        return self._st

    def save_state(self):
        np.savez_compressed(self.state_path, **self._st)

    # ------------------------------------------------------------------ graphs
    def compiled(self, prec, bs):
        """compile record of (task, prec) on this account, or None while it is still compiling / after failure."""
        comp = self.a.compile(self.task, prec, self.device, "qnn_dlc", prefer=self.pinned(prec))
        code = bs.get(comp["job_id"]) or self.a.C.get_job(comp["job_id"]).get_status().code
        if code == "FAILED":
            msg = (self.a.C.get_job(comp["job_id"]).get_status().message or "")[:600]
            self.meta["failed"] = f"compile {prec} failed: " + msg
            self.save()
            print(f"[ctl {self.task} {self.tag}] compile {prec} FAILED: {msg[:160]}", flush=True)
            return None
        if code != "SUCCESS":
            return None
        guard = self.meta.setdefault("guard", {})
        if (guard.get(prec) or {}).get("compile_job") != comp["job_id"]:
            try:
                ver, g = self.a.check(comp, self.task, prec)
            except RuntimeError as exc:
                guard[prec] = {"compile_job": comp["job_id"], "verdict": "fault", "error": str(exc)[:600]}
                self.meta["failed"] = f"guard {prec}: " + str(exc)[:600]
                self.save()
                print(f"FAULT [ctl {self.task} {self.tag}] {prec}: {str(exc)[:400]}", flush=True)
                return None
            guard[prec] = {**g, "qairt": comp.get("qairt"), "qairt_sdk_version": ver}
            self.save()
            if g["verdict"] != "match":
                print(f"WARN [ctl {self.task} {self.tag}] guard {prec} {g['verdict']} on {comp['job_id']} (QAIRT {ver})",
                      flush=True)
        return comp

    def pinned(self, prec):
        """compile job this chain passed the guard on (sticky for the chain's lifetime)."""
        return ((self.meta.get("guard") or {}).get(prec) or {}).get("compile_job")

    def graph(self, prec):
        if prec not in self._comp:
            cj = self.a.C.get_job(self.a.compile(self.task, prec, self.device, "qnn_dlc", prefer=self.pinned(prec))["job_id"])
            self._comp[prec] = cj
            self._tshapes[prec] = cj.get_target_shapes()
            self._enc[prec] = json.loads((I.idir(self.task) / f"io_enc_{prec}.json").read_text())["inputs"] \
                if prec != "fp32" else {}
        return self._comp[prec], self._tshapes[prec], self._enc[prec]

    def feeds(self, prec, idx):
        _, tshapes, enc = self.graph(prec)
        inp, _, (_, _, emb) = self.data()
        st = self.state()
        f = {"z_H": [st["zH"][i] for i in idx], "z_L": [st["zL"][i] for i in idx],
             "inputs": [inp[i:i + 1] for i in idx], "puzzle_embedding": [emb for _ in idx]}
        for name in list(f):
            want = tshapes[name][1]
            if want in ("uint8", "int8", "uint16", "int16"):
                e = dict(enc[name])
                if e["dtype"] != want:
                    assert {e["dtype"], want} == {"uint8", "int8"}, (name, e, want)
                    e = {"scale": e["scale"], "zero_point": e["zero_point"] + (-128 if want == "int8" else 128),
                         "dtype": want}
                f[name] = [I.quantize_input(x, e) for x in f[name]]
        return {k: f[k] for k in tshapes}

    # ------------------------------------------------------------------ jobs
    def inflight(self, bs):
        """job ids of this chain not yet terminal (inference + profile)."""
        out = []
        cur = self.meta.get("current") or {}
        for kind in ("cheap", "precise"):
            j = cur.get("jobs", {}).get(kind)
            if j and bs.get(j["job_id"], "PENDING") not in ("SUCCESS", "FAILED"):
                out.append(j["job_id"])
        for prec, p in self.meta["profile"].items():
            if p.get("job_id") and "ms" not in p and bs.get(p["job_id"], "PENDING") not in ("SUCCESS", "FAILED"):
                out.append(p["job_id"])
        return out

    def submit(self, kind, k, idx):
        prec = self.cheap if kind == "cheap" else self.precise
        cj, _, _ = self.graph(prec)
        ij = self.a.C.submit_inference_job(model=cj.get_target_model(),
                                           device=self.a.hub.Device(self.device, X.OS[self.device]),
                                           inputs=self.feeds(prec, idx),
                                           name=f"x3_{self.a.name}_ctl_{self.task}_{self.tag}_s{k}_{kind}")
        self.a.add(kind="inference", ctl=True, task=self.task, prec=prec, device=self.device, runtime="qnn_dlc",
                   step=k, graph_role=kind, n=len(idx), job_id=ij.job_id, from_compile=cj.job_id)
        return {"job_id": ij.job_id, "idx": [int(i) for i in idx], "prec": prec}

    def profile(self, prec, bs, free):
        """one profile job per graph on this device; reuse the account's record if it has one. Returns slots used."""
        p = self.meta["profile"].get(prec)
        if p is None:
            cj0 = self.a.compile(self.task, prec, self.device, "qnn_dlc", prefer=self.pinned(prec))
            have = self.a.find(kind="profile", task=self.task, prec=prec, device=self.device, from_compile=cj0["job_id"])
            if have:
                self.meta["profile"][prec] = {"job_id": have[-1]["job_id"], "reused": True}
                self.save()
                return 0
            if free <= 0:
                return 0
            cj, _, _ = self.graph(prec)
            pj = self.a.C.submit_profile_job(model=cj.get_target_model(),
                                             device=self.a.hub.Device(self.device, X.OS[self.device]),
                                             name=f"x3_{self.a.name}_prof_{self.task}_stepc_{prec}_{self.device}")
            self.a.add(kind="profile", task=self.task, prec=prec, device=self.device, runtime="qnn_dlc",
                       job_id=pj.job_id, from_compile=cj.job_id)
            self.meta["profile"][prec] = {"job_id": pj.job_id}
            self.save()
            print(f"[ctl {self.task} {self.tag}] profile {prec} -> {pj.job_id}", flush=True)
            return 1
        if "ms" in p or p.get("error"):
            return 0
        code = bs.get(p["job_id"]) or self.a.C.get_job(p["job_id"]).get_status().code
        if code == "SUCCESS":
            row = profile_row(self.a.C.get_job(p["job_id"]).download_profile())
            p.update(ms=row["ms"], summary=row)
            self.save()
            print(f"[ctl {self.task} {self.tag}] profile {prec}: {row['ms']:.3f} ms per step", flush=True)
        elif code == "FAILED":
            p["error"] = (self.a.C.get_job(p["job_id"]).get_status().message or "")[:300]
            self.save()
        return 0

    # ------------------------------------------------------------------ ingest
    def ingest(self, kind, rec):
        job = self.a.C.get_job(rec["job_id"])
        o = named_outputs(job.download_output_data())
        idx = np.asarray(rec["idx"])
        st = self.state()
        _, lab, _ = self.data()
        zH = np.stack([x.astype(np.float32) for x in o["z_H_out"]])
        zL = np.stack([x.astype(np.float32) for x in o["z_L_out"]])
        lg = np.concatenate([x.astype(np.float32) for x in o["logits"]])
        q = np.array([float(x.reshape(-1)[0]) for x in o["q_halt"]], np.float32)
        assert zH.shape[0] == len(idx) == lg.shape[0], (self.tag, kind, zH.shape, len(idx), lg.shape)
        pred = lg.argmax(-1).astype(np.int32)
        st["zH"][idx], st["zL"][idx] = zH, zL
        if kind == "cheap":
            k = int(st["cheap_steps"][idx[0]])
            assert (st["cheap_steps"][idx] == k).all() and (st["status"][idx] == 0).all(), (self.tag, k)
            st["cheap_preds"][idx, k], st["cheap_q"][idx, k] = pred, q
            st["cheap_steps"][idx] = k + 1
            if (k + 1) % HCYC[self.task] == 0:
                f, st["run"][idx] = fires(self.task, q, k, st["run"][idx])
            else:
                f = np.zeros(len(idx), bool)
            capped = (k + 1 >= self.cap) & ~f
            leave = f | capped
            st["fired_at"][idx[f]] = k + 1
            st["exit_pred"][idx[leave]] = pred[leave]
            st["status"][idx[leave]] = 1 if self.fin > 0 else 2
            if self.fin == 0:
                st["final_pred"][idx[leave]] = pred[leave]
            ex = float((pred == lab[idx]).all(-1).mean())
            print(f"[ctl {self.task} {self.tag}] cheap step {k}: n {len(idx)} exact {ex:.3f} fired {int(f.sum())}"
                  f" capped {int(capped.sum())}", flush=True)
        else:
            fs = st["fin_steps"][idx]
            assert (st["status"][idx] == 1).all() and (fs < self.fin).all(), (self.tag, fs)
            st["fin_preds"][idx, fs], st["fin_q"][idx, fs] = pred, q
            st["fin_steps"][idx] = fs + 1
            fin = fs + 1 >= self.fin
            st["final_pred"][idx[fin]] = pred[fin]
            st["status"][idx[fin]] = 2
            ex = float((pred == lab[idx]).all(-1).mean())
            print(f"[ctl {self.task} {self.tag}] precise: n {len(idx)} exact {ex:.3f} finished {int(fin.sum())}",
                  flush=True)
        self.save_state()

    def summary(self):
        st = self.state()
        lab = st["lab"]
        ok = lambda p: float((p == lab).all(-1).mean())  # noqa: E731
        fired = st["fired_at"] >= 0
        h = HCYC[self.task]
        j1 = st["fin_preds"][:, h - 1] if self.fin >= h else st["final_pred"]
        finish_fired = np.where(fired[:, None], st["final_pred"], st["exit_pred"])
        return {"n": int(len(lab)), "fired_frac": float(fired.mean()), "cheap_steps_mean": float(st["cheap_steps"].mean()),
                "fin_steps_mean": float(st["fin_steps"].mean()),
                "exact_exit_only": ok(st["exit_pred"]), "exact_finish_all_j2": ok(st["final_pred"]),
                "exact_finish_all_j1": ok(j1), "exact_finish_fired_only_j2": ok(finish_fired),
                "exact_cheap_per_step": [float((st["cheap_preds"][:, s] == lab).all(-1).mean())
                                         for s in range(self.cap)],   # -1 rows (left) count as wrong
                "device_steps": len(self.meta["steps"]),
                "profile_ms": {p: v.get("ms") for p, v in self.meta["profile"].items()}}

    # ------------------------------------------------------------------ one loop pass
    def step(self, bs, free):
        """advance this chain given the account's job statuses; returns the number of device slots it took."""
        if self.done:
            return 0
        used = 0
        cur = self.meta.get("current")
        if cur:
            for kind in list(cur["jobs"]):
                j = cur["jobs"][kind]
                code = bs.get(j["job_id"]) or self.a.C.get_job(j["job_id"]).get_status().code
                if code == "SUCCESS" and not j.get("ingested"):
                    self.ingest(kind, j)
                    j["ingested"] = True
                    self.save()
                elif code == "FAILED":
                    msg = (self.a.C.get_job(j["job_id"]).get_status().message or "")[:600]
                    self.meta["errors"].append({"step": cur["step"], "kind": kind, "job_id": j["job_id"], "error": msg})
                    cur["jobs"].pop(kind)
                    cur["todo"].append(kind)                       # resubmit next pass
                    if len(self.meta["errors"]) >= 3:
                        self.meta["failed"] = msg
                    self.save()
                    print(f"[ctl {self.task} {self.tag}] step {cur['step']} {kind} FAILED: {msg[:160]}", flush=True)
                    if self.done:
                        return 0
            if not cur["todo"] and all(j.get("ingested") for j in cur["jobs"].values()):
                self.meta["steps"].append({"step": cur["step"], **{k: {"job_id": j["job_id"], "n": len(j["idx"])}
                                                                    for k, j in cur["jobs"].items()}})
                self.meta.pop("current")
                cur = None
                self.save()
        if cur is None:
            for prec in (self.cheap, self.precise):
                if self.compiled(prec, bs) is None:
                    return 0
            st = self.state()
            if (st["status"] == 2).all():
                self.meta["done"] = True
                self.meta["summary"] = self.summary()
                self.save()
                print(f"[ctl {self.task} {self.tag}] DONE {json.dumps(self.meta['summary'])}", flush=True)
                return 0
            todo = [k for k, s in (("cheap", 0), ("precise", 1)) if (st["status"] == s).any()]
            cur = {"step": len(self.meta["steps"]), "jobs": {}, "todo": todo}
            self.meta["current"] = cur
            self.save()
        for prec in (self.cheap, self.precise):      # both graphs compiled on the pin and past the guard before any job
            if self.compiled(prec, bs) is None:
                return used
        for prec in (self.cheap, self.precise):
            n = self.profile(prec, bs, free - used)
            used += n
        st = self.state()
        for kind in list(cur["todo"]):
            if free - used <= 0:
                break
            idx = np.nonzero(st["status"] == (0 if kind == "cheap" else 1))[0]
            cur["jobs"][kind] = self.submit(kind, cur["step"], idx)
            cur["todo"].remove(kind)
            used += 1
            self.save()
            print(f"[ctl {self.task} {self.tag}] step {cur['step']} {kind} n {len(idx)} -> {cur['jobs'][kind]['job_id']}",
                  flush=True)
        return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accts", default="acct2,acct3")
    ap.add_argument("--smoke", action="store_true", help="one chain: acct2, S24, Sudoku-MLP, cap 2, fin 3")
    ap.add_argument("--sleep", type=int, default=60)
    a = ap.parse_args()
    accts = {n: X.Acct(n) for n in a.accts.split(",")}
    for acct in accts.values():
        X.OS.update(device_os(acct.C))
    if a.smoke:
        chains = [CtlChain(accts["acct2"], S24, "sudoku_mlp", "w8a8", "w8a16", cap=2, fin=3)]
    else:
        chains = [CtlChain(accts[n], *spec) for n, plan in PLAN.items() if n in accts for spec in plan]
    hosted = {n: set(device_os(acct.C)) for n, acct in accts.items()}
    for c in [c for c in chains if c.device not in hosted[c.a.name]]:
        print(f"[ctl] DROPPED {c.task}/{c.tag}: {c.device} is not hosted on {c.a.name}", flush=True)
        chains.remove(c)
    print(f"[ctl] {len(chains)} chains on {list(accts)}: " + ", ".join(f"{c.task}/{c.tag}" for c in chains), flush=True)
    while True:
        for acct in accts.values():
            bs = acct.status()
            mine = [c for c in chains if c.a is acct]
            free = SLOTS - sum(len(c.inflight(bs)) for c in mine)
            for c in mine:
                try:
                    free -= c.step(bs, free)
                except Exception as exc:  # noqa: BLE001
                    print(f"[ctl {c.task} {c.tag}] ERROR {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        active = [c for c in chains if not c.done]
        done = [c for c in chains if c.meta.get("done")]
        failed = [c for c in chains if c.meta.get("failed")]
        print(f"[ctl] {time.strftime('%H:%M:%S', time.gmtime())} active {len(active)} done {len(done)} failed {len(failed)}",
              flush=True)
        if not active:
            break
        time.sleep(a.sleep)


if __name__ == "__main__":
    main()
