#!/usr/bin/env python3
"""Priority scheduler for AI Hub device jobs (profiles and on-device inference steps).

Measured on Sep 24 (19:04-19:45 UTC): this account runs about two device jobs at a time, strictly in
submission order, and a scarce device (QCS8550 proxy, IQ-9075) can sit in PROVISIONING_DEVICE for
30+ minutes while everything behind it waits. Submitting every profile at once therefore puts the
on-device inference chains (16 sequential jobs each) behind hundreds of profiles. This loop keeps at
most `--depth` of our device jobs outstanding and fills the slots by priority:
  0  the next step of an on-device inference chain (aihub_infer.py state files)
  1+ profiles from PLAN, in tier order, scarce devices last within a tier.
A profile uses the first finished compile of its (task, graph, H, precision) for the device's class
(qnn_dlc and tflite targets are device-agnostic). Everything is recorded in the same jobs.json /
infer state files as aihub_h1.py / aihub_infer.py, so the scheduler can be killed and restarted.

    python aihub_sched.py --depth 3            (runs until STOP_SCHED exists or nothing is left)
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402
import aihub_infer as I  # noqa: E402

QNN = [d[0] for d in A.DEVICES if d[2]]
TFL = [d[0] for d in A.DEVICES if not d[2]]
SCARCE = ["QCS8550 (Proxy)", "Dragonwing IQ-9075 EVK", "Arduino VENTUNO Q", "SA8295P ADP", "SA8775P ADP",
          "SA7255P ADP", "Dragonwing Q-6690 MTP"]
CORE6 = ["Samsung Galaxy S24", "Dragonwing RB3 Gen 2 Vision Kit", "Samsung Galaxy S21", "Snapdragon X Elite CRD",
         "Samsung Galaxy S26", "SA8775P ADP"]
GEN10 = ["Samsung Galaxy S24", "Dragonwing RB3 Gen 2 Vision Kit", "Samsung Galaxy S21", "Samsung Galaxy S22 5G",
         "Samsung Galaxy S23", "Snapdragon 7 Gen 4 QRD", "Samsung Galaxy S25", "Samsung Galaxy S26",
         "Snapdragon X Elite CRD", "SA8295P ADP"]
GRANVARS = ["w4a8_t", "w8a8_t", "w4a8_g32", "w4a8_g128", "w4a8_l64", "w4a8_l32", "w4a16_g32", "w4a16_l64"]
# on-device correctness chains (task, device, precision); 8 supervision steps each (NSTEPS below)
# Order = priority. The S24 w8a8 Sudoku-MLP chain comes first: whether the device collapses like the host
# QDQ simulation (0/100; localised to the RMSNorm variance/rsqrt tensors, which QNN fuses into one op on
# device) decides what the paper may say about device INT8 correctness (REWRITE_NOTES 29).
CHAINS = [("sudoku_mlp", "Samsung Galaxy S24", "w8a8"), ("sudoku_mlp", "Samsung Galaxy S24", "w8a16"),
          ("sudoku_attn", "Samsung Galaxy S24", "w8a8"),
          ("sudoku_mlp", "Samsung Galaxy S24", "w8a8rms16"),   # wa_collapse B2 fix: RMSNorm at uint16
          ("sudoku_mlp", "Samsung Galaxy S24", "w4a16"), ("sudoku_mlp", "Dragonwing RB3 Gen 2 Vision Kit", "w4a16"),
          ("sudoku_mlp", "Samsung Galaxy S24", "fp32"),
          ("maze", "Samsung Galaxy S24", "w4a16"), ("maze", "Dragonwing RB3 Gen 2 Vision Kit", "w4a16"),
          ("maze", "Samsung Galaxy S24", "fp32"), ("maze", "Samsung Galaxy S24", "w8a8"),
          ("sudoku_attn", "Samsung Galaxy S24", "w4a16"), ("sudoku_attn", "Samsung Galaxy S24", "w8a8")]
CHAINS = list(dict.fromkeys(CHAINS))   # a duplicated entry gave two Chain objects one state file (Sep 25 fix)
# W4A16 retired Sep 25 02:40 UTC: device outputs diverge from the host QDQ graph from step 1 (X3.md)
CHAINS = [c for c in CHAINS if c[2] != "w4a16"]
NSTEPS = {"sudoku_mlp": 8, "sudoku_attn": 8, "maze": 8}


def named_outputs(o):
    """qnn_dlc targets return outputs as output_0..output_3 in graph order (z_H_out, z_L_out,
    logits, q_halt); map them back to the ONNX names."""
    names = ["z_H_out", "z_L_out", "logits", "q_halt"]
    if all(n in o for n in names):
        return o
    return {names[i]: o[f"output_{i}"] for i in range(4)}


def order(devs):
    return [d for d in devs if d not in SCARCE] + [d for d in devs if d in SCARCE]


def plan():
    P = []
    t = 1
    for task in ("maze", "sudoku_mlp"):                                  # T1: Q4a core
        for prec in ("fp32", "w4a8", "w8a8"):
            P += [(t, task, "step", 1, prec, d) for d in order(QNN)]
    t += 1
    for task in ("maze", "sudoku_mlp"):                                  # T2: H=3 same-day ratio
        for prec in ("fp32", "w8a8", "w4a8"):
            P += [(t, task, "step", 3, prec, d) for d in CORE6]
    t += 1
    for task in ("maze", "sudoku_mlp"):                                  # T3: TF-Lite-only phones
        P += [(t, task, "step", 1, "fp32", d) for d in TFL]
    for task in ("sudoku_mlp", "sudoku_attn", "maze"):                   # T3: our own INT8 / W4A8 TFLite
        for prec in ("tfl_int8", "tfl_w4a8"):                            # (AI Edge Quantizer, mcu_h1.py)
            P += [(t, task, "step", 1, prec, d) for d in TFL + ["Samsung Galaxy S21", "Dragonwing RB3 Gen 2 Vision Kit",
                                                                 "Samsung Galaxy S24"]]
    t += 1
    for task in ("maze", "sudoku_mlp"):                                  # T4: A16 on every HTP
        for prec in ("w4a16", "w8a16"):
            P += [(t, task, "step", 1, prec, d) for d in order(QNN)]
    t += 1
    for task in ("maze", "sudoku_mlp"):                                  # T5: granularity matrix
        for prec in GRANVARS:
            P += [(t, task, "step", 1, prec, d) for d in GEN10]
    t += 1
    for task in ("sudoku_attn", "arc"):                                  # T6: the other two tasks
        for prec in ("fp32", "w4a8", "w8a8", "w4a16"):
            P += [(t, task, "step", 1, prec, d) for d in CORE6]
    t += 1
    for task in ("maze", "sudoku_mlp", "sudoku_attn", "arc"):            # T7: deployable step, f
        for prec in ("fp32", "w8a8", "w4a16"):
            P += [(t, task, "stepc", 1, prec, d) for d in ("Samsung Galaxy S24", "Dragonwing RB3 Gen 2 Vision Kit")]
    for task in ("maze", "sudoku_mlp", "sudoku_attn"):
        for prec in ("fp32", "w8a8", "w4a8"):
            P += [(t, task, "f", 1, prec, d) for d in ("Samsung Galaxy S24", "Dragonwing RB3 Gen 2 Vision Kit",
                                                         "Snapdragon X Elite CRD")]
    t += 1
    for task in ("sudoku_attn", "arc"):                                  # T8: the rest
        for prec in ("fp32", "w4a8", "w8a8", "w4a16", "w8a16"):
            P += [(t, task, "step", 1, prec, d) for d in order(QNN) if d not in CORE6]
        for prec in GRANVARS:
            P += [(t, task, "step", 1, prec, d) for d in GEN10]
        for prec in ("fp32", "w8a8", "w4a8"):
            P += [(t, task, "step", 3, prec, d) for d in CORE6]
    seen, out = set(), []
    for p in P:
        if p[1:] not in seen:
            seen.add(p[1:])
            out.append(p)
    return out


def source_compile(rec, graph, H, prec, cls, bs, device=None):
    """First finished compile of (graph, H, prec) for the runtime class; prefer the device's own."""
    cands = [j for j in rec["jobs"] if j["kind"] == "compile" and (j.get("job_id") or j.get("model_id"))
             and j["graph"] == graph and j["H"] == H and j["prec"] == prec
             and (("tflite" in j["runtime"]) == (cls == "tflite"))]
    cands.sort(key=lambda j: j["device"] != device)
    for j in cands:
        if j.get("model_id") and not j.get("job_id"):
            return j            # a directly uploaded target model (e.g. our own TFLite flatbuffer)
        if bs.get(j["job_id"], (None, None))[1] == "SUCCESS":
            return j
    return None


def target_model(src):
    import qai_hub as hub
    if src.get("model_id") and not src.get("job_id"):
        return hub.get_model(src["model_id"])
    return hub.get_job(src["job_id"]).get_target_model()


class Chain:
    def __init__(self, task, device, prec):
        self.task, self.device, self.prec = task, device, prec
        self.tag = f"{prec}_{device.replace(' ', '_')}"
        self.d = I.idir(task)
        self.meta_path = self.d / f"dev_{self.tag}.json"
        self.state_path = self.d / f"dev_{self.tag}.npz"
        self.meta = json.loads(self.meta_path.read_text()) if self.meta_path.exists() else \
            {"steps": [], "jobs": [], "n_steps": NSTEPS[task]}
        self._data = None

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

    def poll(self, bs):
        """Handle the in-flight job. Returns True if a job is still outstanding."""
        import qai_hub as hub
        cur = self.meta.get("current")
        if not cur:
            return False
        code = bs.get(cur["job_id"], (None, None))[1] or hub.get_job(cur["job_id"]).get_status().code
        if code not in ("SUCCESS", "FAILED"):
            return True
        job = hub.get_job(cur["job_id"])
        if code == "FAILED":
            msg = (job.get_status().message or "")[:600]
            self.meta.setdefault("errors", []).append({"step": cur["step"], "job_id": cur["job_id"], "error": msg})
            self.meta.pop("current")
            if len(self.meta["errors"]) >= 3:
                self.meta["failed"] = msg
            self.save()
            print(f"[chain {self.task} {self.tag}] step {cur['step']} FAILED: {msg[:160]}", flush=True)
            return False
        o = named_outputs(job.download_output_data())
        inp, lab, _ = self.data()
        zH = np.stack([x.astype(np.float32) for x in o["z_H_out"]])
        zL = np.stack([x.astype(np.float32) for x in o["z_L_out"]])
        lg = np.concatenate([x.astype(np.float32) for x in o["logits"]])
        q = np.array([float(x.reshape(-1)[0]) for x in o["q_halt"]])
        if self.state_path.exists():
            st = np.load(self.state_path)
            preds, qs = list(st["preds"]), list(st["q"])
        else:
            preds, qs = [], []
        preds.append(lg.argmax(-1))
        qs.append(q)
        np.savez_compressed(self.state_path, zH=zH, zL=zL, preds=np.stack(preds), q=np.stack(qs))
        ex = float((preds[-1] == lab).all(-1).mean())
        self.meta["steps"].append({"step": cur["step"], "job_id": cur["job_id"], "exact": ex,
                                   "cell": float((preds[-1] == lab).mean())})
        self.meta.pop("current")
        if len(preds) >= NSTEPS[self.task]:
            self.meta["done"] = True
        self.save()
        print(f"[chain {self.task} {self.tag}] step {cur['step']}: exact {ex:.3f}", flush=True)
        return False

    def submit(self, rec, bs):
        import qai_hub as hub
        cls = "qnn"
        src = source_compile(rec, "stepc", 1, self.prec, cls, bs, self.device)
        if src is None:
            return False
        cj = hub.get_job(src["job_id"])
        tshapes = cj.get_target_shapes()
        inp, lab, (zH0, zL0, emb) = self.data()
        N = inp.shape[0]
        if self.state_path.exists():
            st = np.load(self.state_path)
            zH, zL, s = st["zH"], st["zL"], st["preds"].shape[0]
        else:
            zH, zL, s = np.repeat(zH0, N, 0)[:, None], np.repeat(zL0, N, 0)[:, None], 0
        enc = json.loads((self.d / f"io_enc_{self.prec}.json").read_text())["inputs"] if self.prec != "fp32" else {}
        feeds = {"z_H": [zH[i] for i in range(N)], "z_L": [zL[i] for i in range(N)],
                 "inputs": [inp[i:i + 1] for i in range(N)], "puzzle_embedding": [emb for _ in range(N)]}
        for name in list(feeds):
            want = tshapes[name][1]
            if want in ("uint8", "int8", "uint16", "int16"):
                assert enc[name]["dtype"] == want, (name, enc[name], want)
                feeds[name] = [I.quantize_input(x, enc[name]) for x in feeds[name]]
        feeds = {k: feeds[k] for k in tshapes}
        ij = hub.submit_inference_job(model=cj.get_target_model(), device=hub.Device(self.device, A.DEV_BY_NAME[self.device][1]),
                                      inputs=feeds, name=f"x3_infer_{self.task}_{self.tag}_s{s}")
        A.add_job(self.task, {"kind": "inference", "graph": "stepc", "H": 1, "prec": self.prec, "device": self.device,
                              "os": A.DEV_BY_NAME[self.device][1], "step": s, "job_id": ij.job_id,
                              "from_compile": src["job_id"],
                              "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        self.meta["current"] = {"step": s, "job_id": ij.job_id, "from_compile": src["job_id"]}
        self.meta["jobs"].append({"step": s, "job_id": ij.job_id})
        self.save()
        print(f"[chain {self.task} {self.tag}] submitted step {s} -> {ij.job_id}", flush=True)
        return True


def outstanding(bs, ours):
    return sum(1 for jid in ours if bs.get(jid, (None, "SUCCESS"))[1] not in ("SUCCESS", "FAILED"))


def main():
    import qai_hub as hub
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--sleep", type=int, default=60)
    ap.add_argument("--no_chains", action="store_true")
    a = ap.parse_args()
    P = plan()
    print(f"[sched] plan: {len(P)} profile items, {len(CHAINS)} chains", flush=True)
    chains = [] if a.no_chains else [Chain(*c) for c in CHAINS]
    while not (A.ART.parent.parent / "x3" / "STOP_SCHED").exists():
        bs = A.bulk_status()
        recs = {t: A.load_rec(t) for t in A.TASKS}
        ours = [j["job_id"] for r in recs.values() for j in r["jobs"]
                if j["kind"] in ("profile", "inference") and j.get("job_id")]
        for c in chains:
            if not c.done:
                c.poll(bs)
        free = a.depth - outstanding(bs, ours)
        submitted = 0
        for c in chains:          # chains are never held back by the profile depth limit
            if c.done or c.meta.get("current"):
                continue
            if c.submit(recs[c.task], bs):
                free -= 1
                submitted += 1
        have = {(t, j["graph"], j["H"], j["prec"], j["device"]) for t, r in recs.items() for j in r["jobs"]
                if j["kind"] == "profile" and j.get("job_id")}
        left = 0
        for (tier, task, graph, H, prec, dev) in P:
            if (task, graph, H, prec, dev) in have:
                continue
            left += 1
            if free <= 0:
                continue
            d = A.DEV_BY_NAME[dev]
            cls = "tflite" if (prec.startswith("tfl_") or not d[2]) else "qnn"
            src = source_compile(recs[task], graph, H, prec, cls, bs, dev)
            if src is None:
                continue
            try:
                pj = hub.submit_profile_job(model=target_model(src),
                                            device=hub.Device(dev, d[1]),
                                            name=f"x3_prof_{task}_{graph}_H{H}_{prec}_{dev}")
                pid, err = pj.job_id, None
            except Exception as exc:  # noqa: BLE001
                pid, err = None, str(exc)[:300]
            A.add_job(task, {"kind": "profile", "graph": graph, "H": H, "prec": prec, "device": dev, "os": d[1],
                             "runtime": src["runtime"], "job_id": pid, "submit_error": err,
                             "from_compile": src.get("job_id") or src.get("model_id"),
                             "fanout_from_device": src["device"], "tier": tier,
                             "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            have.add((task, graph, H, prec, dev))
            free -= 1
            submitted += 1
            print(f"[sched] T{tier} profile {task} {graph} H{H} {prec:9s} {dev:32s} -> {pid or err[:80]}", flush=True)
        n_chain = sum(not c.done for c in chains)
        print(f"[sched] {time.strftime('%H:%M:%S', time.gmtime())} outstanding {outstanding(bs, ours)} "
              f"submitted {submitted} plan-left {left} chains-active {n_chain}", flush=True)
        if left == 0 and n_chain == 0:
            break
        time.sleep(a.sleep)


if __name__ == "__main__":
    main()
