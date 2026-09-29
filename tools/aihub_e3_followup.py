#!/usr/bin/env python3
"""E3 follow-ups to E3_DIAG_V79_CLASSA.md (reports/edgetrm_exp/E3_FOLLOWUP.md), acct2 + acct3, QAIRT pinned at 2.45
(aihub_acct.QAIRT). Tokens are never read or printed here: qai_hub.Client(profile=...) loads them. Runs on p4d from ~/x3.

(1) hostref: the host reference column of the fault-table graphs in ORT *reference* QDQ semantics
    (graph_optimization_level = ORT_DISABLE_ALL: every Q/DQ pair executed as written, which is what the devices follow),
    written beside the existing host_<prec>.npz (made with ORT's default, fused QDQ kernels) as host_<prec>_ref.npz in
    the same layout (preds per step, q, final zH / zL) plus host_<prec>_ref.json; nothing is overwritten.
    classdiff: every completed device chain of those graphs classed against the fused and against the reference
    column (the fault table's PASS >= 0.95 / FAIL < 0.5 cuts on cell s2 and exact s8), and the rows that change.
(2) build: Maze w8a8 with the RMSNorm pattern broken at export (aihub_v79_classa.break_rmsnorm, N from shape
    inference = 512 for the Maze norms), maze/qdq_w8a8/variant_CA_nofuse2.onnx; compile on acct2 (qairt-converter
    path, no float fallback expected); run on the S23 (v73) and S25 (v79), two steps then eight.
(3) The Sudoku-MLP nofuse2 graph on the S21 to eight steps, on the 7 Gen 4 QRD (acct2), RB3 Gen 2 and Q-6690 (acct3):
    two-step probe first everywhere (run --steps 2), then --steps 8 on the named chains.
(4) bisect (only if jobs remain): nofuse variants of the Maze graph with the MatMul dtype rule respected.

    python aihub_e3_followup.py build
    python aihub_e3_followup.py hostref sudoku_mlp:w8a8 [--threads 16]          # one (task:prec) per process
    python aihub_e3_followup.py hostfused maze:w8a8rope16                        # default-ORT column where none exists
    python aihub_e3_followup.py compile [acct2|acct3] [task:prec ...]
    python aihub_e3_followup.py run --steps 2 [--only tag ...]                   # every chain in CHAINS, gate first
    python aihub_e3_followup.py run --steps 8 --only maze:w8a8_nofuse2:S23 ...
    python aihub_e3_followup.py status
    python aihub_e3_followup.py classdiff > classdiff.md
"""
import argparse
import glob
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_acct as X  # noqa: E402
import aihub_infer as I  # noqa: E402
import aihub_v79_classa as V  # noqa: E402  (break_rmsnorm, annotate; installs its own source_path for its PRECS)

S21, S23, S25 = "Samsung Galaxy S21", "Samsung Galaxy S23", "Samsung Galaxy S25"
G7, RB3, Q6690 = "Snapdragon 7 Gen 4 QRD", "Dragonwing RB3 Gen 2 Vision Kit", "Dragonwing Q-6690 MTP"
SHORT = {S21: "S21", S23: "S23", S25: "S25", G7: "7Gen4", RB3: "RB3", Q6690: "Q6690"}

# (task, prec) -> (dir under <task>/infer, onnx file). w8a8_nofuse2 = the qdq_w8a8 graph with every RMSNorm's
# ReduceMean rewritten to ReduceSum * 1/N (both quantized), so the qairt-converter has no RMSNorm to fuse and nothing
# falls back to float; the graph's arithmetic is unchanged (host reference identical to the w8a8 graph).
SRC = {("maze", "w8a8_nofuse2"): ("qdq_w8a8", "variant_CA_nofuse2.onnx"),
       ("sudoku_mlp", "w8a8_nofuse2"): ("qdq_w8a8", "variant_CA_nofuse2.onnx"),
       # (4) v79 bisect on the nofuse family, MatMul dtype rule respected (both inputs of q.k^T and of softmax.V in one
       # dtype set): attn16 = RoPE (q, k), scores, softmax and every matmul output (so V) at 16 bit, the rest 8 bit;
       # w8a16_nofuse2 = the w8a16 graph itself with the RMSNorm pattern broken (all-16 control).
       ("maze", "w8a8_nofuse2_attn16"): ("qdq_w8a16", "variant_CA_nofuse2_attn16.onnx"),
       ("maze", "w8a16_nofuse2"): ("qdq_w8a16", "variant_CA_nofuse2_w8a16.onnx")}
BISECT = {"w8a8_nofuse2_attn16": dict(eight={"attn_rope", "attn_rope_const", "attn_concat", "attn_scores", "attn", "linear_out"},
                                     sixteen_default=False, propagate_movement=True),
          "w8a16_nofuse2": None}   # None: start from qdq_w8a16/model.onnx
# breadth-layout chains: (acct, task, prec, device)
CHAINS = [("acct2", "maze", "w8a8_nofuse2", S23), ("acct2", "maze", "w8a8_nofuse2", S25),
          ("acct2", "sudoku_mlp", "w8a8_nofuse2", S21), ("acct2", "sudoku_mlp", "w8a8_nofuse2", G7),
          ("acct3", "sudoku_mlp", "w8a8_nofuse2", RB3), ("acct3", "sudoku_mlp", "w8a8_nofuse2", Q6690),
          ("acct2", "maze", "w8a8_nofuse2_attn16", S25), ("acct2", "maze", "w8a16_nofuse2", S25)]
# fault-table graphs whose host column is regenerated in reference semantics
HOSTREF = [("sudoku_mlp", "w8a8"), ("sudoku_mlp", "w8a16"), ("sudoku_attn", "w8a8"), ("maze", "w8a8"),
           ("maze", "w8a8rope16")]
PASS, FAIL = 0.95, 0.5
TRANSIENT_MAX = 6


def tag(acct, task, prec, dev):
    return f"{task}:{prec}:{SHORT.get(dev, dev)}@{acct}"


# ------------------------------------------------------------------ source_path (keyed on task AND prec)
_prev_source_path = X.source_path


def source_path(task, prec):
    if (task, prec) not in SRC:
        return _prev_source_path(task, prec)
    base, fname = SRC[(task, prec)]
    d = I.idir(task)
    up = d / f"upload_{prec}"
    if not (up / "model.onnx").exists():
        up.mkdir(exist_ok=True)
        for f in up.iterdir():
            f.unlink()
        os.link(d / base / "model.data", up / "model.data")
        os.link(d / base / fname, up / "model.onnx")
    if not (d / f"io_enc_{prec}.json").exists():
        X.write_io_enc(up / "model.onnx", d / f"io_enc_{prec}.json")
    return up


X.source_path = source_path


def break_rmsnorm(m):
    """aihub_v79_classa.break_rmsnorm for graphs whose ReduceMean carries `axes` as an input (opset >= 18: the
    qdq_w8a16-based variants are opset 21) as well as as an attribute (opset 17, the qdq_w8a8 graphs)."""
    import onnx
    from onnx import helper, numpy_helper
    g = m.graph
    import wa_collapse as W
    vals = W.const_map(g)                     # initializers and Constant-node outputs
    for n in g.node:
        if n.op_type == "ReduceMean" and not any(a.name == "axes" for a in n.attribute) and len(n.input) > 1:
            axes = numpy_helper.to_array(vals[n.input[1]]).astype(np.int64).tolist()
            del n.input[1:]
            n.attribute.append(helper.make_attribute("axes", axes))
    return V.break_rmsnorm(m)


def graph_path(task, prec):
    if prec == "fp32":
        return V.I.idir(task).parent / "stepc_H1.onnx"
    if (task, prec) in SRC:
        base, fname = SRC[(task, prec)]
        return I.idir(task) / base / fname
    if prec in V.PRECS:
        _, base, fname, _ = V.PRECS[prec]
        return I.idir(task) / base / fname
    p = X.source_path(task, prec)
    return p / "model.onnx" if p.is_dir() else p


# ------------------------------------------------------------------ build
def build(names):
    import onnx
    from onnx import numpy_helper
    for task, prec in SRC:
        if names and f"{task}:{prec}" not in names:
            continue
        d = I.idir(task)
        base, fname = SRC[(task, prec)]
        out = d / base / fname
        if out.exists() and task == "sudoku_mlp":
            print(task, prec, "exists (built by aihub_v79_classa build):", out, flush=True)
            continue
        if prec in BISECT:
            import wa_collapse as W
            if BISECT[prec] is None:
                m = onnx.load(str(d / base / "model.onnx"), load_external_data=False)
            else:
                info = W.build_variant(task, f"CA_{prec}_base", **BISECT[prec])   # qdq_w8a16/variant_CA_<prec>_base.onnx
                print(task, prec, "base variant:", json.dumps(info), flush=True)
                m = onnx.load(str(d / base / f"variant_CA_{prec}_base.onnx"), load_external_data=False)
        else:
            m = onnx.load(str(d / base / "model.onnx"), load_external_data=False)
        n = break_rmsnorm(m)
        onnx.save(m, str(out))
        Ns = sorted({int(round(1.0 / float(numpy_helper.to_array(i)))) for i in m.graph.initializer
                     if i.name.endswith("_invN")})
        print(task, prec, "rewrote", n, "ReduceMean nodes; reduction sizes N =", Ns, "->", out, flush=True)


# ------------------------------------------------------------------ host columns
def host_run(task, prec, steps, threads, reference):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    if reference:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    path = graph_path(task, prec)
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp, lab = I.test_set(task)
    init = I.initial_state(task)
    zH0, zL0, emb = init
    N = inp.shape[0]
    zH, zL = np.repeat(zH0, N, 0)[:, None], np.repeat(zL0, N, 0)[:, None]
    preds, qs, rows = [], [], []
    t0 = time.time()
    for s in range(steps):
        nzH, nzL, lg, qh = [], [], [], []
        for i in range(N):
            o = sess.run(I.OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1], "puzzle_embedding": emb})
            nzH.append(o[0].astype(np.float32)); nzL.append(o[1].astype(np.float32)); lg.append(o[2]); qh.append(float(o[3].reshape(-1)[0]))
        zH, zL = np.stack(nzH), np.stack(nzL)
        preds.append(np.concatenate(lg).argmax(-1))
        qs.append(qh)
        rows.append({"step": s, "exact": float((preds[-1] == lab).all(-1).mean()), "cell": float((preds[-1] == lab).mean())})
        print(task, prec, "ref" if reference else "fused", "step", s, rows[-1], f"{time.time() - t0:.0f}s", flush=True)
    return {"preds": np.stack(preds), "q": np.array(qs), "zH": zH[:, 0], "zL": zL[:, 0]}, rows, path


def hostref(names, threads, reference=True):
    pairs = [(t, p) for t, p in HOSTREF + list(SRC)] if not names else [tuple(n.split(":")) for n in names]
    for task, prec in pairs:
        d = I.idir(task)
        out = d / (f"host_{prec}_ref.npz" if reference else f"host_{prec}.npz")
        if out.exists():
            print("exists, not overwritten:", out, flush=True)
            continue
        r, rows, path = host_run(task, prec, I.NSTEPS[task], threads, reference)
        X.atomic_npz(out, **r)
        js = d / (out.stem + ".json")
        if js.exists():                       # e.g. the 4-step host_w8a8_nofuse2_ref.json of E3_DIAG: keep it
            js = d / (out.stem + ".16.json")
        js.write_text(json.dumps({"graph": str(path), "ref": "ORT_DISABLE_ALL" if reference else "ORT default (fused QDQ kernels)",
                                                          "n_steps": I.NSTEPS[task], "steps": rows}, indent=1))
        print("wrote", out, flush=True)


# ------------------------------------------------------------------ hub side
def fallback_ops(txt):
    """qairt-converter prints '... Following OPs fallback to float:' + one line of ops, or '... fallback to float.'
    (nothing) when the list is empty; aihub_v79_classa.annotate took the next line in both cases."""
    ls = txt.splitlines()
    for i, l in enumerate(ls):
        if "fallback to float" in l:
            if l.rstrip().endswith(":"):
                return [x.strip().rstrip(".") for x in ls[i + 1].split("[INFO] ", 1)[-1].split(",") if x.strip()]
            return []
    return []


def annotate(acct, comp):
    info = V.annotate(acct, comp)
    d = V.LOGS / comp["job_id"]
    txt = open(glob.glob(str(d / "*.log"))[0], errors="replace").read()
    info["float_fallback_ops"] = fallback_ops(txt)
    return info


def accounts(names):
    accts = {n: X.Acct(n) for n in names}
    for a in accts.values():
        X.OS.update({d.name: d.os for d in a.C.get_devices() if d.name not in X.OS})
    return accts


def compile_(args):
    names = [a for a in args if a in ("acct2", "acct3")] or ["acct2", "acct3"]
    cells = [tuple(a.split(":")) for a in args if ":" in a]
    accts = accounts(names)
    todo = sorted({(acct, t, p, dev) for acct, t, p, dev in CHAINS if acct in accts and (not cells or (t, p) in cells)},
                  key=lambda x: x[:3])
    seen = set()
    for acct, t, p, dev in todo:
        if (acct, t, p) in seen:
            continue
        seen.add((acct, t, p))
        comp = accts[acct].compile(t, p, dev, "qnn_dlc")
        print(acct, t, p, "compile", comp["job_id"], flush=True)
    while True:
        pend = 0
        for acct in accts.values():
            bs = acct.status()
            for a, t, p in sorted(seen):
                if a != acct.name:
                    continue
                dev = next(d for aa, tt, pp, d in CHAINS if (aa, tt, pp) == (a, t, p))
                comp = acct.compile(t, p, dev, "qnn_dlc")
                code = bs.get(comp["job_id"], "PENDING")
                if code == "SUCCESS" and "converter" not in comp:
                    comp.update(annotate(acct, comp))
                    acct.verify(comp)
                    acct.save()
                if code not in ("SUCCESS", "FAILED"):
                    pend += 1
                print(a, t, p, comp["job_id"], code, comp.get("converter"), "fallback", comp.get("float_fallback_ops"),
                      comp.get("dlc_dtype_counts"), "rms_norm ops", comp.get("dlc_rms_norm_ops"), comp.get("qairt_sdk_version"), flush=True)
                if code == "FAILED":
                    print("   ", (acct.C.get_job(comp["job_id"]).get_status().message or "")[:300], flush=True)
        if not pend:
            break
        time.sleep(45)


def run(only, steps, slots):
    sel = [c for c in CHAINS if not only or tag(*c) in only]
    accts = accounts(sorted({c[0] for c in sel}))
    chains = []
    for acct, task, prec, dev in sel:
        if dev not in X.OS:
            print(f"[e3] DROPPED {tag(acct, task, prec, dev)}: not hosted on {acct}", flush=True)
            continue
        c = X.Chain(accts[acct], task, prec, dev, "qnn_dlc", steps)
        if c.meta.get("done") and c.steps_done < steps:   # extend a finished probe
            c.meta.pop("done")
            c.meta["n_steps"] = steps
            c.save()
        chains.append(c)
    print(f"[e3] {len(chains)} chains, {steps} steps, slots {slots or 'unlimited'} per account", flush=True)
    while True:
        for acct in accts.values():
            bs = acct.status()
            mine = sorted([c for c in chains if c.a is acct], key=lambda c: c.steps_done)   # gate first
            free = (slots - sum(c.inflight(bs) for c in mine)) if slots else None
            for c in mine:
                try:
                    used = c.step(bs, free)
                    if free is not None:
                        free -= used
                except Exception as exc:  # noqa: BLE001
                    print(f"[chain {c.task} {c.tag}] ERROR {type(exc).__name__}: {str(exc)[:300]}", flush=True)
                errs = c.meta.get("errors") or []
                # QNN_DEVICE_ERROR_INVALID_CONFIG is transient on the S21 (fault table 2.6; every retry so far ran):
                # it does not count towards the driver's three-error stop, up to TRANSIENT_MAX attempts per chain
                if c.meta.get("failed") and "INVALID_CONFIG" in c.meta["failed"] \
                        and all("INVALID_CONFIG" in e["error"] for e in errs) and len(errs) < TRANSIENT_MAX:
                    c.meta.pop("failed")
                    c.meta.setdefault("errors_transient", []).extend(errs)
                    c.meta["errors"] = []
                    c.save()
                    print(f"[chain {c.task} {c.tag}] {len(c.meta['errors_transient'])} transient INVALID_CONFIG so far, retrying", flush=True)
                # a compose error is deterministic: stop after the first one instead of the driver's three tries
                if errs and not c.meta.get("failed") and "MODEL_GRAPH_ERROR" in errs[-1]["error"]:
                    c.meta["failed"] = "deterministic compose error, no retry: " + errs[-1]["error"][:300]
                    c.save()
                    print(f"[chain {c.task} {c.tag}] stopped after one MODEL_GRAPH_ERROR", flush=True)
                comp = c.meta.get("compile_job")
                if comp and "converter" not in c.meta:
                    rec = [j for j in acct.find(job_id=comp) if j.get("kind") in ("compile", "compile_orphan")]
                    if rec and rec[0].get("converter"):
                        c.meta["converter"] = {k: rec[0].get(k) for k in ("converter", "float_fallback_ops", "dlc_dtype_counts", "dlc_rms_norm_ops")}
                        c.save()
        active = [c for c in chains if not c.done]
        print(f"[e3] {time.strftime('%H:%M:%S', time.gmtime())} active {len(active)}: "
              + ", ".join(f"{c.task}/{c.tag} s{c.steps_done}" for c in active), flush=True)
        if not active:
            break
        time.sleep(60)


def status():
    for acct, task, prec, dev in CHAINS:
        d = I.idir(task)
        f = d / f"dev_{prec}_{dev.replace(' ', '_')}_{acct}.json"
        m = json.loads(f.read_text()) if f.exists() else {}
        h = d / f"host_{prec}_ref.json"
        hh = json.loads(h.read_text())["steps"] if h.exists() else []
        print(tag(acct, task, prec, dev), "| compile", m.get("compile_job"), (m.get("converter") or {}).get("converter"),
              "| guard", (m.get("guard") or {}).get("verdict"),
              "| device", [(s["step"], round(s["exact"], 2), round(s["cell"], 4)) for s in m.get("steps", [])],
              "| host ref", [(s["step"], round(s["exact"], 2), round(s["cell"], 4)) for s in hh[:8]],
              "| jobs", len(m.get("jobs", [])), "| failed:", (m.get("failed") or "")[:100])


# ------------------------------------------------------------------ class diff
def verdict(r):
    return "n/a" if r is None else ("PASS" if r >= PASS else ("FAIL" if r < FAIL else "WARN"))


def ratio(a, b):
    return None if b == 0 else a / b


def classdiff():
    print("## Host column: fused (host_<prec>.npz, ORT default) vs reference (host_<prec>_ref.npz, ORT_DISABLE_ALL)\n")
    print("| task | graph | step | fused exact / cell | ref exact / cell | cells that differ (fused vs ref) |\n|---|---|---|---|---|---|")
    hosts = {}
    for task, prec in HOSTREF:
        d = I.idir(task)
        _, lab = I.test_set(task)
        fz, rz = d / f"host_{prec}.npz", d / f"host_{prec}_ref.npz"
        if not rz.exists():
            print(f"| {task} | {prec} | - | - | no ref file | |")
            continue
        r = np.load(rz)["preds"]
        f = np.load(fz)["preds"] if fz.exists() else None
        hosts[(task, prec)] = (f, r, lab)
        for s in (0, 1, 7):
            rf = f"{100 * (f[s] == lab).all(-1).mean():.0f} / {100 * (f[s] == lab).mean():.1f}" if f is not None else "no fused file"
            dif = f"{100 * (f[s] != r[s]).mean():.2f} %" if f is not None else "-"
            print(f"| {task} | {prec} | {s + 1} | {rf} | {100 * (r[s] == lab).all(-1).mean():.0f} / {100 * (r[s] == lab).mean():.1f} | {dif} |")
    print("\n## Device rows against each host column (probe = cell s2 ratio, verdict = exact s8 ratio; PASS >= 0.95, FAIL < 0.5)\n")
    print("| task | graph | device record | steps | dev exact s8 / cell s8 | vs fused: probe, s8, agree s8 | vs ref: probe, s8, agree s8 | change |\n|---|---|---|---|---|---|---|---|")
    changes = []
    for (task, prec), (f, r, lab) in hosts.items():
        d = I.idir(task)
        for mp in sorted(d.glob(f"dev_{prec}_*.json")):
            if mp.name.count("_nofuse") or not mp.suffix == ".json":
                continue
            m = json.loads(mp.read_text())
            sp = d / (mp.stem + ".npz")
            if m.get("failed") or not sp.exists():
                continue
            p = np.load(sp)["preds"]
            k = p.shape[0]
            if k < 2:
                continue
            last = min(k, 8) - 1
            def cls(h):
                if h is None:
                    return ("-", "-", "-", "-", "-")
                pr = ratio((p[1] == lab).mean(), (h[1] == lab).mean())
                vr = ratio((p[last] == lab).all(-1).mean(), (h[last] == lab).all(-1).mean())
                return (f"{pr:.2f}" if pr is not None else "-", verdict(pr), f"{vr:.2f}" if vr is not None else "n/a (host 0)", verdict(vr),
                        f"{100 * (p[last] == h[last]).mean():.1f}")
            cf, cr = cls(f), cls(r)
            ch = "" if (cf[1], cf[3]) == (cr[1], cr[3]) else f"{cf[1]}/{cf[3]} -> {cr[1]}/{cr[3]}"
            if ch:
                changes.append((task, prec, mp.stem[4:], ch))
            print(f"| {task} | {prec} | {mp.stem[4 + len(prec) + 1:]} | {k} | {100 * (p[last] == lab).all(-1).mean():.0f} / {100 * (p[last] == lab).mean():.1f} | "
                  f"{cf[0]} {cf[1]}, {cf[2]} {cf[3]}, {cf[4]} | {cr[0]} {cr[1]}, {cr[2]} {cr[3]}, {cr[4]} | {ch} |")
    print("\nrows whose (probe, s8) class changes against the reference column:", len(changes))
    for c in changes:
        print("-", *c)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "hostref", "hostfused", "compile", "run", "status", "classdiff"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    {"build": lambda: build(a.names), "hostref": lambda: hostref(a.names, a.threads, True),
     "hostfused": lambda: hostref(a.names, a.threads, False), "compile": lambda: compile_(a.names),
     "run": lambda: run(a.only, a.steps, a.slots), "status": status, "classdiff": classdiff}[a.cmd]()
