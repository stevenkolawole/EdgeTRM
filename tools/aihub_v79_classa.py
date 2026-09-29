#!/usr/bin/env python3
"""Two E3 diagnostics on AI Hub (reports/edgetrm_exp/E3_DIAG_V79_CLASSA.md), acct2 only, tokens never read here.

(1) class A (devices without an fp16 HTP: S21, RB3 Gen 2, 7 Gen 4, Q-6690): the qairt-converter path of every
    qdq_* compile fuses each RMSNorm into an `rms_norm` op and lets it (plus the injection Adds and Transposes
    next to it) "fallback to float" (compile log), so the DLC carries FLOAT_16 tensors that the composer refuses on
    a SoC "that doesn't support FP16" (runtime log). Probe: a Sudoku-MLP w8a8 graph whose RMSNorm pattern is broken
    (ReduceMean -> ReduceSum * 1/N, both quantized), compiled once; the compile log's fallback list and a dtype scan
    of the DLC decide whether it is worth the one device job on the S21.
(2) v79 Maze: the rope16 graph the S25 / 8 Elite fail on (0 / 90 at step 8) was compiled through the hub's other
    path (ONNX Runtime QNN EP: no fusion, no float fallback, 7 KB log). Variants of that graph move one op group to
    float (Q/DQ pairs stripped inside the group, the group's outputs stay quantized) or to 16 bit; each is compiled
    (path recorded), run two steps on the S25 (breadth chain record layout, prec maze_v79_<variant>) and compared
    with the ORT reference run of the same graph.

    python aihub_v79_classa.py build                 # variant graphs (no hub)
    python aihub_v79_classa.py compile [names]       # upload + compile on acct2, annotate converter path / DLC dtypes
    python aihub_v79_classa.py host [names] --steps 2
    python aihub_v79_classa.py run [names] --steps 2 # device chains (S25 for maze_v79_*, S21 for w8a8_nofuse)
    python aihub_v79_classa.py status
"""
import argparse
import collections
import glob
import json
import os
import pathlib
import struct
import sys
import time
import zipfile

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_acct as X  # noqa: E402
import aihub_infer as I  # noqa: E402
import wa_collapse as W  # noqa: E402

S25, S21 = "Samsung Galaxy S25", "Samsung Galaxy S21"
MAZE = {  # prec suffix -> how the graph is made from variant_Ma_rope16.onnx (or built by wa_collapse)
    "scoresfp": dict(base="Ma_rope16", strip={"attn_scores"}),
    "ropefp": dict(base="Ma_rope16", strip={"attn_rope", "attn_rope_const", "attn_concat"}),
    "attnfp": dict(base="Ma_rope16", strip={"attn_rope", "attn_rope_const", "attn_concat", "attn_scores", "attn"}),
    "all16s8": dict(build=dict(eight=set(W.GROUPS) - {"attn_scores"}, sixteen_default=False, propagate_movement=True)),
}
PRECS = {**{f"maze_v79_{v}": ("maze", "qdq_w8a16", f"variant_V79_{v}.onnx", S25) for v in MAZE},
         # nofuse: first build, N taken as 512 (no value_info in the graph) where the reduction runs over 97: a wrong
         # graph (host 0.128 -> 0.220 vs w8a8 0.340 -> 0.488); kept for the record. nofuse2: N from shape inference.
         "w8a8_nofuse": ("sudoku_mlp", "qdq_w8a8", "variant_CA_nofuse.onnx", S21),
         "w8a8_nofuse2": ("sudoku_mlp", "qdq_w8a8", "variant_CA_nofuse2.onnx", S21)}
DTYPE = {0x0216: "FLOAT_16", 0x0232: "FLOAT_32", 0x0408: "UFIXED_8", 0x0416: "UFIXED_16", 0x0308: "SFIXED_8",
         0x0316: "SFIXED_16", 0x0332: "SFIXED_32"}
LOGS = pathlib.Path.home() / "x3" / "joblogs" / "v79_classa"


# ------------------------------------------------------------------ source_path extension (aihub_acct looks it up by name)
_orig_source_path = X.source_path


def source_path(task, prec):
    if prec not in PRECS:
        return _orig_source_path(task, prec)
    t, base, fname, _ = PRECS[prec]
    assert t == task, (task, prec)
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


# ------------------------------------------------------------------ graph edits
def strip_qdq(g, groups):
    """Remove the Q/DQ pair of every activation tensor whose group is in `groups` and whose consumers all lie in
    those groups (interior tensors); the group's boundary tensors keep their encodings. Returns the stripped names."""
    grp = W.classify(g)
    qn = W.act_quant_nodes(g)
    prod = {o: n for n in g.node for o in n.output}
    cons = collections.defaultdict(list)
    for n in g.node:
        for x in n.input:
            cons[x].append(n)

    def dq_out(t):
        q = qn[t]
        return [n.output[0] for n in cons[q.output[0]] if n.op_type == "DequantizeLinear"]

    def out_group(n):
        """group of the quantized tensor a consumer produces (through pass-through ops)"""
        for o in n.output:
            if o in grp:
                return grp[o]
            if o in prod or cons[o]:
                for c in cons[o]:
                    r = out_group(c)
                    if r:
                        return r
        return None

    interior = []
    for t, gname in grp.items():
        if gname not in groups or t not in qn:
            continue
        users = [c for o in dq_out(t) for c in cons[o]]
        if users and all(out_group(c) in groups for c in users):
            interior.append(t)
    remove, rename = set(), {}
    for t in interior:
        q = qn[t]
        for dq in cons[q.output[0]]:
            if dq.op_type == "DequantizeLinear":
                rename[dq.output[0]] = t
                remove.add(id(dq))
        remove.add(id(q))
    keep = [n for n in g.node if id(n) not in remove]
    for n in keep:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    del g.node[:]
    g.node.extend(keep)
    return interior


def break_rmsnorm(m):
    """ReduceMean(x) -> Mul(ReduceSum(x), 1/N): same arithmetic, both new tensors quantized (sum at N x the mean's
    scale, the constant exactly representable), so qairt-converter's RMSNorm matcher has nothing to fuse.
    N = the size of the reduced (last) axis from shape inference (97 for the Sudoku-MLP token-mixing norms)."""
    g = m.graph
    inf = onnx.shape_inference.infer_shapes(m)
    dims = {}
    for vi in list(inf.graph.value_info) + list(inf.graph.input):
        if vi.type.tensor_type.shape.dim:
            dims[vi.name] = [d.dim_value for d in vi.type.tensor_type.shape.dim]
    vals = W.const_map(g)
    prod = {o: n for n in g.node for o in n.output}
    new_nodes, new_inits, n_done = [], [], 0
    for n in list(g.node):
        if n.op_type != "ReduceMean":
            new_nodes.append(n)
            continue
        axes = next(helper.get_attribute_value(a) for a in n.attribute if a.name == "axes")
        keep = next((helper.get_attribute_value(a) for a in n.attribute if a.name == "keepdims"), 1)
        # the mean's own Q node gives the encoding of the mean; N from the ReduceMean input's shape via the DQ chain
        q = next(c for c in g.node if c.op_type == "QuantizeLinear" and c.input[0] == n.output[0])
        s_mean = float(numpy_helper.to_array(vals[q.input[1]]))
        zp = vals[q.input[2]]
        shape_src = n.input[0]
        # the hidden size: the last dim of the Mul(x, x) input; read from the graph's value_info if present else 512
        assert axes == [-1] or axes == [len(dims[shape_src]) - 1], (n.name, axes)
        N = dims[shape_src][-1]
        assert N > 0, (n.name, dims.get(shape_src))
        base = n.name
        sum_out, sum_q, sum_dq = base + "_sum", base + "_sum_q", base + "_sum_dq"
        c_name, c_q, c_dq = base + "_invN", base + "_invN_q", base + "_invN_dq"
        new_inits += [numpy_helper.from_array(np.array(axes, np.int64), base + "_axes"),
                      numpy_helper.from_array(np.array(s_mean * N, np.float32), sum_q + "_s"),
                      numpy_helper.from_array(np.array(1.0 / N, np.float32), c_name),
                      numpy_helper.from_array(np.array((1.0 / N) / 255.0, np.float32), c_q + "_s"),
                      helper.make_tensor(c_q + "_z", zp.data_type, [], [0])]
        new_nodes += [helper.make_node("ReduceSum", [shape_src, base + "_axes"], [sum_out], name=base + "_ReduceSum", keepdims=keep),
                      helper.make_node("QuantizeLinear", [sum_out, sum_q + "_s", q.input[2]], [sum_q], name=sum_q),
                      helper.make_node("DequantizeLinear", [sum_q, sum_q + "_s", q.input[2]], [sum_dq], name=sum_dq),
                      helper.make_node("QuantizeLinear", [c_name, c_q + "_s", c_q + "_z"], [c_q], name=c_q),
                      helper.make_node("DequantizeLinear", [c_q, c_q + "_s", c_q + "_z"], [c_dq], name=c_dq),
                      helper.make_node("Mul", [sum_dq, c_dq], [n.output[0]], name=base + "_MulInvN")]
        n_done += 1
    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(new_inits)
    return n_done


def build(names):
    d = I.idir("maze")
    for v, spec in MAZE.items():
        if names and f"maze_v79_{v}" not in names:
            continue
        out = d / "qdq_w8a16" / f"variant_V79_{v}.onnx"
        if "build" in spec:
            info = W.build_variant("maze", f"V79_{v}", **spec["build"])
            print(v, json.dumps(info), flush=True)
        else:
            m = onnx.load(str(d / "qdq_w8a16" / f"variant_{spec['base']}.onnx"), load_external_data=False)
            stripped = strip_qdq(m.graph, spec["strip"])
            onnx.save(m, str(out))
            print(v, "stripped Q/DQ on", len(stripped), "tensors; first:", stripped[:4], flush=True)
        (d / f"variant_V79_{v}.json").write_text(json.dumps({"spec": {k: sorted(x) if isinstance(x, set) else
                                                                     ({kk: sorted(vv) if isinstance(vv, set) else vv
                                                                       for kk, vv in x.items()} if isinstance(x, dict) else x)
                                                                     for k, x in spec.items()}}, indent=1))
    if not names or "w8a8_nofuse2" in names:
        d = I.idir("sudoku_mlp")
        m = onnx.load(str(d / "qdq_w8a8" / "model.onnx"), load_external_data=False)
        n = break_rmsnorm(m)
        onnx.save(m, str(d / "qdq_w8a8" / "variant_CA_nofuse2.onnx"))
        print("w8a8_nofuse2: rewrote", n, "ReduceMean nodes", flush=True)


# ------------------------------------------------------------------ hub side
def annotate(acct, comp):
    """converter path (qairt-converter | QNN-EP), float-fallback ops and DLC tensor dtype counts of a compile job."""
    jid = comp["job_id"]
    d = LOGS / jid
    d.mkdir(parents=True, exist_ok=True)
    if not glob.glob(str(d / "*.log")):
        acct.C.get_job(jid).download_job_logs(str(d))
    txt = open(glob.glob(str(d / "*.log"))[0], errors="replace").read()
    path = "QNN-EP" if "QNN ONNX EP converter" in txt else ("qairt-converter" if "qairt-converter" in txt else "?")
    ls = txt.splitlines()
    fb = ""
    for i, l in enumerate(ls):
        if "fallback to float" in l:
            fb = ls[i + 1].split("[INFO] ", 1)[-1]
    if not glob.glob(str(d / "*.dlc")):
        acct.C.get_job(jid).get_target_model().download(str(d))
    dlc = glob.glob(str(d / "*.dlc"))[0]
    buf = None
    with zipfile.ZipFile(dlc) as z:
        with z.open("model") as f:
            try:
                buf = f.read()
            except zipfile.BadZipFile:
                pass
    if buf is None:  # the hub's DLC zips carry a zero CRC; read the stored member raw
        os.system(f"cd {d} && unzip -o -q {os.path.basename(dlc)} model 2>/dev/null")
        buf = open(d / "model", "rb").read()
    cnt = collections.Counter()
    for off in range(0, len(buf) - 4, 4):
        v = struct.unpack_from("<I", buf, off)[0]
        if v in DTYPE:
            cnt[DTYPE[v]] += 1
    info = {"converter": path, "log_bytes": len(txt), "float_fallback_ops": [x.strip().rstrip(".") for x in fb.split(",")] if fb else [],
            "dlc_dtype_counts": dict(cnt), "dlc_rms_norm_ops": buf.count(b"rms_norm")}
    return info


def compile_(names):
    acct = X.Acct("acct2")
    X.OS.update({d.name: d.os for d in acct.C.get_devices() if d.name not in X.OS})
    for prec, (task, _, _, dev) in PRECS.items():
        if names and prec not in names:
            continue
        comp = acct.compile(task, prec, dev, "qnn_dlc")
        print(prec, "compile", comp["job_id"], flush=True)
    while True:
        bs = acct.status()
        pend = 0
        for prec, (task, _, _, dev) in PRECS.items():
            if names and prec not in names:
                continue
            comp = acct.compile(task, prec, dev, "qnn_dlc")
            code = bs.get(comp["job_id"], "PENDING")
            if code == "SUCCESS" and "converter" not in comp:
                comp.update(annotate(acct, comp))
                acct.verify(comp)
                acct.save()
            if code not in ("SUCCESS", "FAILED"):
                pend += 1
            print(prec, comp["job_id"], code, comp.get("converter"), "fallback", len(comp.get("float_fallback_ops", [])),
                  comp.get("dlc_dtype_counts"), comp.get("qairt_sdk_version"), flush=True)
            if code == "FAILED":
                print("   ", (acct.C.get_job(comp["job_id"]).get_status().message or "")[:300], flush=True)
        if not pend:
            break
        time.sleep(45)


def host(names, steps):
    import onnxruntime as ort
    inp, lab = I.test_set("maze")
    zH0, zL0, emb = I.initial_state("maze")
    d = I.idir("maze")
    for prec, (task, base, fname, _) in PRECS.items():
        if task != "maze" or (names and prec not in names):
            continue
        so = ort.SessionOptions()
        so.intra_op_num_threads = 32
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(str(d / base / fname), so, providers=["CPUExecutionProvider"])
        N = inp.shape[0]
        zH, zL = [zH0] * N, [zL0] * N
        rows = []
        t0 = time.time()
        for s in range(steps):
            preds = []
            for i in range(N):
                o = sess.run(I.OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1].astype(np.int32), "puzzle_embedding": emb})
                zH[i], zL[i] = o[0].astype(np.float32), o[1].astype(np.float32)
                preds.append(o[2].argmax(-1))
            p = np.concatenate(preds)
            rows.append({"step": s, "exact": float((p == lab).all(-1).mean()), "cell": float((p == lab).mean())})
            print(prec, "host step", s, rows[-1], f"{time.time() - t0:.0f}s", flush=True)
        (d / f"host_{prec}.json").write_text(json.dumps({"graph": str(d / base / fname), "ref": "ORT_DISABLE_ALL", "steps": rows}, indent=1))


def run(names, steps):
    acct = X.Acct("acct2")
    X.OS.update({d.name: d.os for d in acct.C.get_devices() if d.name not in X.OS})
    chains = []
    for prec, (task, _, _, dev) in PRECS.items():
        if names and prec not in names:
            continue
        c = X.Chain(acct, task, prec, dev, "qnn_dlc", steps)
        if c.meta.get("done") and c.steps_done < steps:   # extend a finished probe to more steps
            c.meta.pop("done")
            c.meta["n_steps"] = steps
            c.save()
        chains.append(c)
    while True:
        bs = acct.status()
        for c in chains:
            try:
                c.step(bs)
            except Exception as exc:  # noqa: BLE001
                print(f"[chain {c.task} {c.tag}] ERROR {type(exc).__name__}: {str(exc)[:300]}", flush=True)
            comp = c.meta.get("compile_job")
            if comp and "converter" not in c.meta:
                rec = [j for j in acct.find(job_id=comp) if j.get("kind") in ("compile", "compile_orphan")]
                if rec and rec[0].get("converter"):
                    c.meta["converter"] = {k: rec[0][k] for k in ("converter", "float_fallback_ops", "dlc_dtype_counts", "dlc_rms_norm_ops")}
                    c.save()
        active = [c for c in chains if not c.done]
        print(f"[v79] {time.strftime('%H:%M:%S', time.gmtime())} active {len(active)}: "
              + ", ".join(f"{c.prec} s{c.steps_done}" for c in active), flush=True)
        if not active:
            break
        time.sleep(60)


def status():
    for prec, (task, _, _, dev) in PRECS.items():
        d = I.idir(task)
        f = d / f"dev_{prec}_{dev.replace(' ', '_')}_acct2.json"
        h = d / f"host_{prec}.json"
        m = json.loads(f.read_text()) if f.exists() else {}
        hh = json.loads(h.read_text())["steps"] if h.exists() else []
        print(prec, "| compile", m.get("compile_job"), (m.get("converter") or {}).get("converter"),
              "| guard", (m.get("guard") or {}).get("verdict"),
              "| device", [(s["step"], round(s["exact"], 2), round(s["cell"], 4)) for s in m.get("steps", [])],
              "| host", [(s["step"], round(s["exact"], 2), round(s["cell"], 4)) for s in hh],
              "| failed:", (m.get("failed") or "")[:120], "| errors", len(m.get("errors", [])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "compile", "host", "run", "status"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--steps", type=int, default=2)
    a = ap.parse_args()
    {"build": lambda: build(a.names), "compile": lambda: compile_(a.names), "host": lambda: host(a.names, a.steps),
     "run": lambda: run(a.names, a.steps), "status": status}[a.cmd]()
