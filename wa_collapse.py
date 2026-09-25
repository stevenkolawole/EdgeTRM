#!/usr/bin/env python3
"""Localise the w8a8 collapse of the AI Hub Sudoku-MLP graph (w8a8 and w4a8 solve 0/100 on the host
simulation, w8a16 and w4a16 recover fp32) by switching the activation precision of one tensor group
at a time, on the exact QDQ graphs AI Hub compiles (H=1 carry step, `stepc`).

Both AI Hub graphs come from the same fp32 ONNX and calibration set and carry identical INT8
per-channel weights; they differ only in the activation encodings (UINT16 vs UINT8, per tensor, on
every op output). A variant is built from the w8a16 graph by giving each activation Quantize/
Dequantize pair either its w8a16 encoding or the w8a8 graph's encoding of the same tensor (same
calibration, AIMET's own 8-bit range), so "all groups 8-bit" must reproduce the w8a8 graph.

Groups (by the op that produces the quantized tensor; value-preserving ops inherit their input's
group):
  carry_in    graph inputs z_H, z_L (the carry as it enters the step)
  carry_out   the z_H_out / z_L_out tensors the step hands back
  embed       token-embedding gather, puzzle row, padding/concat, embed scale
  inject      input-injection adds (z_H + embeddings, z_L + injection)
  residual    residual adds inside the block (x + mixer(x), x + mlp(x))
  rms_int     RMSNorm internals: x*x, mean, +eps, sqrt, 1/sqrt
  rms_out     RMSNorm output (x * 1/sqrt), i.e. the normalised stream
  linear_out  gate_up / down projection outputs
  swiglu      gate/up halves, sigmoid, silu, silu*up
  head        lm_head / q_head outputs
Edge-only variant `linear_in_edge`: an extra 8-bit Q/DQ inserted only on the activation edge into
every MatMul (the harness's +a8), all tensors otherwise 16-bit.
Per-channel variant: the named groups quantized UINT8 per channel along the hidden axis, ranges from
the fp32 graph on the calibration carries (a host-side what-if; QNN HTP takes per-tensor activation
encodings, so the vendor-expressible fix is a 16-bit override on those tensors).

    python wa_collapse.py --stage stats                   # activation ranges per tensor/group
    python wa_collapse.py --stage build                   # variant graphs (small .onnx next to model.data)
    python wa_collapse.py --stage eval --variants A1,A2 --count 100 --start 0
"""
import argparse
import collections
import json
import pathlib
import sys
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402
import aihub_infer as I  # noqa: E402

TASK = "sudoku_mlp"
GROUPS = ["carry_in", "carry_out", "embed", "inject", "residual", "rms_sq", "rms_var", "rms_rsqrt", "rms_out",
          "rms_const", "linear_out", "swiglu_prod", "swiglu_other", "attn", "attn_rope", "attn_rope_const",
          "attn_concat", "attn_scores", "head", "const"]
RMS = {"rms_sq", "rms_var", "rms_rsqrt", "rms_out"}
# attention sub-groups (op-name suffix inside self_attn, TRM Attention export): RoPE = q/k * cos + rotate_half(q/k) * sin
ATTN_CONCAT = {"Concat_1", "Concat_2", "Neg", "Neg_1"}                      # rotate_half: -x2 ++ x1
ATTN_ROPE = {"Mul", "Mul_1", "Mul_2", "Mul_3", "Add", "Add_1"}              # * cos, * sin, sum
ATTN_SCORES = {"Mul_4", "Mul_5", "Softmax"}                                 # scale, softmax (+ the two matmuls)
ATTN_ALL = {"attn", "attn_rope", "attn_rope_const", "attn_concat", "attn_scores"}
import re  # noqa: E402
_RMS_CONST = re.compile(r"^/inner/L_level/layers\.\d+(_\d+)?/Constant(_\d+)?_output_0$")
PASS = {"Cast", "Transpose", "Reshape", "Identity", "Squeeze", "Unsqueeze", "Pad", "Concat", "Slice", "Flatten"}


def paths(task):
    d = I.idir(task)
    return d / "qdq_w8a16", d / "qdq_w8a8", d / "ablate"


def const_map(g):
    vals = {i.name: i for i in g.initializer}
    for n in g.node:
        if n.op_type == "Constant":
            vals[n.output[0]] = n.attribute[0].t
    return vals


def act_quant_nodes(g):
    """{float tensor name: QuantizeLinear node} for activation Q nodes (input is not an initializer)."""
    inits = {i.name for i in g.initializer}
    return {n.input[0]: n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] not in inits}


def encoding(vals, q):
    s = numpy_helper.to_array(vals[q.input[1]]).astype(np.float64)
    zt = vals[q.input[2]]
    return {"scale": s, "zp": numpy_helper.to_array(zt).astype(np.int64), "zp_type": zt.data_type,
            "axis": next((helper.get_attribute_value(a) for a in q.attribute if a.name == "axis"), None)}


def classify(g):
    """Group of every activation-quantized tensor, from the fp32-level producer ops."""
    inits = {i.name for i in g.initializer}
    prod = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
    # Q/DQ are transparent: the float tensor a DQ output stands for is the Q node's input
    dq_src = {}
    for n in g.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in prod and prod[n.input[0]].op_type == "QuantizeLinear":
            dq_src[n.output[0]] = prod[n.input[0]].input[0]

    def src(t):
        """float tensor a (de)quantized tensor stands for"""
        for _ in range(8):
            if t in dq_src:
                t = dq_src[t]
            elif t in prod and prod[t].op_type == "QuantizeLinear":
                t = prod[t].input[0]
            else:
                break
        return t
    carry_out_src = set()
    for n in g.node:
        for o in n.output:
            if o in ("z_H_out", "z_L_out"):
                carry_out_src.add(src(o) if n.op_type in ("DequantizeLinear", "QuantizeLinear") else o)
    memo = {}

    def group(t):
        t = src(t)
        if t in memo:
            return memo[t]
        if t in ("z_H", "z_L"):
            r = "carry_in"
        elif t == "puzzle_embedding":
            r = "embed"
        elif t not in prod:
            r = "const"
        else:
            n = prod[t]
            nm, op = n.name, n.op_type
            act_in = [x for x in n.input if x and x not in inits and not x.startswith("/inner/Constant")
                      and "Constant" not in x]
            if op == "Constant":
                r = "rms_const" if _RMS_CONST.match(t) else ("attn_rope_const" if "self_attn" in t else "const")
            elif "self_attn" in nm and op not in ("MatMul", "Gemm") and op not in PASS:
                suf = nm.rsplit("/", 1)[-1]
                r = ("attn_concat" if suf in ATTN_CONCAT else "attn_rope" if suf in ATTN_ROPE
                     else "attn_scores" if suf in ATTN_SCORES else "attn")
            elif "self_attn" in nm and op in ("MatMul", "Gemm") and all(x not in inits for x in n.input) \
                    and not any("Identity" in x or "onnx::MatMul" in x for x in n.input):
                r = "attn_scores"     # q.k^T and attn.v: activation x activation
            elif op in PASS or op == "QuantizeLinear" or op == "DequantizeLinear":
                r = group(act_in[0]) if act_in else "const"
            elif op == "Gather":
                if "embed_tokens" in nm:
                    r = "embed"
                else:
                    gi = group(n.input[0])
                    r = "shape" if gi in ("shape", "const") else gi
            elif "q_head" in nm or "lm_head" in nm:
                r = "head"
            elif op in ("MatMul", "Gemm"):
                r = "linear_out"
            elif op == "Sigmoid":
                r = "swiglu_other"
            elif "mlp" in nm and op == "Mul":
                ga = [group(x) for x in act_in]
                if "shape" in ga or not act_in:
                    r = "shape"
                else:
                    r = "swiglu_prod" if any(g_ == "swiglu_other" for g_ in ga) and "Sigmoid" not in \
                        "".join(x for x in act_in) and nm.endswith("Mul_3") else "swiglu_other"
            elif op == "ReduceMean":
                r = "rms_var"
            elif op in ("Sqrt",) or (op == "Div" and "layers" in nm and "mlp" not in nm):
                r = "rms_rsqrt"
            elif op == "Mul" and "layers" in nm and len(n.input) == 2 and src(n.input[0]) == src(n.input[1]):
                r = "rms_sq"
            elif op == "Mul" and "layers" in nm and "mlp" not in nm:
                r = "rms_out"
            elif op == "Add" and "layers" in nm and "mlp" not in nm:
                ga = [group(x) for x in act_in]
                r = "rms_var" if len(act_in) < 2 or "rms_var" in ga else "residual"
            elif op == "Add":
                ga = [group(x) for x in act_in]
                r = "shape" if "shape" in ga or len(act_in) < 2 else "inject"
            elif op == "Mul" and nm.startswith("/inner/Mul"):
                r = "embed"
            elif op in ("Shape", "ConstantOfShape"):
                r = "shape"
            else:
                r = "other:" + op
        memo[t] = r
        return r
    qn = act_quant_nodes(g)
    out = {}
    for t in qn:
        out[t] = "carry_out" if src(t) in carry_out_src else group(t)
    return out


def stage_stats(task, n_cal=8):
    """Per quantized tensor: absmax, std, 8-bit step / std (from the w8a8 encoding), outlier ratios."""
    import onnxruntime as ort
    p16, p8, abl = paths(task)
    abl.mkdir(parents=True, exist_ok=True)
    g16 = onnx.load(str(p16 / "model.onnx"), load_external_data=False).graph
    g8 = onnx.load(str(p8 / "model.onnx"), load_external_data=False).graph
    grp = classify(g16)
    v8, q8 = const_map(g8), act_quant_nodes(g8)
    # values from the fp32 graph (same tensor names) on the calibration carries
    m = onnx.load(str(A.ART / task / "stepc_H1.onnx"))
    names = {o.name for o in m.graph.output}
    want = [t for t in grp if t not in names and t not in ("z_H", "z_L", "puzzle_embedding")]
    have = {n for node in m.graph.node for n in node.output}
    want = [t for t in want if t in have]
    for t in want:
        m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    so = ort.SessionOptions()
    so.intra_op_num_threads = 8
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    cal = A.calib_dict(task, 1, "stepc")
    acc = collections.defaultdict(list)
    for i in range(n_cal):
        feed = {k: v[i] for k, v in cal.items()}
        vals = sess.run(want, feed)
        for t, x in zip(want, vals):
            acc[t].append(x.astype(np.float32))
        for t in ("z_H", "z_L", "puzzle_embedding"):
            acc[t].append(feed[t].astype(np.float32))
    rows = []
    for t, g in grp.items():
        if t not in acc:
            continue
        x = np.concatenate([a.reshape(-1, a.shape[-1]) if a.ndim else a.reshape(1, 1) for a in acc[t]])
        absmax, std = float(np.abs(x).max()), float(x.std())
        ch = np.abs(x).max(0)
        e8 = encoding(v8, q8[t]) if t in q8 else None
        step8 = float(e8["scale"]) if e8 is not None and np.ndim(e8["scale"]) == 0 else None
        rows.append({"tensor": t, "group": g, "absmax": absmax, "std": std, "absmax_over_std": absmax / (std + 1e-12),
                     "p999_abs": float(np.quantile(np.abs(x), 0.999)), "min": float(x.min()), "max": float(x.max()),
                     "channel_absmax_max_over_median": float(ch.max() / (np.median(ch) + 1e-12)),
                     "step8": step8, "step8_over_std": (step8 / (std + 1e-12)) if step8 else None,
                     "frac_below_half_step8": float((np.abs(x - x.mean()) < step8 / 2).mean()) if step8 else None})
    (abl / "act_stats.json").write_text(json.dumps(rows, indent=1))
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    summary = {}
    for gname, rs in sorted(by.items()):
        summary[gname] = {"n_tensors": len(rs), "absmax_max": max(r["absmax"] for r in rs),
                          "std_median": float(np.median([r["std"] for r in rs])),
                          "absmax_over_std_median": float(np.median([r["absmax_over_std"] for r in rs])),
                          "absmax_over_std_max": max(r["absmax_over_std"] for r in rs),
                          "step8_over_std_median": float(np.median([r["step8_over_std"] for r in rs if r["step8_over_std"]]))
                          if any(r["step8_over_std"] for r in rs) else None,
                          "step8_over_std_max": max((r["step8_over_std"] for r in rs if r["step8_over_std"]), default=None),
                          "frac_below_half_step8_median": float(np.median([r["frac_below_half_step8"] for r in rs
                                                                          if r["frac_below_half_step8"] is not None]))
                          if any(r["frac_below_half_step8"] is not None for r in rs) else None}
        print(gname, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summary[gname].items()}),
              flush=True)
    (abl / "act_stats_by_group.json").write_text(json.dumps(summary, indent=1))
    other = sorted({g for g in grp.values() if g not in GROUPS})
    print("unassigned groups:", other, collections.Counter(grp.values()), flush=True)


# ----------------------------------------------------------------------------- variants
def per_channel_encodings(task, tensors, n_cal=8):
    """UINT8 per-channel (last axis) asymmetric encodings from fp32 values on the calibration carries."""
    import onnxruntime as ort
    m = onnx.load(str(A.ART / task / "stepc_H1.onnx"))
    have = {n for node in m.graph.node for n in node.output}
    tensors = [t for t in tensors if t in have or t in ("z_H", "z_L")]
    extra = [t for t in tensors if t in have and t not in {o.name for o in m.graph.output}]
    for t in extra:
        m.graph.output.append(helper.make_tensor_value_info(t, TensorProto.FLOAT, None))
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    cal = A.calib_dict(task, 1, "stepc")
    lo, hi = {}, {}
    for i in range(n_cal):
        feed = {k: v[i] for k, v in cal.items()}
        vals = dict(zip(extra, sess.run(extra, feed)))
        vals.update({t: feed[t] for t in ("z_H", "z_L") if t in tensors})
        for t in tensors:
            x = vals[t].reshape(-1, vals[t].shape[-1]) if vals[t].ndim else vals[t].reshape(1, 1)
            lo[t] = np.minimum(lo.get(t, np.inf), x.min(0))
            hi[t] = np.maximum(hi.get(t, -np.inf), x.max(0))
    enc = {}
    for t in tensors:
        l, h = np.minimum(lo[t], 0), np.maximum(hi[t], 0)
        s = np.maximum((h - l) / 255.0, 1e-12).astype(np.float32)
        z = np.clip(np.round(-l / s), 0, 255).astype(np.uint8)
        enc[t] = (s, z)
    return enc


def build_variant(task, name, eight, sixteen_default=True, edge_linear=False, perchannel=(), propagate_movement=False):
    """eight: set of groups at the w8a8 encoding; everything else at w8a16 (or the reverse when
    sixteen_default=False: `eight` then lists the groups kept at 16 bit). perchannel: groups given
    per-channel UINT8 encodings. edge_linear: extra 8-bit Q/DQ on every MatMul activation input."""
    p16, p8, abl = paths(task)
    out = p16 / f"variant_{name}.onnx"
    m = onnx.load(str(p16 / "model.onnx"), load_external_data=False)
    g = m.graph
    g8 = onnx.load(str(p8 / "model.onnx"), load_external_data=False).graph
    grp = classify(g)
    v16, v8 = const_map(g), const_map(g8)
    q16, q8 = act_quant_nodes(g), act_quant_nodes(g8)
    cons = collections.defaultdict(list)
    for n in g.node:
        for x in n.input:
            cons[x].append(n)
    new, n8, n_pc, missing = [], 0, 0, []
    pc_tensors = [t for t, gname in grp.items() if gname in perchannel]
    pc = per_channel_encodings(task, pc_tensors) if pc_tensors else {}
    for t, q in q16.items():
        gname = grp[t]
        to8 = (gname in eight) if sixteen_default else (gname not in eight)
        if gname in perchannel and t in pc:
            s, z = pc[t]
            sn, zn = q.name + "_pcs", q.name + "_pcz"
            new += [numpy_helper.from_array(s, sn), numpy_helper.from_array(z, zn)]
            attrs = {"axis": -1}
            n_pc += 1
        elif to8:
            if t not in q8:
                missing.append(t)
                continue
            e = encoding(v8, q8[t])
            sn, zn = q.name + "_e8s", q.name + "_e8z"
            new.append(numpy_helper.from_array(np.asarray(e["scale"], np.float32), sn))
            new.append(helper.make_tensor(zn, e["zp_type"], list(np.shape(e["zp"])),
                                          np.asarray(e["zp"]).reshape(-1).tolist()))
            attrs = {} if e["axis"] is None else {"axis": e["axis"]}
            n8 += 1
        else:
            continue
        for node in [q] + [c for c in cons[q.output[0]] if c.op_type == "DequantizeLinear"]:
            node.input[1], node.input[2] = sn, zn
            keep = [a for a in node.attribute if a.name not in ("axis",)]
            del node.attribute[:]
            node.attribute.extend(keep + [helper.make_attribute(k, v) for k, v in attrs.items()])
    # The HTP requires a Gather's table and output to share one encoding. The embedding table is a weight
    # (initializer Q/DQ), so when the gathered output moves to 8 bit the table's encoding follows it; the
    # w8a8 graph's encoding of the same table is used (Sep 25: mixed graph failed on device with
    # "embed_tokens/Gather ... has incorrect Value -34869, expected -34816").
    inits = {i.name for i in g.initializer}
    wq16 = {n.input[0]: n for n in g.node if n.op_type == "QuantizeLinear" and n.input[0] in inits}
    wq8 = {n.input[0]: n for n in g8.node if n.op_type == "QuantizeLinear" and n.input[0] in {i.name for i in g8.initializer}}
    prod = {o: n for n in g.node for o in n.output}
    n_table = 0
    for n in g.node:
        if n.op_type != "Gather":
            continue
        out_q = next((c for c in cons.get(n.output[0], []) if c.op_type == "QuantizeLinear"), None)
        if out_q is None or out_q.input[1].endswith("_e8s") is False:
            continue
        dq = prod.get(n.input[0])
        q = prod.get(dq.input[0]) if dq is not None and dq.op_type == "DequantizeLinear" else None
        if q is None or q.op_type != "QuantizeLinear" or q.input[0] not in wq8:
            continue
        e = encoding(v8, wq8[q.input[0]])
        sn, zn = q.name + "_e8s", q.name + "_e8z"
        new.append(numpy_helper.from_array(np.asarray(e["scale"], np.float32), sn))
        new.append(helper.make_tensor(zn, e["zp_type"], list(np.shape(e["zp"])), np.asarray(e["zp"]).reshape(-1).tolist()))
        for node in [q, dq]:
            node.input[1], node.input[2] = sn, zn
        n_table += 1
    # The HTP also requires data-movement ops (Gather on activations, Reshape, Transpose, Slice, ...) to keep
    # their input's encoding (Sep 25: "/inner/Gather ... incorrect Value -26880, expected -30915" when the
    # q_head's z_H[:, 0] gather had a 16-bit input and an 8-bit output). The output takes the input's encoding.
    n_prop = 0
    if propagate_movement:
        prod = {o: n for n in g.node for o in n.output}
        cons = collections.defaultdict(list)
        for n in g.node:
            for x in n.input:
                cons[x].append(n)
        for n in g.node:
            if n.op_type not in ("Gather", "Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze", "Flatten"):
                continue
            dq_in = prod.get(n.input[0])
            if dq_in is None or dq_in.op_type != "DequantizeLinear" or n.input[0] in inits:
                continue
            q_in = prod.get(dq_in.input[0])
            if q_in is None or q_in.op_type != "QuantizeLinear" or q_in.input[0] in inits:
                continue          # embedding tables (weights) are handled above
            q_out = next((c for c in cons.get(n.output[0], []) if c.op_type == "QuantizeLinear"), None)
            if q_out is None or (q_out.input[1], q_out.input[2]) == (dq_in.input[1], dq_in.input[2]):
                continue
            for node in [q_out] + [c for c in cons[q_out.output[0]] if c.op_type == "DequantizeLinear"]:
                node.input[1], node.input[2] = dq_in.input[1], dq_in.input[2]
                keep = [a for a in node.attribute if a.name != "axis"]
                del node.attribute[:]
                node.attribute.extend(keep + [a for a in dq_in.attribute if a.name == "axis"])
            n_prop += 1
    n_edge = 0
    if edge_linear:
        inits = {i.name for i in g.initializer}
        prod = {o: n for n in g.node for o in n.output}
        add_nodes = []
        for n in list(g.node):
            if n.op_type != "MatMul":
                continue
            a = n.input[0]
            if a in inits or a not in prod:
                continue
            # float tensor behind the (16-bit) DQ feeding this MatMul
            p = prod[a]
            t = p.input[0]
            if p.op_type == "DequantizeLinear" and t in prod and prod[t].op_type == "QuantizeLinear":
                t = prod[t].input[0]
            while t not in q8 and t in prod and prod[t].op_type in PASS:
                t = prod[t].input[0]
            if t not in q8:
                missing.append("edge:" + a)
                continue
            e = encoding(v8, q8[t])
            base = f"{n.name}_edge8"
            new.append(numpy_helper.from_array(np.asarray(e["scale"], np.float32), base + "_s"))
            new.append(helper.make_tensor(base + "_z", e["zp_type"], [], [int(np.asarray(e["zp"]).reshape(-1)[0])]))
            add_nodes.append((n, helper.make_node("QuantizeLinear", [a, base + "_s", base + "_z"], [base + "_q"],
                                                  name=base + "_Q"),
                              helper.make_node("DequantizeLinear", [base + "_q", base + "_s", base + "_z"],
                                               [base + "_dq"], name=base + "_DQ")))
        before = {id(n): (qn, dqn) for n, qn, dqn in add_nodes}
        nodes = []
        for n in g.node:
            if id(n) in before:
                qn, dqn = before[id(n)]
                nodes += [qn, dqn]
                n.input[0] = dqn.output[0]
                n_edge += 1
            nodes.append(n)
        nodes = [helper.make_node("Identity", ["__x"], ["__y"])] and [onnx.NodeProto.FromString(x.SerializeToString())
                                                                      for x in nodes]
        del g.node[:]
        g.node.extend(nodes)
    g.initializer.extend(new)
    onnx.save_model(m, str(out))            # big initializers stay external in model.data
    info = {"name": name, "path": str(out), "eight_groups": sorted(eight) if sixteen_default else
            sorted(set(GROUPS) - set(eight)), "sixteen_default": sixteen_default, "n_tensors_8bit": n8,
            "n_edges_8bit": n_edge, "n_gather_tables_8bit": n_table, "n_movement_propagated": n_prop, "n_per_channel": n_pc, "perchannel_groups": list(perchannel),
            "missing": missing[:20], "n_missing": len(missing)}
    return info


VARIANTS = {
    # from w8a16: one group at a time to the w8a8 encoding
    "A0_w8a16": dict(eight=set()),
    "A1_carry_io": dict(eight={"carry_in", "carry_out"}),
    "A2_rms_all": dict(eight=RMS | {"const"}),
    "A2a_rms_sq": dict(eight={"rms_sq"}),
    "A2b_rms_var_rsqrt": dict(eight={"rms_var", "rms_rsqrt", "const"}),
    # same perturbation without the embed-scale scalar (rms_const = RMSNorm's own eps / 1.0 constants);
    # the definition used for the cross-mixer comparison
    "A2b2_rms_var_rsqrt": dict(eight={"rms_var", "rms_rsqrt", "rms_const"}),
    "B2b2_rms_var_rsqrt16": dict(eight={"rms_var", "rms_rsqrt", "rms_const"}, sixteen_default=False),
    # B2 (all RMSNorm at 16) rescues w8a8; is the RMSNorm output needed at 16 too?
    "B2ab_sq_var16": dict(eight={"rms_sq", "rms_var", "rms_rsqrt", "rms_const"}, sixteen_default=False),
    # device-legal rebuilds (Gather table encoding follows its 8-bit output; built after the Sep 25 fix)
    "B2t_rms16": dict(eight=RMS | {"const"}, sixteen_default=False),
    "B2abt_sq_var16": dict(eight={"rms_sq", "rms_var", "rms_rsqrt", "rms_const"}, sixteen_default=False),
    # + data-movement ops keep their input's encoding (second HTP rejection, Sep 25)
    "B2u_rms16": dict(eight=RMS | {"const"}, sixteen_default=False, propagate_movement=True),
    "B2abu_sq_var16": dict(eight={"rms_sq", "rms_var", "rms_rsqrt", "rms_const"}, sixteen_default=False,
                           propagate_movement=True),
    # Maze w8a8 device-failure bisection (all 8 bit except the named groups at 16; both HTP legality rules)
    "M0_all8": dict(eight=set(), sixteen_default=False, propagate_movement=True),
    "Ma_rope16": dict(eight={"attn_rope", "attn_rope_const", "attn_concat"}, sixteen_default=False, propagate_movement=True),
    "Mb_concat16": dict(eight={"attn_concat"}, sixteen_default=False, propagate_movement=True),
    "Mc_scores16": dict(eight={"attn_scores"}, sixteen_default=False, propagate_movement=True),
    "Md_notlinear16": dict(eight=set(GROUPS) - {"linear_out"}, sixteen_default=False, propagate_movement=True),
    # second round (Ma restores the device, Md does not): split RoPE; test whether 16-bit scores break the device
    "Me_ropeconst16": dict(eight={"attn_rope_const"}, sixteen_default=False, propagate_movement=True),
    "Mf_ropemul16": dict(eight={"attn_rope"}, sixteen_default=False, propagate_movement=True),
    "Mg_rope_scores16": dict(eight={"attn_rope", "attn_rope_const", "attn_concat", "attn_scores"},
                             sixteen_default=False, propagate_movement=True),
    "Mh_notlinear_notscores16": dict(eight=set(GROUPS) - {"linear_out", "attn_scores"}, sixteen_default=False,
                                     propagate_movement=True),
    # B2b2 (RMSNorm variance path alone at 16) does not rescue w8a8: greedy unions kept at 16 bit
    "B23_rms_resid16": dict(eight=RMS | {"rms_const", "residual", "inject"}, sixteen_default=False),
    "B26_rms_swiglu16": dict(eight=RMS | {"rms_const", "swiglu_prod"}, sixteen_default=False),
    "B236_rms_resid_swiglu16": dict(eight=RMS | {"rms_const", "residual", "inject", "swiglu_prod"},
                                    sixteen_default=False),
    "A2c_rms_out": dict(eight={"rms_out"}),
    "A3_residual_inject": dict(eight={"residual", "inject"}),
    "A4_linear_in_edge": dict(eight=set(), edge_linear=True),
    "A5_embed": dict(eight={"embed"}),
    "A6_swiglu_prod": dict(eight={"swiglu_prod"}),
    "A6b_linout_swiglu_other": dict(eight={"linear_out", "swiglu_other"}),
    "A7_head": dict(eight={"head"}),
    "A8_all8": dict(eight=set(GROUPS)),
    # from w8a8: one group kept at 16 bit
    "B1_carry_io16": dict(eight={"carry_in", "carry_out"}, sixteen_default=False),
    "B2_rms16": dict(eight=RMS | {"const"}, sixteen_default=False),
    "B2a_rms_sq16": dict(eight={"rms_sq"}, sixteen_default=False),
    "B2b_rms_var_rsqrt16": dict(eight={"rms_var", "rms_rsqrt", "const"}, sixteen_default=False),
    "B3_residual_inject16": dict(eight={"residual", "inject"}, sixteen_default=False),
    "B5_embed16": dict(eight={"embed"}, sixteen_default=False),
    "B6_swiglu_prod16": dict(eight={"swiglu_prod"}, sixteen_default=False),
    "B6b_linout_swiglu_other16": dict(eight={"linear_out", "swiglu_other"}, sixteen_default=False),
}


def fix_variants(kill):
    """From w8a8: keep the killing group(s) at 16 bit, or per-channel 8 bit."""
    return {f"B_{'+'.join(sorted(kill))}_16": dict(eight=set(kill), sixteen_default=False),
            f"C_{'+'.join(sorted(kill))}_pc8": dict(eight=set(), sixteen_default=False, perchannel=tuple(kill))}


def stage_build(task, names, kill=None):
    p16, p8, abl = paths(task)
    abl.mkdir(parents=True, exist_ok=True)
    V = dict(VARIANTS)
    if kill:
        V.update(fix_variants(kill))
    reg_path = abl / "variants.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    for name in names or V:
        info = build_variant(task, name, **V[name])
        reg[name] = info
        print(json.dumps(info), flush=True)
    reg_path.write_text(json.dumps(reg, indent=1))


def stage_eval(task, names, start, count, nsteps, threads, ref=False):
    """ref=True: reference QDQ semantics (ORT graph optimizations disabled, so every Q/DQ pair is executed
    as written). The default ORT_ENABLE_ALL fuses QDQ patterns into integer kernels, which changes the
    arithmetic when uint8 and uint16 encodings mix (M3; REWRITE_NOTES 40)."""
    import onnxruntime as ort
    p16, p8, abl = paths(task)
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    if ref:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    inp, lab = A.load_data(task, list(range(start, start + count)))
    inp = inp.astype(np.int32)
    init = I.initial_state(task)
    for name in names:
        out = abl / f"eval_{name}_{start}_{count}{'_ref' if ref else ''}.json"
        if out.exists():
            continue
        path = p16 / f"variant_{name}.onnx" if not name.startswith("ref_") else \
            (A.ART / task / "stepc_H1.onnx" if name == "ref_fp32" else I.idir(task) / f"qdq_{name[4:]}" / "model.onnx")
        t0 = time.time()
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        r = I.host_loop(sess, task, inp, nsteps, init)
        ex = [(r["preds"][s] == lab).all(-1).mean() for s in range(nsteps)]
        cell = [(r["preds"][s] == lab).mean() for s in range(nsteps)]
        res = {"variant": name, "task": task, "start": start, "count": count, "nsteps": nsteps,
               "runtime": "ORT reference QDQ (ORT_DISABLE_ALL)" if ref else "ORT fused integer kernels (ORT_ENABLE_ALL)",
               "exact": float(ex[-1]), "cell": float(cell[-1]), "exact_per_step": [float(e) for e in ex],
               "cell_per_step": [float(c) for c in cell], "correct": (r["preds"][-1] == lab).all(-1).astype(int).tolist(),
               "secs": time.time() - t0}
        out.write_text(json.dumps(res))
        print(f"[{task}] {name}{' [ref]' if ref else ''} n={count}: exact {ex[-1]:.3f} cell {cell[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)


def stage_report(task):
    """Table of every evaluated variant (exact %, cell %, per n) with a Wilson 95% interval."""
    p16, p8, abl = paths(task)
    reg = json.loads((abl / "variants.json").read_text()) if (abl / "variants.json").exists() else {}
    rows = []
    for f in sorted(abl.glob("eval_*.json")):
        r = json.loads(f.read_text())
        k, n = sum(r["correct"]), r["count"]
        z = 1.96
        c = (k + z * z / 2) / (n + z * z)
        h = z * ((k * (n - k) / n + z * z / 4) ** 0.5) / (n + z * z)
        info = reg.get(r["variant"], {})
        rows.append({**{x: r[x] for x in ("variant", "start", "count", "nsteps", "exact", "cell")},
                     "wilson95": [c - h, c + h], "eight_groups": info.get("eight_groups"),
                     "n_tensors_8bit": info.get("n_tensors_8bit"), "n_edges_8bit": info.get("n_edges_8bit"),
                     "exact_per_step": r["exact_per_step"]})
    (abl / "report.json").write_text(json.dumps(rows, indent=1))
    for r in rows:
        print(f"{r['variant']:28s} n={r['count']:4d} (start {r['start']}) exact {r['exact']*100:5.1f} "
              f"[{r['wilson95'][0]*100:.1f}, {r['wilson95'][1]*100:.1f}] cell {r['cell']*100:5.1f} "
              f"8-bit tensors {r['n_tensors_8bit']} edges {r['n_edges_8bit']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["stats", "build", "eval", "report"])
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--variants", default="")
    ap.add_argument("--kill", default="", help="comma list of groups for the B/C fix variants")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--nsteps", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--ref", action="store_true", help="reference QDQ semantics (graph optimizations off)")
    a = ap.parse_args()
    names = [v for v in a.variants.split(",") if v]
    if a.stage == "stats":
        stage_stats(a.task)
    elif a.stage == "build":
        stage_build(a.task, names, [k for k in a.kill.split(",") if k] or None)
    elif a.stage == "report":
        stage_report(a.task)
    else:
        stage_eval(a.task, names, a.start, a.count, a.nsteps, a.threads, a.ref)


if __name__ == "__main__":
    main()
