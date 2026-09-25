#!/usr/bin/env python3
"""Pre-deployment release check for compressed recursive reasoners (M4).

Question it answers, before any device run and without labels: does this compressed / compiled
recursive step (TRM-style carry step: (z_H, z_L, inputs, puzzle_embedding) -> (z_H', z_L', logits,
q_halt)) still solve what the fp32 step solves when it is iterated?

Evidence, in three tiers (all label-free; only the fp32 reference and unlabeled inputs are needed):
  T0  per-layer reference comparison (ML-EXray pattern): every compute tensor of the candidate graph
      against the fp32 graph on the same input at fp32's end state (propagated relative error), the
      local error of every activation quantize/dequantize pair, and the static error of every
      quantized weight. For TFLite graphs also the same graph on two runtimes (deployment kernels vs
      reference kernels, and TFLite Micro), op by op in output LSBs.
  T1  one-step map perturbation delta_T = ||F~(s_T) - F(s_T)|| / ||s_T|| at fp32's end state s_T
      (X2), and the halting-logit sign agreement after that one step.
  T2  rollout from the initial carry for the deployment step count: per-cell agreement of the
      final answer with fp32 (the cell unit, X8), whole-answer agreement, phi_H (cosine of the final
      z_H with fp32's), halting-head sign agreement.
Verdict: PASS / WARN / FAIL with the rule that fired (thresholds in `THRESHOLDS`, fitted and
validated held-out by `m4_table.py`).

One command per artifact:
    # unlabeled check inputs (initial carry, puzzle embedding, inputs) for a task
    python release_check.py prep  --task sudoku_mlp --n 32 --out check_sudoku_mlp.npz
    # an exported graph (ONNX QDQ from AI Hub / AIMET, or any ONNX with the carry-step interface)
    python release_check.py onnx  --ref stepc_H1.onnx --graph qdq_w8a8/model.onnx \
                                  --check check_sudoku_mlp.npz --steps 16 --out ev.json
    # a TFLite carry-step graph (interpreter; optional reference-kernel and TFLM cross-runtime check)
    python release_check.py tflite --ref stepc_fp32.tflite --graph stepc_int8.tflite --check ... --xrt
    # a torch model plus a quantization config (weights and/or activations; see torch_check)
    python release_check.py torch --task sudoku_mlp --config w8c+a8all --H 1 --steps 16
    # device-facing tier: the device's predictions for its first 2 steps on the check inputs
    python release_check.py probe --ref stepc_H1.onnx --check check_sudoku_mlp.npz --device_preds dev.npz
"""
import argparse
import hashlib
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
for extra in (HERE, pathlib.Path.home() / "x3", pathlib.Path.home() / "edgetrm_src"):
    if (extra / "aihub_h1.py").exists() and str(extra) not in sys.path:
        sys.path.append(str(extra))

# Decision thresholds. Defaults are the values fitted in m4_table.py (see M4.md); a JSON written by
# `m4_table.py fit` overrides them with --thresholds.
THRESHOLDS = {
    # T2 (rollout; primary): agreement with fp32 on the cells fp32 writes, over inputs fp32 itself settled on,
    # and whole-answer agreement on the same inputs. Rounded from the fits in m4_table.py (M4.md): on 176
    # same-runtime cells no BROKEN cell escapes FAIL and no not-OK cell gets PASS.
    "written_fail": 0.80,        # below -> FAIL
    "written_pass": 0.95,        # at or above (with answer agreement >= answer_pass) -> PASS, else WARN
    "answer_fail": 0.50,         # whole-answer agreement below -> FAIL (long outputs: a maze keeps 0.92
    "answer_pass": 0.90,         #   written-cell agreement while its answers break)
    # T1 (one step at fp32's end state; the cheap tier when no rollout is run). 251 cells incl. X2's:
    # no not-OK cell below delta_pass; 68/73 BROKEN above delta_fail.
    "delta_pass": 0.12,
    "delta_fail": 0.40,
    # T0: an op whose output differs by more than this many LSB between reference and optimized kernels
    # while its inputs agree within 1 LSB -> FAIL for the reference-kernel runtime (TFLite Micro)
    "xrt_lsb_fail": 16,
    "layer_local_warn": 0.5,     # evidence only (localises a failure)
    "device_written_pass": 0.5,  # device probe: written-cell agreement with fp32 after 2 device steps
}

OUTS = ["z_H_out", "z_L_out", "logits", "q_halt"]
INS = ["z_H", "z_L", "inputs", "puzzle_embedding"]


# =============================================================================== check inputs
def prep(task, n, out, start=None, steps_ref=None):
    """Unlabeled check inputs, disjoint from every evaluation slice used in X3/M4: Sudoku test puzzles
    200.. (eval 0-99 and 1000-1999, calibration 900+), maze 100.., ARC canonical rows 100..; the
    initial carry (H_init, L_init) and the puzzle-embedding row each input uses."""
    import aihub_h1 as A
    t = A.TASKS[task]
    m, cfg, sd = A.build(task, 1)
    seq = cfg["seq_len"] + A.PLEN
    zH0 = np.broadcast_to(m.inner.H_init.detach().numpy(), (1, seq, 512)).astype(np.float32)
    zL0 = np.broadcast_to(m.inner.L_init.detach().numpy(), (1, seq, 512)).astype(np.float32)
    W = sd["inner.puzzle_emb.weights"].float().numpy()
    if t["data"] == "arc":
        import aihub_infer as AI
        s0 = 100 if start is None else start
        inp, _lab, pids = AI.arc_rows(s0 + n)
        inp, pids = inp[s0:], pids[s0:]
        emb = W[pids][:, None].astype(np.float32)
        idx = np.arange(s0, s0 + n)
    else:
        s0 = (200 if t["data"] == "sudoku" else 100) if start is None else start
        idx = np.arange(s0, s0 + n)
        inp, _lab = A.load_data(task, list(idx))
        emb = np.repeat(W[:1][None], n, 0).astype(np.float32)
    np.savez_compressed(out, task=task, index=idx, inputs=inp.astype(np.int32), puzzle_embedding=emb,
                        z_H0=zH0, z_L0=zL0)
    print(f"[prep] {task}: {n} unlabeled inputs (indices {idx[0]}..{idx[-1]}) -> {out}", flush=True)


def load_check(path, n=None):
    d = np.load(path, allow_pickle=False)
    k = d["inputs"].shape[0] if n is None else min(n, d["inputs"].shape[0])
    return dict(task=str(d["task"]), inputs=d["inputs"][:k], emb=d["puzzle_embedding"][:k],
                zH0=d["z_H0"], zL0=d["z_L0"], index=d["index"][:k])


def _external_locations(m):
    locs = set()
    for t in m.graph.initializer:
        for kv in t.external_data:
            if kv.key == "location":
                locs.add(kv.value)
    return sorted(locs)


def fuse_rmsnorm_qdq(m, out_float=False):
    """Vendor-kernel emulation (QNN): the QNN converter fuses RMSNorm (x*x -> mean -> +eps -> sqrt -> 1/.
    -> x * .) into one op whose internals are not quantized; only its input and output carry
    activation encodings. On an exported QDQ graph this removes the Q/DQ pairs inside every RMSNorm:
    each chain node (ReduceMean, Add eps, Sqrt, Div, and the normalising Mul) reads the float source
    of a dequantized chain tensor or constant directly. The norm input (the residual stream) and the
    norm output keep their encodings. Returns (model, number of norms fused)."""
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}

    def src_of(t):
        """float tensor behind DQ(Q(t)), else t"""
        d = prod.get(t)
        if d is not None and d.op_type == "DequantizeLinear":
            q = prod.get(d.input[0])
            if q is not None and q.op_type == "QuantizeLinear":
                return q.input[0]
        return t

    fused = 0
    for r in [n for n in g.node if n.op_type == "ReduceMean"]:
        sq = prod.get(src_of(r.input[0]))
        if sq is None or sq.op_type not in ("Mul", "Pow"):
            continue
        if sq.op_type == "Mul" and src_of(sq.input[0]) != src_of(sq.input[1]):
            continue
        xin = src_of(sq.input[0])
        chain = [r]
        cur = r
        ok = True
        for want in ("Add", "Sqrt", "Div"):
            nxt = [n for n in g.node if any(src_of(i) == cur.output[0] for i in n.input) and n.op_type == want]
            if not nxt:
                ok = False
                break
            cur = nxt[0]
            chain.append(cur)
        if not ok:
            continue
        # the normalising multiply: x * (1/sqrt(.))
        muls = [n for n in g.node if n.op_type == "Mul" and any(src_of(i) == cur.output[0] for i in n.input)]
        chain += muls[:1]
        if out_float and muls:
            # variant: the fused norm's output also stays in high precision (its consumers read it raw)
            # the normalised stream stays high precision through layout ops (transpose, cast, reshape)
            PASS_OPS = {"Cast", "Transpose", "Reshape", "Identity", "Squeeze", "Unsqueeze", "Flatten"}
            F = {muls[0].output[0]}
            grew = True
            while grew:
                grew = False
                for n in g.node:
                    if n.op_type in PASS_OPS and src_of(n.input[0]) in F and n.output[0] not in F:
                        F.add(n.output[0]); grew = True
            for n in g.node:
                if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
                    continue
                for k, i in enumerate(n.input):
                    if i not in F and src_of(i) in F:
                        n.input[k] = src_of(i)
        for n in chain:
            for k, i in enumerate(n.input):
                s_ = src_of(i)
                if s_ != i and s_ != xin:        # keep the quantized norm input x
                    n.input[k] = s_
        fused += 1
    return m, fused


def _float_constants(m):
    """Emulation variant: scalar/elementwise Constant-node operands are used unquantized (a kernel that
    folds them in higher precision); weight initializers keep their encodings."""
    g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    consts = {n.output[0] for n in g.node if n.op_type == "Constant"}
    k_ = 0
    for n in g.node:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for k, i in enumerate(n.input):
            d = prod.get(i)
            if d is not None and d.op_type == "DequantizeLinear":
                q = prod.get(d.input[0])
                if q is not None and q.op_type == "QuantizeLinear" and q.input[0] in consts:
                    n.input[k] = q.input[0]
                    k_ += 1
    return k_


def _inline_external(m, base):
    import onnx
    for t in m.graph.initializer:
        ext = {kv.key: kv.value for kv in t.external_data}
        if ext.get("location"):
            with open(base / ext["location"], "rb") as f:
                f.seek(int(ext.get("offset", 0)))
                raw = f.read(int(ext["length"])) if "length" in ext else f.read()
            del t.external_data[:]
            t.data_location = onnx.TensorProto.DEFAULT
            t.raw_data = raw


def _init_arrays(path):
    """Initializer values without onnx's external-data loader (reads the external file directly)."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path), load_external_data=False)
    base = pathlib.Path(path).resolve().parent
    out = {}
    for t in m.graph.initializer:
        ext = {kv.key: kv.value for kv in t.external_data}
        if ext.get("location"):
            with open(base / ext["location"], "rb") as f:
                f.seek(int(ext.get("offset", 0)))
                raw = f.read(int(ext["length"])) if "length" in ext else f.read()
            t2 = onnx.TensorProto()
            t2.CopyFrom(t)
            del t2.external_data[:]
            t2.data_location = onnx.TensorProto.DEFAULT
            t2.raw_data = raw
            out[t.name] = numpy_helper.to_array(t2)
        else:
            out[t.name] = numpy_helper.to_array(t)
    return out


# =============================================================================== runners
class OnnxStep:
    """One carry step on ONNX Runtime (batch 1 graphs, as AI Hub exports them)."""

    def __init__(self, path, threads=4, extra_outputs=None, vendor=None, opt_level="basic"):
        """opt_level 'basic' keeps the reference QDQ semantics (float ops between Q/DQ); 'all' lets ORT
        fuse Q/DQ into integer QLinear kernels, which changes the arithmetic of graphs that mix 8- and
        16-bit encodings (M3, Sep 25): a different runtime, not a different model."""
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        so.graph_optimization_level = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                       "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
                                       "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[opt_level]
        self.opt_level = opt_level
        self.path = str(path)
        self.extra = list(extra_outputs or [])
        self.n_fused = 0
        if self.extra or vendor:
            # graph with extra outputs, external data read into memory by hand (onnx's loader refuses
            # hard-linked external files and ORT refuses paths outside the model directory; the artifact
            # directories stay read-only for the check)
            import onnx
            m = onnx.load(self.path, load_external_data=False)
            _inline_external(m, pathlib.Path(self.path).resolve().parent)
            if vendor in ("qnn", "qnn16", "qnn16c"):
                m, self.n_fused = fuse_rmsnorm_qdq(m, out_float=vendor != "qnn")
            if vendor == "qnn16c":
                _float_constants(m)
            have = {o.name for o in m.graph.output}
            for name in self.extra:
                if name not in have:
                    m.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
            self.sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
            del m
        else:
            self.sess = ort.InferenceSession(self.path, so, providers=["CPUExecutionProvider"])

    def __call__(self, zH, zL, inp, emb, with_extra=False):
        names = OUTS + (self.extra if with_extra else [])
        o = self.sess.run(names, {"z_H": zH, "z_L": zL, "inputs": inp, "puzzle_embedding": emb})
        out = (o[0], o[1], o[2], float(np.asarray(o[3]).reshape(-1)[0]))
        if with_extra:
            return out, dict(zip(self.extra, o[4:]))
        return out


def rollout(step, chk, T, keep_states=(), log=None):
    """T steps from the initial carry for every check input. Returns per-step predictions [T, N, L],
    halting logits [T, N], final (z_H, z_L) [N, ...], and the states entering the listed steps."""
    N = chk["inputs"].shape[0]
    preds, qs, fin_H, fin_L = [], [], [], []
    kept = {s: [] for s in keep_states}
    for i in range(N):
        zH, zL = chk["zH0"], chk["zL0"]
        inp, emb = chk["inputs"][i:i + 1], chk["emb"][i]
        p_i, q_i = [], []
        for s in range(T):
            if s in kept:
                kept[s].append((zH, zL))
            zH, zL, lg, q = step(zH, zL, inp, emb)
            p_i.append(lg.argmax(-1)[0]); q_i.append(q)
        if T in kept:
            kept[T].append((zH, zL))
        preds.append(np.stack(p_i)); qs.append(q_i); fin_H.append(zH[0]); fin_L.append(zL[0])
        if log:
            log(f"  rollout {i + 1}/{N}")
    return dict(preds=np.stack(preds, 1), q=np.array(qs).T, zH=np.stack(fin_H), zL=np.stack(fin_L),
                kept={s: v for s, v in kept.items()})


def _rel(a, b):
    """Relative L2 error; a non-finite candidate value (inf/nan where the reference is finite) is
    reported as 1e9 so it sorts first and survives JSON."""
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    if not np.isfinite(a).all() and np.isfinite(b).all():
        return 1e9
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def _cos(a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# =============================================================================== T1 / T2 (any runner)
def changed_cell_agreement(fin_c, fin_r, inputs, settled):
    """Agreement on the cells fp32 writes (its final prediction differs from the input token): the
    Sudoku blanks, the maze path, the ARC output grid. Copied cells (givens, walls) agree trivially and
    swamp a whole-grid agreement on long outputs (a maze with 900 cells keeps 0.98 agreement while
    every path is wrong). Label-free: uses the input and fp32's own answer."""
    fin_c, fin_r, inputs = np.asarray(fin_c), np.asarray(fin_r), np.asarray(inputs)
    L = min(fin_r.shape[-1], inputs.shape[-1])
    ch = fin_r[:, :L] != inputs[:, :L]
    agree = (fin_c[:, :L] == fin_r[:, :L]) & ch
    per = agree.sum(-1) / np.maximum(ch.sum(-1), 1)
    out = dict(changed_cells_mean=float(ch.sum(-1).mean()), changed_agree=float(per.mean()),
               changed_agree_min=float(per.min()))
    settled = np.asarray(settled, bool)
    out["changed_agree_settled"] = float(per[settled].mean()) if settled.any() else None
    return out


def map_and_rollout(ref_step, cand_step, chk, T, ref_cache=None, log=None, skip_rollout=False):
    """T1 and T2 evidence for a candidate step against the fp32 step. `ref_cache` (npz path) keeps the
    fp32 rollout so a sweep over candidates computes it once."""
    t0 = time.time()
    ref = None
    if ref_cache and pathlib.Path(ref_cache).exists():
        z = np.load(ref_cache)
        if z["preds"].shape[:2] == (T, chk["inputs"].shape[0]):
            ref = dict(preds=z["preds"], q=z["q"], zH=z["zH"], zL=z["zL"],
                       sT=(z["sT_H"], z["sT_L"]), next=(z["nx_H"], z["nx_L"], z["nx_lg"], z["nx_q"]))
    if ref is None:
        r = rollout(ref_step, chk, T, keep_states=(T,))
        sT = r["kept"][T]
        nx = [ref_step(zH, zL, chk["inputs"][i:i + 1], chk["emb"][i]) for i, (zH, zL) in enumerate(sT)]
        ref = dict(preds=r["preds"], q=r["q"], zH=r["zH"], zL=r["zL"],
                   sT=(np.stack([a for a, _ in sT]), np.stack([b for _, b in sT])),
                   next=(np.stack([o[0] for o in nx]), np.stack([o[1] for o in nx]),
                         np.stack([o[2] for o in nx]), np.array([o[3] for o in nx])))
        if ref_cache:
            np.savez_compressed(ref_cache, preds=ref["preds"], q=ref["q"], zH=ref["zH"], zL=ref["zL"],
                                sT_H=ref["sT"][0], sT_L=ref["sT"][1], nx_H=ref["next"][0], nx_L=ref["next"][1],
                                nx_lg=ref["next"][2], nx_q=ref["next"][3])
    N = chk["inputs"].shape[0]
    # ---- T1: one candidate step at fp32's end state
    deltas, d_H, pred1, qsign1 = [], [], [], []
    for i in range(N):
        zH, zL = ref["sT"][0][i], ref["sT"][1][i]
        cH, cL, clg, cq = cand_step(zH, zL, chk["inputs"][i:i + 1], chk["emb"][i])
        rH, rL, rlg, rq = ref["next"][0][i], ref["next"][1][i], ref["next"][2][i], ref["next"][3][i]
        num = np.linalg.norm(np.concatenate([(cH - rH).ravel(), (cL - rL).ravel()]).astype(np.float64))
        den = np.linalg.norm(np.concatenate([zH.ravel(), zL.ravel()]).astype(np.float64))
        deltas.append(float(num / den)); d_H.append(_rel(cH, rH))
        pred1.append(float((clg.argmax(-1) == rlg.argmax(-1)).mean()))
        qsign1.append(float((cq > 0) == (rq > 0)))
    ev = dict(n=N, T=T,
              delta_T_median=float(np.median(deltas)), delta_T_mean=float(np.mean(deltas)),
              delta_T_max=float(np.max(deltas)), delta_T_H_median=float(np.median(d_H)),
              onestep_cell_agree=float(np.mean(pred1)), onestep_qsign_agree=float(np.mean(qsign1)),
              delta_T_per_input=deltas, t1_seconds=round(time.time() - t0, 1))
    if skip_rollout:
        return ev
    # ---- T2: rollout
    t1 = time.time()
    c = rollout(cand_step, chk, T, log=log)
    fin_c, fin_r = c["preds"][-1], ref["preds"][-1]
    cell = (fin_c == fin_r).mean(-1)
    # label-free subsets where fp32's own answer is meaningful: settled (unchanged over the last 3 steps)
    # and halted (fp32's halting logit > 0). Disagreement on inputs fp32 itself has not settled on is
    # not a loss (both answers are wrong there, differently).
    k = min(3, T)
    settled = (ref["preds"][-k:] == fin_r[None]).all(-1).all(0)
    halted = ref["q"][-1] > 0
    for name, msk in (("settled", settled), ("halted", halted)):
        ev[f"n_{name}"] = int(msk.sum())
        ev[f"answer_agree_{name}"] = float((fin_c[msk] == fin_r[msk]).all(-1).mean()) if msk.any() else None
        ev[f"cell_agree_{name}"] = float((fin_c[msk] == fin_r[msk]).mean()) if msk.any() else None
    ev.update(changed_cell_agreement(fin_c, fin_r, chk["inputs"], settled))
    ev.update(cell_agree=float(cell.mean()), cell_agree_min=float(cell.min()),
              answer_agree=float((fin_c == fin_r).all(-1).mean()),
              cell_agree_per_step=[float((c["preds"][s] == ref["preds"][s]).mean()) for s in range(T)],
              phi_H=float(np.mean([_cos(c["zH"][i], ref["zH"][i]) for i in range(N)])),
              qsign_agree=float(((c["q"][-1] > 0) == (ref["q"][-1] > 0)).mean()),
              q_fp32_pos=float((ref["q"][-1] > 0).mean()), q_cand_pos=float((c["q"][-1] > 0).mean()),
              t2_seconds=round(time.time() - t1, 1))
    ev["_cand_rollout"] = c
    return ev


# =============================================================================== T0: ONNX per-layer
COMPUTE_OPS = {"MatMul", "Gemm", "Add", "Sub", "Mul", "Div", "Sqrt", "Reciprocal", "ReduceMean", "Pow",
               "Sigmoid", "Softmax", "Gather", "Tanh", "Erf", "Relu", "Neg", "Exp", "Log", "Conv",
               "LayerNormalization", "SimplifiedLayerNormalization"}


def onnx_layer_map(cand_path, vendor=None):
    """Tensor names for the per-layer comparison: compute-op outputs of the candidate that also exist
    in the fp32 graph (AIMET/QDQ graphs keep the fp32 names), activation Q/DQ pairs (x -> DQ(Q(x))),
    and quantized weights (initializer -> DQ(Q(w)) or initializer Q-weights)."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(cand_path), load_external_data=False)
    if vendor in ("qnn", "qnn16", "qnn16c"):
        m, _ = fuse_rmsnorm_qdq(m, out_float=vendor != "qnn")
    if vendor == "qnn16c":
        _float_constants(m)
    g = m.graph
    inits = {i.name for i in g.initializer}
    consts = {n.output[0] for n in g.node if n.op_type == "Constant"}
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    compute = [n.output[0] for n in g.node if n.op_type in COMPUTE_OPS]
    pairs, wpairs = [], []
    for n in g.node:
        if n.op_type != "QuantizeLinear":
            continue
        x = n.input[0]
        dqs = [c for c in cons.get(n.output[0], []) if c.op_type == "DequantizeLinear"]
        dqs = [c for c in dqs if cons.get(c.output[0])]          # skip pairs a fusion left unused
        if not dqs:
            continue
        if x in inits or x in consts:
            wpairs.append((x, dqs[0].output[0]))
        else:
            src = prod.get(x)
            pairs.append((x, dqs[0].output[0], src.op_type if src is not None else "input"))
    # weights stored already quantized (DQ on an integer initializer)
    for n in g.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in inits and n.input[0] not in {w for w, _ in wpairs}:
            src = n.input[0]
            if src in inits:
                wpairs.append((None, n.output[0]))
    return dict(compute=compute, act_pairs=pairs, weight_pairs=wpairs,
                op_of={o: prod[o].op_type for o in compute}, n_nodes=len(g.node))


def onnx_per_layer(ref_path, cand_path, chk, ref_states, samples=(0, 1), threads=4, top=12, vendor=None,
                   opt_level="basic"):
    """T0 on an ONNX candidate: one step at fp32's end state for `samples` check inputs, every shared
    compute tensor compared with the fp32 graph, every activation Q/DQ pair's local error."""
    lm = onnx_layer_map(cand_path, vendor)
    import onnx
    ref_names = {o for n in onnx.load(str(ref_path), load_external_data=False).graph.node for o in n.output}
    shared = [t for t in lm["compute"] if t in ref_names]
    pair_names = [x for x, _, _ in lm["act_pairs"]] + [y for _, y, _ in lm["act_pairs"]]
    cand = OnnxStep(cand_path, threads, extra_outputs=sorted(set(shared + pair_names)), vendor=vendor,
                    opt_level=opt_level)
    ref = OnnxStep(ref_path, threads, extra_outputs=shared)
    prop, local, nonfinite = {}, {}, set()
    for i in samples:
        zH, zL = ref_states[0][i], ref_states[1][i]
        _, ce = cand(zH, zL, chk["inputs"][i:i + 1], chk["emb"][i], with_extra=True)
        _, re_ = ref(zH, zL, chk["inputs"][i:i + 1], chk["emb"][i], with_extra=True)
        for t in shared:
            if t in ce and t in re_ and np.issubdtype(np.asarray(re_[t]).dtype, np.floating) \
                    and np.asarray(re_[t]).size > 1:
                prop.setdefault(t, []).append(_rel(ce[t], re_[t]))
        for x, y, op in lm["act_pairs"]:
            if x in ce and y in ce and np.asarray(ce[x]).size >= 64:      # tiny tensors (the 2 halting
                local.setdefault((x, op), []).append(_rel(ce[y], ce[x]))   # logits) are not layers
        nonfinite.update(t for t, v in ce.items() if np.issubdtype(np.asarray(v).dtype, np.floating)
                         and not np.isfinite(v).all())
    prop = {t: float(np.median(v)) for t, v in prop.items()}
    local = {k: float(np.median(v)) for k, v in local.items()}
    # weights: static error of every quantized weight (needs the initializer values)
    werr = []
    if lm["weight_pairs"]:
        vals = _init_arrays(cand_path)
        # evaluate DQ(Q(w)) by running only those nodes would need a subgraph; use the scale/zp directly
        wq = _onnx_weight_errors(cand_path, vals)
        werr = wq
    order_local = sorted(local.items(), key=lambda kv: -kv[1])
    order_prop = sorted(prop.items(), key=lambda kv: -kv[1])
    ev = dict(n_shared_tensors=len(shared), n_act_pairs=len(lm["act_pairs"]), n_nonfinite_tensors=len(nonfinite),
              nonfinite_examples=sorted(nonfinite)[:5],
              layer_local_max=order_local[0][1] if order_local else 0.0,
              layer_local_p99=float(np.quantile([v for _, v in order_local], 0.99)) if order_local else 0.0,
              layer_local_top=[dict(tensor=k[0], op=k[1], rel_err=v) for k, v in order_local[:top]],
              layer_prop_out=float(np.median([prop.get(t, np.nan) for t in shared[-3:]])) if shared else None,
              layer_prop_max=order_prop[0][1] if order_prop else 0.0,
              layer_prop_top=[dict(tensor=k, op=lm["op_of"].get(k), rel_err=v) for k, v in order_prop[:top]],
              layer_local_by_op=_by_op(local), layer_prop_by_op=_by_op({(k, lm["op_of"].get(k)): v
                                                                        for k, v in prop.items()}),
              weight_err_max=max((w["rel_err"] for w in werr), default=0.0),
              weight_err_median=float(np.median([w["rel_err"] for w in werr])) if werr else 0.0,
              weight_err_top=sorted(werr, key=lambda w: -w["rel_err"])[:5])
    # first tensor (in graph order) whose propagated error exceeds 0.5: where the damage enters
    first = next((t for t in shared if prop.get(t, 0) > 0.5), None)
    ev["layer_first_prop_gt_0.5"] = dict(tensor=first, op=lm["op_of"].get(first)) if first else None
    return ev


def onnx_xrt(cand_path, chk, ref_states, threads=4, vendor=None, samples=(0, 1, 2, 3)):
    """T0 across runtimes for an ONNX QDQ graph (ML-EXray): one step at fp32's end state under reference
    QDQ semantics (ORT basic) and under ORT's integer-kernel fusions (ORT all); relative difference of
    the new carry. A large value means the graph's numerics depend on the runtime's kernels."""
    a = OnnxStep(cand_path, threads, vendor=vendor, opt_level="basic")
    b = OnnxStep(cand_path, threads, vendor=vendor, opt_level="all")
    d, agree = [], []
    for i in samples:
        if i >= len(ref_states[0]):
            break
        args = (ref_states[0][i], ref_states[1][i], chk["inputs"][i:i + 1], chk["emb"][i])
        oa, ob = a(*args), b(*args)
        d.append(_rel(np.concatenate([oa[0].ravel(), oa[1].ravel()]), np.concatenate([ob[0].ravel(), ob[1].ravel()])))
        agree.append(float((oa[2].argmax(-1) == ob[2].argmax(-1)).mean()))
    return dict(onnx_xrt_rel_median=float(np.median(d)), onnx_xrt_argmax_agree=float(np.mean(agree)))


def _by_op(d):
    out = {}
    for (name, op), v in d.items():
        out.setdefault(op, []).append(v)
    return {op: dict(n=len(v), median=float(np.median(v)), max=float(np.max(v))) for op, v in out.items()}


def _onnx_weight_errors(cand_path, vals):
    """Static relative error of every weight Q/DQ pair: w vs (clip(round(w/s)+zp) - zp) * s, with the
    graph's own scale / zero point / axis / block size."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(cand_path), load_external_data=False)
    g = m.graph
    cvals = dict(vals)
    for n in g.node:
        if n.op_type == "Constant":
            cvals[n.output[0]] = numpy_helper.to_array(n.attribute[0].t)
    cons = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    out = []
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] not in cvals:
            continue
        w = cvals[n.input[0]].astype(np.float64)
        s = cvals.get(n.input[1]); zp = cvals.get(n.input[2]) if len(n.input) > 2 else None
        if s is None or w.size < 64:
            continue
        s = s.astype(np.float64)
        zpv = zp.astype(np.float64) if zp is not None else np.zeros_like(s)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        axis = attrs.get("axis", 1)
        block = attrs.get("block_size", 0)
        if zp is not None:
            dt = zp.dtype
            if dt == np.int8:
                lo, hi = -128, 127
            elif dt == np.uint8:
                lo, hi = 0, 255
            elif dt == np.int16:
                lo, hi = -32768, 32767
            elif dt == np.uint16:
                lo, hi = 0, 65535
            else:  # int4 / uint4 stored as int8/uint8 by numpy_helper
                name = onnx.TensorProto.DataType.Name(
                    next(i for i in g.initializer if i.name == n.input[2]).data_type
                    if any(i.name == n.input[2] for i in g.initializer) else 3)
                lo, hi = (-8, 7) if name == "INT4" else (0, 15) if name == "UINT4" else (-128, 127)
            tname = None
            for i in g.initializer:
                if i.name == n.input[2]:
                    tname = onnx.TensorProto.DataType.Name(i.data_type)
            if tname == "INT4":
                lo, hi = -8, 7
            elif tname == "UINT4":
                lo, hi = 0, 15
        else:
            lo, hi = 0, 255
        if s.size == 1:
            S, Z = s.reshape(()), zpv.reshape(())
        elif block:
            S = np.repeat(s, block, axis=axis)[tuple(slice(0, d) for d in w.shape)]
            Z = np.repeat(zpv, block, axis=axis)[tuple(slice(0, d) for d in w.shape)]
        else:
            shape = [1] * w.ndim
            shape[axis] = -1
            S, Z = s.reshape(shape), zpv.reshape(shape)
        q = np.clip(np.round(w / S) + Z, lo, hi)
        wd = (q - Z) * S
        out.append(dict(weight=n.input[0], rel_err=_rel(wd, w), bits=int(np.log2(hi - lo + 1)),
                        granularity="tensor" if s.size == 1 else ("block%d" % block if block else "channel")))
    return out


# =============================================================================== TFLite runner
class TfliteStep:
    """One carry step on the TFLite interpreter. Inputs/outputs are matched by signature names
    (args_0..3 -> z_H, z_L, inputs, puzzle_embedding; outputs in the Step wrapper's order);
    quantized I/O is (de)quantized with the graph's own parameters, as a deployment would."""

    def __init__(self, path, threads=4, resolver="auto", preserve=False):
        self._args = (path, threads, preserve)
        self.resolver_used = resolver
        self._build(path, threads, resolver, preserve)

    def _build(self, path, threads, resolver, preserve):
        from ai_edge_litert import interpreter as I
        kind = {"auto": I.OpResolverType.AUTO, "builtin": I.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
                "ref": I.OpResolverType.BUILTIN_REF}[resolver]
        self.it = I.Interpreter(model_path=str(path), num_threads=threads, experimental_op_resolver_type=kind,
                                experimental_preserve_all_tensors=preserve)
        self.it.allocate_tensors()
        sig = self.it.get_signature_list()
        self.key = list(sig)[0]
        self.runner = self.it.get_signature_runner(self.key)
        self.in_det = self.runner.get_input_details()
        self.out_det = self.runner.get_output_details()
        self.in_names = sorted(self.in_det)            # args_0..args_3
        self.out_names = sorted(self.out_det, key=lambda s: int(s.split("_")[-1]))   # output_0..3

    @staticmethod
    def _q(x, det):
        dt = det["dtype"]
        if dt in (np.int8, np.int16, np.uint8):
            s, zp = det["quantization"]
            info = np.iinfo(dt)
            return np.clip(np.round(x / s + zp), info.min, info.max).astype(dt)
        return np.asarray(x).astype(dt)

    @staticmethod
    def _dq(y, det):
        dt = det["dtype"]
        if dt in (np.int8, np.int16, np.uint8):
            s, zp = det["quantization"]
            return (y.astype(np.float32) - zp) * s
        return y.astype(np.float32)

    def __call__(self, zH, zL, inp, emb):
        src = [zH, zL, inp, emb.reshape(1, -1) if emb.ndim == 1 else emb]
        feed = {n: self._q(src[k], self.in_det[n]) for k, n in enumerate(self.in_names)}
        try:
            o = self.runner(**feed)
        except ValueError:
            if self.resolver_used != "auto":
                raise
            # the XNNPACK delegate leaves some outputs unallocated on a few graphs (FC-only int8 with
            # float I/O): fall back to the builtin kernels, as a phone runtime would
            self.resolver_used = "builtin (fallback)"
            path, threads, preserve = self._args
            self._build(path, threads, "builtin", preserve)
            o = self.runner(**feed)
        vals = [self._dq(o[n], self.out_det[n]) for n in self.out_names]
        return vals[0], vals[1], vals[2], float(np.asarray(vals[3]).reshape(-1)[0])


def tflite_xrt(graph, chk, ref_states, sample=0, tflm_python=None):
    """T0 across runtimes for one quantized TFLite graph (ML-EXray): the same inputs through the
    optimized builtin kernels and the reference kernels (what TFLite Micro shares), op by op, output
    difference in LSB of each op's output scale. Optionally the TFLM runtime itself (separate venv;
    `tflm_python` = its interpreter) via `release_check.py _tflm_dump`."""
    from ai_edge_litert import interpreter as I
    from ai_edge_litert.tools import flatbuffer_utils as fu
    zH, zL = ref_states[0][sample], ref_states[1][sample]
    src = [zH, zL, chk["inputs"][sample:sample + 1], chk["emb"][sample].reshape(1, -1)]

    def run(kind):
        it = I.Interpreter(model_path=str(graph), experimental_preserve_all_tensors=True, num_threads=1,
                           experimental_op_resolver_type=kind)
        it.allocate_tensors()
        sig = it.get_signature_runner(list(it.get_signature_list())[0])
        det = sig.get_input_details()
        feed = {n: TfliteStep._q(src[k], det[n]) for k, n in enumerate(sorted(det))}
        sig(**feed)
        return it, feed
    try:
        a, feed = run(I.OpResolverType.BUILTIN_REF)
    except RuntimeError as e:
        # the reference kernels (TFLite Micro's) cannot even execute the graph on this input
        import re
        mm = re.search(r"Node number (\d+) \((\w+)\)", str(e))
        bad = dict(op_index=int(mm.group(1)) if mm else -1, op=mm.group(2) if mm else "?", maxdiff=float("inf"),
                   in_diff_lsb=0.0, unit="lsb", error=str(e)[:300])
        return dict(xrt_pair="builtin_ref_vs_builtin_opt", xrt_ref_error=str(e)[:300], xrt_first_bad=bad,
                    xrt_max_lsb=float("inf"))
    b, _ = run(I.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    m = fu.read_model(str(graph))
    sg = m.subgraphs[0]
    rows = []
    for k, op in enumerate(sg.operators):
        name = fu.opcode_to_name(m, op.opcodeIndex)
        for o in op.outputs:
            try:
                x, y = a.get_tensor(o), b.get_tensor(o)
            except ValueError:
                continue
            if x.dtype == np.float32:
                d = float(np.abs(x - y).max()); unit = "abs"
            else:
                d = float(np.abs(x.astype(np.int64) - y.astype(np.int64)).max()); unit = "lsb"
            rows.append(dict(op_index=k, op=name, tensor=int(o), dtype=str(x.dtype), maxdiff=d, unit=unit))
    # local disagreement: an op whose output differs by far more than its inputs do (a kernel that
    # computes something else), as opposed to divergence inherited from upstream ops. Input differences
    # are expressed in LSB of the op's output scale (float inputs: |diff| / output scale).
    tdiff = {}
    for k, op in enumerate(sg.operators):
        for o in op.outputs:
            try:
                x, y = a.get_tensor(o), b.get_tensor(o)
            except ValueError:
                continue
            tdiff[int(o)] = (x, y)
    for r in rows:
        op = sg.operators[r["op_index"]]
        q = sg.tensors[r["tensor"]].quantization
        oscale = float(q.scale[0]) if (q is not None and q.scale is not None and len(q.scale)) else None
        ind = 0.0
        for i in op.inputs:
            if i < 0 or int(i) not in tdiff:
                continue
            x, y = tdiff[int(i)]
            if x.dtype == np.float32:
                d = float(np.abs(x - y).max())
                ind = max(ind, d / oscale if oscale else d)
            else:
                qi = sg.tensors[i].quantization
                iscale = float(qi.scale[0]) if (qi is not None and qi.scale is not None and len(qi.scale)) else 1.0
                d = float(np.abs(x.astype(np.int64) - y.astype(np.int64)).max())
                ind = max(ind, d * iscale / oscale if oscale else d)
        r["in_diff_lsb"] = ind
    worst = sorted([r for r in rows if r["unit"] == "lsb"], key=lambda r: -r["maxdiff"])
    thr = THRESHOLDS["xrt_lsb_fail"]
    first_bad = next((r for r in rows if r["unit"] == "lsb" and r["maxdiff"] > thr
                      and r["in_diff_lsb"] <= 1.0), None)
    first_any = next((r for r in rows if r["unit"] == "lsb" and r["maxdiff"] > thr), None)
    out = dict(xrt_pair="builtin_ref_vs_builtin_opt", xrt_ops=len(rows),
               xrt_max_lsb=worst[0]["maxdiff"] if worst else 0.0, xrt_worst=worst[:6], xrt_first_bad=first_bad,
               xrt_first_propagated=first_any)
    if tflm_python:
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            np.savez(f"{td}/in.npz", __idx__=np.array([r["tensor"] for r in rows]),
                     **{f"in{j}": v for j, v in enumerate(feed[n] for n in sorted(feed))})
            ref_dump = {f"t{r['tensor']}": a.get_tensor(r["tensor"]) for r in rows}
            np.savez(f"{td}/lite.npz", **ref_dump)
            p = subprocess.run([tflm_python, __file__, "_tflm_dump", str(graph), f"{td}/in.npz", f"{td}/micro.npz"],
                               capture_output=True, text=True, timeout=3600)
            if p.returncode != 0:
                out["tflm_error"] = (p.stderr or p.stdout)[-600:]
            else:
                mic = np.load(f"{td}/micro.npz")
                trows = []
                for r in rows:
                    key = f"t{r['tensor']}"
                    if key in mic.files and r["unit"] == "lsb":
                        x, y = mic[key], ref_dump[key]
                        if x.shape == y.shape:
                            trows.append(dict(r, maxdiff=float(np.abs(x.astype(np.int64) - y.astype(np.int64)).max())))
                opt = {r["tensor"]: b.get_tensor(r["tensor"]) for r in rows if r["unit"] == "lsb"}
                trows2 = []
                for r in rows:
                    key = f"t{r['tensor']}"
                    if key in mic.files and r["unit"] == "lsb" and mic[key].shape == opt[r["tensor"]].shape:
                        trows2.append(dict(r, maxdiff=float(np.abs(mic[key].astype(np.int64)
                                                                    - opt[r["tensor"]].astype(np.int64)).max())))
                w2 = sorted(trows2, key=lambda r: -r["maxdiff"])
                out.update(tflm_vs_opt_max_lsb=w2[0]["maxdiff"] if w2 else None, tflm_vs_opt_worst=w2[:6],
                           tflm_first_bad=next((r for r in trows2 if r["maxdiff"] > THRESHOLDS["xrt_lsb_fail"]), None),
                           tflm_vs_ref_max_lsb=max((r["maxdiff"] for r in trows), default=None),
                           tflm_out_distinct=int(len(np.unique(mic["__out__"]))) if "__out__" in mic.files else None)
    return out


def _tflm_dump(graph, in_npz, out_npz):
    """(runs in the TFLM venv) the listed tensors (`__idx__` in the input file) after one TFLM invoke."""
    from tflite_micro.python.tflite_micro import runtime
    d = np.load(in_npz)
    it = runtime.Interpreter.from_file(graph, arena_size=1 << 30,
                                       intrepreter_config=runtime.InterpreterConfig.kPreserveAllTensors)
    j = 0
    while f"in{j}" in d.files:
        it.set_input(d[f"in{j}"], j); j += 1
    it.invoke()
    out = {}
    for k in d["__idx__"].tolist():
        t = it.GetTensor(int(k), 0)
        if isinstance(t, dict):
            t = t.get("tensor_data", t.get("data"))
        out[f"t{k}"] = np.asarray(t)
    out["__out__"] = np.asarray(it.get_output(2))
    np.savez(out_npz, **out)


# =============================================================================== torch: model + config
class ActFakeQuant:
    """Toolchain emulation for activations, as a torch function mode: every floating-point op output
    inside one recursion step gets a static per-tensor asymmetric fake quantization (min/max
    calibrated per call site on fp32 trajectories of calibration inputs), which is what the AI Hub /
    AIMET and AI Edge Quantizer default recipes do ('all' mode). 'lin' quantizes only the inputs of
    linear layers (the breadth harness's +a8). Carry in/out are quantized by the caller.
    Call sites are numbered in execution order within one step, so the unrolled applications of the
    shared block get separate encodings, as in the exported graphs."""
    SKIP = {"to", "type", "float", "double", "half", "view", "reshape", "permute", "transpose", "contiguous",
            "__getitem__", "expand", "expand_as", "size", "dim", "unsqueeze", "squeeze", "chunk", "split",
            "flatten", "unflatten", "detach", "numel", "__get__", "clone", "repeat", "narrow", "select",
            "cat", "stack", "__setitem__", "new_zeros", "zeros_like", "ones_like", "full_like", "view_as",
            "masked_fill", "__len__", "__iter__", "__format__", "__repr__", "item", "tolist", "movedim",
            "unbind", "t", "is_floating_point", "requires_grad_", "__bool__", "dtype", "device", "shape"}

    def __init__(self, bits=8, mode="all"):
        import torch
        from torch.overrides import TorchFunctionMode
        self.bits, self.mode = bits, mode
        self.ranges, self.calibrating, self.site = {}, True, 0
        outer = self

        class _M(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                name = getattr(func, "__name__", str(func))
                if outer.mode == "lin" and name == "linear":
                    args = (outer.q(args[0]),) + tuple(args[1:])
                    return func(*args, **kwargs)
                out = func(*args, **kwargs)
                if outer.mode == "all" and name not in outer.SKIP and isinstance(out, torch.Tensor) \
                        and out.is_floating_point() and out.numel() > 1 \
                        and not isinstance(out, torch.nn.Parameter):
                    out = outer.q(out)
                return out
        self._mode_cls = _M

    def q(self, x):
        import torch
        k = self.site
        self.site += 1
        if self.calibrating:
            lo, hi = float(x.min()), float(x.max())
            if k in self.ranges:
                a, b = self.ranges[k]
                self.ranges[k] = (min(a, lo), max(b, hi))
            else:
                self.ranges[k] = (lo, hi)
            return x
        lo, hi = self.ranges.get(k, (float(x.min()), float(x.max())))
        lo, hi = min(lo, 0.0), max(hi, 0.0)
        n = 2 ** self.bits - 1
        scale = max(hi - lo, 1e-12) / n
        zp = round(-lo / scale)
        return (torch.clamp(torch.round(x / scale) + zp, 0, n) - zp) * scale

    def __call__(self):
        self.site = 0
        return self._mode_cls()


def _torch_models(task, H, T, wspec, draw=0, batch=64):
    """fp32 and candidate models from the breadth harness (weight quantizers / noise by spec, optionally
    restricted to one weight group with spec@group, as in contraction.make_model)."""
    import torch
    import breadth_sweep as bs
    ref, _ = bs.build(task, H, T, batch)
    cand, _ = bs.build(task, H, T, batch)
    if wspec not in ("fp32", "", None):
        spec, group = (wspec.split("@") + [None])[:2]
        q = bs.parse_quant(spec)
        target = cand
        if group is not None:
            import contraction as C
            mods = [mod for name, mod in cand.named_modules()
                    if mod.__class__.__name__ == "CastedLinear" and C.GROUPS[group](name)]
            target = torch.nn.ModuleList(mods)
        if q["gran"] == "noise":
            torch.manual_seed(10_000 + 1000 * draw + int(round(q["sigma"] * 1000)))
        if q["gran"] is not None:
            bs.quantize_weights_(target, q["bits"], q["gran"], q.get("sigma"))
    for m in (ref, cand):
        for p_ in m.parameters():
            p_.requires_grad_(False)
    return ref.eval(), cand.eval()


class TorchStep:
    """F(z) of one supervision step (contraction.Recursion), optionally with activation fake quant
    (ActFakeQuant) inside and the carry quantized per tensor on the way in and out."""

    def __init__(self, m, family, x_tok, act=None, carry_bits=None):
        import contraction as C
        self.R = C.Recursion(m, family, x_tok)
        self.act, self.carry_bits = act, carry_bits
        self.cq = ActFakeQuant(carry_bits, "all") if carry_bits else None

    def __call__(self, z):
        import torch
        with torch.no_grad():
            if self.cq is not None:
                self.cq.site = 0
                z = torch.stack([self.cq.q(z[:, 0]), self.cq.q(z[:, 1])], 1)
            if self.act is None:
                out = self.R(z)
            else:
                with self.act():
                    out = self.R(z)
            if self.cq is not None:
                out = torch.stack([self.cq.q(out[:, 0]), self.cq.q(out[:, 1])], 1)
        return out

    def calibrating(self, flag):
        for fq in (self.act, self.cq):
            if fq is not None:
                fq.calibrating = flag


def torch_check(task, wspec="fp32", amode="none", H=None, T=16, check=None, n=32, calib_start=900,
                n_calib=16, draw=0, kicks=True, threads=16, batch=32, carry_bits=None, device="cpu",
                label_n=0, label_start=0):
    """T1/T2 evidence (and random-kick growth) for a torch model + quantization config.
    wspec: breadth_sweep spec for weights (w8c, w4c, w4t, w3g32, noise0.1, ...; '@mixer' etc. restricts
    it to a weight group); amode: none | a8all | a16all | a8lin (activation emulation)."""
    import torch
    import breadth_sweep as bs
    import contraction as C
    torch.set_num_threads(threads)
    family = "hrm" if task.startswith("hrm") else "trm"
    kind = bs.TRM_TASKS[task]["data"] if task in bs.TRM_TASKS else bs.HRM_TASKS[task]["data"]
    ref_m, cand_m = _torch_models(task, H, T, wspec, draw, batch)
    ref_m, cand_m = ref_m.to(device), cand_m.to(device)
    if check is not None:
        x_np = load_check(check, n)["inputs"].astype(np.int64)
    else:
        s0 = 200 if kind == "sudoku" else 100
        x_np = bs.load_qa(kind, s0 + n)[0][s0:]
    x = torch.as_tensor(x_np).to(device)
    act = None
    if amode != "none":
        bits = 16 if amode.startswith("a16") else 8
        act = ActFakeQuant(bits, "lin" if amode.endswith("lin") else "all")
    cb = carry_bits if carry_bits else (int(amode[1:-3]) if amode.endswith("all") else None)
    t0 = time.time()
    if act is not None or cb:
        # calibration: fp32 trajectories of calibration inputs (disjoint from the check inputs), every step
        xc = torch.as_tensor(bs.load_qa(kind, calib_start + n_calib)[0][calib_start:]).to(device)
        Fr_c = C.Recursion(ref_m, family, xc)
        Fq_c = TorchStep(cand_m, family, xc, act, cb)
        Fq_c.calibrating(True)
        z = Fr_c.z0()
        for _ in range(T):
            Fq_c(z)
            z = Fr_c(z)
        Fq_c.calibrating(False)
    Fr = C.Recursion(ref_m, family, x)
    Fq = TorchStep(cand_m, family, x, act, cb)
    Fq.calibrating(False)
    with torch.no_grad():
        z = Fr.z0()
        zs_r = []
        for _ in range(T):
            z = Fr(z); zs_r.append(z)
        sT = zs_r[-1]
        nxt = Fr(sT)
        d = (Fq(sT) - nxt).flatten(1).norm(dim=1) / sT.flatten(1).norm(dim=1)
        zc = Fr.z0()
        zs_c = []
        for _ in range(T):
            zc = Fq(zc); zs_c.append(zc)
        ro_r, ro_c = Fr.readout(sT, torch.zeros_like(x)), Fr.readout(zc, torch.zeros_like(x))
        pr, pc = ro_r["pred"], ro_c["pred"]
        cell = (pr == pc).float().mean(-1)
        prev = [Fr.readout(zs_r[-k], torch.zeros_like(x))["pred"] for k in (2, 3)]
        settled = ((prev[0] == pr).all(-1) & (prev[1] == pr).all(-1))
        halted = ro_r["qhalt"] > 0
        ev = dict(kind="torch", task=task, wspec=wspec, amode=amode, H=Fr.H, T=T, n=int(x.shape[0]),
                  delta_T_median=float(d.median()), delta_T_mean=float(d.mean()), delta_T_per_input=d.tolist(),
                  cell_agree=float(cell.mean()), answer_agree=float((pr == pc).all(-1).float().mean()),
                  phi_H=float(torch.nn.functional.cosine_similarity(sT[:, 0].flatten(1), zc[:, 0].flatten(1)).mean()),
                  qsign_agree=float(((ro_r["qhalt"] > 0) == (ro_c["qhalt"] > 0)).float().mean()))
        for name, msk in (("settled", settled), ("halted", halted)):
            ev[f"n_{name}"] = int(msk.sum())
            ev[f"answer_agree_{name}"] = float((pr[msk] == pc[msk]).all(-1).float().mean()) if msk.any() else None
            ev[f"cell_agree_{name}"] = float((pr[msk] == pc[msk]).float().mean()) if msk.any() else None
        ev.update(changed_cell_agreement(pc.cpu().numpy(), pr.cpu().numpy(), x.cpu().numpy(), settled.cpu().numpy()))
        if kicks and act is None and not cb:
            # random-kick growth along the candidate's own trajectory (X2): 16-step nonlinear growth of
            # a random kick of relative size 1e-3 at the candidate's end state (label- and reference-free)
            g = torch.Generator().manual_seed(0)
            u = torch.randn(zc.shape, generator=g).to(device)
            u = u / u.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * zc.flatten(1).norm(dim=1).view(-1, 1, 1, 1) * 1e-3
            a_, b_ = zc + u, zc
            for _ in range(16):
                a_, b_ = Fq(a_), Fq(b_)
            growth = (a_ - b_).flatten(1).norm(dim=1) / u.flatten(1).norm(dim=1)
            ev.update(kick_growth_median=float(growth.median()), kick_growth_per_input=growth.tolist())
    if label_n:
        # the emulated model's own accuracy on labeled evaluation puzzles (a label run, kept apart
        # from the check evidence: it is the torch-emulation backend's ground truth, not an input)
        xl_np, yl_np = bs.load_qa(kind, label_start + label_n)
        xl = torch.as_tensor(xl_np[label_start:]).to(device)
        yl = torch.as_tensor(yl_np[label_start:]).to(device)
        res = {}
        for tag, Fm in (("fp32", None), ("cand", "cand")):
            accs, cells = [], []
            for b in range(0, xl.shape[0], batch):
                xb, yb = xl[b:b + batch], yl[b:b + batch]
                Rb = C.Recursion(ref_m, family, xb)
                Fb = Rb if Fm is None else TorchStep(cand_m, family, xb, act, cb)
                if Fm is not None:
                    if act is not None:
                        Fb.act.ranges = act.ranges
                    if cb:
                        Fb.cq.ranges = Fq.cq.ranges
                    Fb.calibrating(False)
                with torch.no_grad():
                    zb = Rb.z0()
                    for _ in range(T):
                        zb = Fb(zb)
                    ro = Rb.readout(zb, yb)
                accs.append(ro["correct"].float()); cells.append(ro["cellacc"].float())
            res[tag] = (torch.cat(accs), torch.cat(cells))
        ev.update(label_n=label_n, label_start=label_start, label_exact=float(res["cand"][0].mean()),
                  label_exact_fp32=float(res["fp32"][0].mean()), label_cell=float(res["cand"][1].mean()),
                  label_cell_fp32=float(res["fp32"][1].mean()),
                  label_correct=res["cand"][0].bool().tolist(), label_correct_fp32=res["fp32"][0].bool().tolist())
    ev["seconds"] = round(time.time() - t0, 1)
    return ev


# =============================================================================== device probe
def device_probe(ref_step, chk, device_preds, k=2, threshold=0.5):
    """Device-facing tier: the device's own predictions for the first k supervision steps on the check
    inputs (from the initial carry), compared with the host fp32 model on the cells fp32 writes. Validated on
    10 AI Hub chains (M4.md): working 0.72-1.00 at step 2, failing 0.00-0.27. The reference is the fp32
    model, not the exported graph's host simulation (which itself can be wrong, e.g. Sudoku-MLP w8a8)."""
    device_preds = np.asarray(device_preds)
    K = min(k, device_preds.shape[0])
    r = rollout(ref_step, dict(chk, inputs=chk["inputs"][:device_preds.shape[1]],
                               emb=chk["emb"][:device_preds.shape[1]]), K)
    inp = chk["inputs"][:device_preds.shape[1]]
    per = []
    for s_ in range(K):
        ch = r["preds"][s_] != inp
        per.append(float((((device_preds[s_] == r["preds"][s_]) & ch).sum(-1) / np.maximum(ch.sum(-1), 1)).mean()))
    return dict(device_probe_steps=K, device_written_agree_per_step=per, device_written_agree=per[-1],
                device_probe_pass=bool(per[-1] >= threshold))  # threshold = THRESHOLDS["device_written_pass"]


# =============================================================================== decision
def decide(ev, thr=None):
    """PASS / WARN / FAIL with the rules that fired. The primary rule is T2 (per-cell agreement of the
    rollout); T1 (delta_T) decides when no rollout was run; T0 cross-runtime disagreement is a hard
    FAIL (kernel bug: the host simulation does not describe that runtime); T0 per-layer errors are
    evidence that localises a FAIL, not a verdict on their own."""
    thr = dict(THRESHOLDS, **(thr or {}))
    rules, verdict = [], "PASS"

    def bump(v):
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        return v if order[v] > order[verdict] else verdict
    if ev.get("xrt_first_bad"):
        bad = ev["xrt_first_bad"]
        if bad.get("error"):
            rules.append(f"T0 cross-runtime: the reference kernels fail at op {bad['op_index']} {bad['op']}: "
                         f"{bad['error'][:120]}")
        else:
          rules.append(f"T0 cross-runtime: op {bad['op_index']} {bad['op']} differs by {bad['maxdiff']:.0f} LSB "
                       f"between reference and optimized kernels while its inputs differ by "
                       f"{bad.get('in_diff_lsb', 0):.0f} (kernel disagreement, > {thr['xrt_lsb_fail']})")
        verdict = bump("FAIL")
    if ev.get("device_probe_pass") is False:
        rules.append(f"DEVICE probe: written-cell agreement with fp32 after {ev['device_probe_steps']} device "
                     f"steps {ev['device_written_agree']:.3f} < 0.5 (the vendor kernels compute a different function)")
        verdict = bump("FAIL")
    if ev.get("n_nonfinite_tensors"):
        rules.append(f"T0 per-layer: {ev['n_nonfinite_tensors']} intermediate tensors are inf/nan where fp32 is "
                     f"finite (e.g. {(ev.get('nonfinite_examples') or ['?'])[0]})")
        verdict = bump("FAIL")
    if "cell_agree" in ev:
        # T2 on inputs fp32 itself has settled on (label-free); all inputs when too few settle
        use_settled = ev.get("n_settled", 0) >= 5
        key = "changed_agree" if ev.get("changed_agree") is not None else "cell_agree"
        ca = ev.get(key + "_settled") if use_settled and ev.get(key + "_settled") is not None else ev[key]
        # whole-answer agreement only where fp32's own answer has settled (on unsettled inputs, e.g. ARC at 8
        # steps, fp32's answer is still moving and disagreement is not a loss)
        aa = ev.get("answer_agree_settled") if use_settled else None
        tag = ("cells fp32 writes" if key == "changed_agree" else "all cells") + \
              (", fp32-settled inputs" if use_settled else ", all inputs")
        if ca < thr["written_fail"] or (aa is not None and aa < thr["answer_fail"]):
            rules.append(f"T2 agreement with fp32 ({tag}) {ca:.3f} (< {thr['written_fail']}) / whole answers "
                         f"{aa if aa is None else round(aa, 3)} (< {thr['answer_fail']})")
            verdict = bump("FAIL")
        elif ca < thr["written_pass"] or (aa is not None and aa < thr["answer_pass"]):
            rules.append(f"T2 agreement with fp32 ({tag}) {ca:.3f} / whole answers "
                         f"{aa if aa is None else round(aa, 3)}: between FAIL and PASS bands")
            verdict = bump("WARN")
    elif "delta_T_median" in ev:
        d = ev["delta_T_median"]
        if d > thr["delta_fail"]:
            rules.append(f"T1 one-step perturbation {d:.3f} > {thr['delta_fail']}")
            verdict = bump("FAIL")
        elif d >= thr["delta_pass"]:
            rules.append(f"T1 one-step perturbation {d:.3f} in [{thr['delta_pass']}, {thr['delta_fail']}]")
            verdict = bump("WARN")
    if ev.get("layer_local_max", 0) > thr["layer_local_warn"]:
        top = ev["layer_local_top"][0]
        rules.append(f"T0 per-layer: activation {top['tensor']} ({top['op']}) local 8/16-bit error "
                     f"{top['rel_err']:.2f} (evidence)")
        if verdict == "PASS":
            verdict = "WARN"
    return verdict, rules


def _strip(ev):
    return {k: v for k, v in ev.items() if not k.startswith("_")}


# =============================================================================== CLI
def cmd_onnx(a):
    chk = load_check(a.check, a.n)
    log = (lambda s: print(s, flush=True)) if a.verbose else None
    ref = OnnxStep(a.ref, a.threads, opt_level=a.ort_level)
    cand = OnnxStep(a.graph, a.threads, vendor=a.vendor or None, opt_level=a.ort_level)
    cache = a.ref_cache or None
    ev = dict(kind="onnx", ref=str(a.ref), graph=str(a.graph), check=str(a.check), steps=a.steps,
              vendor=a.vendor or None, n_fused_norms=cand.n_fused, ort_level=a.ort_level)
    ev.update(map_and_rollout(ref, cand, chk, a.steps, ref_cache=cache, log=log, skip_rollout=a.no_rollout))
    if not a.no_layers:
        z = np.load(cache) if cache and pathlib.Path(cache).exists() else None
        states = (z["sT_H"], z["sT_L"]) if z is not None else None
        if states is None:
            r = rollout(ref, chk, a.steps, keep_states=(a.steps,))
            states = (np.stack([s[0] for s in r["kept"][a.steps]]), np.stack([s[1] for s in r["kept"][a.steps]]))
        t0 = time.time()
        ev.update(onnx_per_layer(a.ref, a.graph, chk, states, samples=tuple(range(min(2, chk["inputs"].shape[0]))),
                                 threads=a.threads, vendor=a.vendor or None, opt_level=a.ort_level))
        ev.update(onnx_xrt(a.graph, chk, states, threads=a.threads, vendor=a.vendor or None))
        ev["t0_seconds"] = round(time.time() - t0, 1)
    finish(ev, a)


def cmd_tflite(a):
    chk = load_check(a.check, a.n)
    ref = TfliteStep(a.ref, a.threads) if a.ref.endswith(".tflite") else OnnxStep(a.ref, a.threads)
    cand = TfliteStep(a.graph, a.threads, resolver=a.resolver)
    ev = dict(kind="tflite", ref=str(a.ref), graph=str(a.graph), resolver=a.resolver, check=str(a.check),
              steps=a.steps)
    if a.xrt_only and a.out and pathlib.Path(a.out).exists():
        ev = json.loads(pathlib.Path(a.out).read_text())
        for k in [k for k in ev if k.startswith(("xrt_", "tflm_"))]:
            ev.pop(k)
    else:
        ev.update(map_and_rollout(ref, cand, chk, a.steps, ref_cache=a.ref_cache or None,
                                  skip_rollout=a.no_rollout))
    if a.xrt:
        z = np.load(a.ref_cache)
        ev.update(tflite_xrt(a.graph, chk, (z["sT_H"], z["sT_L"]), tflm_python=a.tflm_python))
    finish(ev, a)


def finish(ev, a):
    thr = json.loads(pathlib.Path(a.thresholds).read_text()) if getattr(a, "thresholds", None) else None
    v, rules = decide(ev, thr)
    ev["verdict"], ev["rules"] = v, rules
    # per-runtime verdicts: the cross-runtime evidence condemns the runtime whose kernels disagree
    # (reference kernels = TFLite Micro), not the graph on the runtime that computes it correctly
    if ev.get("kind") == "tflite" and ev.get("xrt_first_bad"):
        e2 = {k: x for k, x in ev.items() if not k.startswith("xrt_")}
        v2, r2 = decide(e2, thr)
        ev["verdict_by_runtime"] = {"optimized kernels (XNNPACK / phone CPU)": v2,
                                    "reference kernels (TFLite Micro)": v}
    if ev.get("kind") == "onnx":
        ev["predicts_backend"] = ("QDQ graph under " + ("reference QDQ semantics" if ev.get("ort_level", "basic") == "basic"
                                  else "ONNX Runtime fused integer kernels")
                                  + (f", vendor kernel model {ev['vendor']}" if ev.get("vendor") else ""))
    ev = _strip(ev)
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(ev, indent=1, default=float))
    print(f"VERDICT {v}" + (f"  (by runtime: {ev['verdict_by_runtime']})" if ev.get("verdict_by_runtime") else "")
          + (f"  [predicts: {ev['predicts_backend']}]" if ev.get("predicts_backend") else ""))
    for r in rules:
        print(f"  - {r}")
    keys = ["delta_T_median", "onestep_cell_agree", "cell_agree", "answer_agree", "phi_H", "qsign_agree",
            "layer_local_max", "layer_prop_out", "weight_err_max", "xrt_max_lsb", "tflm_vs_opt_max_lsb"]
    print("  " + "  ".join(f"{k}={ev[k]:.4g}" for k in keys if isinstance(ev.get(k), (int, float))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest="cmd", required=True)
    q = sp.add_parser("prep"); q.add_argument("--task", required=True); q.add_argument("--n", type=int, default=32)
    q.add_argument("--out", required=True); q.add_argument("--start", type=int, default=None)
    for name in ("onnx", "tflite"):
        q = sp.add_parser(name)
        q.add_argument("--ref", required=True); q.add_argument("--graph", required=True)
        q.add_argument("--check", required=True); q.add_argument("--steps", type=int, default=16)
        q.add_argument("--n", type=int, default=None); q.add_argument("--threads", type=int, default=4)
        q.add_argument("--ref_cache", default=""); q.add_argument("--out", default="")
        q.add_argument("--thresholds", default=""); q.add_argument("--no_rollout", action="store_true")
        q.add_argument("--verbose", action="store_true")
        if name == "onnx":
            q.add_argument("--no_layers", action="store_true")
            q.add_argument("--ort_level", default="basic", choices=["basic", "all"],
                           help="ONNX Runtime graph optimisation: basic = reference QDQ semantics; all = "
                                "ORT integer-kernel fusions (a different runtime)")
            q.add_argument("--vendor", default="", choices=["", "qnn", "qnn16", "qnn16c"],
                           help="emulate a vendor's fused kernels on the QDQ graph (qnn: RMSNorm fused, "
                                "internals not quantized; qnn16: norm output also unquantized)")
        else:
            q.add_argument("--resolver", default="auto", choices=["auto", "builtin", "ref"])
            q.add_argument("--xrt", action="store_true")
            q.add_argument("--xrt_only", action="store_true", help="recompute only the cross-runtime T0 into --out")
            q.add_argument("--tflm_python", default="")
    q = sp.add_parser("torch")
    q.add_argument("--task", required=True); q.add_argument("--config", default="fp32",
                   help="weights[+activations]: e.g. w8c, w4c+a8all, fp32+a8all, w8c+a16all, noise0.1, w4t@mixer")
    q.add_argument("--H", type=int, default=None); q.add_argument("--steps", type=int, default=16)
    q.add_argument("--check", default=None); q.add_argument("--n", type=int, default=32)
    q.add_argument("--draw", type=int, default=0); q.add_argument("--threads", type=int, default=16)
    q.add_argument("--out", default=""); q.add_argument("--thresholds", default="")
    q = sp.add_parser("probe", help="device-facing tier: device predictions for the first steps vs host fp32")
    q.add_argument("--ref", required=True); q.add_argument("--check", required=True)
    q.add_argument("--device_preds", required=True, help="npz with preds [K, N, L] from the device, same inputs")
    q.add_argument("--k", type=int, default=2); q.add_argument("--threads", type=int, default=4)
    q.add_argument("--out", default=""); q.add_argument("--thresholds", default="")
    q = sp.add_parser("_tflm_dump"); q.add_argument("graph"); q.add_argument("inp"); q.add_argument("out")
    a = p.parse_args()
    if a.cmd == "prep":
        prep(a.task, a.n, a.out, a.start)
    elif a.cmd == "onnx":
        cmd_onnx(a)
    elif a.cmd == "tflite":
        cmd_tflite(a)
    elif a.cmd == "torch":
        w, _, am = a.config.partition("+")
        am = am if am in ("a8all", "a16all", "a8lin") else "none"
        ev = torch_check(a.task, w, am, a.H, a.steps, a.check, a.n, draw=a.draw, threads=a.threads)
        finish(ev, a)
    elif a.cmd == "probe":
        chk = load_check(a.check)
        ref = TfliteStep(a.ref, a.threads) if a.ref.endswith(".tflite") else OnnxStep(a.ref, a.threads)
        ev = dict(kind="device_probe", ref=a.ref, device_preds=a.device_preds)
        ev.update(device_probe(ref, chk, np.load(a.device_preds)["preds"], a.k))
        finish(ev, a)
    elif a.cmd == "_tflm_dump":
        _tflm_dump(a.graph, a.inp, a.out)


if __name__ == "__main__":
    main()
