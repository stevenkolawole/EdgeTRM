#!/usr/bin/env python3
"""Batched torch executor for the AI Hub QDQ step graphs (`stepc`, H=1), for mixed-precision sweeps.

ONNX Runtime runs these graphs one puzzle at a time (the export has batch 1), 5-13 min per 100-puzzle
16-step host simulation. This executor runs the same graph node by node in torch, vmapped over puzzles,
on a GPU. Quantize/Dequantize pairs are executed as fake quantization with the graph's own encodings
(round half to even, clamp to the zero-point type's range), so the arithmetic is the QDQ graph's; only
float reduction order differs from ONNX Runtime (validated against the ORT host results in M3.md).

Activation precision per tensor: the runner is built from the w8a16 graph and, optionally, the w8a8 graph
of the same export; every activation QuantizeLinear (input not an initializer; keyed by the float tensor
it quantizes, as in wa_collapse.py) carries both encodings, and `set_act8(names)` selects the 8-bit one for
the named tensors. Weight precision per tensor: `set_weight_fmt({init name: fmt})` replaces a weight's
Q/DQ by fake quantization of the float initializer in `fmt` (w4t, w8c, ... ; axis = output dim of the
[in, out] MatMul weight).

    from qdq_torch import QDQRunner
    r = QDQRunner(dir16 / "model.onnx", dir8 / "model.onnx", device="cuda")
    preds = r.rollout(inputs, init, nsteps=16)["preds"]
"""
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

ONNX2TORCH = {1: torch.float32, 2: torch.uint8, 3: torch.int8, 4: torch.int32, 5: torch.int16, 6: torch.int32,
              7: torch.int64, 9: torch.bool, 10: torch.float16, 11: torch.float64, 16: torch.bfloat16}
QRANGE = {2: (0, 255), 3: (-128, 127), 4: (0, 65535), 5: (-32768, 32767), 21: (0, 15), 22: (-8, 7)}


def attrs(n):
    return {a.name: helper.get_attribute_value(a) for a in n.attribute}


def _np(t):
    return numpy_helper.to_array(t)


def load_graph(path):
    """onnx.load with external data read by hand (onnx refuses hard-linked data files, which the shared
    AI Hub artifacts on p4d are)."""
    path = Path(path)
    m = onnx.load(str(path), load_external_data=False)
    cache = {}
    for t in m.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = {e.key: e.value for e in t.external_data}
        f = path.parent / info["location"]
        if f not in cache:
            cache[f] = np.memmap(f, dtype=np.uint8, mode="r")
        off = int(info.get("offset", 0))
        n = int(info["length"]) if "length" in info else len(cache[f]) - off
        t.raw_data = cache[f][off:off + n].tobytes()
        del t.external_data[:]
        t.data_location = onnx.TensorProto.DEFAULT
    return m


def act_encodings(g):
    """{float tensor name: (scale np, zp np, zp onnx type, axis)} for activation Q nodes."""
    inits = {i.name: i for i in g.initializer}
    consts = {n.output[0]: n.attribute[0].t for n in g.node if n.op_type == "Constant"}
    vals = {**inits, **consts}
    out = {}
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in inits:
            continue
        zt = vals[n.input[2]]
        out[n.input[0]] = (_np(vals[n.input[1]]).astype(np.float32), _np(zt).astype(np.float64), zt.data_type,
                           attrs(n).get("axis", 1))
    return out


class QDQRunner:
    def __init__(self, path16, path8=None, device="cuda", fp32=False):
        """path16: the w8a16 QDQ graph (or any QDQ graph); path8: the w8a8 graph of the same export (its
        activation encodings become selectable per tensor). fp32=True: a float graph (no Q/DQ)."""
        self.dev = torch.device(device)
        m = load_graph(path16)
        self.g = m.graph
        g = self.g
        self.inits = {i.name: i for i in g.initializer}
        self.inputs = [i.name for i in g.input if i.name not in self.inits]
        self.outputs = [o.name for o in g.output]
        self.nodes = list(g.node)
        self.const = {}
        for name, t in self.inits.items():
            a = _np(t)
            self.const[name] = torch.from_numpy(a.copy()).to(self.dev) if a.dtype != np.float64 else \
                torch.from_numpy(a.astype(np.float32)).to(self.dev)
        for n in self.nodes:
            if n.op_type == "Constant":
                self.const[n.output[0]] = torch.from_numpy(_np(n.attribute[0].t).copy()).to(self.dev)
        last = {}
        for i, n in enumerate(self.nodes):
            for x in n.input:
                last[x] = i
        keep = set(self.outputs) | set(self.const)
        self.free_after = {}
        for x, i in last.items():
            if x not in keep:
                self.free_after.setdefault(i, []).append(x)
        self.enc16 = act_encodings(g)
        g8 = load_graph(path8).graph if path8 else None
        self.enc8 = act_encodings(g8) if g8 is not None else {}
        # w8a8 weight encodings (for the HTP Gather-table rule in device mode)
        self.wenc8 = {}
        if g8 is not None:
            v8 = {i.name: i for i in g8.initializer}
            v8.update({n.output[0]: n.attribute[0].t for n in g8.node if n.op_type == "Constant"})
            for n in g8.node:
                if n.op_type == "QuantizeLinear" and n.input[0] in v8 and n.input[0] in {i.name for i in g8.initializer}:
                    zt = v8[n.input[2]]
                    self.wenc8[n.input[0]] = (_np(v8[n.input[1]]).astype(np.float32), _np(zt).astype(np.float64),
                                              zt.data_type, attrs(n).get("axis", 1))
        self.device_mode = False
        self._device_structure()
        self.act_names = sorted(self.enc16)
        # weight Q nodes (input = float initializer): folded once per weight configuration
        self.wq = {n.output[0]: n for n in self.nodes if n.op_type == "QuantizeLinear" and n.input[0] in self.inits}
        self.w_of = {k: n.input[0] for k, n in self.wq.items()}
        self.weight_fmt = {}
        self.act8 = set()
        self._table8_on = set()
        self.act_noise = {}                                  # {tensor: sigma}: Gaussian noise, std = sigma x std(x)
        self._enc_t = {}
        self._build_enc_tensors()
        self._fold_weights()

    # ------------------------------------------------------------------ configuration
    def _build_enc_tensors(self):
        """per activation tensor: (scale, zp, qmin, qmax) torch tensors for 16 and 8 bit."""
        for name in self.act_names:
            e16 = self.enc16[name]
            self._enc_t[(name, 16)] = self._enc(e16)
            if name in self.enc8:
                self._enc_t[(name, 8)] = self._enc(self.enc8[name])

    def _enc(self, e):
        s, z, zt, axis = e
        lo, hi = QRANGE[zt]
        return (torch.tensor(s, dtype=torch.float32, device=self.dev), torch.tensor(z, dtype=torch.float32, device=self.dev),
                float(lo), float(hi), axis)

    def set_act8(self, names):
        names = set(names)
        missing = [n for n in names if (n, 8) not in self._enc_t]
        assert not missing, f"no 8-bit encoding for {missing[:5]}"
        self.act8 = names
        # device mode: a Gather table switches to its 8-bit (w8a8) encoding when its output is 8 bit
        self._table8_on = set()
        if self.device_mode:
            for out_t, (wq_out, wname) in self.table.items():
                if self.root(out_t) in names and wname in self.wenc8:
                    self._table8_on.add(wq_out)

    def set_device_mode(self, on=True):
        self.device_mode = on
        self._table8_fold = {}
        self._table8_on = set()
        for out_t, (wq_out, wname) in self.table.items():
            if wname in self.wenc8:
                sc, zp, zt, axis = self.wenc8[wname]
                lo, hi = QRANGE[zt]
                self._table8_fold[wq_out] = self._fq(self.const[wname].float(), torch.tensor(sc, device=self.dev),
                                                     torch.tensor(zp, dtype=torch.float32, device=self.dev), lo, hi, axis)
        self.set_act8(self.act8)

    def set_act_encoding(self, name, scale, zp, zp_type, bits_tag):
        """a custom encoding (e.g. 4-bit, or a re-calibrated 8-bit) for one tensor, selectable by bits_tag."""
        self._enc_t[(name, bits_tag)] = self._enc((np.asarray(scale, np.float32), np.asarray(zp, np.float64), zp_type, 1))

    def matmul_weights(self):
        """float initializer names of the weights that feed a MatMul / Gemm (through their Q/DQ)."""
        cons = {}
        for n in self.nodes:
            for x in n.input:
                cons.setdefault(x, []).append(n)
        out = []
        for qout, qn in self.wq.items():
            for dq in cons.get(qout, []):
                if dq.op_type == "DequantizeLinear" and any(c.op_type in ("MatMul", "Gemm") for c in cons.get(dq.output[0], [])):
                    out.append(qn.input[0])
                    break
        return sorted(set(out))

    MOVE = ("Gather", "Reshape", "Transpose", "Slice", "Squeeze", "Unsqueeze", "Flatten")

    def _device_structure(self):
        """HTP legality (X3, Sep 25): (1) a data-movement op's output keeps its input's encoding -> alias
        {movement output tensor: input tensor}; (2) a Gather's table (a weight) shares its output's encoding ->
        table {gathered output tensor: (weight Q output, weight name)}."""
        prod = {o: n for n in self.nodes for o in n.output}
        cons = {}
        for n in self.nodes:
            for x in n.input:
                cons.setdefault(x, []).append(n)
        self.alias, self.table = {}, {}
        for n in self.nodes:
            if n.op_type not in self.MOVE:
                continue
            q_out = next((c for c in cons.get(n.output[0], []) if c.op_type == "QuantizeLinear"), None)
            if q_out is None:
                continue
            dq_in = prod.get(n.input[0])
            if dq_in is None or dq_in.op_type != "DequantizeLinear":
                continue
            q_in = prod.get(dq_in.input[0])
            if q_in is None or q_in.op_type != "QuantizeLinear":
                continue
            if q_in.input[0] in self.inits:
                if n.op_type == "Gather":
                    self.table[q_out.input[0]] = (q_in.output[0], q_in.input[0])
                continue
            self.alias[q_out.input[0]] = q_in.input[0]

    def root(self, t):
        seen = 0
        while t in self.alias and seen < 64:
            t = self.alias[t]
            seen += 1
        return t

    def matmul_input_acts(self):
        """activation tensors (keys of the activation encodings) whose Q/DQ output is the activation input of a
        MatMul/Gemm (Ethos-U with INT4 weights accepts only 8-bit activations there)."""
        prod = {o: n for n in self.nodes for o in n.output}
        out = set()
        for n in self.nodes:
            if n.op_type not in ("MatMul", "Gemm"):
                continue
            for x in n.input[:2]:
                p = prod.get(x)
                hops = 0
                while p is not None and hops < 6 and p.op_type in ("DequantizeLinear", "QuantizeLinear", "Transpose",
                                                                     "Reshape", "Identity", "Cast"):
                    if p.op_type == "QuantizeLinear" and p.input[0] in self.enc16:
                        out.add(p.input[0])
                        break
                    p = prod.get(p.input[0])
                    hops += 1
        return out

    def set_weight_fmt(self, fmts):
        """{weight initializer name: fmt or None (graph's own encoding)}."""
        self.weight_fmt = {k: v for k, v in fmts.items() if v}
        self._fold_weights()

    @torch.no_grad()
    def _fold_weights(self):
        self.wfold = {}
        for qout, n in self.wq.items():
            wname = n.input[0]
            W = self.const[wname].float()
            fmt = self.weight_fmt.get(wname)
            if fmt is None:
                s = self.const[n.input[1]].float()
                ztp = self.inits[n.input[2]].data_type if n.input[2] in self.inits else 3
                z = self.const[n.input[2]].float()
                lo, hi = QRANGE[ztp]
                axis = attrs(n).get("axis", 1)
                self.wfold[qout] = self._fq(W, s, z, lo, hi, axis)
            else:
                self.wfold[qout] = fq_weight(W, fmt)

    @staticmethod
    def _fq(x, s, z, lo, hi, axis):
        if s.dim() == 1 and s.numel() > 1:
            ax = axis if axis >= 0 else x.dim() + axis
            shp = [1] * x.dim()
            shp[ax] = -1
            s, z = s.view(shp), z.view(shp)
        return (torch.clamp(torch.round(x / s) + z, lo, hi) - z) * s

    # ------------------------------------------------------------------ execution
    def _run_one(self, feeds, return_env=False):
        """per-sample graph execution (called under vmap)."""
        env = dict(self.const)
        env.update(feeds)
        env[""] = None
        fq_out = set()
        for i_n, n in enumerate(self.nodes):
            op = n.op_type
            if not return_env and i_n - 1 in self.free_after:
                for x in self.free_after[i_n - 1]:
                    env.pop(x, None)
            if op == "Constant":
                continue
            ins = [env[i] if i else None for i in n.input]
            a = attrs(n)
            if op == "QuantizeLinear":
                if n.output[0] in self.wfold:
                    y = self.wfold[n.output[0]]
                    if self.device_mode and n.output[0] in self._table8_on:
                        y = self._table8_fold[n.output[0]]
                else:
                    name = n.input[0]
                    if self.device_mode:
                        name = self.root(name)
                    key = (name, 8) if name in self.act8 else (name, 16)
                    s, z, lo, hi, axis = self._enc_t[key]
                    x = ins[0].float()
                    if name in self.act_noise:
                        x = x + torch.randn_like(x) * (self.act_noise[name] * x.std())
                    y = self._fq(x, s, z, lo, hi, axis)
                env[n.output[0]] = y
                fq_out.add(n.output[0])
                continue
            if op == "DequantizeLinear":
                if n.input[0] in fq_out:
                    y = ins[0]
                else:                                        # stored integer initializer
                    ztp = self.inits[n.input[2]].data_type if n.input[2] in self.inits else 3
                    y = (ins[0].float() - ins[2].float() if ins[2] is not None else ins[0].float())
                    s = ins[1].float()
                    if s.dim() == 1 and s.numel() > 1:
                        ax = a.get("axis", 1)
                        ax = ax if ax >= 0 else y.dim() + ax
                        shp = [1] * y.dim()
                        shp[ax] = -1
                        s = s.view(shp)
                        if ins[2] is not None:
                            y = ins[0].float() - ins[2].float().view(shp)
                    y = y * s
                env[n.output[0]] = y
                fq_out.add(n.output[0])
                continue
            try:
                outs = OPS[op](ins, a)
            except Exception as ex:
                raise RuntimeError(f"node {n.name} ({op}) inputs {list(n.input)} shapes "
                                   f"{[tuple(x.shape) if hasattr(x, 'shape') else x for x in ins]} "
                                   f"values {[x.tolist() if hasattr(x, 'numel') and x.numel() < 8 and not x.is_floating_point() else '' for x in ins]} attrs {a}") from ex
            if not isinstance(outs, (list, tuple)):
                outs = [outs]
            for o, v in zip(n.output, outs):
                env[o] = v
        if return_env:
            return env
        return tuple(env[o] for o in self.outputs)

    @torch.no_grad()
    def step(self, feeds, batched):
        """feeds: {input name: tensor}; batched: set of input names with a leading puzzle dim."""
        names = self.inputs
        in_dims = tuple(0 if k in batched else None for k in names)

        def f(*args):
            return self._run_one(dict(zip(names, args)))
        outs = torch.func.vmap(f, in_dims=in_dims, randomness="different")(*[feeds[k] for k in names])
        return dict(zip(self.outputs, outs))

    @torch.no_grad()
    def rollout(self, inputs, init, nsteps, batch=100, keep_logits_last=False, carry0=None):
        """inputs: (N, seq) int array; init = (z_H0 (1,S,D), z_L0 (1,S,D), puzzle_embedding (1,E) or (N,E)).
        carry0: optional (z_H (N,S,D), z_L (N,S,D)) to start from. Returns per-step predictions
        (nsteps, N, seq), q_halt (nsteps, N), the final carry, and (optionally) the last logits."""
        zH0, zL0 = [torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.dev) for x in init[:2]]
        emb = np.asarray(init[2], dtype=np.float32)
        N = inputs.shape[0]
        per_row = emb.shape[0] == N and N > 1
        preds, qs, zHs, zLs, lgs = [], [], [], [], []
        for s0 in range(0, N, batch):
            inp = torch.as_tensor(inputs[s0:s0 + batch], device=self.dev).to(torch.int32)[:, None]   # (B,1,seq)
            B = inp.shape[0]
            if carry0 is not None:
                zH = torch.as_tensor(carry0[0][s0:s0 + B], dtype=torch.float32, device=self.dev)[:, None]
                zL = torch.as_tensor(carry0[1][s0:s0 + B], dtype=torch.float32, device=self.dev)[:, None]
            else:
                zH = zH0.expand(B, *zH0.shape).contiguous()
                zL = zL0.expand(B, *zL0.shape).contiguous()
            if per_row:
                e = torch.as_tensor(emb[s0:s0 + B], device=self.dev)[:, None]
                batched = {"z_H", "z_L", "inputs", "puzzle_embedding"}
            else:
                e = torch.as_tensor(emb[:1], device=self.dev)
                batched = {"z_H", "z_L", "inputs"}
            P, Q = [], []
            o = None
            for _ in range(nsteps):
                o = self.step({"z_H": zH, "z_L": zL, "inputs": inp, "puzzle_embedding": e}, batched)
                zH, zL = o["z_H_out"], o["z_L_out"]
                P.append(o["logits"][:, 0].argmax(-1).cpu())
                Q.append(o["q_halt"].reshape(B, -1)[:, 0].float().cpu())
            preds.append(torch.stack(P))
            qs.append(torch.stack(Q))
            zHs.append(zH[:, 0].cpu())
            zLs.append(zL[:, 0].cpu())
            if keep_logits_last:
                lgs.append(o["logits"][:, 0].float().cpu())
        r = {"preds": torch.cat(preds, 1).numpy(), "q": torch.cat(qs, 1).numpy(),
             "zH": torch.cat(zHs).numpy(), "zL": torch.cat(zLs).numpy()}
        if keep_logits_last:
            r["logits"] = torch.cat(lgs).numpy()
        return r

    @torch.no_grad()
    def tensor_stats(self, feeds):
        """one un-vmapped pass on a single sample: {activation tensor: (numel, std, absmax)} of the float
        tensor each activation Q node quantizes."""
        env = self._run_one(feeds, return_env=True)
        out = {}
        for n in self.nodes:
            if n.op_type == "QuantizeLinear" and n.output[0] not in self.wfold:
                x = env[n.input[0]]
                x = x.float()
                out[n.input[0]] = (int(x.numel()), float(x.std()) if x.numel() > 1 else 0.0, float(x.abs().max()))
        return out


# ----------------------------------------------------------------------------- weight formats
@torch.no_grad()
def fq_weight(W, fmt):
    """W: [in, out] MatMul weight (or [out, in] Gemm with transB; per-channel = last axis for MatMul).
    fmt: w<b>t (per tensor), w<b>c (per output channel), w<b>g<k> (blocks of k along the input dim)."""
    body = fmt[1:]
    i = 0
    while body[i].isdigit():
        i += 1
    b, gran = int(body[:i]), body[i:]
    qmax = 2 ** (b - 1) - 1
    if W.dim() < 2:
        gran = "t"
    if gran == "t":
        s = (W.abs().max() / qmax).clamp_min(1e-12)
        return torch.round(W / s).clamp(-qmax, qmax) * s
    Wt = W.transpose(-1, -2) if W.dim() == 2 else W      # [out, in]
    if gran == "c":
        s = (Wt.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        Q = torch.round(Wt / s).clamp(-qmax, qmax) * s
    else:
        k = int(gran[1:])
        o, n = Wt.shape
        pad = (-n) % k
        Wp = torch.nn.functional.pad(Wt, (0, pad)).view(o, -1, k)
        s = (Wp.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-12)
        Q = (torch.round(Wp / s).clamp(-qmax, qmax) * s).view(o, -1)[:, :n]
    return Q.transpose(-1, -2) if W.dim() == 2 else Q


# ----------------------------------------------------------------------------- ops (per sample)
def _ints(t):
    return [int(v) for v in t.reshape(-1).tolist()]


def _reshape(ins, a):
    x, shp = ins
    shp = _ints(shp)
    if not a.get("allowzero", 0):
        shp = [x.shape[i] if v == 0 else v for i, v in enumerate(shp)]
    return x.reshape(shp)


def _slice(ins, a):
    x, starts, ends = ins[0], _ints(ins[1]), _ints(ins[2])
    axes = _ints(ins[3]) if len(ins) > 3 and ins[3] is not None else list(range(len(starts)))
    steps = _ints(ins[4]) if len(ins) > 4 and ins[4] is not None else [1] * len(starts)
    for s, e, ax, st in zip(starts, ends, axes, steps):
        ax = ax if ax >= 0 else x.dim() + ax
        d = x.shape[ax]
        s = s + d if s < 0 else s
        e = e + d if e < 0 else e
        if st > 0:
            s, e = max(0, min(d, s)), max(0, min(d, e))
            idx = [slice(None)] * x.dim()
            idx[ax] = slice(s, e, st)
            x = x[tuple(idx)]
        else:                                            # ONNX negative step: start in [0, d-1], end in [-1, d-1]
            s, e = max(0, min(d - 1, s)), max(-1, min(d - 1, e))
            ii = torch.arange(s, e, st, device=x.device)
            x = torch.index_select(x, ax, ii)
    return x


def _gather(ins, a):
    x, idx = ins
    ax = a.get("axis", 0)
    ax = ax if ax >= 0 else x.dim() + ax
    idx = idx.long()
    idx = torch.where(idx < 0, idx + x.shape[ax], idx)          # ONNX allows negative indices
    if idx.dim() == 0:
        return x.select(ax, int(idx))
    flat = torch.index_select(x, ax, idx.reshape(-1))
    return flat.reshape(tuple(x.shape[:ax]) + tuple(idx.shape) + tuple(x.shape[ax + 1:]))


def _pad(ins, a):
    x, pads = ins[0], _ints(ins[1])
    val = float(ins[2]) if len(ins) > 2 and ins[2] is not None else 0.0
    n = x.dim()
    axes = _ints(ins[3]) if len(ins) > 3 and ins[3] is not None else list(range(n))
    tp = [0] * (2 * n)
    k = len(axes)
    for j, ax in enumerate(axes):
        ax = ax if ax >= 0 else n + ax
        tp[2 * (n - 1 - ax)] = pads[j]
        tp[2 * (n - 1 - ax) + 1] = pads[j + k]
    return torch.nn.functional.pad(x, tp, value=val)


def _reduce(fn):
    def r(ins, a):
        x = ins[0]
        axes = a.get("axes")
        if axes is None and len(ins) > 1 and ins[1] is not None:
            axes = _ints(ins[1])
        kd = bool(a.get("keepdims", 1))
        if not axes:
            return fn(x, dim=tuple(range(x.dim())), keepdim=kd)
        return fn(x, dim=tuple(axes), keepdim=kd)
    return r


def _gemm(ins, a):
    A, B = ins[0], ins[1]
    if a.get("transA", 0):
        A = A.transpose(-1, -2)
    if a.get("transB", 0):
        B = B.transpose(-1, -2)
    y = a.get("alpha", 1.0) * (A @ B)
    if len(ins) > 2 and ins[2] is not None:
        y = y + a.get("beta", 1.0) * ins[2]
    return y


def _split(ins, a):
    x = ins[0]
    ax = a.get("axis", 0)
    if len(ins) > 1 and ins[1] is not None:
        sizes = _ints(ins[1])
    elif "split" in a:
        sizes = list(a["split"])
    else:
        k = a.get("num_outputs")
        sizes = [x.shape[ax] // k] * k
    return list(torch.split(x, sizes, dim=ax))


def _unsqueeze(ins, a):
    x = ins[0]
    axes = _ints(ins[1]) if len(ins) > 1 and ins[1] is not None else list(a["axes"])
    n = x.dim() + len(axes)
    for ax in sorted(v if v >= 0 else n + v for v in axes):
        x = x.unsqueeze(ax)
    return x


def _squeeze(ins, a):
    x = ins[0]
    axes = _ints(ins[1]) if len(ins) > 1 and ins[1] is not None else list(a.get("axes", []))
    if not axes:
        return x.squeeze()
    for ax in sorted((v if v >= 0 else x.dim() + v for v in axes), reverse=True):
        x = x.squeeze(ax)
    return x


def _expand(ins, a):
    x, shp = ins
    shp = _ints(shp)
    out = torch.broadcast_shapes(tuple(x.shape), tuple(shp))
    return x.expand(out)


OPS = {
    "Identity": lambda i, a: i[0],
    "Cast": lambda i, a: i[0].to(ONNX2TORCH[a["to"]]),
    "Shape": lambda i, a: torch.tensor(list(i[0].shape), dtype=torch.int64, device=i[0].device),
    "ConstantOfShape": lambda i, a: torch.full(tuple(_ints(i[0])), float(_np(a["value"]).reshape(-1)[0]) if "value" in a else 0.0,
                                               dtype=ONNX2TORCH[a["value"].data_type] if "value" in a else torch.float32,
                                               device=i[0].device),
    "Concat": lambda i, a: torch.cat([x for x in i if x is not None and x.numel() > 0] or [i[0]], dim=a["axis"]),
    "Reshape": _reshape,
    "Slice": _slice,
    "Transpose": lambda i, a: i[0].permute(*a["perm"]) if "perm" in a else i[0].permute(*reversed(range(i[0].dim()))),
    "Gather": _gather,
    "Pad": _pad,
    "MatMul": lambda i, a: torch.matmul(i[0], i[1]),
    "Gemm": _gemm,
    "Add": lambda i, a: i[0] + i[1],
    "Sub": lambda i, a: i[0] - i[1],
    "Mul": lambda i, a: i[0] * i[1],
    "Div": lambda i, a: torch.div(i[0], i[1], rounding_mode="trunc") if not (i[0].is_floating_point() or
                                                                            i[1].is_floating_point()) else i[0] / i[1],
    "Pow": lambda i, a: torch.pow(i[0], i[1]),
    "Sqrt": lambda i, a: torch.sqrt(i[0]),
    "Reciprocal": lambda i, a: 1.0 / i[0],
    "Sigmoid": lambda i, a: torch.sigmoid(i[0]),
    "Tanh": lambda i, a: torch.tanh(i[0]),
    "Relu": lambda i, a: torch.relu(i[0]),
    "Exp": lambda i, a: torch.exp(i[0]),
    "Neg": lambda i, a: -i[0],
    "Sin": lambda i, a: torch.sin(i[0]),
    "Cos": lambda i, a: torch.cos(i[0]),
    "Erf": lambda i, a: torch.erf(i[0]),
    "Softmax": lambda i, a: torch.softmax(i[0], dim=a.get("axis", -1)),
    "ReduceMean": _reduce(torch.mean),
    "ReduceSum": _reduce(torch.sum),
    "ReduceMax": _reduce(lambda x, dim, keepdim: torch.amax(x, dim=dim, keepdim=keepdim)),
    "Split": _split,
    "Unsqueeze": _unsqueeze,
    "Squeeze": _squeeze,
    "Expand": _expand,
    "Equal": lambda i, a: i[0] == i[1],
    "Where": lambda i, a: torch.where(i[0], i[1], i[2]),
    "Max": lambda i, a: torch.maximum(i[0], i[1]),
    "Min": lambda i, a: torch.minimum(i[0], i[1]),
}
