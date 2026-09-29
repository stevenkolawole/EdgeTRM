#!/usr/bin/env python3
"""v79 Maze fault: sequence-length probe and softmax localisation (reports/edgetrm_exp/E3_V79_LENGTH.md).
Follow-up to E3_FOLLOWUP.md section 4. acct2 only, QAIRT pinned at 2.45 (aihub_acct.QAIRT), the nofuse recipe
(RMSNorm pattern broken at export, aihub_v79_classa.break_rmsnorm). Tokens are never read or printed here:
qai_hub.Client(profile=...) loads them. Runs on p4d from ~/x3; drives aihub_e3_followup's compile / run machinery
on its own cells only (E3.CHAINS is replaced for this process).

Cells (task : prec -> graph):
  maze_s97 : w8a8_nofuse2       the Maze weights exported at seq 81 + 16 = 97 (aihub_h1.TASKS['maze_s97'], own
                                calibration and AI Hub w8a8 quantize job jp2rowlxg), qdq_w8a8/model.onnx with the
                                RMSNorm pattern broken exactly as maze/qdq_w8a8/variant_CA_nofuse2.onnx was made.
                                Labels are the first 81 cells of each maze, so accuracy is meaningless: the reading
                                is device-vs-host-reference agreement per step (argmax agreement over 50 x 81 cells).
  maze     : w8a8_nofuse2_smfp  the all-8 nofuse Maze graph (916 keys) with the Q/DQ pair stripped on every
                                Softmax output AND on V (Transpose_1_output_0, the qkv Slice that feeds softmax.V),
                                so each Softmax runs in float (fp16 on the HTP) over the 8-bit scores and softmax.V
                                is float x float (MatMul dtype rule); the softmax.V output, q.k^T and everything
                                else stay 8-bit.
  maze     : w8a8_nofuse2_smfp2 as smfp plus the scores (MatMul_output_0) pair stripped: q.k^T output, Softmax and
                                softmax.V all in float. Fallback if the composer refuses a uint8-in / float-out Softmax.
  maze     : w8a8_nofuse2_smfp3 the bounded float island: the S25 refused smfp at softmax.V (`MatMul_1`: fp16 inputs,
                                uint8 output, "mismatch for the combination of inputs and outputs"), so an HTP MatMul
                                needs inputs AND output in one dtype set. smfp3 strips the pairs on V's Slice_2 and
                                Transpose_1, Softmax, MatMul_1, Transpose_3, Reshape_1 and the o_proj MatMul output:
                                softmax -> softmax.V -> o_proj run in float, the island ends at the residual Add
                                (an elementwise op, which the fused DLCs' fp16 islands already ended at), q.k^T,
                                RoPE and the qkv projection stay 8-bit.

    python aihub_v79_length.py build
    python aihub_v79_length.py hostref maze_s97:w8a8_nofuse2 [--steps 8] [--threads 12]
    python aihub_v79_length.py compile [task:prec ...]
    python aihub_v79_length.py run --steps 2 [--only tag ...]
    python aihub_v79_length.py run --steps 8 --only maze_s97:w8a8_nofuse2:S25@acct2
    python aihub_v79_length.py status
    python aihub_v79_length.py poserr > poserr.md        # per-position error of the existing S25 / S23 records
"""
import argparse
import json
import pathlib
import re
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_acct as X  # noqa: E402
import aihub_e3_followup as E3  # noqa: E402  (source_path keyed on (task, prec); break_rmsnorm; compile_/run drivers)
import aihub_infer as I  # noqa: E402

S23, S25 = E3.S23, E3.S25
PLEN = 16                                   # puzzle-embedding prefix: grid cell c sits at sequence position c + PLEN
I.NSTEPS.setdefault("maze_s97", 8)
E3.SRC.update({("maze_s97", "w8a8_nofuse2"): ("qdq_w8a8", "variant_CA_nofuse2.onnx"),
               ("maze", "w8a8_nofuse2_smfp"): ("qdq_w8a8", "variant_CA_nofuse2_smfp.onnx"),
               ("maze", "w8a8_nofuse2_smfp2"): ("qdq_w8a8", "variant_CA_nofuse2_smfp2.onnx"),
               ("maze", "w8a8_nofuse2_smfp3"): ("qdq_w8a8", "variant_CA_nofuse2_smfp3.onnx")})
STRIP = {"w8a8_nofuse2_smfp": ("Softmax", "Transpose_1"),                  # regex fragments, see strip_pairs
         "w8a8_nofuse2_smfp2": ("Softmax", "Transpose_1", "MatMul"),
         "w8a8_nofuse2_smfp3": ("Softmax", "Transpose_1", "Slice_2", "MatMul_1", "Transpose_3", "Reshape_1",
                                r"o_proj(_\d+)?/MatMul")}
CHAINS = [("acct2", "maze_s97", "w8a8_nofuse2", S25), ("acct2", "maze_s97", "w8a8_nofuse2", S23),
          ("acct2", "maze", "w8a8_nofuse2_smfp", S25), ("acct2", "maze", "w8a8_nofuse2_smfp2", S25),
          ("acct2", "maze", "w8a8_nofuse2_smfp3", S25)]
E3.CHAINS[:] = CHAINS
tag = E3.tag


# ------------------------------------------------------------------ graph edits
def strip_pairs(g, suffixes):
    """Remove the Q/DQ pair of every tensor `.../self_attn(_k)/<suffix>_output_0` (one per attention block per
    suffix; suffixes are regex fragments, e.g. `o_proj(_\\d+)?/MatMul` for the o_proj output, which the export
    names `self_attn/o_proj_k/MatMul_output_0`); every consumer of the DQ output reads the float tensor instead.
    Returns the stripped tensor names."""
    pat = re.compile(r"/self_attn(_\d+)?/(" + "|".join(suffixes) + r")_output_0$")
    prod = {o: n for n in g.node for o in n.output}
    cons = {}
    for n in g.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    inits = {i.name for i in g.initializer}
    remove, rename, done = set(), {}, []
    for n in g.node:
        if n.op_type != "QuantizeLinear" or n.input[0] in inits or not pat.search(n.input[0]):
            continue
        t = n.input[0]
        dqs = [c for c in cons.get(n.output[0], []) if c.op_type == "DequantizeLinear"]
        assert dqs and len(cons.get(n.output[0], [])) == len(dqs), (t, [c.op_type for c in cons.get(n.output[0], [])])
        for dq in dqs:
            rename[dq.output[0]] = t
            remove.add(id(dq))
        remove.add(id(n))
        done.append(t)
    keep = [n for n in g.node if id(n) not in remove]
    for n in keep:
        for i, x in enumerate(n.input):
            if x in rename:
                n.input[i] = rename[x]
    del g.node[:]
    g.node.extend(keep)
    return done


def build(names):
    import onnx
    from onnx import numpy_helper
    for (task, prec), (base, fname) in E3.SRC.items():
        if (task, prec) not in {c[1:3] for c in CHAINS} or (names and f"{task}:{prec}" not in names):
            continue
        d = I.idir(task)
        out = d / base / fname
        if out.exists():
            print(task, prec, "exists, not rebuilt:", out, flush=True)
            continue
        if prec in STRIP:
            src = d / base / "variant_CA_nofuse2.onnx"            # the all-8 nofuse Maze graph of E3_FOLLOWUP section 2
            m = onnx.load(str(src), load_external_data=False)
            done = strip_pairs(m.graph, STRIP[prec])
            onnx.save(m, str(out))
            print(task, prec, "from", src.name, "stripped Q/DQ on", len(done), "tensors:",
                  sorted({t.rsplit("/", 1)[-1] for t in done}), "->", out, flush=True)
        else:
            m = onnx.load(str(d / base / "model.onnx"), load_external_data=False)
            n = E3.break_rmsnorm(m)
            onnx.save(m, str(out))
            Ns = sorted({int(round(1.0 / float(numpy_helper.to_array(i)))) for i in m.graph.initializer
                         if i.name.endswith("_invN")})
            print(task, prec, "rewrote", n, "ReduceMean nodes; reduction sizes N =", Ns, "->", out, flush=True)


# ------------------------------------------------------------------ host reference (reference QDQ semantics)
def hostref(names, steps, threads):
    for name in names:
        task, prec = name.split(":")
        d = I.idir(task)
        k = steps or I.NSTEPS[task]
        out = d / (f"host_{prec}_ref.npz" if k == I.NSTEPS[task] else f"host_{prec}_ref_n{k}.npz")
        if out.exists():
            print("exists, not overwritten:", out, flush=True)
            continue
        r, rows, path = E3.host_run(task, prec, k, threads, True)
        X.atomic_npz(out, **r)
        (d / (out.stem + ".json")).write_text(json.dumps({"graph": str(path), "ref": "ORT_DISABLE_ALL", "n_steps": k,
                                                          "steps": rows}, indent=1))
        print("wrote", out, flush=True)


# ------------------------------------------------------------------ readouts
def agree(task, prec, dev, acct="acct2"):
    """per step: device exact / cell vs labels, and argmax agreement with the host reference of the same graph."""
    d = I.idir(task)
    f = d / f"dev_{prec}_{dev.replace(' ', '_')}_{acct}.npz"
    h = d / f"host_{prec}_ref.npz"
    if not f.exists() or not h.exists():
        return []
    p, r = np.load(f)["preds"], np.load(h)["preds"]
    _, lab = I.test_set(task)
    rows = []
    for s in range(min(p.shape[0], r.shape[0])):
        rows.append({"step": s + 1, "dev_exact": float((p[s] == lab).all(-1).mean()), "dev_cell": float((p[s] == lab).mean()),
                     "host_exact": float((r[s] == lab).all(-1).mean()), "host_cell": float((r[s] == lab).mean()),
                     "agree_cell": float((p[s] == r[s]).mean()), "agree_all": float((p[s] == r[s]).all(-1).mean())})
    return rows


def status():
    for acct, task, prec, dev in CHAINS:
        d = I.idir(task)
        f = d / f"dev_{prec}_{dev.replace(' ', '_')}_{acct}.json"
        m = json.loads(f.read_text()) if f.exists() else {}
        print(tag(acct, task, prec, dev), "| compile", m.get("compile_job"), (m.get("converter") or {}).get("converter"),
              "fallback", len((m.get("converter") or {}).get("float_fallback_ops") or []),
              "| guard", (m.get("guard") or {}).get("verdict"), "| jobs", len(m.get("jobs", [])),
              "| failed:", (m.get("failed") or "")[:120])
        for r in agree(task, prec, dev, acct):
            print(f"    step {r['step']}: device {100 * r['dev_exact']:.0f} / {100 * r['dev_cell']:.1f}  host ref "
                  f"{100 * r['host_exact']:.0f} / {100 * r['host_cell']:.1f}  agree {100 * r['agree_cell']:.2f} % cells, "
                  f"{100 * r['agree_all']:.0f} % puzzles all-cells", flush=True)


def poserr():
    """Per-position disagreement (device argmax != host-reference argmax of the same graph) for the existing Maze
    records on the S25 (v79) and S23 (v73), steps 1 and 2, plus the final-step z_H cosine per sequence position."""
    d = I.idir("maze")
    _, lab = I.test_set("maze")
    side = int(round(lab.shape[1] ** 0.5))
    recs = [("S25 v79, all-8 nofuse2", "w8a8_nofuse2", "Samsung_Galaxy_S25"), ("S23 v73, all-8 nofuse2", "w8a8_nofuse2", "Samsung_Galaxy_S23"),
            ("S25 v79, attn16", "w8a8_nofuse2_attn16", "Samsung_Galaxy_S25"), ("S25 v79, all-16", "w8a16_nofuse2", "Samsung_Galaxy_S25"),
            ("S25 v79, softmax fp16 (smfp)", "w8a8_nofuse2_smfp", "Samsung_Galaxy_S25"),
            ("S25 v79, scores+softmax fp16 (smfp2)", "w8a8_nofuse2_smfp2", "Samsung_Galaxy_S25"),
            ("S25 v79, softmax..o_proj fp16 (smfp3)", "w8a8_nofuse2_smfp3", "Samsung_Galaxy_S25")]
    print("## Per-position disagreement with the host reference (cells of the 30x30 grid; grid cell c = sequence position c + 16)\n")
    print("| record | step | cells != host | head (cells 0-599) | cells 600-749 | 750-779 | 780-899 | first 10-cell window > 4x head median (cell / seq pos) | per-maze min / median / max |\n|---|---|---|---|---|---|---|---|---|")
    curves = {}
    for label, prec, dev in recs:
        f, h = d / f"dev_{prec}_{dev}_acct2.npz", d / f"host_{prec}_ref.npz"
        if not f.exists() or not h.exists():
            continue
        p, r = np.load(f)["preds"], np.load(h)["preds"]
        for s in range(min(2, p.shape[0], r.shape[0])):
            e = (p[s] != r[s]).mean(0)
            base = np.median(e[:600])
            sm = np.convolve(e, np.ones(10) / 10, "same")
            hit = np.where(sm > 4 * base)[0]
            first = int(hit[0]) if len(hit) else -1
            pm = (p[s] != r[s]).mean(1)
            print(f"| {label} | {s + 1} | {100 * e.mean():.2f} % | {100 * e[:600].mean():.1f} % | {100 * e[600:750].mean():.1f} % | "
                  f"{100 * e[750:780].mean():.1f} % | {100 * e[780:].mean():.1f} % | {first} / {first + PLEN if first >= 0 else '-'} | "
                  f"{100 * pm.min():.1f} / {100 * np.median(pm):.1f} / {100 * pm.max():.1f} |")
            curves[(label, s + 1)] = e
    print("\n## Disagreement by grid row (30 cells each; x100)\n")
    print("| record | step | " + " | ".join(f"r{i}" for i in range(side)) + " |\n|---|---|" + "---|" * side)
    for (label, s), e in curves.items():
        print(f"| {label} | {s} | " + " | ".join(f"{100 * x:.0f}" for x in e.reshape(side, side).mean(1)) + " |")
    print("\n## Disagreement by grid column (x100)\n")
    print("| record | step | " + " | ".join(f"c{i}" for i in range(side)) + " |\n|---|---|" + "---|" * side)
    for (label, s), e in curves.items():
        print(f"| {label} | {s} | " + " | ".join(f"{100 * x:.0f}" for x in e.reshape(side, side).mean(0)) + " |")
    print("\n## Final-step z_H carry: cosine(device, host reference) by sequence position (blocks of 64; the first 16 positions are the puzzle embedding)\n")
    print("| record | step | " + " | ".join(f"p{i * 64}" for i in range(15)) + " | prefix 0-15 | 16-787 | 788-915 |\n|---|---|" + "---|" * 18)
    for label, prec, dev in recs:
        f, h = d / f"dev_{prec}_{dev}_acct2.npz", d / f"host_{prec}_ref.npz"
        if not f.exists() or not h.exists():
            continue
        zd, zh = np.load(f)["zH"].reshape(-1, side * side + PLEN, 512), np.load(h)["zH"].reshape(-1, side * side + PLEN, 512)
        n = min(len(zd), len(zh))
        zd, zh = zd[:n], zh[:n]
        c = ((zd * zh).sum(-1) / (np.linalg.norm(zd, axis=-1) * np.linalg.norm(zh, axis=-1) + 1e-9)).mean(0)
        k = np.load(f)["preds"].shape[0]
        print(f"| {label} | {k} | " + " | ".join(f"{c[i * 64:(i + 1) * 64].mean():.3f}" for i in range(15))
              + f" | {c[:PLEN].mean():.3f} | {c[PLEN:788].mean():.3f} | {c[788:].mean():.3f} |")
    # maze_s97, if it ran
    d97 = I.idir("maze_s97")
    for dev in ("Samsung_Galaxy_S25", "Samsung_Galaxy_S23"):
        f, h = d97 / f"dev_w8a8_nofuse2_{dev}_acct2.npz", d97 / "host_w8a8_nofuse2_ref.npz"
        if f.exists() and h.exists():
            p, r = np.load(f)["preds"], np.load(h)["preds"]
            print(f"\n## maze_s97 (97 keys) on the {dev}: per-position disagreement with the host reference, 9 blocks of 9 cells (x100)\n")
            for s in range(min(p.shape[0], r.shape[0])):
                e = (p[s] != r[s]).mean(0)
                print(f"- step {s + 1}: {100 * e.mean():.2f} % of cells; by block: " + " ".join(f"{100 * x:.0f}" for x in e.reshape(9, 9).mean(1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "hostref", "compile", "run", "status", "poserr"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    {"build": lambda: build(a.names), "hostref": lambda: hostref(a.names, a.steps, a.threads),
     "compile": lambda: E3.compile_(["acct2"] + a.names), "run": lambda: E3.run(a.only, a.steps or 2, a.slots),
     "status": status, "poserr": poserr}[a.cmd]()
