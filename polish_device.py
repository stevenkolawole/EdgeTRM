#!/usr/bin/env python3
"""M1 on the device graphs: switch ACTIVATION precision between recursion steps, weights fixed.

The AI Hub QDQ graphs of the TRM step (H=1 carry step `stepc`, X3) share identical INT8 (or INT4) per-channel
weights and differ only in their activation encodings (8-bit vs 16-bit per tensor).  On Sudoku-MLP the
w8a8 graph solves 0/1000 (item 32 of REWRITE_NOTES), w8a16 recovers fp32.  Here the cheap steps run the
a8 graph and the last k steps the a16 graph of the same weights: a precision switch that needs no second
copy of the weights (on the device: two graphs over one weight buffer, or an encoding switch).

Host simulation with onnxruntime (CPU), batch 1 per puzzle as `aihub_infer.host_loop`.  Schedules
(T = 16 steps): a8 16, a16 16, fp32 16, a8(16-k)+a16(k), a8(16-k)+fp32(k), a8 16 + a16 k (append),
fp32 from scratch k (from the fp32 trajectory), a16(16-k)... also w4a8 -> w4a16 when --wbits 4.

    cd ~/x3 && nice -n 15 ~/venvs/kvcache/bin/python polish_device.py --task sudoku_mlp --start 1000 --count 200 \
        --out ~/m1/device_sudoku_mlp.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402
import aihub_infer as I  # noqa: E402


def main():
    import onnxruntime as ort
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sudoku_mlp")
    ap.add_argument("--start", type=int, default=1000)
    ap.add_argument("--count", type=int, default=200)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--wbits", default="8")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    T, ks = a.T, [int(k) for k in a.ks.split(",")]
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    sess = {"fp32": ort.InferenceSession(str(A.ART / a.task / "stepc_H1.onnx"), so, providers=["CPUExecutionProvider"])}
    for wb in a.wbits.split(","):
        for ab in ("8", "16"):
            p = I.idir(a.task) / f"qdq_w{wb}a{ab}" / "model.onnx"
            if p.exists():
                sess[f"w{wb}a{ab}"] = ort.InferenceSession(str(p), so, providers=["CPUExecutionProvider"])
    print("graphs:", list(sess), flush=True)
    if a.task == "arc":                            # canonical ARC-AGI-1 rows, each with its own puzzle embedding
        inp, lab, pids = I.arc_rows(a.start + a.count)
        inp, lab, pids = inp[a.start:], lab[a.start:], pids[a.start:]
        m, cfg, sd = A.build(a.task, 1)
        embs = sd["inner.puzzle_emb.weights"][I.torch_index(pids)].float().numpy()
        seq = cfg["seq_len"] + A.PLEN
        zH0 = np.broadcast_to(m.inner.H_init.detach().numpy(), (1, seq, 512)).astype(np.float32).copy()
        zL0 = np.broadcast_to(m.inner.L_init.detach().numpy(), (1, seq, 512)).astype(np.float32).copy()
        del m
    else:
        inp, lab = A.load_data(a.task, list(range(a.start, a.start + a.count)))
        zH0, zL0, emb = I.initial_state(a.task)
        embs = None
    inp = inp.astype(np.int32)
    N = inp.shape[0]

    def step(key, zH, zL, i):
        e = embs[i:i + 1] if embs is not None else emb
        o = sess[key].run(I.OUTS, {"z_H": zH, "z_L": zL, "inputs": inp[i:i + 1], "puzzle_embedding": e})
        return o[0], o[1], o[2].argmax(-1)[0]

    def traj(key, i, zH, zL, n, keep=()):
        """run n steps; returns per-step predictions and the states after the steps in keep."""
        preds, st = [], {}
        for s in range(n):
            zH, zL, p = step(key, zH, zL, i)
            preds.append(p)
            if s + 1 in keep:
                st[s + 1] = (zH, zL)
        return preds, st, (zH, zL)

    out_path = Path(a.out)
    part = out_path.with_suffix(".partial.json")
    t0 = time.time()
    corr, start = {}, 0                            # schedule -> per-puzzle correct (final) / per-step lists
    if part.exists():                              # resume (p4d reboots daily)
        pj = json.loads(part.read_text())
        if pj.get("start") == a.start and pj.get("wbits") == a.wbits:
            corr, start = pj["corr"], pj["done"]
            print(f"resuming at puzzle {start}", flush=True)
    cheap = [k for k in sess if k.endswith("a8")]
    for i in range(start, N):
        y = lab[i]
        msk = y != 0                               # ARC: token exact over non-pad cells (arc_breadth pexact_token)
        ok = (lambda p: int((p[msk] == y[msk]).all())) if a.task == "arc" else (lambda p: int((p == y).all()))
        # fp32 from scratch: per-step curve
        fp, _, _ = traj("fp32", i, zH0, zL0, T)
        corr.setdefault("fp32_curve", []).append([ok(p) for p in fp])
        for c in cheap:
            pol16 = c[:-1] + "16"
            cp, st, endc = traj(c, i, zH0, zL0, T, keep={T - k for k in ks if k < T})
            corr.setdefault(f"{c}_curve", []).append([ok(p) for p in cp])
            if pol16 in sess:
                pp, _, _ = traj(pol16, i, zH0, zL0, T)
                corr.setdefault(f"{pol16}_curve", []).append([ok(p) for p in pp])
            for pol in [pol16, "fp32"]:
                if pol not in sess:
                    continue
                for k in ks:
                    if k >= T:
                        continue
                    zh, zl = st[T - k]
                    p2, _, _ = traj(pol, i, zh, zl, k)
                    corr.setdefault(f"{c}{T-k}+{pol}{k}", []).append(ok(p2[-1]))
                ap_, _, _ = traj(pol, i, endc[0], endc[1], max(ks))
                corr.setdefault(f"{c}{T}+{pol}_append", []).append([ok(p) for p in ap_])
        if (i + 1) % 10 == 0 or i + 1 == N:
            tmp = part.with_suffix(".tmp")
            tmp.write_text(json.dumps({"start": a.start, "wbits": a.wbits, "done": i + 1, "corr": corr}))
            tmp.replace(part)
            print(f"  {i+1}/{N} puzzles ({time.time()-t0:.0f}s)", flush=True)
    summ = {"task": a.task, "start": a.start, "count": N, "T": T, "ks": ks, "graphs": list(sess),
            "secs": time.time() - t0, "per_puzzle": corr, "acc": {}}
    for kname, v in corr.items():
        arr = np.asarray(v)
        summ["acc"][kname] = arr.mean(0).tolist() if arr.ndim == 2 else float(arr.mean())
    out_path.write_text(json.dumps(summ))
    for kname, v in summ["acc"].items():
        if isinstance(v, list):
            print(f"{kname}: " + ", ".join(f"{s+1}:{100*x:.1f}" for s, x in enumerate(v) if s + 1 in set(ks) | {T}))
        else:
            print(f"{kname}: {100*v:.1f}")
    print("wrote", out_path)


if __name__ == "__main__":
    main()
