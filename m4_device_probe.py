#!/usr/bin/env python3
"""M4: the release check's device-facing tier, a one-step on-device probe.

For every AI Hub on-device chain X3 ran (completed or retired), take the device's own output of the
FIRST supervision step (the step-0 inference job; its outputs are downloaded, no new job is submitted)
and compare it with the host references on the same inputs and initial carry:
  host fp32 graph (stepc_H1.onnx), the QDQ graph under reference semantics (ORT basic), and under ORT's
  fused integer kernels (ORT all).
Label-free quantities per chain: median z_H cosine and logits argmax agreement at step 1, on the first
`--n` inputs. Ground truth: the chain's exact accuracy at its last step vs the S24 fp32 chain / host fp32
at the same step. The question (coordinator, REWRITE_NOTES 48): does one step on the device, compared
with the host fp32 reference, separate the device graphs that work from those that fail?

    ~/venvs/kvcache/bin/python m4_device_probe.py --n 16     -> ~/m4/device_probe.json
"""
import argparse
import json
import pathlib
import sys

import numpy as np

HOME = pathlib.Path.home()
sys.path.insert(0, str(HOME / "x3"))
M4 = HOME / "m4"
sys.path.insert(0, str(M4))


def named(o):
    names = ["z_H_out", "z_L_out", "logits", "q_halt"]
    if all(n in o for n in names):
        return o
    return {names[i]: o[f"output_{i}"] for i in range(4)}


def cos_rows(a, b):
    a = a.reshape(a.shape[0], -1).astype(np.float64)
    b = b.reshape(b.shape[0], -1).astype(np.float64)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=16)
    a = p.parse_args()
    import onnxruntime as ort
    import qai_hub as hub
    import aihub_infer as I
    import release_check as R
    out = []
    cache = M4 / "device_probe_cache"
    cache.mkdir(exist_ok=True)
    for meta_f in sorted((HOME / "aihub_artifacts" / "h1").glob("*/infer/dev_*.json")):
        task = meta_f.parts[-3]
        prec, dev = meta_f.stem[4:].split("_", 1)
        meta = json.loads(meta_f.read_text())
        if not meta.get("jobs"):
            continue
        job0 = next((j for j in meta["jobs"] if j["step"] == 0), None)
        rec = dict(task=task, prec=prec, device=dev.replace("_", " "), done=bool(meta.get("done")),
                   failed=meta.get("failed") or meta.get("retired") or meta.get("reason"),
                   steps_run=len(meta.get("steps", [])))
        if meta.get("steps"):
            rec["exact_per_step"] = [s_["exact"] for s_ in meta["steps"]]
        try:
            cf = cache / f"{task}__{meta_f.stem}_step0.npz"
            if cf.exists():
                z = np.load(cf)
                o = {k: z[k] for k in z.files}
            else:
                job = hub.get_job(job0["job_id"])
                if job.get_status().code != "SUCCESS":
                    rec["error"] = f"step-0 job {job0['job_id']} {job.get_status().code}"
                    out.append(rec)
                    continue
                raw = named(job.download_output_data())
                o = {k: np.stack([np.asarray(x, np.float32) for x in v]) for k, v in raw.items()}
                np.savez_compressed(cf, **o)
            inp, lab = I.test_set(task)
            zH0, zL0, emb = I.initial_state(task)
            N = min(a.n, o["z_H_out"].shape[0])
            dev_zH = o["z_H_out"][:N].reshape(N, -1)
            dev_lg = o["logits"][:N].reshape(N, -1, o["logits"].shape[-1])
            refs = {"host_fp32": (I.A.ART / task / "stepc_H1.onnx", "basic")}
            qdq = I.idir(task) / f"qdq_{prec}" / "model.onnx"
            if prec != "fp32" and qdq.exists():
                refs["host_qdq_ref"] = (qdq, "basic")
                refs["host_qdq_ortall"] = (qdq, "all")
            for name, (path, lvl) in refs.items():
                st = R.OnnxStep(path, 4, opt_level=lvl)
                hz, hl = [], []
                for i in range(N):
                    zH, zL, lg, q = st(zH0, zL0, inp[i:i + 1], emb)
                    hz.append(zH.reshape(-1)); hl.append(lg.reshape(-1, lg.shape[-1]))
                hz, hl = np.stack(hz), np.stack(hl)
                c = cos_rows(dev_zH, hz)
                rec[f"zH_cos_vs_{name}"] = float(np.median(c))
                rec[f"zH_cos_min_vs_{name}"] = float(c.min())
                rec[f"argmax_agree_vs_{name}"] = float((dev_lg.argmax(-1) == hl.argmax(-1)).mean())
            rec["n_probe"] = N
            # k-step probe from the chain's own per-step predictions (no extra job): agreement with host
            # fp32 on the cells fp32 writes, per step
            dnpz = meta_f.with_suffix(".npz")
            hf = I.idir(task) / "host_fp32.npz"
            if dnpz.exists() and hf.exists():
                P = np.load(dnpz)["preds"]
                H = np.load(hf)["preds"]
                M = min(P.shape[1], H.shape[1])
                per = []
                for k in range(min(P.shape[0], H.shape[0])):
                    ch = H[k][:M] != inp[:M]
                    per.append(float((((P[k][:M] == H[k][:M]) & ch).sum(-1) / np.maximum(ch.sum(-1), 1)).mean()))
                rec["written_agree_vs_host_fp32_per_step"] = per
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e}"
        out.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("exact_per_step",)}), flush=True)
    (M4 / "device_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
