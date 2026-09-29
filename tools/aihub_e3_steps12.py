#!/usr/bin/env python3
"""E3 label-free gate at steps 1 and 2 (E3_GATE_PHI_STEP2.md): recover the step-1 / step-2 device carries of the
existing chains from AI Hub without a device job, produce the matching host carries, and score carry fidelity.

The chain records (dev_<tag>.npz) keep only the carry after the LAST step (aihub_acct.Chain, aihub_infer.chain
overwrote it every step until Sep 29). AI Hub keeps every inference job's outputs and every dev_<tag>.json lists the
per-step inference job ids, so the earlier carries are a credentialed download (`download_output_data`), not a job.
Credentials: qai_hub.Client(profile="acct2" | "acct3") / qai_hub.Client() for the main profile, as the record's
`account` field names it; nothing is printed or logged.

Steps in the record's json are 0-based (record step s = readout step s+1); "step 1" / "step 2" here are the readout
steps (after 1 / 2 device steps) = record json steps 0 / 1.

    python aihub_e3_steps12.py download --chains e3_steps12_chains.json [--only ...]
        -> dev_<tag>_steps12.npz beside the record (never touches the record): zH_steps / zL_steps (2, N, 1, seq, 512)
           in the dtype the hub returned, preds_steps (2, N, seq), q_steps (2, N), job_ids; skipped when present.
    python aihub_e3_steps12.py check --chains ...      # download one chain's LAST step and compare with its record
    python aihub_e3_steps12.py host --task T --prec P --sem fp32|ref|fused --threads k
        -> host_fp32_n2.npz | host_<prec>_ref_n2.npz | host_<prec>_n2.npz with zH_steps / zL_steps (2, N, seq, 512)
           (ref = ORT_DISABLE_ALL as aihub_e3_followup.host_run; fused = ORT default as aihub_infer.stage_host)
    python aihub_e3_steps12.py phi --chains ... --out e3_steps12_phi.json
        -> per chain, per step (1, 2), per reference (fp32, ref, fused): per-puzzle cosines of z_H and z_L
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_infer as I  # noqa: E402
import aihub_e3_followup as F  # noqa: E402  (graph_path: the same graph files host_<prec>_ref.npz came from)

STEPS = (0, 1)     # record json steps -> readout steps 1, 2


def clients():
    import qai_hub as hub
    c = {}

    def get(name):
        if name not in c:
            c[name] = hub.Client() if name == "main" else hub.Client(profile=name)
        return c[name]
    return get


def sidecar(task, file):
    return I.idir(task) / file.replace(".json", "_steps12.npz")


def fetch_step(client, job_id):
    o = I.named_outputs(client.get_job(job_id).download_output_data())
    zH = np.stack(list(o["z_H_out"]))          # dtype as returned by the hub (float32 for every chain seen so far)
    zL = np.stack(list(o["z_L_out"]))
    lg = np.concatenate([x.astype(np.float32) for x in o["logits"]])
    q = np.array([float(x.reshape(-1)[0]) for x in o["q_halt"]])
    return zH, zL, lg.argmax(-1), q


def download(a):
    get = clients()
    chains = json.load(open(a.chains))
    done = skipped = failed = 0
    for ch in chains:
        if a.only and not any(s in ch["file"] for s in a.only):
            continue
        task, file, acct = ch["task"], ch["file"], ch.get("acct", "main")
        out = sidecar(task, file)
        if out.exists():
            skipped += 1
            continue
        d = I.idir(task)
        meta = json.loads((d / file).read_text())
        jobs = {j["step"]: j["job_id"] for j in meta["jobs"]}
        if not all(s in jobs for s in STEPS):
            print("no step-0/1 job ids on file:", task, file, flush=True)
            failed += 1
            continue
        t0 = time.time()
        try:
            rows = [fetch_step(get(acct), jobs[s]) for s in STEPS]
        except Exception as exc:                       # noqa: BLE001
            print("download failed:", task, file, type(exc).__name__, str(exc)[:200], flush=True)
            failed += 1
            continue
        zH = np.stack([r[0] for r in rows]); zL = np.stack([r[1] for r in rows])
        preds = np.stack([r[2] for r in rows]); q = np.stack([r[3] for r in rows])
        # consistency with the record's own predictions at those steps
        rec = d / file.replace(".json", ".npz")
        agree = "no record npz"
        if rec.exists():
            rp = np.load(rec)["preds"]
            agree = "preds agree" if rp.shape[0] >= 2 and (rp[:2] == preds).all() else f"PREDS DIFFER (record steps {rp.shape[0]})"
        tmp = out.with_name(out.name + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, zH_steps=zH, zL_steps=zL, preds_steps=preds, q_steps=q,
                                job_ids=np.array([jobs[s] for s in STEPS]), record_steps=np.array(STEPS))
        tmp.replace(out)
        done += 1
        print(f"{task} {file}: {zH.shape} {zH.dtype} / {zL.dtype}; {agree}; {time.time() - t0:.0f}s", flush=True)
    print(f"downloaded {done}, skipped {skipped}, failed {failed}", flush=True)


def check(a):
    """download the LAST step of one chain and compare its carry with the record: validates the download path."""
    get = clients()
    ch = [c for c in json.load(open(a.chains)) if a.only[0] in c["file"]][0]
    d = I.idir(ch["task"])
    meta = json.loads((d / ch["file"]).read_text())
    last = max(j["step"] for j in meta["jobs"])
    jid = [j["job_id"] for j in meta["jobs"] if j["step"] == last][0]
    zH, zL, preds, q = fetch_step(get(ch.get("acct", "main")), jid)
    rec = np.load(d / ch["file"].replace(".json", ".npz"))
    print(ch["file"], "last step", last, "hub dtype", zH.dtype, "record dtype", rec["zH"].dtype)
    print("zH max abs diff", float(np.abs(zH.astype(np.float32) - rec["zH"]).max()),
          "zL max abs diff", float(np.abs(zL.astype(np.float32) - rec["zL"]).max()),
          "preds equal", bool((preds == rec["preds"][last]).all()), "q max diff", float(np.abs(q - rec["q"][last]).max()))


def host(a):
    import onnxruntime as ort
    task, prec, sem = a.task, a.prec, a.sem
    d = I.idir(task)
    out = d / {"fp32": "host_fp32_n2.npz", "ref": f"host_{prec}_ref_n2.npz", "fused": f"host_{prec}_n2.npz"}[sem]
    if out.exists():
        print("exists, not overwritten:", out); return
    so = ort.SessionOptions()
    so.intra_op_num_threads = a.threads
    if sem == "ref":
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    path = F.graph_path(task, "fp32" if sem == "fp32" else prec)
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    inp, lab = I.test_set(task)
    zH0, zL0, emb = I.initial_state(task)
    N = inp.shape[0]
    zH, zL = np.repeat(zH0, N, 0)[:, None], np.repeat(zL0, N, 0)[:, None]
    zHs, zLs, preds, qs = [], [], [], []
    t0 = time.time()
    for s in range(2):                     # aihub_infer.host_loop, keeping the carry of every step
        nzH, nzL, lg, qh = [], [], [], []
        for i in range(N):
            o = sess.run(I.OUTS, {"z_H": zH[i], "z_L": zL[i], "inputs": inp[i:i + 1], "puzzle_embedding": emb})
            nzH.append(o[0].astype(np.float32)); nzL.append(o[1].astype(np.float32)); lg.append(o[2]); qh.append(float(o[3].reshape(-1)[0]))
        zH, zL = np.stack(nzH), np.stack(nzL)
        zHs.append(zH[:, 0]); zLs.append(zL[:, 0])
        preds.append(np.concatenate(lg).argmax(-1)); qs.append(qh)
        print(task, prec, sem, "step", s, "exact", float((preds[-1] == lab).all(-1).mean()), "cell", float((preds[-1] == lab).mean()),
              f"{time.time() - t0:.0f}s", flush=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, zH_steps=np.stack(zHs), zL_steps=np.stack(zLs), preds=np.stack(preds), q=np.array(qs))
    tmp.replace(out)
    json.dump({"graph": str(path), "semantics": {"fp32": "ORT default, stepc_H1.onnx", "ref": "ORT_DISABLE_ALL",
                                                 "fused": "ORT default (fused QDQ kernels)"}[sem], "steps": 2,
               "exact": [float((p == lab).all(-1).mean()) for p in preds], "cell": [float((p == lab).mean()) for p in preds]},
              open(out.with_suffix(".json"), "w"), indent=1)
    print("wrote", out, flush=True)


def cos_rows(x, y):
    x = x.reshape(x.shape[0], -1).astype(np.float64); y = y.reshape(y.shape[0], -1).astype(np.float64)
    return (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12)


def phi(a):
    chains = json.load(open(a.chains))
    res, hosts = [], {}

    def hostf(task, name):
        p = I.idir(task) / name
        if p not in hosts:
            hosts[p] = np.load(p) if p.exists() else None
        return hosts[p]
    for ch in chains:
        task, prec, file = ch["task"], ch["prec"], ch["file"]
        d = I.idir(task)
        sc = sidecar(task, file)
        row = dict(ch)
        if sc.exists():
            z = np.load(sc)
            dev = {s + 1: (z["zH_steps"][s][:, 0], z["zL_steps"][s][:, 0]) for s in range(2)}
            row["source"] = "hub download (steps12 sidecar)"
        else:
            rec = d / file.replace(".json", ".npz")
            if not rec.exists():
                row["source"] = "no sidecar, no record"; res.append(row); continue
            z = np.load(rec)
            k = int(z["preds"].shape[0])
            if k not in (1, 2):
                row["source"] = f"no sidecar; record ends at step {k}"; res.append(row); continue
            dev = {k: (z["zH"][:, 0], z["zL"][:, 0])}
            row["source"] = f"record carry (chain ended at step {k})"
        refs = {"fp32": hostf(task, "host_fp32_n2.npz")}
        if prec != "fp32":
            refs["ref"] = hostf(task, f"host_{prec}_ref_n2.npz")
            refs["fused"] = hostf(task, f"host_{prec}_n2.npz")
        row["phi"] = {}
        for s, (zH, zL) in dev.items():
            for rname, h in refs.items():
                if h is None:
                    continue
                cH, cL = cos_rows(zH, h["zH_steps"][s - 1]), cos_rows(zL, h["zL_steps"][s - 1])
                row["phi"][f"s{s}_{rname}"] = dict(H=float(cH.mean()), L=float(cL.mean()), H_min=float(cH.min()), L_min=float(cL.min()),
                                                   per_puzzle_H=cH.round(6).tolist(), per_puzzle_L=cL.round(6).tolist())
        res.append(row)
        f = lambda k: f"{row['phi'][k]['H']:.4f}" if k in row["phi"] else "-"   # noqa: E731
        print(f"{task} {prec} {ch['device']} ({ch.get('acct', 'main')}): s1 fp32 {f('s1_fp32')} ref {f('s1_ref')} | s2 fp32 {f('s2_fp32')} ref {f('s2_ref')} fused {f('s2_fused')}", flush=True)
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out, len(res), "chains")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("download", "check", "phi"):
        p = sub.add_parser(name)
        p.add_argument("--chains", required=True)
        p.add_argument("--only", nargs="*", default=None)
        p.add_argument("--out", default="e3_steps12_phi.json")
    p = sub.add_parser("host")
    p.add_argument("--task", required=True); p.add_argument("--prec", default="fp32")
    p.add_argument("--sem", choices=("fp32", "ref", "fused"), required=True); p.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    {"download": download, "check": check, "host": host, "phi": phi}[a.cmd](a)


if __name__ == "__main__":
    main()
