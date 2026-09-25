#!/usr/bin/env python3
"""M4: run the release check (release_check.py, ONNX path) on every AI Hub QDQ graph and X3 ablation
variant that has a host-simulation label, on unlabeled check inputs disjoint from the evaluation set.

    ~/venvs/kvcache/bin/python m4_onnx_sweep.py --workers 4 --threads 4
Outputs: ~/m4/ev/onnx__<task>__<cell>.json (one per cell; resumable), ref caches ~/m4/refcache/.
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import sys
import time
import traceback

HOME = pathlib.Path.home()
ART = HOME / "aihub_artifacts" / "h1"
M4 = HOME / "m4"
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

TASK_CFG = {"sudoku_mlp": dict(n=32, T=16), "sudoku_attn": dict(n=32, T=16), "maze": dict(n=16, T=8),
            "arc": dict(n=16, T=8)}
PRECS = ["w8a8", "w4a8", "w8a16", "w4a16"]


def cells():
    out = []
    for task in TASK_CFG:
        out.append((task, "fp32self", ART / task / "stepc_H1.onnx"))
        for p in PRECS:
            g = ART / task / "infer" / f"qdq_{p}" / "model.onnx"
            if g.exists():
                out.append((task, p, g))
                if p in ("w8a8", "w4a8"):                   # QNN kernel emulation (fused RMSNorm)
                    out.append((task, p + "@qnnfused", g))
                    out.append((task, p + "@qnnfused16", g))
        vj = ART / task / "infer" / "ablate" / "variants.json"
        if vj.exists():
            for name, v in json.loads(vj.read_text()).items():
                if pathlib.Path(v["path"]).exists():
                    out.append((task, "abl_" + name, pathlib.Path(v["path"])))
    return out


def ref_cache(task):
    return M4 / "refcache" / f"ref_{task}_n{TASK_CFG[task]['n']}_T{TASK_CFG[task]['T']}.npz"


LEVEL = "all"


def run_cell(args):
    task, name, graph, threads, level = args
    vendor = {"qnnfused": "qnn", "qnnfused16": "qnn16"}.get(name.split("@")[-1]) if "@" in name else None
    import numpy as np
    import release_check as R
    out = M4 / "ev" / (f"onnx__{task}__{name}.json" if level == "all" else f"onnx__{task}__{name}#{level}.json")
    if out.exists():
        return f"{task}/{name}", "exists"
    t0 = time.time()
    try:
        cfg = TASK_CFG[task]
        chk = R.load_check(M4 / "check" / f"check_{task}.npz", cfg["n"])
        ref = R.OnnxStep(ART / task / "stepc_H1.onnx", threads, opt_level=level)
        cand = R.OnnxStep(graph, threads, vendor=vendor, opt_level=level)
        ev = dict(kind="onnx", backend="qnn_qdq_host", task=task, cell=name, graph=str(graph), n=cfg["n"],
                  steps=cfg["T"], vendor=vendor, n_fused_norms=cand.n_fused, ort_level=level)
        ev.update(R.map_and_rollout(ref, cand, chk, cfg["T"], ref_cache=str(ref_cache(task))))
        z = np.load(ref_cache(task))
        t1 = time.time()
        ev.update(R.onnx_per_layer(ART / task / "stepc_H1.onnx", graph, chk, (z["sT_H"], z["sT_L"]),
                                   samples=(0, 1), threads=threads, vendor=vendor, opt_level=level))
        ev.update(R.onnx_xrt(graph, chk, (z["sT_H"], z["sT_L"]), threads=threads, vendor=vendor))
        ev["t0_seconds"] = round(time.time() - t1, 1)
        c = ev.pop("_cand_rollout", None)
        if c is not None:
            np.savez_compressed(out.with_suffix(".npz"), preds=c["preds"], q=c["q"])
        v, rules = R.decide(ev)
        ev.update(verdict=v, rules=rules, seconds=round(time.time() - t0, 1))
        out.write_text(json.dumps(R._strip(ev), indent=1, default=float))
        return f"{task}/{name}", f"{v} delta={ev['delta_T_median']:.3f} cell={ev['cell_agree']:.3f} ({ev['seconds']}s)"
    except Exception as e:  # noqa: BLE001
        (M4 / "ev" / f"onnx__{task}__{name}.err").write_text(traceback.format_exc())
        return f"{task}/{name}", f"ERROR {type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--tasks", default=",".join(TASK_CFG))
    p.add_argument("--ort_level", default="all", choices=["all", "basic"])
    a = p.parse_args()
    tasks = a.tasks.split(",")
    todo = [c for c in cells() if c[0] in tasks]
    # fp32 reference rollouts first (one per task), so the workers share them
    import release_check as R
    for task in tasks:
        if not ref_cache(task).exists():
            t0 = time.time()
            cfg = TASK_CFG[task]
            chk = R.load_check(M4 / "check" / f"check_{task}.npz", cfg["n"])
            ref = R.OnnxStep(ART / task / "stepc_H1.onnx", a.threads * a.workers)
            R.map_and_rollout(ref, ref, chk, cfg["T"], ref_cache=str(ref_cache(task)), skip_rollout=True)
            print(f"[ref] {task} n={cfg['n']} T={cfg['T']} ({time.time() - t0:.0f}s)", flush=True)
    print(f"{len(todo)} cells", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, msg in pool.imap_unordered(run_cell, [(t, n, g, a.threads, a.ort_level) for t, n, g in todo]):
            print(f"[{time.strftime('%H:%M:%S')}] {name}: {msg}", flush=True)


if __name__ == "__main__":
    main()
