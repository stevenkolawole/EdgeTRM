#!/usr/bin/env python3
"""Sequence-length control for the Maze w8a8 device fault (REWRITE_NOTES 111 / audit item 111).

The Maze weights exported at seq 97 (81 grid cells + 16 puzzle-embedding positions) instead of 916, with
everything else the same: export, calibration from real carries (first 81 cells of the calibration
mazes), AI Hub quantize jobs (w8a8 and w8a16) on acct2, then the wa_collapse bisection variants
M0_all8 / Ma_rope16 / Mc_scores16 built from them. `BISECT_TASK=maze_s97 maze_bisect.py` then runs step 1
on the S24 and compares device z_H with the host run of the same graph. If M0_all8 matches its host
graph here (cosine > 0.99) while it fails at seq 916 (0.710), the RoPE fault depends on sequence length
with the model held fixed.

    python maze_s97.py        (idempotent; waits for the quantize jobs)
"""
import json
import pathlib
import shutil
import sys
import time
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_h1 as A  # noqa: E402
import aihub_infer as I  # noqa: E402

TASK, ACCT = "maze_s97", "acct2"


def main():
    import qai_hub as hub
    C = hub.Client(profile=ACCT)
    A.stage_export([TASK], 1)
    A.stage_calib([TASK], 1)
    path = A.ART / TASK / A.onnx_name("stepc", 1)
    for prec in ("w8a8", "w8a16"):
        if A.find(A.load_rec(TASK), kind="quantize", graph="stepc", H=1, prec=prec):
            continue
        wd, ad = A.QDTYPE[prec][:2]
        qj = C.submit_quantize_job(model=str(path), calibration_data=A.calib_dict(TASK, 1, "stepc"),
                                   weights_dtype=getattr(hub.QuantizeDtype, wd),
                                   activations_dtype=getattr(hub.QuantizeDtype, ad),
                                   name=f"x3_{ACCT}_{TASK}_stepc_H1_{prec}")
        A.add_job(TASK, {"kind": "quantize", "graph": "stepc", "H": 1, "prec": prec, "job_id": qj.job_id,
                         "account": ACCT, "submitted": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        print(f"[{TASK}] quantize {prec} -> {qj.job_id} ({ACCT})", flush=True)
    for prec in ("w8a8", "w8a16"):
        d = I.idir(TASK) / f"qdq_{prec}"
        if (d / "model.onnx").exists():
            continue
        q = A.find(A.load_rec(TASK), kind="quantize", graph="stepc", H=1, prec=prec)[-1]
        job = C.get_job(q["job_id"])
        st = job.wait()
        if st.code != "SUCCESS":
            print(f"[{TASK}] quantize {prec} FAILED {st.message}", flush=True)
            sys.exit(1)
        z = job.get_target_model().download(str(I.idir(TASK) / f"qdq_{prec}_dl"))
        tmp = I.idir(TASK) / f"unz_{prec}"
        with zipfile.ZipFile(z) as f:
            f.extractall(tmp)
        next(tmp.iterdir()).rename(d)
        shutil.rmtree(tmp)
        pathlib.Path(z).unlink()
        print(f"[{TASK}] downloaded {prec} -> {d}", flush=True)
    import wa_collapse as W
    W.stage_build(TASK, ["M0_all8", "Ma_rope16", "Mc_scores16"])


if __name__ == "__main__":
    main()
