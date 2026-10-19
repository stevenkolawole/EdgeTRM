#!/usr/bin/env python3
"""Save the predicted grids behind one x2_retk cell, for a worked example (the media image).

Reruns the first batch of puzzles with x2_retk.py's code path and batch size, and saves per puzzle: input, label,
the fp32 model's answer after T steps, the compressed model's answer after T steps, and the answer after k steps of
the returning map from the compressed end state. Prints how many correct/wrong flags agree with the x2_retk record.
    python dump_sudoku_grids.py --task hrm_sudoku --cell w2g32 --map w8c --k 16 --n 64 \
        --check results/breadth/babel/x2_retk/retk_hrm_sudoku_a.json --out sudoku_grids.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from x2_return import parse_cell  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="hrm_sudoku")
    ap.add_argument("--cell", default="w2g32")
    ap.add_argument("--map", default="w8c")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--check", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    import breadth_sweep as bs
    import contraction as C
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available(), "no CUDA device"
    family = "trm" if a.task in bs.TRM_TASKS else "hrm"
    t_ = bs.TRM_TASKS.get(a.task) or bs.HRM_TASKS[a.task]
    batch = 64 if t_["seq"] < 200 else 8
    inp, lab = bs.load_qa(t_["data"], a.n)
    mk = lambda net, x: C.Recursion(net, family, x)
    m0 = C.make_model(a.task, ("fp32", 0), a.T, batch)[0]
    mr = C.make_model(a.task, (a.map, 0), a.T, batch)[0]
    wspec, act, an, draw = parse_cell(a.cell)
    assert act is None and an is None, "weight-only cells only"
    mq = C.make_model(a.task, (wspec, draw), a.T, batch)[0]
    out = {k: [] for k in ("pred_f", "pred_q", "pred_r", "corr_f", "corr_q", "corr_r")}
    with torch.no_grad(), C.sdpa_math():
        for s in range(0, len(inp), batch):
            e = min(s + batch, len(inp))
            x = torch.from_numpy(inp[s:e]).to(bs.DEV)
            y = torch.from_numpy(lab[s:e]).to(bs.DEV)
            for tag, net in (("f", m0), ("q", mq)):
                F = mk(net, x)
                z = F.z0()
                for _ in range(a.T):
                    z = F(z)
                ro = F.readout(z, y)
                out["pred_" + tag].append(ro["pred"].cpu())
                out["corr_" + tag].append(ro["correct"].int().cpu())
                if tag == "q":
                    Fr = mk(mr, x)
                    w = z
                    for _ in range(a.k):
                        w = Fr(w)
                    ro = Fr.readout(w, y)
                    out["pred_r"].append(ro["pred"].cpu())
                    out["corr_r"].append(ro["correct"].int().cpu())
    out = {k: torch.cat(v).numpy() for k, v in out.items()}
    np.savez_compressed(a.out, inputs=inp, labels=lab, **out)
    n = len(inp)
    print(f"{a.task} {a.cell} -> {a.map} k={a.k}, n={n}: acc fp32 {out['corr_f'].mean():.3f} "
          f"compressed {out['corr_q'].mean():.3f} finished {out['corr_r'].mean():.3f}")
    if a.check:
        rec = next(r for r in json.load(open(a.check))["records"] if r["cell"] == a.cell)
        ref = {"f": rec["per"]["corr_f"], "q": rec["per"]["corr_q"], "r": rec["maps"][a.map]["per_corr"][str(a.k)]}
        print("flags agreeing with " + Path(a.check).name + ": "
              + ", ".join(f"{t} {int((np.array(ref[t][:n]) == out['corr_' + t]).sum())}/{n}" for t in "fqr"))


if __name__ == "__main__":
    main()
