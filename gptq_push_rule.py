#!/usr/bin/env python3
"""Reviewer LXVH W3: does the tolerance law's room classify GPTQ and RTN models as collapsed vs surviving from their measured push,
and do GPTQ models sit on the RTN push-vs-retained curve?

Push: results/m5/push_gptq_vs_rtn/{sudoku_mlp,sudoku_attn}.json (median push at T=16, rows 0-255). Retained = accuracy at 16
default-H steps / fp32 accuracy at 16 steps, from results/m5/ctl/p4d det records (rows 0-1999 and rows 0-255 for the push rows).
Rule: predict collapse iff push > room. Rooms: pooled delta* (App F, 0.344), TRM+HRM refit 0.327 (x2_sens_fit_same.md), and per-mixer
rooms = median of the implied rooms S*sigma_half of the sudoku TRM models of that mixer (x2_sens_fit_same.md, low-fp32 rows excluded).
Label: collapsed iff retained < thr (0.5; sensitivity 0.4, 0.6).   python gptq_push_rule.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import m5_ctl_ho as C

def load_acc(task, spec, rows):
    d = [C.load_det(C.ROOT / f"det_o{o}" / task / f"{spec}.npz") for o in (0, 1000)]
    if any(x is None for x in d):
        return None
    c = np.concatenate([x[0][:, 15] for x in d])
    return float(c[rows].mean())

def mixer_rooms():
    out = {"sudoku_mlp": [], "sudoku_attn": []}
    for ln in open(H / "results/breadth/x2_sens_fit_same.md"):
        if "low fp32" in ln or not ln.startswith("|"):
            continue
        p = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(p) < 9: continue
        for t, k in (("mlp", "sudoku_mlp"), ("attn", "sudoku_attn")):
            if re.fullmatch(rf"(trm_sudoku_{t}_w\d+_s\d|sudoku_{t})", p[0]):
                try: out[k].append(float(p[8]))
                except ValueError: pass
    return {k: (float(np.median(v)), len(v)) for k, v in out.items()}

def main():
    rooms_mixer = mixer_rooms()
    print("per-mixer room (median implied room, n models):", rooms_mixer)
    rows = []
    for task in ("sudoku_mlp", "sudoku_attn"):
        P = json.load(open(H / f"results/m5/push_gptq_vs_rtn/{task}.json"))["cells"]
        for key, v in P.items():
            spec = key
            for rs, nm in ((slice(0, 2000), "2000"), (slice(0, 256), "256")):
                pass
            r = dict(task=task, spec=key, gptq=v["gptq"], fmt=v["fmt"], push=v["push"]["16"]["median"], push64=v["push"]["64"]["median"],
                     wdist=v["w_dist_pooled"])
            if load_acc(task, key, slice(0, 2000)) is None:
                print("no det record, skipped:", task, key); continue
            f2000, a2000 = load_acc(task, "fp32", slice(0, 2000)), load_acc(task, key, slice(0, 2000))
            f256, a256 = load_acc(task, "fp32", slice(0, 256)), load_acc(task, key, slice(0, 256))
            r.update(fp32=f2000, acc=a2000, ret=a2000 / f2000, ret256=a256 / f256)
            rows.append(r)
    print(f"\n{'task':11s} {'spec':9s} {'type':5s} {'push16':>6s} {'wdist':>5s} {'acc':>5s} {'fp32':>5s} {'ret2000':>7s} {'ret256':>7s}")
    for r in rows:
        print(f"{r['task']:11s} {r['spec']:9s} {'GPTQ' if r['gptq'] else 'RTN':5s} {r['push']:6.3f} {r['wdist']:5.3f} {100*r['acc']:5.1f} {100*r['fp32']:5.1f} {r['ret']:7.3f} {r['ret256']:7.3f}")
    res = {"rows": rows, "rooms": {}}
    for thr in (0.5, 0.4, 0.6):
        print(f"\n=== label: collapsed iff retained (rows 0-1999) < {thr}")
        for rname, room in (("pooled 0.344", lambda r: 0.344), ("refit 0.327", lambda r: 0.327), ("per-mixer", lambda r: rooms_mixer[r["task"]][0])):
            out = {}
            for grp, sel in (("GPTQ", [r for r in rows if r["gptq"]]), ("RTN", [r for r in rows if not r["gptq"]]), ("all", rows)):
                ok = [(r["push"] > room(r)) == (r["ret"] < thr) for r in sel]
                miss = [f"{r['task'][7:]} {r['spec']} push {r['push']:.2f} room {room(r):.2f} ret {r['ret']:.2f}" for r, o in zip(sel, ok) if not o]
                out[grp] = (int(sum(ok)), len(ok), miss)
            print(f"  {rname:13s} " + " | ".join(f"{g}: {a}/{b}" for g, (a, b, m) in out.items()))
            for g, (a, b, m) in out.items():
                for x in m: print(f"      miss [{g}] {x}")
            res["rooms"][f"{thr}|{rname}"] = {g: dict(correct=a, n=b, misses=m) for g, (a, b, m) in out.items()}
    # curve: isotonic (decreasing) fit of retained vs push on RTN only, GPTQ residuals; and the same on weight distance (control)
    def iso_pred(x, y, x0):
        o = np.argsort(x); x, y = np.array(x)[o], np.array(y)[o]
        yy = y.copy()
        for _ in range(100):                                  # pool adjacent violators for a non-increasing fit
            for i in range(len(yy) - 1):
                if yy[i] < yy[i + 1]:
                    yy[i] = yy[i + 1] = (yy[i] + yy[i + 1]) / 2
        return float(np.interp(x0, x, yy))
    print("\n=== push-vs-retained curve: isotonic fit on RTN points (leave-one-out for RTN), GPTQ predicted from the RTN curve")
    for xk in ("push", "wdist"):
        rtn = [r for r in rows if not r["gptq"]]
        loo = [r["ret"] - iso_pred([q[xk] for q in rtn if q is not r], [q["ret"] for q in rtn if q is not r], r[xk]) for r in rtn]
        gp = [r["ret"] - iso_pred([q[xk] for q in rtn], [q["ret"] for q in rtn], r[xk]) for r in rows if r["gptq"]]
        print(f"  x = {xk}: RTN leave-one-out residuals (measured - predicted retained) mean abs {np.mean(np.abs(loo)):.3f}, max {np.max(np.abs(loo)):.3f};"
              f" GPTQ residuals {[round(g, 3) for g in gp]} mean abs {np.mean(np.abs(gp)):.3f}, mean signed {np.mean(gp):+.3f}")
        res[f"curve_{xk}"] = dict(rtn_loo=loo, gptq=gp)
    json.dump(res, open(H / "results/m5/gptq_push_rule.json", "w"), indent=1, default=float)
    print("wrote results/m5/gptq_push_rule.json")

main()
