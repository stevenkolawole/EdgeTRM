#!/usr/bin/env python3
"""Second-route re-derivation of the same-location fixed-vs-fresh control (REBUTTAL_NOTES.md section 1).
Reads results/breadth/babel/x1rb/rb_<task>_<arm><sigma>_s<k>.json directly (no analyze_difficulty import): per puzzle correctness
over 512 supervision steps is a hex bitset (bit k = correct after step k+1). For every arm x sigma: per draw gap(k) =
acc_q(k) - acc_fp32(k) over the 1800 puzzles, D = gap(512) - gap(16); pooled over the 5 draws as the mean with a t-interval
across draws; phi = mean of fid_H_mean over draws; per-draw regime from paired normal intervals (95%) over puzzles:
catch-up if D>0 with interval above 0, full if the gap(512) interval includes 0 (else partial); widening if D<0 with interval
below 0; tracks/constant otherwise. Arms: noise = fixed weight, afnoise = fixed activation, anoise = fresh activation,
wfresh = fresh weight.
"""
import glob, json, math, re
import numpy as np
R = "results/breadth/babel/x1rb/"
T = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78}
def bits(hexes, K=512):
    return np.array([[(int(h, 16) >> k) & 1 for k in range(K)] for h in hexes], dtype=np.int8)
def load(f):
    d = json.load(open(f)); r = d["records"][0]
    return bits(r["bits"], r["K"]), r["fid_H_mean"], r["spec"], d["rows"]
out = {}
print("task arm sigma | phi | D pooled [t-int] | draws D>0 | regimes (full/partial/constant+tracks/widening) | gap16 gap512 pooled")
for task in ("sudoku_mlp", "sudoku_attn", "hrm_sudoku"):
    F, _, _, rows0 = load(f"{R}rb_{task}_fp32.json")
    for arm in ("noise", "afnoise", "anoise", "wfresh"):
        sigmas = sorted({re.search(rf"{task}_{arm}([0-9.]+)_s\d", f).group(1) for f in glob.glob(f"{R}rb_{task}_{arm}*_s*.json") if re.search(rf"{task}_{arm}([0-9.]+)_s\d", f)}, key=float)
        for sg in sigmas:
            Ds, phis, reg, g16, g512 = [], [], {"full": 0, "partial": 0, "flat": 0, "widen": 0}, [], []
            files = sorted(glob.glob(f"{R}rb_{task}_{arm}{sg}_s*.json"))
            for f in files:
                B, phi, spec, rows = load(f)
                assert rows == rows0
                a = lambda k: B[:, k - 1].astype(float) - F[:, k - 1]
                d16, d512 = a(16), a(512)
                D = d512 - d16
                se = lambda x: x.std(ddof=1) / math.sqrt(len(x))
                mD, sD = D.mean(), se(D); m5, s5 = d512.mean(), se(d512)
                Ds.append(mD); phis.append(phi); g16.append(d16.mean()); g512.append(d512.mean())
                if mD - 1.96 * sD > 0:
                    reg["full" if (m5 - 1.96 * s5 <= 0 <= m5 + 1.96 * s5) else "partial"] += 1
                elif mD + 1.96 * sD < 0:
                    reg["widen"] += 1
                else:
                    reg["flat"] += 1
            Ds = np.array(Ds); n = len(Ds)
            h = T.get(n, 2.0) * Ds.std(ddof=1) / math.sqrt(n) if n > 1 else 0
            out[f"{task}|{arm}|{sg}"] = dict(n=n, phi=float(np.mean(phis)), D=100 * Ds.mean(), lo=100 * (Ds.mean() - h), hi=100 * (Ds.mean() + h), posD=int((Ds > 0).sum()), reg=reg,
                                              gap16=100 * np.mean(g16), gap512=100 * np.mean(g512))
            o = out[f"{task}|{arm}|{sg}"]
            print(f"{task:11s} {arm:8s} {sg:5s} n={n} phi {o['phi']:.2f} | D {o['D']:+.1f} [{o['lo']:+.1f},{o['hi']:+.1f}] | {o['posD']}/{n} | {reg} | {o['gap16']:+.1f} {o['gap512']:+.1f}")
json.dump(out, open("results/breadth/fixed_fresh_control_check.json", "w"), indent=1)
