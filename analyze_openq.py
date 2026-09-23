#!/usr/bin/env python3
"""Read-outs for the open questions (reports/EDGETRM_OPEN_QUESTIONS.md) from results/breadth/.

  python analyze_openq.py q1     # extrapolation under compression: tax or cap (dt_extrap_*.json)
  python analyze_openq.py q2     # masked fidelity vs the bands (dt_extrap_*.json)
  python analyze_openq.py q3     # residual / halting-head / fidelity AUROC for a wrong answer (residual_*.json)
  python analyze_openq.py q5     # activation-noise dial (anoise_*.json)
  python analyze_openq.py q8     # basin width vs width / mixer / seed (variants_noise/*.json)
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

B = Path(__file__).resolve().parent / "results" / "breadth"


def q1():
    for f in sorted(glob.glob(str(B / "p4d" / "dt_extrap_*.json"))):
        if "smoke" in f:
            continue
        j = json.load(open(f))
        recs = j["records"]
        its = sorted({r["iters"] for r in recs})
        quants = []
        for r in recs:
            s = r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
            if s not in quants:
                quants.append(s)
        acc = {(r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"], r["iters"]): 100 * r["acc"] for r in recs}
        print(f"\n### {Path(f).stem}  (n={recs[0]['n']}, train max_iters={recs[0]['max_iters']})")
        print("| quant | " + " | ".join(str(i) for i in its) + " | max | iters@90%fp32max |")
        print("|---|" + "---|" * (len(its) + 2))
        fmax = max(acc.get(("fp32", i), 0) for i in its)
        for q in quants:
            row = [acc.get((q, i)) for i in its]
            m = max(v for v in row if v is not None)
            # first iteration count at which the quantizer reaches 90% of fp32's best
            hit = next((i for i, v in zip(its, row) if v is not None and v >= 0.9 * fmax), None)
            print(f"| {q} | " + " | ".join("--" if v is None else f"{v:.1f}" for v in row) + f" | {m:.1f} | {hit} |")


def q2():
    print("| file | quant | iters | acc | Δacc vs fp32 | φ whole | φ target | φ pred | φ cell-target |")
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for f in sorted(glob.glob(str(B / "p4d" / "dt_extrap_*.json"))):
        if "smoke" in f:
            continue
        recs = json.load(open(f))["records"]
        fp = {r["iters"]: r for r in recs if (r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]) == "fp32"}
        for r in recs:
            q = r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
            if q == "fp32" or r["iters"] not in fp:
                continue
            d = 100 * (r["acc"] - fp[r["iters"]]["acc"])
            rows.append((Path(f).stem, q, r["iters"], 100 * r["acc"], d, r["fidelity"], r["fid_target"], r["fid_pred"], r["fid_cell_target"]))
    # the band violations: whole-map fidelity >= 0.85 but accuracy loss > 10 points
    viol = [x for x in rows if x[5] >= 0.85 and x[4] < -10]
    print(f"cells with whole-map φ ≥ 0.85 and Δacc < −10: {len(viol)} of {len(rows)}")
    for x in sorted(viol, key=lambda x: x[4])[:40]:
        print("| " + " | ".join([x[0], x[1], str(x[2]), f"{x[3]:.1f}", f"{x[4]:+.1f}", f"{x[5]:.3f}", f"{x[6]:.3f}", f"{x[7]:.3f}", f"{x[8]:.3f}"]) + " |")
    # does any masked fidelity separate the violators from the rest?  compare φ distributions
    import statistics as st
    ok = [x for x in rows if x[4] >= -3 and x[3] > 20]
    for k, name in ((5, "whole"), (6, "target"), (7, "pred"), (8, "cell-target")):
        a = [x[k] for x in viol if x[k] is not None]; b = [x[k] for x in ok if x[k] is not None]
        print(f"φ {name}: violators median {st.median(a):.3f} (n={len(a)})  vs  intact cells median {st.median(b):.3f} (n={len(b)})")
    # exact-match amplification: how many target cells does each task have?
    for f in sorted(glob.glob(str(B / "p4d" / "dt_extrap_*.json"))):
        if "smoke" in f:
            continue
        r = json.load(open(f))["records"][0]
        print(f"{Path(f).stem}: mean target cells per instance = {r.get('mean_target_cells')}")


def q3():
    for plat in ("p4d", "babel"):
        for f in sorted(glob.glob(str(B / plat / "residual*_*.json"))):
            if "smoke" in f:
                continue
            recs = json.load(open(f))["records"]
            nsup = recs[0]["nsup"]
            print(f"\n### {plat}/{Path(f).stem}  (n={recs[0]['n']}, nsup={nsup}); AUROC for 'answer wrong' at steps 2 / 4 / 8 / {nsup}")
            print("| quant | acc@k | resid_H | resid_L | q-halt | φ_H (ref) |")
            print("|---|---|---|---|---|---|")
            for r in recs:
                q = r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
                s = r["summary"]
                ks = [k for k in ("2", "4", "8", str(nsup)) if k in s]
                def col(key):
                    return "/".join("--" if s[k].get(key) is None else f"{s[k][key]:.2f}" for k in ks)
                accs = "/".join(f"{100*s[k]['acc']:.0f}" for k in ks)
                print(f"| {q} | {accs} | {col('auroc_resid_H')} | {col('auroc_resid_L')} | {col('auroc_qhalt')} | {col('auroc_fid_H')} |")


def q5():
    for plat in ("babel", "p4d"):
        for f in sorted(glob.glob(str(B / plat / "anoise_*.json"))):
            recs = json.load(open(f))["records"]
            print(f"\n### {plat}/{Path(f).stem} (n={recs[0]['n']})")
            print("| quant | nsup | exact | cell | φ_H | φ_L |")
            print("|---|---|---|---|---|---|")
            for r in recs:
                q = r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
                print(f"| {q} | {r['nsup']} | {100*r['pexact']:.1f} | {100*r['cell']:.1f} | {r['fidelity_H']:.3f} | {r['fidelity_L']:.3f} |")


def q8():
    print("| variant | nsup | fp32 | w8c | w4c | w4t | w3g32 | σ.05 | σ.1 | σ.15 | σ.2 | σ.3 | σ.5 | σ.7 | σ½ (exact halves) |")
    print("|---|" + "---|" * 14)
    for f in sorted(glob.glob(str(B / "p4d" / "variants_noise" / "*.json"))):
        recs = json.load(open(f))["records"]
        by = defaultdict(dict)
        for r in recs:
            q = r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
            by[r["nsup"]][q] = 100 * r["pexact"]
        for nsup in sorted(by):
            d = by[nsup]
            order = ["fp32", "w8c", "w4c", "w4t", "w3g32", "noise0.05", "noise0.1", "noise0.15", "noise0.2", "noise0.3", "noise0.5", "noise0.7"]
            sig = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
            half = next((s for s in sig if d.get(f"noise{s}", 0) < 0.5 * d["fp32"]), None)
            print(f"| {Path(f).stem.replace('trm_sudoku_','')} | {nsup} | " + " | ".join("--" if d.get(q) is None else f"{d[q]:.1f}" for q in order) + f" | {half} |")



def q1b():
    """Do compressed models fail on the instances fp32 finds hardest?  For each maze size, bin the
    instances by the fp32 iterations-to-solve (first iteration count at which fp32 is correct) and
    report each quantizer's failure rate at its largest iteration count within each bin."""
    import numpy as np
    for f in sorted(glob.glob(str(B / "p4d" / "dt_extrap_mazes_*.json"))):
        recs = json.load(open(f))["records"]
        its = sorted({r["iters"] for r in recs})
        spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
        by = {(spec(r), r["iters"]): np.array(r["correct"], bool) for r in recs}
        fp = np.stack([by[("fp32", i)] for i in its])            # [iters, n]
        n = fp.shape[1]
        solve_at = np.full(n, np.inf)
        for j, i in enumerate(its):
            solve_at = np.where((solve_at == np.inf) & fp[j], i, solve_at)
        bins = sorted(set(solve_at.tolist()))
        print(f"\n### {Path(f).stem}: failure rate at {its[-1]} iterations, by fp32 iterations-to-solve")
        print("| quant | " + " | ".join(("never" if b == np.inf else f"≤{int(b)}") + f" (n={int((solve_at==b).sum())})" for b in bins) + " |")
        print("|---|" + "---|" * len(bins))
        for q in ["w4c", "w4t", "w3a", "w3g32", "noise0.2", "noise0.3"]:
            if (q, its[-1]) not in by:
                continue
            ok = by[(q, its[-1])]
            cells = []
            for b in bins:
                m = solve_at == b
                cells.append(f"{100*(1-ok[m].mean()):.1f}" if m.any() else "--")
            print(f"| {q} | " + " | ".join(cells) + " |")


def q1c():
    """Angle G: for the instances a compressed model gets wrong at the largest iteration count,
    was the model ever right at an earlier count (arrived, then left) or never (never arrived)?
    Also the same for fp32 on the same instances, and whether the failing instances' fp32
    iterations-to-solve differs from the rest."""
    import numpy as np
    for f in sorted(glob.glob(str(B / "p4d" / "dt_extrap_mazes_*.json"))):
        recs = json.load(open(f))["records"]
        its = sorted({r["iters"] for r in recs})
        spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
        by = {(spec(r), r["iters"]): np.array(r["correct"], bool) for r in recs}
        print(f"\n### {Path(f).stem}: failures at {its[-1]} iterations")
        print("| quant | failures | were right earlier (arrived then left) | right at the previous count | never right |")
        print("|---|---|---|---|---|")
        for q in ["w4c", "w4t", "w3a", "noise0.2", "noise0.3"]:
            if (q, its[-1]) not in by:
                continue
            last = by[(q, its[-1])]
            fail = ~last
            ever = np.zeros_like(last)
            for i in its[:-1]:
                ever |= by[(q, i)]
            prev = by[(q, its[-2])]
            n = int(fail.sum())
            if n == 0:
                print(f"| {q} | 0 | -- | -- | -- |")
                continue
            print(f"| {q} | {n} | {100*(fail & ever).sum()/n:.0f}% | {100*(fail & prev).sum()/n:.0f}% | {100*(fail & ~ever).sum()/n:.0f}% |")


def q2b():
    """Q2 closure from the dt_cell_* reruns: exact accuracy, per-cell accuracy (all cells / target
    cells) and fidelity per quantizer; the spread over three noise draws; and whether failures
    concentrate on long paths."""
    import numpy as np
    for size in ("33", "25"):
        files = sorted(glob.glob(str(B / "p4d" / f"dt_cell_mazes_{size}_s*.json")))
        print(f"\n### mazes {size}: exact vs per-cell accuracy vs fidelity (seed 0), iterations 100/200/300")
        print("| quant | exact | cell acc (all) | cell acc (target) | φ whole | φ target |")
        print("|---|---|---|---|---|---|")
        recs0 = json.load(open(files[0]))["records"]
        spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
        order = []
        for r in recs0:
            if spec(r) not in order:
                order.append(spec(r))
        for q in order:
            rows = [r for r in recs0 if spec(r) == q]
            rows.sort(key=lambda r: r["iters"])
            f = lambda key: "/".join("--" if r.get(key) is None else f"{100*r[key]:.1f}" for r in rows)
            g = lambda key: "/".join("--" if r.get(key) is None else f"{r[key]:.3f}" for r in rows)
            print(f"| {q} | {f('acc')} | {f('cell_acc')} | {f('cell_acc_target')} | {g('fidelity')} | {g('fid_target')} |")
        print(f"\n### mazes {size}: three noise draws, exact accuracy at 300 iterations (φ whole)")
        print("| σ | seed 0 | seed 1 | seed 2 |")
        print("|---|---|---|---|")
        for q in ("noise0.2", "noise0.3"):
            cells = []
            for fpath in files:
                rr = [r for r in json.load(open(fpath))["records"] if spec(r) == q and r["iters"] == 300]
                cells.append(f"{100*rr[0]['acc']:.1f} ({rr[0]['fidelity']:.3f})" if rr else "--")
            print(f"| {q} | " + " | ".join(cells) + " |")
        # failures vs path length (seed 0): split instances by target-cell count quartile
        fp = [r for r in recs0 if spec(r) == "fp32" and r["iters"] == 300][0]
        if "n_target_cells" not in fp:      # seed-0 files predate the field; seeds 1/2 carry it
            alt = [r for r in json.load(open(files[1]))["records"] if spec(r) == "fp32" and r["iters"] == 300][0]
            assert alt["correct"] == fp["correct"], "instance order differs between seed files"
            fp = alt
        nt = np.array(fp["n_target_cells"])
        qs = np.quantile(nt, [0.25, 0.5, 0.75])
        bins = np.digitize(nt, qs)
        print(f"\n### mazes {size}: failure rate at 300 iterations by path-length quartile (target cells: quartiles {qs.astype(int).tolist()})")
        print("| quant | Q1 (short) | Q2 | Q3 | Q4 (long) |")
        print("|---|---|---|---|---|")
        for q in order:
            rr = [r for r in recs0 if spec(r) == q and r["iters"] == 300]
            if not rr:
                continue
            ok = np.array(rr[0]["correct"], bool)
            print(f"| {q} | " + " | ".join(f"{100*(1-ok[bins==b].mean()):.1f}" for b in range(4)) + " |")


def q5b():
    """Risk 0: three weight-noise draws per sigma for the TRM/HRM tasks. Seed 0 is the original
    single-draw sweep (results/breadth/<task>_noise.json, H=3), seeds 1 and 2 are the Babel reruns
    (results/breadth/babel/noise_<task>_seed<s>.json). Maze originals ran n=1000; the reruns n=500,
    so the originals are cut to their first 500 puzzles. Prints exact accuracy (phi_H) per draw and
    the spread, for nsup 1 and 16."""
    import numpy as np
    spec = lambda r: r["quant"]["spec"] if isinstance(r["quant"], dict) else r["quant"]
    for task in ("sudoku_mlp", "sudoku_attn", "maze_attn", "hrm_sudoku", "hrm_maze"):
        draws = {}
        f0 = B / f"{task}_noise.json"
        if f0.exists():
            recs = json.load(open(f0))["records"]
            draws[0] = [r for r in recs if r.get("H_cycles") in (3, None) or task.startswith("hrm")]
        for sd in (1, 2):
            fs = B / "babel" / f"noise_{task}_seed{sd}.json"
            if fs.exists():
                draws[sd] = json.load(open(fs))["records"]
        if not draws:
            continue
        n_new = min(r["n"] for sd in draws if sd else () for r in draws[sd]) if any(draws.get(s) for s in (1, 2)) else None
        print(f"\n### {task}: exact accuracy (phi_H) per draw; seeds {sorted(draws)}; reruns n={n_new}")
        print("| nsup | sigma | " + " | ".join(f"seed {sd}" for sd in sorted(draws)) + " | spread |")
        print("|---|---|" + "---|" * (len(draws) + 1))
        for nsup in (1, 16):
            for sig in ("0.05", "0.1", "0.15", "0.2", "0.3", "0.5", "0.7", "1.0"):
                cells, accs = [], []
                for sd in sorted(draws):
                    rr = [r for r in draws[sd] if spec(r) == f"noise{sig}" and r["nsup"] == nsup]
                    if not rr:
                        cells.append("--"); continue
                    r = rr[0]
                    if sd == 0 and n_new and r["n"] > n_new and isinstance(r.get("correct"), list):
                        acc = 100 * float(np.mean(r["correct"][:n_new]))
                    else:
                        acc = 100 * r["pexact"]
                    accs.append(acc)
                    cells.append(f"{acc:.1f} ({r['fidelity_H']:.2f})")
                spread = f"{max(accs) - min(accs):.1f}" if len(accs) > 1 else "--"
                flag = " **" if len(accs) > 1 and max(accs) - min(accs) > 10 else ""
                print(f"| {nsup} | {sig} | " + " | ".join(cells) + f" | {spread}{flag} |")


if __name__ == "__main__":
    globals()[sys.argv[1]]()
