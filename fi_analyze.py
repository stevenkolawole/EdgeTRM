#!/usr/bin/env python3
"""FI read-out: hardware fault injection (fault_inject.py) on the compressed recursive reasoners, fresh vs fixed.

Every number is computed from the cell files (corr (K, n, D) uint8, per-seed per-puzzle exact correctness per step),
never from the index's acc_at. Estimator: per puzzle i the mean over the K seeds of corr[k, i, s-1] (the base has
K = 1); accuracy = mean over puzzles. Intervals: paired puzzle bootstrap, B = 2000 resamples of the puzzle index,
95 % percentile; every difference (arm - base, arm@D - arm@16, fixed - fresh, p4d - babel) is a difference of the
same resampled puzzles, so puzzle difficulty is paired out; the K = 3 fault draws are averaged, not resampled.

    python fi_analyze.py [--root results/fault] [--B 2000] [--out reports/edgetrm_exp/fi_tables.md]

Prints markdown tables (1) accuracy at 16 / 64 / D per task x base, (2) the three pre-registered tests (a, b, c),
(3) fixed vs fresh at equal magnitude at D, (4) the perturb-every-step reading, (5) p4d vs Babel platform check,
(6) flip counts vs weights.
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

RATES = ["1e-6", "1e-5", "1e-4", "1e-3", "1e-2"]
FRACS = ["0.05", "0.1", "0.2", "0.5"]
TASKS = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze_attn", "hrm_maze"]
BASES = ["fp32", "w8c"]
HOSTS = {"babel_l40s": "Babel L40S", "p4d_a100": "p4d A100"}


def load_host(root, host):
    """{(task, base, arm): dict(rec, p=(n, D) per-puzzle seed-mean correctness at every step)}"""
    out = {}
    for f in sorted(glob.glob(f"{root}/{host}/index_fault_*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            z = np.load(f"{root}/{host}/cells/{r['file']}")
            corr = z["corr"].astype(np.float32)            # (K, n, D)
            r["p"] = corr.mean(0)                          # (n, D) seed-mean per puzzle
            out[(r["task"], r["base"], r["arm"])] = r
    return out


class Boot:
    def __init__(self, n, B, seed=0):
        rng = np.random.default_rng(seed)
        self.idx = rng.integers(0, n, (B, n))

    def stat(self, v):
        """v: (n,) per-puzzle value -> (mean, lo, hi) in points (x100)."""
        v = np.asarray(v, np.float64)
        m = v[self.idx].mean(1)
        return 100 * v.mean(), 100 * np.percentile(m, 2.5), 100 * np.percentile(m, 97.5)


def fmt(t, sign=False):
    m, lo, hi = t
    s = f"{m:+.1f}" if sign else f"{m:.1f}"
    return f"{s} [{lo:+.1f}, {hi:+.1f}]" if sign else f"{s} [{lo:.1f}, {hi:.1f}]"


def sig(t):
    """CI excludes zero."""
    return t[1] > 0 or t[2] < 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent / "results" / "fault"))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = {h: load_host(a.root, h) for h in HOSTS}
    boots = {}

    def boot_for(n):
        if n not in boots:
            boots[n] = Boot(n, a.B)
        return boots[n]

    L = []
    pr = L.append
    src_script = "EdgeTRM/fi_analyze.py"

    def cell(host, task, base, arm):
        return data[host].get((task, base, arm))

    def acc(r, s):
        return r["p"][:, s - 1]

    # ---------------------------------------------------------------- (1) accuracy tables, Babel
    pr("## T1. Accuracy at 16 / 64 / D (Babel L40S; mean over K = 3 fault draws, per-puzzle seed mean; 95 % puzzle bootstrap)")
    pr("")
    pr("Delta = arm minus the fault-free base at the same step on the same puzzles (paired). Sudoku D = 512, n = 1000 "
       "(rows 1000-1999); Maze D = 128, n = 500 (rows 500-999).")
    pr("")
    host = "babel_l40s"
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            if b is None:
                continue
            D = b["D"]
            bt = boot_for(b["n"])
            pr(f"### {task} / {base} (D = {D}, n = {b['n']})")
            pr("")
            pr("| arm | kind | acc@16 | acc@64 | acc@D | D - base@D [95 %] | 16 - base@16 [95 %] |")
            pr("|---|---|---|---|---|---|---|")
            rows = [("none", "base")]
            for p in RATES:
                rows += [(f"bitflip{p}", "fresh"), (f"stuck{p}", "fixed")]
            for f in FRACS:
                rows += [(f"scalefresh{f}", "fresh"), (f"scale{f}", "fixed")]
            for arm, kind in rows:
                r = cell(host, task, base, arm)
                if r is None:
                    pr(f"| {arm} | {kind} | - | - | - | - | - |")
                    continue
                a16, a64, aD = (bt.stat(acc(r, s)) for s in (16, 64, D))
                dD = bt.stat(acc(r, D) - acc(b, D))
                d16 = bt.stat(acc(r, 16) - acc(b, 16))
                pr(f"| {arm} | {kind} | {a16[0]:.1f} | {a64[0]:.1f} | {aD[0]:.1f} | "
                   f"{'' if arm == 'none' else fmt(dD, True)} | {'' if arm == 'none' else fmt(d16, True)} |")
            pr("")
    pr(f"Source: results/fault/babel_l40s/index_fault_<task>.jsonl and cells/<task>__<base>__fault_<arm>__D<D>_K<K>_o<off>.npz; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (2a) fresh arms: gain D-16 vs base gain
    pr("## T2a. Test (a), fresh arms: does the loop still improve under fresh faults? gain = acc@D - acc@16 (Babel)")
    pr("")
    pr("gain(arm) and gain(arm) - gain(base), paired over puzzles. Prediction: gain > 0 wherever the arm has not "
       "collapsed, and the arm is within a few points of the base at D for p <= 1e-4.")
    pr("")
    pr("| task | base | base gain | arm | acc@D | gain(arm) [95 %] | gain(arm) - gain(base) [95 %] | reads |")
    pr("|---|---|---|---|---|---|---|---|")
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bt = boot_for(b["n"])
            bg = bt.stat(acc(b, D) - acc(b, 16))
            for arm in [f"bitflip{p}" for p in RATES] + [f"scalefresh{f}" for f in FRACS]:
                r = cell(host, task, base, arm)
                if r is None:
                    continue
                g = bt.stat(acc(r, D) - acc(r, 16))
                dg = bt.stat((acc(r, D) - acc(r, 16)) - (acc(b, D) - acc(b, 16)))
                aD = bt.stat(acc(r, D))
                if aD[0] < 0.5 * 100 * acc(b, D).mean():
                    reads = "collapsed"
                elif g[1] > 0:
                    reads = "improves" + (" (more than base)" if dg[1] > 0 else " (less than base)" if dg[2] < 0 else "")
                elif g[2] < 0:
                    reads = "WORSENS with depth"
                else:
                    reads = "flat" if bg[1] <= 0 else "flat (base improves)"
                pr(f"| {task} | {base} | {bg[0]:+.1f} | {arm} | {aD[0]:.1f} | {fmt(g, True)} | {fmt(dg, True)} | {reads} |")
    pr("")
    pr(f"Source: as T1; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (2b) fixed arms: gap to base at 16/64/D
    pr("## T2b. Test (b), fixed arms: gap to base at 16 / 64 / D, flat or collapse (Babel)")
    pr("")
    pr("gap(s) = acc(arm, s) - acc(base, s), paired. 'flat' = the gap at D is within the 95 % interval of the gap at 16 "
       "(their paired difference includes 0); 'collapse' = acc@D below half the base.")
    pr("")
    pr("| task | base | arm | gap@16 [95 %] | gap@64 | gap@D [95 %] | gap@D - gap@16 [95 %] | reads |")
    pr("|---|---|---|---|---|---|---|---|")
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bt = boot_for(b["n"])
            for arm in [f"stuck{p}" for p in RATES] + [f"scale{f}" for f in FRACS]:
                r = cell(host, task, base, arm)
                if r is None:
                    continue
                g16, g64, gD = (bt.stat(acc(r, s) - acc(b, s)) for s in (16, 64, D))
                dd = bt.stat((acc(r, D) - acc(b, D)) - (acc(r, 16) - acc(b, 16)))
                aD = 100 * acc(r, D).mean()
                if aD < 0.5 * 100 * acc(b, D).mean():
                    reads = "collapse"
                elif not sig(g16) and not sig(gD):
                    reads = "no effect"
                elif not sig(dd):
                    reads = "flat gap"
                elif dd[0] > 0:
                    reads = "gap narrows with depth"
                else:
                    reads = "gap WIDENS with depth"
                pr(f"| {task} | {base} | {arm} | {fmt(g16, True)} | {g64[0]:+.1f} | {fmt(gD, True)} | {fmt(dd, True)} | {reads} |")
    pr("")
    pr(f"Source: as T1; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (2c) feed-forward control
    pr("## T2c. Test (c), feed-forward control: fresh arm minus base at step 16 vs at step D (Babel)")
    pr("")
    pr("The 16-step run is the step-16 column of the same record. Prediction: at 16 the fresh arm is at or below the "
       "base; whatever recovery exists shows only at D.")
    pr("")
    pr("| task | base | arm | arm - base @16 [95 %] | arm - base @D [95 %] | (D gap) - (16 gap) [95 %] | reads |")
    pr("|---|---|---|---|---|---|---|")
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bt = boot_for(b["n"])
            for arm in [f"bitflip{p}" for p in RATES] + [f"scalefresh{f}" for f in FRACS]:
                r = cell(host, task, base, arm)
                if r is None:
                    continue
                g16 = bt.stat(acc(r, 16) - acc(b, 16))
                gD = bt.stat(acc(r, D) - acc(b, D))
                dd = bt.stat((acc(r, D) - acc(b, D)) - (acc(r, 16) - acc(b, 16)))
                if g16[1] > 0:
                    reads = "ABOVE base already at 16"
                elif sig(dd) and dd[0] > 0:
                    reads = "recovers/lifts with depth"
                elif sig(dd) and dd[0] < 0:
                    reads = "loses with depth"
                else:
                    reads = "no depth effect"
                pr(f"| {task} | {base} | {arm} | {fmt(g16, True)} | {fmt(gD, True)} | {fmt(dd, True)} | {reads} |")
    pr("")
    pr(f"Source: as T1; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (3) fixed vs fresh at equal magnitude at D
    pr("## T3. Fixed vs fresh at equal magnitude, at D (Babel): fixed - fresh, paired over puzzles")
    pr("")
    pr("stuck<p> seed k is the step-0 draw of bitflip<p> seed k; scale<f> seed k scales the layer scalefresh<f> seed k "
       "scales at step 0. Prediction: fixed <= fresh at D at every magnitude, widening with magnitude.")
    pr("")
    pr("| task | base | magnitude | fresh arm acc@D | fixed arm acc@D | fixed - fresh @D [95 %] | fixed - fresh @16 [95 %] | reads |")
    pr("|---|---|---|---|---|---|---|---|")
    summary3 = []
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bt = boot_for(b["n"])
            pairs = [(f"bitflip{p}", f"stuck{p}", f"p = {p}") for p in RATES] + \
                    [(f"scalefresh{f}", f"scale{f}", f"f = {f}") for f in FRACS]
            for fr, fx, mag in pairs:
                rf, rx = cell(host, task, base, fr), cell(host, task, base, fx)
                if rf is None or rx is None:
                    continue
                d = bt.stat(acc(rx, D) - acc(rf, D))
                d16 = bt.stat(acc(rx, 16) - acc(rf, 16))
                reads = "fresh better" if d[2] < 0 else "fresh WORSE" if d[1] > 0 else "tie"
                summary3.append((task, base, mag, reads, d[0]))
                pr(f"| {task} | {base} | {mag} | {100 * acc(rf, D).mean():.1f} | {100 * acc(rx, D).mean():.1f} | "
                   f"{fmt(d, True)} | {fmt(d16, True)} | {reads} |")
    pr("")
    cnt = {}
    for _, _, _, reads, _ in summary3:
        cnt[reads] = cnt.get(reads, 0) + 1
    pr(f"Count over {len(summary3)} matched pairs: " + ", ".join(f"{k}: {v}" for k, v in sorted(cnt.items())) + ".")
    pr("")
    pr(f"Source: as T1; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (4) perturb-every-step: fresh scale above base at D
    pr("## T4. Perturb every step: fresh arms above the fault-free base at D? (Babel; arm - base at D, paired)")
    pr("")
    pr("| task | base | base@D | arm | acc@D | arm - base @D [95 %] | reads |")
    pr("|---|---|---|---|---|---|---|")
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bt = boot_for(b["n"])
            for arm in [f"scalefresh{f}" for f in FRACS] + [f"bitflip{p}" for p in RATES[2:]]:
                r = cell(host, task, base, arm)
                if r is None:
                    continue
                d = bt.stat(acc(r, D) - acc(b, D))
                reads = "LIFT above base" if d[1] > 0 else "below base" if d[2] < 0 else "tie"
                pr(f"| {task} | {base} | {100 * acc(b, D).mean():.1f} | {arm} | {100 * acc(r, D).mean():.1f} | {fmt(d, True)} | {reads} |")
    pr("")
    pr(f"Source: as T1; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (5) platform check
    pr("## T5. Platform check: p4d A100 (torch 2.6) vs Babel L40S (torch 2.10), same arm, same puzzles")
    pr("")
    pr("Difference p4d - Babel at 16 and at D, paired over puzzles (the same rows on both hosts; the K = 3 fault draws "
       "are re-drawn on each host from the same seeds, but the CUDA generator stream differs between torch versions, so "
       "the draws are replicates, not copies). Never pooled.")
    pr("")
    pr("| task | base | arm | Babel @16 | p4d @16 | Babel @D | p4d @D | p4d - Babel @D [95 %] | agree |")
    pr("|---|---|---|---|---|---|---|---|---|")
    n_agree = n_tot = 0
    for task in TASKS:
        for base in BASES:
            for arm in ["bitflip1e-2", "stuck1e-2", "scalefresh0.5", "scale0.5"]:
                rb, rp = cell("babel_l40s", task, base, arm), cell("p4d_a100", task, base, arm)
                if rb is None or rp is None:
                    continue
                D = rb["D"]
                bt = boot_for(rb["n"])
                d = bt.stat(acc(rp, D) - acc(rb, D))
                ok = not sig(d)
                n_tot += 1
                n_agree += ok
                pr(f"| {task} | {base} | {arm} | {100 * acc(rb, 16).mean():.1f} | {100 * acc(rp, 16).mean():.1f} | "
                   f"{100 * acc(rb, D).mean():.1f} | {100 * acc(rp, D).mean():.1f} | {fmt(d, True)} | {'yes' if ok else 'NO'} |")
    pr("")
    pr(f"{n_agree} of {n_tot} matched cells agree (interval of the paired difference at D includes 0).")
    pr("")
    pr(f"Source: results/fault/p4d_a100/index_fault_p4d_<task>.jsonl + cells/, results/fault/babel_l40s/ as T1; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (6) flip counts vs weights
    pr("## T6. Flip counts: draw-0 flips (seed 0) vs weights, and the first rate that moves the number (Babel)")
    pr("")
    pr("weights = CastedLinear parameters of the model; bit positions = weights x nbits (23 mantissa bits fp32, 8 bits "
       "int8 incl. sign). 'moves' = |arm - base| at D with a 95 % interval excluding 0 for the fixed (stuck) arm; the "
       "fresh (bitflip) column likewise.")
    pr("")
    pr("| task | base | weights | nbits | rate | flips (draw 0) | flips / weight | stuck - base @D | bitflip - base @D |")
    pr("|---|---|---|---|---|---|---|---|---|")
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bt = boot_for(b["n"])
            for p in RATES:
                rx, rf = cell(host, task, base, f"stuck{p}"), cell(host, task, base, f"bitflip{p}")
                if rx is None:
                    continue
                W = rx["bit_positions"] // rx["nbits"]
                nfl = rx["draw0"]["0"]["nflips"]
                dx = bt.stat(acc(rx, D) - acc(b, D))
                df = bt.stat(acc(rf, D) - acc(b, D)) if rf is not None else None
                pr(f"| {task} | {base} | {W / 1e6:.2f} M | {rx['nbits']} | {p} | {nfl:,} | {nfl / W:.2e} | "
                   f"{fmt(dx, True)}{' *' if sig(dx) else ''} | {fmt(df, True) if df else '-'}{' *' if df and sig(df) else ''} |")
    pr("")
    pr("`*` = interval excludes 0.")
    pr("")
    pr(f"Source: draw0 field of the index records (seed 0) and the cells as T1; {src_script}.")
    pr("")


    # ---------------------------------------------------------------- (7) per-seed (per-layer) read of the scale arms
    pr("## T7. Per-seed read of the fixed scale arms (Babel): scale<f> seed k scales ONE drawn layer, so collapse is per layer")
    pr("")
    pr("acc@D per seed (K = 3 draws = 3 layers; the layer index is draw0[k].layer, an index into the model's CastedLinear "
       "list) against the base@D, and the fresh arm at the same f per seed for reference.")
    pr("")
    pr("| task | base | f | base@D | scale<f> per seed @D (layer) | scalefresh<f> per seed @D | seeds collapsed (< half base) |")
    pr("|---|---|---|---|---|---|---|")
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bD = 100 * acc(b, D).mean()
            for f in FRACS:
                rx, rf = cell(host, task, base, f"scale{f}"), cell(host, task, base, f"scalefresh{f}")
                if rx is None:
                    continue
                zx = np.load(f"{a.root}/{host}/cells/{rx['file']}")["corr"]
                zf = np.load(f"{a.root}/{host}/cells/{rf['file']}")["corr"] if rf is not None else None
                per = [100 * zx[k, :, D - 1].mean() for k in range(zx.shape[0])]
                lay = [rx["draw0"][str(k)]["layer"] for k in range(zx.shape[0])]
                perf = [100 * zf[k, :, D - 1].mean() for k in range(zf.shape[0])] if zf is not None else []
                ncol = sum(v < 0.5 * bD for v in per)
                pr(f"| {task} | {base} | {f} | {bD:.1f} | " + ", ".join(f"{v:.1f} (L{l})" for v, l in zip(per, lay)) +
                   " | " + ", ".join(f"{v:.1f}" for v in perf) + f" | {ncol} of {len(per)} |")
    pr("")
    pr(f"Source: as T1, corr per seed; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (8) trajectories of the arms that move with depth
    pr("## T8. Per-checkpoint trajectory of the arms whose accuracy changes with depth (Babel; mean over seeds)")
    pr("")
    pr("| task | base | arm | " + " | ".join(f"@{s}" for s in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)) + " |")
    pr("|---|---|---|" + "---|" * 10)
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            for arm in ["none", "bitflip1e-2", "stuck1e-2", "scalefresh0.5", "scale0.5", "scalefresh0.2", "scale0.2"]:
                r = cell(host, task, base, arm)
                if r is None:
                    continue
                vals = [f"{100 * acc(r, s).mean():.1f}" if s <= D else "-" for s in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)]
                pr(f"| {task} | {base} | {arm} | " + " | ".join(vals) + " |")
    pr("")
    pr(f"Source: as T1; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (9) per-step statistics of the fresh arms
    pr("## T9. Fresh arms read over EVERY step 17..D, not one snapshot (Babel; seed-mean accuracy per step)")
    pr("")
    pr("A fresh fault is re-drawn before every step and the answer is read after that step, so acc@s is a snapshot of "
       "the draw at step s. Columns: median and minimum over steps 17..D of the seed-mean accuracy; share of steps "
       "within 2 points of the base at the same step; longest run of consecutive steps below half the base at that "
       "step (a run of 1 = the loop is back within one step); acc at D.")
    pr("")
    pr("| task | base | arm | median 17..D | min 17..D (step) | steps within 2 pt of base | longest run < half base | acc@D |")
    pr("|---|---|---|---|---|---|---|---|")
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bs = 100 * b["p"].mean(0)                                          # (D,) base per step
            for arm in [f"bitflip{p}" for p in RATES[2:]] + [f"scalefresh{f}" for f in FRACS[1:]]:
                r = cell(host, task, base, arm)
                if r is None:
                    continue
                ts = 100 * r["p"].mean(0)                                      # (D,)
                seg, bseg = ts[16:], bs[16:]
                med, mn, arg = np.median(seg), seg.min(), 17 + int(seg.argmin())
                near = np.mean(np.abs(seg - bseg) <= 2.0)
                bad = seg < 0.5 * bseg
                run = best = 0
                for v in bad:
                    run = run + 1 if v else 0
                    best = max(best, run)
                pr(f"| {task} | {base} | {arm} | {med:.1f} | {mn:.1f} ({arg}) | {100 * near:.0f} % | {best} | {ts[D - 1]:.1f} |")
    pr("")
    pr(f"Source: as T1; {src_script}.")
    pr("")

    # ---------------------------------------------------------------- (10) two-step probe
    pr("## T10. What a two-step probe sees: gap to base at step 2 against the gap at D (Babel)")
    pr("")
    pr("gap(s) = arm - base at step s, paired. For a fixed fault the step-2 gap should predict the depth gap (bounded "
       "shift or collapse, both visible early); for a fresh fault the step-2 gap overstates the depth loss (the loop "
       "recovers), so the probe is conservative on transients. Only arms whose gap at 2 or at D is outside +-1 point.")
    pr("")
    pr("| task | base | arm | kind | gap@1 | gap@2 [95 %] | gap@16 | gap@D [95 %] | reads |")
    pr("|---|---|---|---|---|---|---|---|---|")
    for task in TASKS:
        for base in BASES:
            b = cell(host, task, base, "none")
            D = b["D"]
            bt = boot_for(b["n"])
            for arm in [f"{fam}{p}" for p in RATES for fam in ("bitflip", "stuck")] + \
                       [f"{fam}{f}" for f in FRACS for fam in ("scalefresh", "scale")]:
                r = cell(host, task, base, arm)
                if r is None:
                    continue
                g1 = 100 * (acc(r, 1) - acc(b, 1)).mean()
                g2, gD = bt.stat(acc(r, 2) - acc(b, 2)), bt.stat(acc(r, D) - acc(b, D))
                g16 = 100 * (acc(r, 16) - acc(b, 16)).mean()
                if abs(g2[0]) < 1 and abs(gD[0]) < 1:
                    continue
                kind = "fresh" if r["fresh"] else "fixed"
                if g2[2] < 0 and gD[1] > 0:
                    reads = "probe sees a loss, depth ends ABOVE base"
                elif g2[2] < 0 and not sig(gD):
                    reads = "probe sees a loss, depth recovers to base"
                elif g2[2] < 0 and gD[0] > g2[0] + 1:
                    reads = "probe sees a loss, depth recovers part"
                elif g2[2] < 0 and gD[0] < g2[0] - 1:
                    reads = "probe sees a loss, depth loses more"
                elif g2[2] < 0:
                    reads = "probe sees the loss, depth keeps it"
                elif not sig(g2) and sig(gD):
                    reads = "probe MISSES it (shows only at depth)"
                else:
                    reads = "probe sees a gain"
                pr(f"| {task} | {base} | {arm} | {kind} | {g1:+.1f} | {fmt(g2, True)} | {g16:+.1f} | {fmt(gD, True)} | {reads} |")
    pr("")
    pr(f"Source: as T1; {src_script}.")
    pr("")
    text = "\n".join(L)
    if a.out:
        Path(a.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
