#!/usr/bin/env python3
"""E1 dither row read-out (MLSys): energy per correct answer of perturb-every-step vs RTN vs fp32 on the eager L40S path.

Reads <root>/<task>/energy_dither.json (e1_dither.py, 14 configurations per model) and <root>/<task>/grid.json
(e1_grid.py, the grid it extends) and prints, per model:
  1. the reproduction check: the in-job fp32 rows against the grid's eager_fp32_b{1,256}_{fixed,exit} rows (same
     harness, another day and card): energy per step, J per correct, accuracy, steps, ms per step, W, card temperature,
     and the window CV of mJ/step on both sides. Strict PASS = |in-job / grid - 1| <= 2 * sqrt(cv_grid^2 + cv_job^2)
     on mJ/step (two window-CVs of the difference of two 5-window medians); because the windows inside one job are
     far tighter (0.1-1%) than two cards on two days, a second verdict with a 2% floor on the tolerance is printed
     as "PASS (2% floor)" where only the floor passes it;
  2. the arm x batch x policy table: accuracy, mean steps, ms per step, W above idle, mJ per step (total and idle-
     subtracted), J per correct (total and idle-subtracted) with 95% t-intervals over the windows, and the ratios
     dither / rtn and dither / fp32 at equal (batch, policy);
  3. the re-draw share (redraw_timing) per batch;
  4. the predictions of e1_dither.py's sbatch header against the outcome.

Every statistic is recomputed from the windows: clean windows whose duration is < 2 x win_s (the two stalled first
windows on babel-u5-28, 5300 s long with the card at idle power, are dropped; the record's medians already ignore
them but its t-intervals do not). J per correct = median over windows of (E_J / puzzles done) / first-pass accuracy;
the t-interval is the windows' 95% t-interval of E_J / done, divided by the accuracy. Ratio intervals are the ratio
of medians with the two relative half-widths combined in quadrature. ms per step = window time / (slot-steps / B),
i.e. the time of one batch step. W above idle = window power - the job's idle floor (median of the 5 start idle
windows, the value the record's E_dyn_J was computed with).

    python e1_dither_readout.py --root results/energy/babel_l40s > results/energy/babel_l40s/e1_dither_tables.md
"""
import argparse
import json
import pathlib
import statistics

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447}
TASKS = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze"]
NICE = {"sudoku_mlp": "TRM Sudoku-MLP", "sudoku_attn": "TRM Sudoku-attn", "hrm_sudoku": "HRM Sudoku",
        "maze": "TRM Maze-Hard"}
ARMS = ["fp32", "rtn", "dither"]
# A100 breadth-harness re-draw shares the predictions were written from (HANDOVER_SEP26 "E1 dither row")
A100_SHARE = {"sudoku_mlp": (0.066, 0.002), "sudoku_attn": (0.066, 0.002), "hrm_sudoku": (0.16, 0.009),
              "maze": (0.066, 0.002)}
# predicted dither / rtn J per correct (sbatch header): (b1 low, b1 high, b256)
PRED = {"sudoku_mlp": (0.97, 1.04, 0.91), "sudoku_attn": (0.93, 0.93, 0.89), "hrm_sudoku": (1.1, 1.4, 0.95),
        "maze": (1.07, 1.07, 1.00)}


def ci95(v):
    if len(v) < 2:
        return None
    m = statistics.mean(v)
    h = T95.get(len(v), 2.0) * statistics.stdev(v) / len(v) ** 0.5
    return [m - h, m + h]


def cv(v):
    return statistics.stdev(v) / statistics.mean(v) if len(v) > 1 else 0.0


def med(v):
    return statistics.median(v) if v else None


def stats(c, win_s):
    """Per-configuration statistics from the clean, un-stalled windows."""
    B = c["B"]
    ws = [w for w in c["windows"] if w["clean"] and w["T_s"] < 2 * win_s]
    stalled = [w for w in c["windows"] if w["T_s"] >= 2 * win_s]
    s = c["summary"]
    acc = s["acc"]
    idle = c["idle_P_W"]
    mj = [1e3 * w["E_J"] / w["slot_steps"] for w in ws]
    mjd = [1e3 * w["E_dyn_J"] / w["slot_steps"] for w in ws]
    jp = [w["E_J"] / w["done"] for w in ws]
    jpd = [w["E_dyn_J"] / w["done"] for w in ws]
    ms = [1e3 * w["T_s"] * B / w["slot_steps"] for w in ws]
    P = [w["P_W"] for w in ws]
    lat = [w["lat_ms_mean"] for w in ws]
    out = dict(B=B, mode=c["mode"], arm=c.get("arm", c.get("tool")), acc=acc, steps=s["mean_steps"], n_win=len(ws), n_stalled=len(stalled),
               ms_step=med(ms), W=med(P), W_dyn=med(P) - idle, idle=idle, mJ=med(mj), mJ_dyn=med(mjd), cv=cv(mj),
               J=med(jp) / acc, J_ci=[v / acc for v in ci95(jp)] if len(jp) > 1 else None,
               J_dyn=med(jpd) / acc, J_dyn_ci=[v / acc for v in ci95(jpd)] if len(jpd) > 1 else None,
               lat=med(lat), MHz=med([w.get("sm_MHz_mean", 0) for w in ws]),
               temp=med([w.get("temp_C_end", 0) for w in ws]), share=c.get("redraw_timing"),
               n_redraws=c.get("n_redraws"), nparam=c.get("params_rerounded"))
    if "polish_mean_cheap_steps" in s:
        out["polish"] = (s["polish_mean_cheap_steps"], s["polish_mean_polish_steps"])
    return out


def ratio(a, b, key="J", cikey="J_ci"):
    """Ratio of medians and an interval: relative half-widths of the two t-intervals combined in quadrature."""
    r = a[key] / b[key]
    ha = (a[cikey][1] - a[cikey][0]) / 2 / a[key] if a.get(cikey) else 0.0
    hb = (b[cikey][1] - b[cikey][0]) / 2 / b[key] if b.get(cikey) else 0.0
    h = r * (ha ** 2 + hb ** 2) ** 0.5
    return r, [r - h, r + h]


def f(v, d=3):
    return "-" if v is None else f"{v:.{d}f}"


def fr(rc, d=3):
    r, c = rc
    return f"{r:.{d}f} [{c[0]:.{d}f}, {c[1]:.{d}f}]"


def fci(v, d=2):
    return "" if not v else f" [{v[0]:.{d}f}, {v[1]:.{d}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = pathlib.Path(a.root)
    print("# E1 dither row (Babel L40S, one exclusive card per model; DATACENTER-GPU energy, not edge)\n")
    print("Generated by `EdgeTRM/e1_dither_readout.py` from `<task>/energy_dither.json` (e1_dither.py) and "
          "`<task>/grid.json` (e1_grid.py). Every number is recomputed from the clean, un-stalled windows as the "
          "docstring states.\n")
    verdict = {}
    for task in TASKS:
        pd = root / task / "energy_dither.json"
        pg = root / task / "grid.json"
        if not pd.exists():
            print(f"## {NICE[task]}: no dither record\n")
            continue
        d = json.loads(pd.read_text())
        g = json.loads(pg.read_text()) if pg.exists() else None
        m = d["meta"]
        idle_start = [w["P_W"] for w in d["idle"] if w["clean"] and w["label"] == "start"]
        idle_end = [w["P_W"] for w in d["idle"] if w["clean"] and w["label"] == "end"]
        print(f"## {NICE[task]} (H={m['H']}, rule {m['rule']}, n={m['n']}, cell {m['spec']} from master {m['master']}, "
              f"compute {m['compute']}, draw seed {m['seed']}; {m['gpu']} on {m['host']}, job {m['slurm_job']}, "
              f"driver {m['driver']}, torch {m['torch']})")
        print(f"Idle floor: start windows median {statistics.median(idle_start):.1f} W ({len(idle_start)}), end windows "
              f"{' / '.join(f'{v:.1f}' for v in idle_end)} W; the record subtracts {d['configs']['fp32_b1_fixed']['idle_P_W']:.1f} W.")
        S = {name: stats(c, m["win_s"]) for name, c in d["configs"].items() if not c.get("skipped")}
        skipped = [name for name, c in d["configs"].items() if c.get("skipped")]
        if skipped:
            print("Skipped configurations: " + ", ".join(f"{n} ({d['configs'][n]['reason'][:80]})" for n in skipped))
        stalled = [(n, s["n_stalled"]) for n, s in S.items() if s["n_stalled"]]
        if stalled:
            print("Stalled windows dropped: " + ", ".join(f"{n} ({k})" for n, k in stalled) + ".")
        print()
        # ---- 1. reproduction check
        print("### 1. Check: in-job fp32 rows vs the grid's eager fp32 rows (`grid.json`, another day and card)\n")
        if g is None:
            print("no grid record\n")
        else:
            gm = g["meta"]
            print(f"Grid: {gm['gpu']} on {gm['host']}, job {gm['slurm_job']}, idle {g['configs']['eager_fp32_b1_fixed']['idle_P_W']:.1f} W.\n")
            print("| batch, policy | source | mJ/step | CV | J/correct | acc % | steps | ms/step | W | T end C | in-job / grid mJ/step | J/correct ratio | verdict |")
            print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            vt = {}
            for B in m["batches"]:
                for mode in ("fixed", "exit"):
                    jn, gn = f"fp32_b{B}_{mode}", f"eager_fp32_b{B}_{mode}"
                    if jn not in S or gn not in g["configs"] or g["configs"][gn].get("skipped"):
                        print(f"| b{B} {mode} | missing |||||||||||")
                        continue
                    sj = S[jn]
                    sg = stats(g["configs"][gn], gm["win_s"])
                    dev = sj["mJ"] / sg["mJ"] - 1
                    tol = 2 * (sg["cv"] ** 2 + sj["cv"] ** 2) ** 0.5
                    ok = abs(dev) <= tol
                    ok2 = abs(dev) <= max(tol, 0.02)
                    vt[(B, mode)] = "PASS" if ok else ("PASS (2% floor)" if ok2 else "FAIL")
                    for lab, s in (("grid", sg), ("in-job", sj)):
                        print(f"| b{B} {mode} | {lab} | {s['mJ']:.1f} | {100*s['cv']:.1f}% | {s['J']:.3f}{fci(s['J_ci'])} | "
                              f"{100*s['acc']:.1f} | {s['steps']:.2f} | {s['ms_step']:.2f} | {s['W']:.1f} | {s['temp']:.0f} | "
                              + (f"{1+dev:.3f} (tol {100*tol:.1f}%) | {sj['J']/sg['J']:.3f} | {vt[(B, mode)]} |" if lab == "in-job" else "| | |"))
            verdict[task] = vt
            print()
        # ---- 2. arm x batch x policy
        print("### 2. Arms x batch x policy (in-job; medians over the windows, 95% t-intervals)\n")
        print("| config | acc % | mean steps | ms/step | W | W above idle | mJ/step | mJ/step dyn | J/correct [95%] | J/correct dyn [95%] | dither/rtn J/correct | dither/fp32 J/correct | SM MHz |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for B in m["batches"]:
            for mode in ("fixed", "exit", "polish"):
                for arm in ARMS:
                    n = f"{arm}_b{B}_{mode}"
                    if n not in S:
                        continue
                    s = S[n]
                    steps = f"{s['steps']:.2f}"
                    if "polish" in s:
                        steps += f" ({s['polish'][0]:.2f} cheap + {s['polish'][1]:.2f} w8c)"
                    rr = rf = ""
                    if arm == "dither":
                        if f"rtn_b{B}_{mode}" in S:
                            rr = fr(ratio(s, S[f"rtn_b{B}_{mode}"]))
                        if f"fp32_b{B}_{mode}" in S:
                            rf = fr(ratio(s, S[f"fp32_b{B}_{mode}"]))
                        elif mode == "polish" and f"fp32_b{B}_exit" in S:
                            rf = fr(ratio(s, S[f"fp32_b{B}_exit"])) + " (vs fp32 exit)"
                    print(f"| {n} | {100*s['acc']:.1f} | {steps} | {s['ms_step']:.2f} | {s['W']:.1f} | {s['W_dyn']:.1f} | {s['mJ']:.1f} | "
                          f"{s['mJ_dyn']:.1f} | {s['J']:.3f}{fci(s['J_ci'])} | {s['J_dyn']:.3f}{fci(s['J_dyn_ci'])} | {rr} | {rf} | {s['MHz']:.0f} |")
        print()
        # dyn ratios and the exit ranking
        print("Idle-subtracted ratios (dither / rtn, dither / fp32 on J/correct dyn): " + "; ".join(
            f"b{B} {mode}: {fr(ratio(S[f'dither_b{B}_{mode}'], S[f'rtn_b{B}_{mode}'], 'J_dyn', 'J_dyn_ci'))} / "
            f"{fr(ratio(S[f'dither_b{B}_{mode}'], S[f'fp32_b{B}_{mode}'], 'J_dyn', 'J_dyn_ci'))}"
            for B in m["batches"] for mode in ("fixed", "exit")
            if all(f"{x}_b{B}_{mode}" in S for x in ARMS)) + ".")
        for B in m["batches"]:
            rows = [(S[n]["J"], n) for n in S if S[n]["B"] == B and S[n]["mode"] in ("exit", "polish")]
            rows.sort()
            print(f"Cheapest per correct at batch {B} among exit/polish rows: " + " < ".join(f"{n} {v:.3f}" for v, n in rows) + ".")
        print()
        # ---- 3. re-draw share
        print("### 3. Re-draw share (redraw_timing: 50 re-draws alone vs 50 plain steps, same batch; not part of the windows)\n")
        print("| config | re-draw ms | plain step ms | share L40S | share A100 (breadth harness) | params re-rounded | re-draws in the run | step-time ratio dither/rtn (windows) | mJ/step ratio dither/rtn |")
        print("|---|---|---|---|---|---|---|---|---|")
        for n, s in S.items():
            if s["arm"] != "dither" or not s["share"]:
                continue
            sh = s["share"]
            a100 = A100_SHARE[task][0 if s["B"] == 1 else 1]
            rt = S.get(f"rtn_b{s['B']}_{s['mode']}")
            print(f"| {n} | {sh['redraw_ms']:.3f} | {sh['step_ms_noredraw']:.3f} | {sh['redraw_share']:.4f} | {a100} | {s['nparam']:,} | {s['n_redraws']:,} | "
                  f"{f(s['ms_step']/rt['ms_step'] if rt else None)} | {f(s['mJ']/rt['mJ'] if rt else None)} |")
        print()
        # ---- 4. predictions
        lo, hi, p256 = PRED[task]
        print("### 4. Prediction vs outcome (dither / rtn J per correct, total energy)\n")
        for B, pred in ((1, f"{lo}-{hi}" if lo != hi else f"~{lo}"), (256, f"~{p256}")):
            for mode in ("fixed", "exit"):
                if all(f"{x}_b{B}_{mode}" in S for x in ("dither", "rtn")):
                    r, c = ratio(S[f"dither_b{B}_{mode}"], S[f"rtn_b{B}_{mode}"])
                    da, ra = S[f"dither_b{B}_{mode}"]["acc"], S[f"rtn_b{B}_{mode}"]["acc"]
                    print(f"- b{B} {mode}: predicted {pred}, measured {r:.3f} [{c[0]:.3f}, {c[1]:.3f}] "
                          f"(acc rtn {100*ra:.1f} -> dither {100*da:.1f}, rel {100*(da/ra-1):+.1f}%; "
                          f"mJ/step x{S[f'dither_b{B}_{mode}']['mJ']/S[f'rtn_b{B}_{mode}']['mJ']:.3f}; "
                          f"steps {S[f'rtn_b{B}_{mode}']['steps']:.2f} -> {S[f'dither_b{B}_{mode}']['steps']:.2f}).")
        if "dither_b1_polish" in S and "dither_b1_exit" in S:
            p, e = S["dither_b1_polish"], S["dither_b1_exit"]
            print(f"- dither_b1_polish vs dither_b1_exit: acc {100*e['acc']:.1f} -> {100*p['acc']:.1f} "
                  f"({100*(p['acc']-e['acc']):+.1f} pt), J/correct {e['J']:.3f} -> {p['J']:.3f} (x{p['J']/e['J']:.3f}); "
                  f"steps {p['polish'][0]:.2f} cheap + {p['polish'][1]:.2f} w8c.")
        if "rtn_b1_polish" in S and "rtn_b1_exit" in S:
            p, e = S["rtn_b1_polish"], S["rtn_b1_exit"]
            print(f"- rtn_b1_polish vs rtn_b1_exit: acc {100*e['acc']:.1f} -> {100*p['acc']:.1f} "
                  f"({100*(p['acc']-e['acc']):+.1f} pt), J/correct {e['J']:.3f} -> {p['J']:.3f} (x{p['J']/e['J']:.3f}).")
        print()
    print("## Check verdicts\n")
    for task, vt in verdict.items():
        print(f"- {NICE[task]}: " + ", ".join(f"b{B} {mode} {v}" for (B, mode), v in vt.items()))


if __name__ == "__main__":
    main()
