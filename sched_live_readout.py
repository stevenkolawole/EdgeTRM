#!/usr/bin/env python3
"""E9C read-out: tables and the prediction scorecard from sched_live.json records (one per task).

    python sched_live_readout.py                                   # results/energy/babel_l40s/<task>/sched_live.json
    python sched_live_readout.py --dir /path/with/<task>/sched_live.json --tasks sudoku_mlp,maze
    python sched_live_readout.py --file sched_live_smoke.json --dir ~/e9/smoke     # a smoke record (NOT results)

Per task and deadline: hit fraction (% of finalised puzzles answered correctly by their deadline; median over clean
windows with the 95 % t-interval), latency of the hits (mean / p95, in loops of time = s / tau, and ms), the hits' own
loop count, throughput (finalised per s), J per hit and idle-subtracted J per hit. Scorecard against the predictions in
sched_live.py / E9C_LIVE.md, with the E9 simulation (results/breadth/sched_sim.json) as the reference for P2.
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
TASKS = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze"]
POL = ["fifo", "rr", "head", "ctl", "ctl_dither"]
SIM_POL = {"fifo": "fifo16", "rr": "rr", "head": "head", "ctl": "ep"}
# closest simulated cell per live cell (the live cell is the E1 dither cell: w4c / w3c / w3c / w4c)
SIM_CELL = {"sudoku_mlp": "sudoku_mlp|w4c", "sudoku_attn": "sudoku_attn|w3c", "hrm_sudoku": "hrm_sudoku|w4c",
            "maze": "maze_attn|w4t"}


def f(v, d=1, pct=False):
    if v is None:
        return "-"
    x = v["median"] if isinstance(v, dict) else v
    if x is None:
        return "-"
    return f"{100 * x:.{d}f}" if pct else f"{x:.{d}f}"


def ci(v, pct=False):
    if not v or not v.get("ci95"):
        return ""
    lo, hi = v["ci95"]
    k = 100 if pct else 1
    return f" [{k * lo:.1f}, {k * hi:.1f}]"


def load_sim(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE / "results" / "energy" / "babel_l40s"))
    ap.add_argument("--file", default="sched_live.json")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--sim", default=str(HERE / "results" / "breadth" / "sched_sim.json"))
    a = ap.parse_args()
    sim = load_sim(a.sim)
    score = {"P1": [], "P2": [], "P3": [], "P4": []}
    for task in a.tasks.split(","):
        p = pathlib.Path(a.dir).expanduser() / task / a.file
        if not p.exists():
            print(f"\n## {task}: no record at {p}")
            continue
        r = json.loads(p.read_text())
        m = r["meta"]
        cf = r["configs"]
        print(f"\n## {task} ({m.get('gpu')}, job {m.get('slurm_job')}, {m.get('spec')} cheap / {m.get('master')} master, "
              f"cap {m.get('cap')}, tau {1e3 * m.get('tau_s', 0):.2f} ms at full batch (b1 {1e3 * m.get('tau', {}).get('tick_s_b1', 0):.2f} ms, "
              f"re-draw {1e3 * m.get('tau', {}).get('redraw_s', 0):.2f} ms), loop rate {m.get('loop_rate_per_s', 0):.0f}/s, "
              f"windows {m.get('nwin')} x {m.get('win_s')} s{'  SMOKE (not a result)' if m.get('smoke') else ''})")
        idle = [w["P_W"] for w in r.get("idle", []) if w.get("clean")]
        if idle:
            print(f"idle floor: median {sorted(idle)[len(idle) // 2]:.1f} W over {len(idle)} clean windows")
        loads, deltas = m.get("loads", []), m.get("deltas", [])
        pols = [q for q in POL if any(k.startswith(q + "_l") for k in cf)]
        for delta in deltas:
            print(f"\n### {task}, deadline {delta} loops: hit % [95% t-interval over windows] per policy vs load")
            print("| load | " + " | ".join(pols) + " |")
            print("|---|" + "---|" * len(pols))
            for load in loads:
                row = []
                for q in pols:
                    c = cf.get(f"{q}_l{load:g}_d{delta}")
                    s = c["summary"] if c else None
                    row.append(f"{f(s['hit_frac'], 1, True)}{ci(s['hit_frac'], True)}" if s else "-")
                print(f"| {load:g} | " + " | ".join(row) + " |")
            print(f"\n### {task}, deadline {delta}: latency of hits mean / p95 (loops of time; ms), own loops per hit, "
                  f"finalised per s, J per hit, idle-subtracted J per hit (medians over clean windows)")
            print("| load | policy | hit % | lat loops | lat ms | loops/hit | final/s | J/hit | J/hit dyn | batch | P W |")
            print("|---|---|---|---|---|---|---|---|---|---|---|")
            for load in loads:
                for q in pols:
                    c = cf.get(f"{q}_l{load:g}_d{delta}")
                    if not c:
                        continue
                    s = c["summary"]
                    print(f"| {load:g} | {q} | {f(s['hit_frac'], 1, True)} | {f(s['lat_hit_mean_loops'])} / {f(s['lat_hit_p95_loops'])} | "
                          f"{f(s['lat_hit_mean_ms'])} / {f(s['lat_hit_p95_ms'])} | {f(s['loops_per_hit'])} | {f(s['final_per_s'])} | "
                          f"{f(s['J_per_hit'], 3)}{ci(s['J_per_hit'])} | {f(s['J_per_hit_dyn'], 3)} | {f(s['batch_mean'])} | {f(s['P_W'], 0)} |")
            # scorecard
            for load in loads:
                hv = {q: cf[f"{q}_l{load:g}_d{delta}"]["summary"]["hit_frac"]["median"] for q in pols
                      if f"{q}_l{load:g}_d{delta}" in cf and cf[f"{q}_l{load:g}_d{delta}"]["summary"]["hit_frac"]}
                jv = {q: cf[f"{q}_l{load:g}_d{delta}"]["summary"]["J_per_hit"]["median"] for q in pols
                      if f"{q}_l{load:g}_d{delta}" in cf and cf[f"{q}_l{load:g}_d{delta}"]["summary"]["J_per_hit"]}
                chain = [q for q in ("fifo", "rr", "head", "ctl") if q in hv]
                if len(chain) >= 2:
                    ok = all(hv[chain[i]] <= hv[chain[i + 1]] + 1e-9 for i in range(len(chain) - 1))
                    score["P1"].append((task, delta, load, ok, " < ".join(f"{q} {100 * hv[q]:.1f}" for q in chain)))
                    okj = all(jv[chain[i]] >= jv[chain[i + 1]] - 1e-9 for i in range(len(chain) - 1) if chain[i] in jv and chain[i + 1] in jv)
                    score["P3"].append((task, delta, load, okj, " > ".join(f"{q} {jv[q]:.2f}" for q in chain if q in jv)))
                if "ctl" in hv and "ctl_dither" in hv:
                    s_c = cf[f"ctl_l{load:g}_d{delta}"]["summary"]["hit_frac"]
                    s_d = cf[f"ctl_dither_l{load:g}_d{delta}"]["summary"]["hit_frac"]
                    score["P4"].append((task, delta, load, hv["ctl_dither"] - hv["ctl"], s_c.get("ci95"), s_d.get("ci95")))
                if load == 4.0 and delta == 16 and "ctl" in hv and "fifo" in hv and sim:
                    cell = sim["cells"].get(SIM_CELL.get(task, ""))
                    if cell:
                        sh = cell["deadline"]["16"]["4.0"]["hit"]
                        gs = (sh["ep"]["hit"] - sh["fifo16"]["hit"]) / 100
                        gl = hv["ctl"] - hv["fifo"]
                        score["P2"].append((task, SIM_CELL[task], gl, gs, gl / gs if gs else None))
    print("\n## Prediction scorecard (from the records above)")
    p1 = score["P1"]
    if p1:
        print(f"- P1 ordering fifo <= rr <= head <= ctl in hit fraction: holds in {sum(x[3] for x in p1)} of {len(p1)} (task, deadline, load) cells")
        for x in p1:
            if not x[3]:
                print(f"    violated: {x[0]} d{x[1]} load {x[2]:g}: {x[4]}")
    for x in score["P2"]:
        print(f"- P2 {x[0]} (sim cell {x[1]}): live gap ctl - fifo at load 4, deadline 16 = {100 * x[2]:.1f} points; simulated "
              f"ep - fifo16 = {100 * x[3]:.1f}; ratio live / sim = {x[4]:.2f} " + ("(within a third)" if x[4] is not None and 2 / 3 <= x[4] <= 4 / 3 else "(OUTSIDE a third)"))
    p3 = score["P3"]
    if p3:
        print(f"- P3 J per hit falls fifo -> rr -> head -> ctl: holds in {sum(x[3] for x in p3)} of {len(p3)} cells")
        for x in p3:
            if not x[3]:
                print(f"    violated: {x[0]} d{x[1]} load {x[2]:g}: {x[4]}")
    for x in score["P4"]:
        fmt = lambda c: "-" if not c else f"[{100 * c[0]:.1f}, {100 * c[1]:.1f}]"  # noqa: E731
        print(f"- P4 {x[0]} d{x[1]} load {x[2]:g}: ctl_dither - ctl = {100 * x[3]:+.1f} points (ctl {fmt(x[4])}, ctl_dither {fmt(x[5])})")


if __name__ == "__main__":
    main()
