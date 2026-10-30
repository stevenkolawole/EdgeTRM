#!/usr/bin/env python3
"""Loop-cap sweep on the E1 eager energy harness (reports/edgetrm_exp/LOOP_CAP_SWEEP.md): does one looped model trace
the accuracy vs joules-per-correct-answer trade-off by its loop cap alone?

Harness, weights, rules and protocol are E1_DITHER's (e1_dither.py): eager PyTorch, fp32 compute, TF32 off; the cell's
low-bit weights by RTN from the w8c master (w4c Sudoku-MLP and Maze, w3c Sudoku-attn and HRM); NVML total-energy
windows (5 idle windows x 30 s at the start, 2 at the end; 10 s warm-up; 5 windows of >= 60 s per configuration;
clean-card check per window; idle floor = median of the start windows); accuracy from the first completion of every
puzzle (rows 0..n-1, as E1). The loop cap replaces E1's 16 (e1_energy.NSUP, read at run time by e1_grid.Stream):
  fixed    rtn weights, exactly `cap` steps
  exit     rtn weights, the M5 halting rule, at most `cap` steps
  polish   rtn weights under the exit rule (<= cap), then <= 4 steps with the w8c master until its head fires
           (batch 1 only, as in E1)
  dither   per-step re-draw of the cell from the w8c master: on top of polish at batch 1 (dither_b1_polish), on top
           of exit at batch 256 (E1 has no finishing step at batch 256)
Exit at a cap where the rule cannot fire before the cap (cap 1 for every rule; cap 2 for patience2 and min2native) is
the same computation as fixed at that cap and is not run (recorded as same_as).

    python loop_cap_sweep.py --task sudoku_mlp --out $HOME/capsweep/sudoku_mlp.json            (one card, one job)
    python loop_cap_sweep.py --readout results/energy/babel_l40s_capsweep > .../capsweep_tables.md
"""
import argparse
import json
import os
import pathlib
import statistics
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

CAPS = (1, 2, 4, 8, 16, 32)
B256_CAPS = (4, 8, 16)          # batch 256 at caps 1, 2 and 32 dropped to fit one 3 h job (LOOP_CAP_SWEEP.md s1)


def configs(task, caps_b1, caps_b256):
    import e1_energy as E
    rule = E.TASKS[task]["rule"]
    forced = lambda c: c == 1 or (c == 2 and rule in ("patience2", "min2native"))   # noqa: E731
    out = []
    for B, caps, pols in ((1, caps_b1, (("rtn", "fixed"), ("rtn", "exit"), ("rtn", "polish"), ("dither", "polish"))),
                          (256, caps_b256, (("rtn", "fixed"), ("rtn", "exit"), ("dither", "exit")))):
        for cap in caps:
            for arm, mode in pols:
                name = f"{arm}_b{B}_{mode}_c{cap}"
                same = f"rtn_b{B}_fixed_c{cap}" if (arm, mode) == ("rtn", "exit") and forced(cap) else None
                out.append((name, arm, B, mode, cap, same))
    # anchors first: the E1 fixed-16 cells of this arm (rtn_b{1,256}_fixed in energy_dither.json)
    anchors = [c for c in out if c[0] in ("rtn_b1_fixed_c16", "rtn_b256_fixed_c16")]
    return anchors + [c for c in out if c not in anchors]


def sweep(a):
    import torch
    import pynvml as nv
    import e1_energy as E
    import e1_grid as G
    import e1_dither as D
    assert torch.cuda.is_available(), "CUDA not available: abort"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    task = a.task
    t = E.TASKS[task]
    H = t["H"]
    spec = D.CELLS[task]
    nv.nvmlInit()
    uuid_t = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
    h = next(nv.nvmlDeviceGetHandleByIndex(i) for i in range(nv.nvmlDeviceGetCount())
             if (lambda u: u.decode() if isinstance(u, bytes) else u)(nv.nvmlDeviceGetUUID(nv.nvmlDeviceGetHandleByIndex(i))) == uuid_t)
    samp = E.Sampler(nv, h)
    samp.start()
    outp = pathlib.Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    rec = json.loads(outp.read_text()) if outp.exists() else {"configs": {}, "idle": []}
    cfgs = configs(task, [int(c) for c in a.caps_b1.split(",")], [int(c) for c in a.caps_b256.split(",") if c])
    rec["meta"] = {"gpu": torch.cuda.get_device_name(0), "uuid": uuid_t, "host": __import__("socket").gethostname(),
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "driver": nv.nvmlSystemGetDriverVersion(),
                   "torch": torch.__version__, "H": H, "rule": t["rule"], "n": t["n"], "win_s": a.win_s, "nwin": a.nwin,
                   "polish_cap": D.POLISH_CAP, "spec": spec, "master": "w8c", "compute": "fp32", "seed": a.seed,
                   "caps_b1": a.caps_b1, "caps_b256": a.caps_b256, "planned": [c[0] for c in cfgs],
                   "harness": "e1_dither.DitherRunner / DitherStream, e1_grid windows; loop cap = e1_energy.NSUP",
                   "platform": "Babel L40S, one card for the whole sweep (datacenter GPU, not edge); clocks not pinned"}

    def clean():
        return not [p.pid for p in nv.nvmlDeviceGetComputeRunningProcesses(h) if p.pid != os.getpid()]

    def idle(k, label):
        for _ in range(k):
            torch.cuda.synchronize()
            samp.take()
            samp.on.set()
            e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            time.sleep(30)
            e1, t1 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            samp.on.clear()
            w = {"label": label, "P_W": (e1 - e0) / 1e3 / (t1 - t0), "clean": clean(),
                 "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **samp.take()}
            rec["idle"].append(w)
            log("idle", label, round(w["P_W"], 2), "W", w.get("sm_MHz_mean"))
        E.dump(outp, rec)

    log = E.log
    if not [w for w in rec["idle"] if w["label"] == "start"]:
        idle(5, "start")
    idle_P = statistics.median([w["P_W"] for w in rec["idle"] if w["clean"] and w["label"] == "start"])
    x, lab = E.load_rows(t["data"], 0, t["n"])
    xg = torch.as_tensor(x, device="cuda")
    labg = torch.as_tensor(lab, device="cuda")
    for name, arm, B, mode, cap, same in cfgs:
        if name in rec["configs"] and (rec["configs"][name].get("n_clean", 0) >= a.nwin or rec["configs"][name].get("skipped")
                                       or rec["configs"][name].get("same_as")):
            continue
        if same:
            rec["configs"][name] = {"arm": arm, "B": B, "mode": mode, "cap": cap, "same_as": same,
                                    "reason": f"rule {t['rule']} cannot fire before step {cap}: identical to fixed"}
            E.dump(outp, rec)
            continue
        E.NSUP = cap                                                 # the loop cap for fixed / exit / cheap phase
        stream = torch.cuda.Stream()
        torch.cuda.set_stream(stream)
        try:
            runner = D.DitherRunner(task, H, B, "fp32", arm, spec, "w8c", a.seed)
            runner.x = runner.x.clone()
            S = D.DitherStream(task, runner, G.reset_eager, B, mode, xg, labg, None)
            S.polish_steps = []
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < 10:
                S.unit()
            torch.cuda.synchronize()
            share = D.time_redraw(runner) if arm == "dither" else None
        except Exception as ex:                                      # OOM etc.: recorded, next configuration
            rec["configs"][name] = {"arm": arm, "B": B, "mode": mode, "cap": cap, "skipped": True, "reason": repr(ex)[:300]}
            E.dump(outp, rec)
            log(name, "SKIPPED", repr(ex)[:200])
            torch.cuda.empty_cache()
            continue
        wins = []
        for k in range(a.nwin):
            ok0 = clean()
            samp.take()
            samp.on.set()
            steps = done = corr = 0
            lat = 0.0
            torch.cuda.synchronize()
            e0, t0 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            while time.perf_counter() - t0 < a.win_s:
                s_, d_, c_, l_ = S.unit()
                steps += s_
                done += d_
                corr += c_
                lat += l_
            torch.cuda.synchronize()
            e1, t1 = nv.nvmlDeviceGetTotalEnergyConsumption(h), time.perf_counter()
            samp.on.clear()
            E_, T_ = (e1 - e0) / 1e3, t1 - t0
            w = {"E_J": E_, "T_s": T_, "P_W": E_ / T_, "E_dyn_J": E_ - idle_P * T_, "slot_steps": steps, "done": done,
                 "correct": corr, "J_per_puzzle": E_ / max(done, 1), "J_per_puzzle_dyn": (E_ - idle_P * T_) / max(done, 1),
                 "puzzles_per_s": done / T_, "lat_ms_mean": 1e3 * lat / max(done, 1),
                 "mJ_per_slot_step": 1e3 * E_ / max(steps, 1), "window_acc": corr / max(done, 1),
                 "clean": bool(ok0 and clean()), "loadavg": os.getloadavg(), **samp.take()}
            wins.append(w)
            log(task, name, f"w{k}", f"{w['P_W']:.1f} W", f"{w['J_per_puzzle']:.4f} J/puz", f"{w['lat_ms_mean']:.2f} ms",
                f"acc {w['window_acc']:.3f}", f"{w.get('sm_MHz_mean')} MHz", "clean" if w["clean"] else "DIRTY")
        tc = time.perf_counter()
        guard = 0
        while len(S.first) < S.n and guard < 10 ** 7:
            S.unit()
            guard += 1
        tc = time.perf_counter() - tc
        acc = float(np.mean([v[1] for v in S.first.values()]))
        msteps = float(np.mean([v[0] for v in S.first.values()]))
        good = [w for w in wins if w["clean"]]
        summ = {"acc": acc, "mean_steps": msteps, "n_clean": len(good),
                "J_solved_median": statistics.median([w["J_per_puzzle"] for w in good]) / acc if good and acc else None,
                "J_solved_dyn_median": statistics.median([w["J_per_puzzle_dyn"] for w in good]) / acc if good and acc else None,
                "lat_ms_median": statistics.median([w["lat_ms_mean"] for w in good]) if good else None}
        if mode == "polish" and S.polish_steps:
            ps = np.array(S.polish_steps)
            summ["polish_mean_cheap_steps"], summ["polish_mean_polish_steps"] = float(ps[:, 0].mean()), float(ps[:, 1].mean())
        rec["configs"][name] = {"arm": arm, "B": B, "mode": mode, "cap": cap, "spec": spec, "master": "w8c",
                                "compute": "fp32", "idle_P_W": idle_P, "redraw_timing": share,
                                "params_rerounded": runner.nparam, "n_redraws": runner.n_redraw, "windows": wins,
                                "summary": summ, "completion_untimed_s": round(tc, 1),
                                "first_pass_correct": "".join("1" if S.first[p][1] else "0" for p in sorted(S.first)),
                                "first_pass_steps": [S.first[p][0] for p in sorted(S.first)]}
        E.dump(outp, rec)
        log(task, name, "acc", round(acc, 4), "steps", round(msteps, 2), "J/solved", summ["J_solved_median"],
            "dyn", summ["J_solved_dyn_median"], "lat ms", summ["lat_ms_median"])
        del S, runner
        torch.cuda.empty_cache()
    idle(2, "end")
    samp.stop_.set()


# ----------------------------------------------------------------------------- read-out
POLICY = {("rtn", "fixed"): "fixed", ("rtn", "exit"): "exit", ("rtn", "polish"): "exit+finish",
          ("dither", "polish"): "exit+finish+redraw", ("dither", "exit"): "exit+redraw"}


def pareto(pts, key):
    """Points not dominated on (acc higher, J lower)."""
    return {p["name"] for p in pts if not any(q["acc"] >= p["acc"] and q[key] <= p[key] and (q["acc"] > p["acc"] or q[key] < p[key])
                                               for q in pts)}


def readout(root, e1root):
    import e1_dither_readout as R
    for path in sorted(pathlib.Path(root).glob("*.json")):
        rec = json.loads(path.read_text())
        m = rec["meta"]
        task = path.stem
        print(f"\n## {task} ({m['spec']} from w8c; {m['host']}, job {m['slurm_job']}, {m['uuid'][:12]}; "
              f"idle {statistics.median([w['P_W'] for w in rec['idle'] if w['label'] == 'start' and w['clean']]):.1f} W)\n")
        st = {}
        for name, c in rec["configs"].items():
            if c.get("windows"):
                s = R.stats(c, m["win_s"])
                lt = [w["lat_ms_mean"] for w in c["windows"] if w["clean"] and w["T_s"] < 2 * m["win_s"]]
                s.update(name=name, cap=c["cap"], policy=POLICY[(c["arm"], c["mode"])], lat_ci=R.ci95(lt))
                st[name] = s
        # anchor: the E1 fixed-16 cell of this arm, same harness, earlier day / card
        e1p = pathlib.Path(e1root) / task / "energy_dither.json"
        if e1p.exists():
            e1 = json.loads(e1p.read_text())
            print("Anchor (in-job cap 16 vs E1_DITHER record " + f"{e1['meta']['host']} job {e1['meta']['slurm_job']}):\n")
            print("| cell | E1 mJ/step | in-job mJ/step | ratio | E1 J/correct | in-job J/correct | E1 acc | in-job acc |")
            print("|---|---|---|---|---|---|---|---|")
            for B in (1, 256):
                k, k1 = f"rtn_b{B}_fixed_c16", f"rtn_b{B}_fixed"
                if k in st and k1 in e1["configs"]:
                    a1 = R.stats(e1["configs"][k1], e1["meta"]["win_s"])
                    b = st[k]
                    print(f"| {k1} | {a1['mJ']:.1f} | {b['mJ']:.1f} | {b['mJ'] / a1['mJ']:.3f} | {a1['J']:.3f} | {b['J']:.3f} | "
                          f"{100 * a1['acc']:.1f} | {100 * b['acc']:.1f} |")
        for B in (1, 256):
            pts = [s for s in st.values() if s["B"] == B]
            if not pts:
                continue
            pt, pd = pareto(pts, "J"), pareto(pts, "J_dyn")
            print(f"\n### batch {B}\n")
            print("| policy | cap | acc % | mean steps | J/correct [95%] | J/correct dyn [95%] | lat ms [95%] | vs cap 16 J/correct | Pareto (total / dyn) |")
            print("|---|---|---|---|---|---|---|---|---|")
            for s in sorted(pts, key=lambda s: (s["policy"], s["cap"])):
                ref = next((q for q in pts if q["policy"] == s["policy"] and q["cap"] == 16), None)
                rr = R.fr(R.ratio(s, ref)) if ref and ref is not s else ""
                pol = f" ({s['polish'][0]:.2f} cheap + {s['polish'][1]:.2f} w8c)" if s.get("polish") else ""
                print(f"| {s['policy']} | {s['cap']} | {100 * s['acc']:.1f} | {s['steps']:.2f}{pol} | {s['J']:.3f}{R.fci(s['J_ci'], 3)} | "
                      f"{s['J_dyn']:.3f}{R.fci(s['J_dyn_ci'], 3)} | {s['lat']:.1f}{R.fci(s['lat_ci'], 1)} | {rr} | "
                      f"{'P' if s['name'] in pt else '-'} / {'P' if s['name'] in pd else '-'} |")
            same = [f"{n} = {c['same_as']}" for n, c in rec["configs"].items() if c.get("same_as") and c["B"] == B]
            if same:
                print("\nNot run (identical computation): " + "; ".join(same))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--caps_b1", default=",".join(map(str, CAPS)))
    ap.add_argument("--caps_b256", default=",".join(map(str, B256_CAPS)))
    ap.add_argument("--seed", type=int, default=17, help="draw seed, as e1_dither.py")
    ap.add_argument("--win_s", type=float, default=60.0)
    ap.add_argument("--nwin", type=int, default=5)
    ap.add_argument("--readout", default=None, help="dir of <task>.json records")
    ap.add_argument("--e1root", default=str(HERE / "results" / "energy" / "babel_l40s"))
    a = ap.parse_args()
    if a.readout:
        readout(a.readout, a.e1root)
    else:
        sweep(a)


if __name__ == "__main__":
    main()
