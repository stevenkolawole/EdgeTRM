#!/usr/bin/env python3
"""Read-out of the controller-on-phone chains (E5, MLSys 2027 "the controller on the chip").

Inputs (pulled read-only from sk-p4d:~/aihub_artifacts/h1/<task>/infer/ into results/devices/ctl/<task>/infer/):
  ctl_<cheap>-<precise>_<device>_<acct>.{json,npz}   the six chains written by tools/aihub_ctl.py (pinned QAIRT 2.45)
  dev_<prec>_<device>[_<acct>].{json,npz}             breadth rows (fixed-step device chains, 8 cycles, same puzzles)
  host_<prec>.npz                                     aihub_infer.py --stage host loops (fp32 / QDQ sim, 16 or 8 steps)

Per (device, task) it prints, as markdown tables:
  1. accuracy: fixed 8 (breadth row, provenance stated), fixed 16 (host only; no device row runs 16 cycles), the
     chain's own all-puzzle accuracy at the last cycle before any puzzle can exit, exit only, exit + finishing j=1 / j=2
     (all puzzles and fired-only), with paired 95% puzzle-bootstrap intervals;
  2. device cost per puzzle from the profiled per-step latencies;
  3. the host controller: the same exit rule replayed on the host loops (finishing is not replayable: the host records
     hold no carry at the exit step);
  4. a compact json of every number at results/devices/ctl/e5_phone_ctl.json.
--slim writes ctl_*.readout.npz (every array except the zH / zL carry) next to each chain record so the per-puzzle
predictions can be committed (the full npz is 27-110 MB, over GitHub's limit for the maze chains).
"""
import argparse
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1] / "results" / "devices" / "ctl"
HCYC = {"sudoku_mlp": 3, "sudoku_attn": 3, "maze": 3}
RULES = {"sudoku_mlp": ("pat", 2), "sudoku_attn": ("native", 0), "maze": ("min", 2)}
CELLS = [("sudoku_mlp", "w8a8", "w8a16"), ("sudoku_attn", "w8a8", "w8a16"), ("maze", "w8a8rope16", "fp32")]
DEVICES = [("Samsung Galaxy S24", "acct2"), ("Samsung Galaxy S25", "acct3")]
SHORT = {"Samsung Galaxy S24": "S24", "Samsung Galaxy S25": "S25"}
NAME = {"sudoku_mlp": "Sudoku-MLP", "sudoku_attn": "Sudoku-attn", "maze": "Maze"}
B, SEED = 10000, 0


def tagof(device):
    return device.replace(" ", "_")


def fires(task, q, k, run):
    """copy of aihub_ctl.fires: head read after device step k (0-based)."""
    base = q > 0
    kind, p = RULES[task]
    sup = (k + 1) // HCYC[task]
    if kind == "native":
        return base, run
    if kind == "pat":
        run = np.where(base, run + 1, 0)
        return run >= p, run
    if kind == "min":
        return base & (sup >= p), run
    raise ValueError(kind)


def first_exit_cycle(task):
    """earliest device cycle at which the rule can fire (all puzzles are still running at every earlier cycle)."""
    kind, p = RULES[task]
    h = HCYC[task]
    return {"native": h, "pat": p * h, "min": p * h}[kind]


def exact(pred, lab):
    return (pred == lab).all(-1)


def boot(diff, rng):
    """95% percentile bootstrap interval of the mean of a per-puzzle difference vector (paired by puzzle)."""
    n = len(diff)
    idx = rng.integers(0, n, (B, n))
    m = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def fmt_ci(t):
    m, lo, hi = t
    return f"{100*m:+.1f} [{100*lo:+.1f}, {100*hi:+.1f}]"


def sim_exit(task, preds, q, cap):
    """replay the exit rule on a host loop record: preds [S, N, L], q [S, N]. Returns exit preds, steps, fired mask."""
    S, N, _ = preds.shape
    cap = min(cap, S)
    run = np.zeros(N, np.int32)
    done = np.zeros(N, bool)
    fired = np.zeros(N, bool)
    steps = np.full(N, cap, np.int32)
    out = preds[cap - 1].copy()
    for k in range(cap):
        if (k + 1) % HCYC[task] == 0:
            f, run_new = fires(task, q[k], k, run)
            run = np.where(done, run, run_new)
            f = f & ~done
            fired |= f
            steps[f] = k + 1
            out[f] = preds[k][f]
            done |= f
    return out, steps, fired


def breadth_row(task, prec, device, acct):
    """the breadth record for (task, prec, device): same account first, then any pinned (qairt) row, then any row."""
    d = ROOT / task / "infer"
    cands = [d / f"dev_{prec}_{tagof(device)}_{acct}.json", *sorted(d.glob(f"dev_{prec}_{tagof(device)}*.json"))]
    seen, rows = set(), []
    for p in cands:
        if p in seen or not p.exists() or not p.with_suffix(".npz").exists():
            continue
        seen.add(p)
        meta = json.loads(p.read_text())
        z = np.load(p.with_suffix(".npz"))
        if "preds" not in z.files or meta.get("failed"):
            continue
        rows.append((p, meta, z["preds"]))
    if not rows:
        return None
    rows.sort(key=lambda r: (r[1].get("account") != acct, r[1].get("qairt") is None))
    p, meta, preds = rows[0]
    return {"file": str(p.relative_to(ROOT.parents[2])), "steps": int(preds.shape[0]), "preds": preds,
            "account": meta.get("account", "main"), "qairt": meta.get("qairt"), "compile": meta.get("compile_job"),
            "done": bool(meta.get("done"))}


def host_rec(task, prec):
    p = ROOT / task / "infer" / f"host_{prec}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {"file": str(p.relative_to(ROOT.parents[2])), "preds": z["preds"], "q": z["q"]}


def chain(task, cheap, precise, device, acct):
    p = ROOT / task / "infer" / f"ctl_{cheap}-{precise}_{tagof(device)}_{acct}"
    meta = json.loads(p.with_suffix(".json").read_text())
    z = {k: v for k, v in np.load(p.with_suffix(".npz")).items()}
    return p, meta, z


def slim(p, z):
    out = p.with_suffix(".readout.npz")
    np.savez_compressed(out, **{k: v for k, v in z.items() if k not in ("zH", "zL")})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", action="store_true", help="write ctl_*.readout.npz (carry stripped) beside each chain")
    ap.add_argument("--json", default=str(ROOT / "e5_phone_ctl.json"))
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    out = {}
    acc_rows, ci_rows, cost_rows, host_rows, fired_rows = [], [], [], [], []
    for task, cheap, precise in CELLS:
        h = HCYC[task]
        fe = first_exit_cycle(task)
        for device, acct in DEVICES:
            p, meta, z = chain(task, cheap, precise, device, acct)
            if a.slim:
                slim(p, z)
            assert meta.get("done") and not meta.get("errors"), (p, meta.get("failed"), meta.get("errors"))
            lab = z["lab"]
            n = len(lab)
            fired = z["fired_at"] >= 0
            e_exit = exact(z["exit_pred"], lab)
            e_j2 = exact(z["final_pred"], lab)
            e_j1 = exact(z["fin_preds"][:, h - 1], lab)
            e_fired_only = np.where(fired, e_j2, e_exit)
            e_chain_fe = exact(z["cheap_preds"][:, fe - 1], lab)         # all puzzles still running at cycle fe
            assert (z["cheap_preds"][:, fe - 1] >= 0).all()
            fin_curve = [float(exact(z["fin_preds"][:, s], lab).mean()) for s in range(z["fin_preds"].shape[1])]
            cheap_curve = [float(exact(z["cheap_preds"][:, s], lab).mean()) for s in range(fe)]
            # fixed 8 on the device: the breadth row
            br = breadth_row(task, cheap, device, acct)
            f8 = None
            if br is not None:
                k8 = min(8, br["steps"])
                f8 = {"exact": exact(br["preds"][k8 - 1], lab), "cycles": k8, **{k: br[k] for k in
                      ("file", "account", "qairt", "compile", "steps", "done")}}
            # fixed 16: host only
            hf = host_rec(task, "fp32")
            hq = host_rec(task, cheap if cheap != "w8a8rope16" else "w8a8")
            f16 = None
            if hf is not None and hf["preds"].shape[0] >= 16:
                f16 = {"exact": exact(hf["preds"][15], lab), "file": hf["file"], "prec": "fp32"}
            key = f"{SHORT[device]}/{task}"
            rec = {"device": device, "account": acct, "task": task, "cheap": cheap, "precise": precise, "n": n,
                   "chain_json": str(p.with_suffix(".json").relative_to(ROOT.parents[2])),
                   "guard": {k: v.get("verdict") for k, v in meta.get("guard", {}).items()},
                   "compiles": {k: v.get("compile_job") for k, v in meta.get("guard", {}).items()},
                   "fired_frac": float(fired.mean()), "cheap_steps_mean": float(z["cheap_steps"].mean()),
                   "fin_steps_mean": float(z["fin_steps"].mean()),
                   "fired_at_hist": {int(k): int(v) for k, v in zip(*np.unique(z["fired_at"], return_counts=True))},
                   "exact_exit_only": float(e_exit.mean()), "exact_finish_all_j1": float(e_j1.mean()),
                   "exact_finish_all_j2": float(e_j2.mean()), "exact_finish_fired_only_j2": float(e_fired_only.mean()),
                   "exact_exit_fired": float(e_exit[fired].mean()) if fired.any() else None,
                   "exact_exit_capped": float(e_exit[~fired].mean()) if (~fired).any() else None,
                   "exact_finish_j2_capped": float(e_j2[~fired].mean()) if (~fired).any() else None,
                   "cell_acc_exit": float((z["exit_pred"] == lab).mean()),
                   "chain_fixed_cycle": fe, "exact_chain_fixed": float(e_chain_fe.mean()),
                   "cheap_curve_all_puzzles": cheap_curve, "fin_curve": fin_curve,
                   "fin_q_pos": [float((z["fin_q"][:, s] > 0).mean()) for s in range(z["fin_q"].shape[1])],
                   "profile_ms": meta["summary"]["profile_ms"],
                   "profile_jobs": {k: v.get("job_id") for k, v in meta["profile"].items()}}
            if f8 is not None:
                rec["fixed8"] = {k: (float(v.mean()) if k == "exact" else v) for k, v in f8.items()}
            if f16 is not None:
                rec["fixed16_host_fp32"] = {"exact": float(f16["exact"].mean()), "file": f16["file"]}
            # paired intervals
            ci = {"j2_minus_exit": boot(e_j2.astype(float) - e_exit, rng),
                  "j1_minus_exit": boot(e_j1.astype(float) - e_exit, rng),
                  "exit_minus_chain_fixed": boot(e_exit.astype(float) - e_chain_fe, rng)}
            if f8 is not None:
                ci["exit_minus_fixed8"] = boot(e_exit.astype(float) - f8["exact"], rng)
                ci["j2_minus_fixed8"] = boot(e_j2.astype(float) - f8["exact"], rng)
            if f16 is not None:
                ci["exit_minus_fixed16_host_fp32"] = boot(e_exit.astype(float) - f16["exact"], rng)
            rec["ci"] = ci
            # cost
            cm, pm = meta["summary"]["profile_ms"][cheap], meta["summary"]["profile_ms"][precise]
            cost = {"cheap_ms": cm, "precise_ms": pm, "exit_only_ms": cm * rec["cheap_steps_mean"],
                    "exit_finish_ms": cm * rec["cheap_steps_mean"] + pm * rec["fin_steps_mean"],
                    "fixed8_ms": 8 * cm, "fixed16_ms": 16 * cm, "precise_fixed8_ms": 8 * pm}
            cost["ratio_exit_vs_fixed8"] = cost["exit_only_ms"] / cost["fixed8_ms"]
            cost["ratio_exit_vs_fixed16"] = cost["exit_only_ms"] / cost["fixed16_ms"]
            cost["ratio_finish_vs_fixed8"] = cost["exit_finish_ms"] / cost["fixed8_ms"]
            cost["ratio_finish_vs_fixed16"] = cost["exit_finish_ms"] / cost["fixed16_ms"]
            rec["cost"] = cost
            # host controller: the exit rule replayed on the host loops
            host = {}
            for prec, hr in (("fp32", hf), (cheap if cheap != "w8a8rope16" else "w8a8", hq)):
                if hr is None:
                    continue
                hp, hs, hfd = sim_exit(task, hr["preds"], hr["q"], cap=16)
                he = exact(hp, lab)
                host[prec] = {"file": hr["file"], "cap": int(min(16, hr["preds"].shape[0])),
                              "fired_frac": float(hfd.mean()), "cheap_steps_mean": float(hs.mean()),
                              "exact_exit_only": float(he.mean()),
                              "exact_fixed_last": float(exact(hr["preds"][-1], lab).mean()),
                              "ci_device_exit_minus_host_exit": boot(e_exit.astype(float) - he, rng)}
            rec["host"] = host
            out[key] = rec
            # ------------------------------------------------------------------ table rows
            f8s = f"{100*rec['fixed8']['exact']:.0f} ({rec['fixed8']['cycles']}c, {rec['fixed8']['account']}" \
                  f"{', 2.45 pin' if rec['fixed8']['qairt'] else ''})" if f8 else "n/a"
            f16s = f"{100*rec['fixed16_host_fp32']['exact']:.0f} (host fp32)" if f16 else "n/a"
            acc_rows.append(f"| {SHORT[device]} | {NAME[task]} | {n} | {f8s} | {f16s} | {100*rec['exact_chain_fixed']:.0f} "
                            f"(c{fe}) | {100*rec['exact_exit_only']:.0f} | {100*rec['exact_finish_all_j1']:.0f} | "
                            f"{100*rec['exact_finish_all_j2']:.0f} | {100*rec['exact_finish_fired_only_j2']:.0f} | "
                            f"{100*rec['fired_frac']:.0f} | {rec['cheap_steps_mean']:.1f} |")
            ci_rows.append(f"| {SHORT[device]} | {NAME[task]} | {fmt_ci(ci['j2_minus_exit'])} | {fmt_ci(ci['j1_minus_exit'])} | "
                           f"{fmt_ci(ci['exit_minus_fixed8']) if f8 else 'n/a'} | {fmt_ci(ci['exit_minus_chain_fixed'])} (c{fe}) | "
                           f"{fmt_ci(ci['exit_minus_fixed16_host_fp32']) if f16 else 'n/a'} |")
            cost_rows.append(f"| {SHORT[device]} | {NAME[task]} | {cm:.2f} | {pm:.2f} | {cost['fixed8_ms']:.1f} | {cost['fixed16_ms']:.1f} | "
                             f"{cost['exit_only_ms']:.1f} ({cost['ratio_exit_vs_fixed8']:.2f}x / {cost['ratio_exit_vs_fixed16']:.2f}x) | "
                             f"{cost['exit_finish_ms']:.1f} ({cost['ratio_finish_vs_fixed8']:.2f}x / {cost['ratio_finish_vs_fixed16']:.2f}x) | "
                             f"{cost['precise_fixed8_ms']:.1f} |")
            fired_rows.append(f"| {SHORT[device]} | {NAME[task]} | " + ", ".join(
                f"{'cap' if k < 0 else 'c' + str(k)}: {v}" for k, v in sorted(rec["fired_at_hist"].items(), key=lambda t: (t[0] < 0, t[0])))
                + f" | {100*rec['exact_exit_fired']:.0f} | {100*rec['exact_exit_capped']:.0f} | {100*rec['exact_finish_j2_capped']:.0f} | "
                + f"{100*rec['cell_acc_exit']:.1f} |")
            for prec, hv in host.items():
                host_rows.append(f"| {SHORT[device]} | {NAME[task]} | host {prec} | {hv['cap']} | {100*hv['fired_frac']:.0f} | "
                                 f"{hv['cheap_steps_mean']:.1f} | {100*hv['exact_exit_only']:.0f} | {100*hv['exact_fixed_last']:.0f} | "
                                 f"{100*rec['exact_exit_only']:.0f} | {fmt_ci(hv['ci_device_exit_minus_host_exit'])} |")
    pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
    print("wrote", a.json)
    print("\n### Table 1 accuracy (exact %, n puzzles)\n")
    print("| device | task | n | fixed 8 (breadth row) | fixed 16 | chain all-puzzle @c | exit only | exit+fin j=1 | "
          "exit+fin j=2 | fin fired-only j=2 | fired % | cheap cycles |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(acc_rows))
    print("\n### Table 2 paired 95% intervals (points, puzzle bootstrap B=10000)\n")
    print("| device | task | j=2 - exit | j=1 - exit | exit - fixed 8 (breadth) | exit - chain fixed | exit - host fp32 fixed 16 |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(ci_rows))
    print("\n### Table 3 fired-at histogram and exit quality\n")
    print("| device | task | fired at cycle: count | exit exact fired % | exit exact capped % | j=2 exact capped % | exit cell acc % |")
    print("|---|---|---|---|---|---|---|")
    print("\n".join(fired_rows))
    print("\n### Table 4 device cost per puzzle (ms, profiled per-step latency)\n")
    print("| device | task | cheap ms/step | precise ms/step | fixed 8 | fixed 16 | exit only (vs 8 / vs 16) | exit+fin j=2 (vs 8 / vs 16) | precise fixed 8 |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(cost_rows))
    print("\n### Table 5 host controller (exit rule replayed on the host loop; no finishing)\n")
    print("| device | task | host loop | cap | fired % | cheap steps | host exit only | host fixed (last step) | device exit only | device - host exit |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(host_rows))


if __name__ == "__main__":
    main()
