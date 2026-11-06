#!/usr/bin/env python3
"""Energy law for looped reasoners: J per correct answer = (steps per answer) x (J per step) / accuracy.

Reads every energy record on disk and, per board / power mode / task / precision / batch, takes the energy per
step from ONE anchor run (fixed depth 16 when it exists) and predicts J per correct answer for every other run of
that group from the run's own mean steps and accuracy (record pass). Compared with the measured median J per
solved over the clean windows, for total-board energy and for idle-subtracted (dynamic) energy.

Also: a FLOP-based estimate of energy per step from published peak FLOPS and rated mode power; achieved FLOP
throughput as a fraction of peak; where per-step, per-puzzle and per-correct energy rank configurations
differently; and a forecast (no measurement) for the head-free exit rules from their step counts.

Inputs (rerun end to end; new boards are picked up by the globs):
  edge/results/<board>/<board>_*.json     edge_loop.py records (agx, nano-a, nano-b, ...); records without
                                          energy windows (gate / check-only runs) are skipped
  results/energy/babel_l40s/<task>/energy.json                 L40S E1 grid (e1_energy.py)
  results/energy/babel_l40s_capsweep{,_rerun}/<task>.json       L40S cap sweep (rerun wins for attn, HRM)
  edge/results/p4d-cpu/hf_check*.json     step counts and accuracy per head-free rule (check.exit_headfree)
Output: edge/results/energy_law.json, and markdown tables on stdout.

    python3 edge/energy_law.py > /tmp/energy_law_tables.md
"""
import glob
import json
import os
import re
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results", "energy_law.json")
MIN_ACC = 0.05  # J per solved is undefined when nothing is solved (L40S int8 maze)

# FLOPs per supervision step of each exported graph (from the ONNX graphs; batch 1)
GFLOP_PER_STEP = {"sudoku_mlp": 25.63, "sudoku_attn": 28.58, "maze": 238.85, "hrm_sudoku": 13.75}

# Published specs. peak in TFLOPS at the max GPU clock; mode power in W.
SPECS = {
    "agx": {
        "gpu_max_MHz": 1377.0,
        "peak_TFLOPS": {"fp16": 11.0, "fp32": 512 * 2 * 1.377e9 / 1e12, "dla0fp16": 2.5},
        "mode_W": {"maxn": 30.0, "30w": 30.0, "15w": 15.0, "10w": 10.0},
        "src": {
            "fp16": "11 TFLOPS FP16 (Volta GPU, 64 Tensor Cores, 1377 MHz): "
                    "https://developer.nvidia.com/blog/nvidia-jetson-agx-xavier-32-teraops-ai-robotics/",
            "fp32": "derived: 512 CUDA cores x 2 FLOP x 1.377 GHz = 1.41 TFLOPS (core count and clock from the same "
                    "NVIDIA page); third-party listing 1.35 TFLOPS: https://siliconcat.net/gpu/nvidia-jetson-agx-xavier-gpu",
            "dla0fp16": "2.5 TFLOPS FP16 per DLA (same NVIDIA page)",
            "mode_W": "10W / 15W / 30W profiles (same NVIDIA page); MAXN has no rated budget, 30 W used",
            "mode_clocks": "GPU cap per nvpmodel: 520 (10W), 670 (15W), 900 (30W), 1377 (MAXN) MHz: "
                           "https://developer.ridgerun.com/wiki/index.php/Xavier/JetPack_5.0.2/Performance_Tuning/Tuning_Power"
                           " (the records' own jetson_clocks read-out is used where present)"}},
    "nano": {
        "gpu_max_MHz": 921.6,
        "peak_TFLOPS": {"fp32": 0.236, "fp16": 0.472, "fp16forced": 0.472},
        "mode_W": {"maxn": 10.0, "10w": 10.0, "5w": 5.0},
        "src": {
            "fp16": "472 GFLOPS (128-core Maxwell @ 921 MHz): https://developer.nvidia.com/blog/jetson-nano-ai-computing/ ;"
                    " 472 = 128 x 2 x 2 x 0.9216 GHz, i.e. the FP16 rate",
            "fp32": "derived: 128 x 2 FLOP x 0.9216 GHz = 236 GFLOPS (half the FP16 figure)",
            "mode_W": "5W / 10W modes (same NVIDIA page)",
            "mode_clocks": "GPU 640 MHz in 5W, 921 MHz in 10W (MAXN): https://en.wikipedia.org/wiki/Nvidia_Jetson"}},
    "l40s": {
        "gpu_max_MHz": None,
        "peak_TFLOPS": {"fp32": 91.6, "fp16": 362.05},
        "mode_W": {"default": 350.0},
        "src": {"fp32": "91.6 TFLOPS FP32: https://www.supermicro.com/datasheet/datasheet_NVIDIA_L40S_Systems.pdf",
                "fp16": "362.05 TFLOPS FP16 Tensor Core dense (733 with sparsity): https://www.pny.com/nvidia-l40s",
                "mode_W": "350 W max board power (same pages)"}},
}
NANO_MODE = {"nano-a": "10w", "nano-b": "5w"}
NV_MODE = {"MAXN": "maxn", "MODE_10W": "10w", "MODE_15W": "15w", "MODE_30W_ALL": "30w", "MODE_5W": "5w", "5W": "5w"}


def spec_of(board):
    return SPECS["nano"] if board.startswith("nano") else SPECS.get(board)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


# ---------------------------------------------------------------- loading

def load_edge():
    runs = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "*", "*.json"))):
        board = os.path.basename(os.path.dirname(f))
        if board in ("p4d-cpu", "laptop") or "no_energy" in f:
            continue
        try:
            d = json.load(open(f))
        except (ValueError, OSError):
            continue
        if not isinstance(d, dict) or not d.get("windows") or not d.get("record"):
            continue
        m = d["meta"]
        cfg, w = next(iter(d["windows"].items()))
        mc = w.get("median_clean") or {}
        if "J_per_solved" not in mc:
            continue
        good = [g for g in w["windows"] if g.get("clean") and "E_J" in g]
        note = m.get("note") or ""
        ce = m.get("clocks_end") if isinstance(m.get("clocks_end"), dict) else (m.get("clocks") or {})
        nv = (ce.get("nvpmodel") or "").split("\n")[0].replace("NV Power Mode:", "").strip()
        gpu = re.search(r"GPU MinFreq=\d+ MaxFreq=(\d+)", ce.get("jetson_clocks") or "")
        mreq = re.search(r"^(maxn|10w|15w|30w|5w)", note)
        req = mreq.group(1) if mreq else NANO_MODE.get(board)
        actual = NV_MODE.get(nv, req) if nv else req
        fin = int(m.get("finish_steps") or 0) if m.get("finish") else 0
        runs.append({
            "src": os.path.relpath(f, ROOT), "board": board.split("-")[0] if board.startswith("agx") else board,
            "power_mode": actual, "requested_mode": req, "gpu_MHz": int(gpu.group(1)) / 1e6 if gpu else None,
            "task": m["task"], "prec": m["precision"], "B": int(m.get("B") or 1), "mode": m["mode"],
            "cap": int(m["depth"]), "finish": fin, "cfg": cfg,
            "label": "%s %s%d%s%s" % (m["precision"], m["mode"], int(m["depth"]), " +%d fp32" % fin if fin else "",
                                      " (req %s)" % req if req and req != actual else ""),
            "acc": d["record"]["exact"], "mean_steps": d["record"]["mean_steps"],
            "acc_win": mc.get("acc_window"), "steps_per_puzzle_win": med([g["steps"] / g["puzzles"] for g in good]),
            "mJ_step": mc["mJ_per_step"], "mJ_step_dyn": mc.get("mJ_per_step_dyn"), "ms_step": mc["ms_per_step"],
            "P_W": mc["P_W"], "J_puzzle": mc["J_per_puzzle"], "J_solved": mc["J_per_solved"],
            "J_solved_dyn": mc.get("J_per_solved_dyn"), "idle_W": med([i.get("P_W") for i in d.get("idle", [])]),
            "n_clean": w.get("n_clean")})
    return runs


def load_l40s():
    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "energy", "babel_l40s", "*", "energy.json"))):
        d = json.load(open(f))
        task = os.path.basename(os.path.dirname(f))
        for cfg, w in d["windows"].items():
            mc, fp = w.get("median_clean") or {}, w.get("first_pass") or {}
            if "J_per_solved" not in mc:
                continue
            rt, prec, b, mode = cfg.split("_")
            good = [g for g in w["windows"] if g.get("clean")]
            runs.append({
                "src": os.path.relpath(f, ROOT), "board": "l40s", "power_mode": "default", "requested_mode": None,
                "gpu_MHz": med([g.get("sm_MHz_mean") for g in good]), "task": task, "prec": rt + "_" + prec,
                "B": int(b[1:]), "mode": mode, "cap": 16, "finish": 0, "cfg": cfg, "label": cfg,
                "acc": fp["exact"], "mean_steps": fp["mean_steps"], "acc_win": mc.get("acc_window"),
                "steps_per_puzzle_win": med([g["steps"] / g["puzzles"] for g in good]),
                "mJ_step": mc["mJ_per_step"], "mJ_step_dyn": mc.get("mJ_per_step_dyn"), "ms_step": mc["ms_per_step"],
                "P_W": mc["P_W"], "J_puzzle": mc["J_per_puzzle"], "J_solved": mc["J_per_solved"],
                "J_solved_dyn": mc.get("J_per_solved_dyn"), "idle_W": med([i.get("P_W") for i in d["idle"]]),
                "n_clean": w.get("n_clean")})
    base = os.path.join(ROOT, "results", "energy")
    for task in GFLOP_PER_STEP:
        f = os.path.join(base, "babel_l40s_capsweep_rerun", task + ".json")
        if not os.path.exists(f):
            f = os.path.join(base, "babel_l40s_capsweep", task + ".json")
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        for cfg, c in d["configs"].items():
            s = c.get("summary")
            good = [g for g in c.get("windows") or [] if g.get("clean")]
            if not s or not good:
                continue
            m = re.match(r"(rtn|dither)_b(\d+)_(fixed|exit|polish)_c(\d+)", cfg)
            arm, b, mode, cap = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
            runs.append({
                "src": os.path.relpath(f, ROOT), "board": "l40s", "power_mode": "default", "requested_mode": None,
                "gpu_MHz": med([g.get("sm_MHz_mean") for g in good]), "task": task, "prec": "capsweep_w4c_fp32",
                "B": b, "mode": mode, "cap": cap, "finish": 0, "cfg": cfg, "label": cfg, "arm": arm,
                "acc": s["acc"], "mean_steps": s["mean_steps"], "acc_win": med([g["window_acc"] for g in good]),
                "steps_per_puzzle_win": med([g["slot_steps"] / g["done"] for g in good]),
                "mJ_step": med([1e3 * g["E_J"] / g["slot_steps"] for g in good]),
                "mJ_step_dyn": med([1e3 * g["E_dyn_J"] / g["slot_steps"] for g in good]),
                "ms_step": med([1e3 * g["T_s"] / g["slot_steps"] for g in good]),
                "P_W": med([g["P_W"] for g in good]), "J_puzzle": med([g["J_per_puzzle"] for g in good]),
                "J_solved": s["J_solved_median"], "J_solved_dyn": s["J_solved_dyn_median"],
                "idle_W": float(c["idle_P_W"]), "n_clean": s["n_clean"]})
    return runs


# ---------------------------------------------------------------- task 1: the law

def pick_anchor(rs):
    """fixed 16 without finishing; else the deepest fixed run; else the deepest plain exit run."""
    plain = [r for r in rs if not r["finish"] and r.get("arm", "rtn") == "rtn"]
    for key in (lambda r: r["mode"] == "fixed" and r["cap"] == 16, lambda r: r["mode"] == "fixed",
                lambda r: r["mode"] == "exit"):
        c = sorted([r for r in plain if key(r)],
                   key=lambda r: (-r["cap"], bool(r["requested_mode"]) and r["requested_mode"] != r["power_mode"]))
        if c:
            return c[0], ("fixed16" if c[0]["mode"] == "fixed" and c[0]["cap"] == 16 else "fallback " + c[0]["label"])
    return None, None


def family(r):
    if r["board"] != "l40s":
        return "finishing" if r["finish"] else "plain"
    if r["prec"].startswith("capsweep"):
        return "capsweep " + r.get("arm", "rtn")
    return "E1 grid"


def pct(p, m):
    return None if p is None or m in (None, 0) else 100.0 * (p - m) / m


def law(runs):
    groups = {}
    for r in runs:
        groups.setdefault((r["board"], r["power_mode"], r["task"], r["prec"], r["B"]), []).append(r)
    rows, untested = [], []
    for key, rs in sorted(groups.items()):
        a, kind = pick_anchor(rs)
        others = [r for r in rs if r is not a]
        if a is None or not others:
            untested.append({"group": list(key), "runs": [r["label"] for r in rs]})
            continue
        for r in others:
            if r["acc"] < MIN_ACC:
                continue
            p = r["mean_steps"] * a["mJ_step"] / 1e3 / r["acc"]
            pd = r["mean_steps"] * a["mJ_step_dyn"] / 1e3 / r["acc"] if a["mJ_step_dyn"] is not None else None
            row = {"board": key[0], "power_mode": key[1], "task": key[2], "prec": key[3], "B": key[4],
                   "anchor": a["label"], "anchor_kind": kind, "run": r["label"], "requested_mode": r["requested_mode"],
                   "family": family(r), "finish": r["finish"], "mean_steps": r["mean_steps"], "acc": r["acc"],
                   "J_solved_meas": r["J_solved"], "J_solved_pred": p, "err_pct": pct(p, r["J_solved"]),
                   "J_solved_dyn_meas": r["J_solved_dyn"], "J_solved_dyn_pred": pd,
                   "err_dyn_pct": pct(pd, r["J_solved_dyn"]),
                   "J_step_dev_pct": pct(r["mJ_step"], a["mJ_step"]),
                   "J_step_dyn_dev_pct": pct(r["mJ_step_dyn"], a["mJ_step_dyn"]),
                   "acc_record_vs_window_pct": pct(r["acc"], r["acc_win"]),
                   "steps_record_vs_window_pct": pct(r["mean_steps"], r["steps_per_puzzle_win"]), "src": r["src"]}
            if r["finish"]:
                # finishing steps run the fp32 graph: price them at the fp32 J/step of the same board/mode/task
                f32 = [x for x in runs if x["board"] == key[0] and x["power_mode"] == key[1] and x["task"] == key[2]
                       and x["prec"] == "fp32" and not x["finish"] and x["B"] == key[4]]
                if f32:
                    f = sorted(f32, key=lambda x: (x["mode"] != "fixed", -x["cap"]))[0]
                    cheap = r["mean_steps"] - r["finish"]
                    pc = (cheap * a["mJ_step"] + r["finish"] * f["mJ_step"]) / 1e3 / r["acc"]
                    pcd = (cheap * a["mJ_step_dyn"] + r["finish"] * f["mJ_step_dyn"]) / 1e3 / r["acc"]
                    row.update({"fp32_ref": f["label"], "J_solved_pred_split": pc, "err_split_pct": pct(pc, r["J_solved"]),
                                "J_solved_dyn_pred_split": pcd, "err_dyn_split_pct": pct(pcd, r["J_solved_dyn"])})
            rows.append(row)
    return rows, untested


def summarize(rows):
    """Per board and family: |err| with record steps and accuracy (the law as stated), and |J/step deviation|
    from the anchor (the law with each window's own steps and accuracy: isolates J/step constancy)."""
    out = {}
    for key in sorted(set((r["board"], r["family"]) for r in rows)):
        rs = [r for r in rows if (r["board"], r["family"]) == key]
        ab = lambda k: [abs(r[k]) for r in rs if r.get(k) is not None]
        e = {k: ab(k) for k in ("err_pct", "err_dyn_pct", "J_step_dev_pct", "J_step_dyn_dev_pct")}
        out["%s | %s" % key] = dict([("n", len(rs))] + [(n + "_" + k, fn(v) if v else None) for k, v in e.items()
                                                        for n, fn in (("median_abs", med), ("max_abs", max))])
        if key[1] == "finishing":
            e = ab("err_split_pct")
            ed = ab("err_dyn_split_pct")
            if e:
                out["%s | finishing, fp32 steps at fp32 J/step" % key[0]] = {
                    "n": len(e), "median_abs_err_pct": med(e), "max_abs_err_pct": max(e),
                    "median_abs_err_dyn_pct": med(ed), "max_abs_err_dyn_pct": max(ed)}
    return out


def cross_mode(runs):
    """Same task/precision/config run at another power mode, predicted with the MAXN (or default) anchor."""
    out = []
    for r in runs:
        if not r["board"] == "agx" or r["power_mode"] == "maxn":
            continue
        base = [x for x in runs if x["board"] == r["board"] and x["power_mode"] == "maxn" and x["task"] == r["task"]
                and x["prec"] == r["prec"] and x["B"] == r["B"]]
        a, kind = pick_anchor(base)
        if a is None:
            continue
        p = r["mean_steps"] * a["mJ_step"] / 1e3 / r["acc"]
        pd = r["mean_steps"] * a["mJ_step_dyn"] / 1e3 / r["acc"]
        out.append({"task": r["task"], "prec": r["prec"], "power_mode": r["power_mode"], "gpu_MHz": r["gpu_MHz"],
                    "run": r["label"], "anchor": "maxn " + a["label"], "P_W": r["P_W"], "ms_step": r["ms_step"],
                    "mJ_step": r["mJ_step"], "anchor_mJ_step": a["mJ_step"], "err_pct": pct(p, r["J_solved"]),
                    "err_dyn_pct": pct(pd, r["J_solved_dyn"])})
    return out


# ---------------------------------------------------------------- tasks 2 and 4: FLOPs

def flops(runs):
    out = []
    seen = set()
    for r in runs:
        sp = spec_of(r["board"])
        prec = r["prec"].split("_")[-1] if r["board"] == "l40s" else r["prec"]
        if sp is None or prec not in sp["peak_TFLOPS"] or r["prec"].startswith("capsweep"):
            continue
        key = (r["board"], r["power_mode"], r["task"], r["prec"], r["B"])
        if key in seen:
            continue
        rs = [x for x in runs if (x["board"], x["power_mode"], x["task"], x["prec"], x["B"]) == key and not x["finish"]]
        a, _ = pick_anchor(rs)
        if a is None:
            continue
        seen.add(key)
        G = GFLOP_PER_STEP[r["task"]] * 1e9
        peak = sp["peak_TFLOPS"][prec] * 1e12
        W = sp["mode_W"].get(r["power_mode"], sp["mode_W"].get("default"))
        est_mJ = 1e3 * G / peak * W
        clk = a["gpu_MHz"] if (a["gpu_MHz"] and sp["gpu_max_MHz"] and prec != "dla0fp16") else None
        peak_clk = peak * clk / sp["gpu_max_MHz"] if clk else peak
        est_mJ_clk = 1e3 * G / peak_clk * W
        achieved = G / (a["ms_step"] / 1e3)
        out.append({"board": r["board"], "power_mode": r["power_mode"], "task": r["task"], "prec": r["prec"], "B": r["B"],
                    "run": a["label"], "GFLOP_step": G / 1e9, "peak_TFLOPS": peak / 1e12, "rated_W": W,
                    "gpu_MHz": clk, "est_mJ_step": est_mJ, "est_mJ_step_clockscaled": est_mJ_clk,
                    "meas_mJ_step": a["mJ_step"], "meas_mJ_step_dyn": a["mJ_step_dyn"],
                    "ratio_meas_over_est": a["mJ_step"] / est_mJ,
                    "ratio_dyn_over_est": a["mJ_step_dyn"] / est_mJ if a["mJ_step_dyn"] else None,
                    "ratio_meas_over_est_clockscaled": a["mJ_step"] / est_mJ_clk, "ms_step": a["ms_step"],
                    "achieved_TFLOPS": achieved / 1e12, "frac_of_peak": achieved / peak,
                    "frac_of_clockscaled_peak": achieved / peak_clk, "meas_P_W": a["P_W"]})
    return out


# ---------------------------------------------------------------- task 3: metrics rank differently

def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0] * len(vals)
    for k, i in enumerate(order):
        r[i] = k + 1
    return r


def metric_ranking(runs):
    out = {}
    groups = {}
    for r in runs:
        if r["board"] == "l40s" and r["prec"].startswith("capsweep"):
            continue
        if r["acc"] < MIN_ACC:
            continue
        groups.setdefault((r["board"], r["power_mode"], r["task"], r["B"]), []).append(r)
    for key, rs in sorted(groups.items()):
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: r["J_solved"])
        a, b, c = (ranks([r[k] for r in rs]) for k in ("mJ_step", "J_puzzle", "J_solved"))
        flips = {"step_vs_correct": 0, "puzzle_vs_correct": 0, "step_vs_puzzle": 0}
        n = len(rs)
        for i in range(n):
            for j in range(i + 1, n):
                flips["step_vs_correct"] += (a[i] - a[j]) * (c[i] - c[j]) < 0
                flips["puzzle_vs_correct"] += (b[i] - b[j]) * (c[i] - c[j]) < 0
                flips["step_vs_puzzle"] += (a[i] - a[j]) * (b[i] - b[j]) < 0
        out["|".join(map(str, key))] = {
            "n_configs": n, "n_pairs": n * (n - 1) // 2, "flipped_pairs": flips,
            "rows": [{"run": r["label"], "acc": r["acc"], "mean_steps": r["mean_steps"], "mJ_step": r["mJ_step"],
                      "J_puzzle": r["J_puzzle"], "J_solved": r["J_solved"], "J_solved_dyn": r["J_solved_dyn"],
                      "rank_step": a[k], "rank_puzzle": b[k], "rank_correct": c[k]} for k, r in enumerate(rs)]}
    return out


# ---------------------------------------------------------------- forecast: head-free rules (no measurement)

def headfree_forecast(runs):
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "results", "p4d-cpu", "hf_check*.json"))):
        d = json.load(open(f))
        c, task = d.get("check") or {}, d["meta"]["task"]
        hf = dict(c.get("exit_headfree") or {})
        hf[c.get("exit_rule", "native") + " (head)"] = {"exact": c.get("exit_exact"), "mean_steps": c.get("exit_mean_steps")}
        for board in sorted(set(r["board"] for r in runs if r["board"] != "l40s")):
            for prec in ("fp16", "fp32", "fp16forced"):
                base = [x for x in runs if x["board"] == board and x["power_mode"] in ("maxn", "10w") and
                        x["task"] == task and x["prec"] == prec and not x["finish"]]
                a, _ = pick_anchor(base)
                if a is None:
                    continue
                for rule, v in hf.items():
                    if not v.get("exact"):
                        continue
                    out.append({"board": board, "power_mode": a["power_mode"], "task": task, "prec": prec,
                                "rule": rule, "steps_src": os.path.relpath(f, ROOT), "n_puzzles": c.get("n"),
                                "acc_cpu_fp32": v["exact"], "mean_steps": v["mean_steps"], "anchor": a["label"],
                                "J_solved_pred": v["mean_steps"] * a["mJ_step"] / 1e3 / v["exact"],
                                "J_solved_dyn_pred": v["mean_steps"] * a["mJ_step_dyn"] / 1e3 / v["exact"]})
    return out


# ---------------------------------------------------------------- tables

def f(x, d=1):
    return "-" if x is None else ("%." + str(d) + "f") % x


def tables(res):
    L = []
    L.append("## Law: per-run error (predicted vs measured J per correct)\n")
    L.append("| board | mode | task | prec | B | anchor | run | steps | acc | J/correct meas | err % | dyn err % "
             "| J/step dev % | fp32-priced err % | fp32-priced dyn err % |")
    L.append("|" + "---|" * 15)
    for r in res["law_rows"]:
        L.append("| %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["board"], r["power_mode"], r["task"], r["prec"], r["B"], r["anchor"], r["run"],
            f(r["mean_steps"], 2), f(r["acc"], 3), f(r["J_solved_meas"], 3), f(r["err_pct"]), f(r["err_dyn_pct"]),
            f(r["J_step_dev_pct"]), f(r.get("err_split_pct")), f(r.get("err_dyn_split_pct"))))
    L.append("\n## Law: summary\n")
    L.append("| board, rows | n | median abs err % | max | median abs dyn err % | max | median abs J/step dev % "
             "| max | median abs dyn J/step dev % | max |")
    L.append("|" + "---|" * 10)
    for k, v in res["law_summary"].items():
        L.append("| %s | %d | %s |" % (k, v["n"], " | ".join(f(v.get(c)) for c in (
            "median_abs_err_pct", "max_abs_err_pct", "median_abs_err_dyn_pct", "max_abs_err_dyn_pct",
            "median_abs_J_step_dev_pct", "max_abs_J_step_dev_pct", "median_abs_J_step_dyn_dev_pct",
            "max_abs_J_step_dyn_dev_pct"))))
    L.append("\n## Cross-mode: MAXN anchor applied to other power modes\n")
    L.append("| task | prec | mode | GPU MHz | run | W | ms/step | mJ/step | MAXN mJ/step | err % | dyn err % |")
    L.append("|" + "---|" * 11)
    for r in res["cross_mode"]:
        L.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["task"], r["prec"], r["power_mode"], f(r["gpu_MHz"], 0), r["run"], f(r["P_W"], 2), f(r["ms_step"], 2),
            f(r["mJ_step"]), f(r["anchor_mJ_step"]), f(r["err_pct"]), f(r["err_dyn_pct"])))
    L.append("\n## FLOP estimate and achieved throughput\n")
    L.append("| board | mode | task | prec | B | GFLOP/step | peak TFLOPS | rated W | est mJ/step | meas mJ/step "
             "| meas/est | dyn/est | meas/est (clock-scaled peak) | ms/step | achieved TFLOPS | frac of peak |")
    L.append("|" + "---|" * 16)
    for r in res["flops"]:
        L.append("| %s | %s | %s | %s | %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["board"], r["power_mode"], r["task"], r["prec"], r["B"], f(r["GFLOP_step"], 2), f(r["peak_TFLOPS"], 2),
            f(r["rated_W"], 0), f(r["est_mJ_step"], 2), f(r["meas_mJ_step"]), f(r["ratio_meas_over_est"]),
            f(r["ratio_dyn_over_est"]), f(r["ratio_meas_over_est_clockscaled"]), f(r["ms_step"], 2),
            f(r["achieved_TFLOPS"], 3), f(r["frac_of_peak"], 3)))
    L.append("\n## Metric ranking (rank 1 = least energy)\n")
    for k, v in res["metric_ranking"].items():
        L.append("%s: %d configs, flipped pairs %s" % (k, v["n_configs"], v["flipped_pairs"]))
    L.append("\n## Head-free forecast (predicted only)\n")
    L.append("| board | task | prec | rule | acc (CPU fp32, n) | steps | anchor | pred J/correct | pred dyn |")
    L.append("|" + "---|" * 9)
    for r in res["headfree_forecast"]:
        L.append("| %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s |" % (
            r["board"], r["task"], r["prec"], r["rule"], f(r["acc_cpu_fp32"], 2), r["n_puzzles"], f(r["mean_steps"], 2),
            r["anchor"], f(r["J_solved_pred"], 3), f(r["J_solved_dyn_pred"], 3)))
    return "\n".join(L)


def main():
    runs = load_edge() + load_l40s()
    rows, untested = law(runs)
    res = {"note": "energy_law.py; J per correct = mean steps x J/step(anchor) / accuracy. err = (pred - meas) / meas.",
           "specs": SPECS, "gflop_per_step": GFLOP_PER_STEP,
           "boards_found": sorted(set(r["board"] for r in runs)), "n_runs": len(runs),
           "runs": runs, "law_rows": rows, "law_summary": summarize(rows), "untested_groups": untested,
           "cross_mode": cross_mode(runs), "flops": flops(runs), "metric_ranking": metric_ranking(runs),
           "headfree_forecast": headfree_forecast(runs)}
    with open(OUT, "w") as fh:
        json.dump(res, fh, indent=1)
    print(tables(res))
    print("\nwrote", OUT, "runs", len(runs), "boards", res["boards_found"])


if __name__ == "__main__":
    main()
