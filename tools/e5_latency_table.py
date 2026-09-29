#!/usr/bin/env python3
"""E5 latency by device tier and Hexagon generation (reports/edgetrm_exp/E5_LATENCY_TABLE.md). Records only: no device job.

Inputs (EdgeTRM/results/devices/):
  latency/aihub_profiles_sep29.json  one row per AI Hub profile job recorded on sk-p4d, read back from the hub on Sep 29
                                     (read-only: hub.get_job(id).download_profile(); credentials via qai_hub.Client(profile=...)).
                                     Main account: the H=1 fan-out of Sep 24-25 (`aihub_h1.py --stage fanout`: the `step`
                                     graph = logits-only H=1 step, and on the S24 the deployable `stepc` graph, on every
                                     hosted device; ~/aihub_artifacts/h1/<task>/jobs.json). acct2 / acct3: the controller
                                     profiles of Sep 29 (`aihub_ctl.py`, stepc qdq graphs on the pinned QAIRT 2.45 compiles,
                                     plus the five Sep 29 00:36-00:49 profiles that ran on the set-aside 2.50 compiles).
  aihub_devices_sep29.json           hub device attributes (chipset, hexagon:vNN, htp-supports-fp16, os)
  device_faithful_acc.json           per-step exact / cell accuracy of every device chain and the host simulations
  breadth_chain_meta.json            chain provenance (failure text for the chains that never ran)
  ctl/e5_phone_ctl.json              the S24 / S25 controller step counts and profiled per-step latencies

Latency numbers: `ms` = the hub's estimated_inference_time (the number every earlier note quotes; the hub derives it from the
100 timed iterations of the profile job), `median` = median of the 100 iterations, `first` = the first iteration.
QAIRT: the hub profile carries no converter version; the column follows DIAG_ACCT_SPLIT.md section 1 (compiles submitted
Sep 24-25 ran QAIRT 2.45.0.260326154327; the Sep 26 compiles listed in aihub_move_qairt250.Q250 ran 2.50) and the pinned
Sep 29 compiles (`aihub_acct.QAIRT`).

    python tools/e5_latency_table.py            # prints the Markdown, writes results/devices/latency/e5_latency_table.json
"""
import collections
import json
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent.parent
DEV = HERE / "results" / "devices"
sys.path.insert(0, str(HERE))
from device_table import CHIP  # noqa: E402  (chip / Hexagon of devices the hub no longer lists)

Q250 = {"jpvln8vz5", "jgjr89e1p", "jpe7nqk85", "j5wlrkq4p", "jgddmkv6g", "jg9z7r08p", "jgzl06r45"}   # aihub_move_qairt250
TASKS = ["sudoku_mlp", "sudoku_attn", "maze"]
TNAME = {"sudoku_mlp": "Sudoku-MLP", "sudoku_attn": "Sudoku-attn", "maze": "Maze"}
PRECS = {"sudoku_mlp": ["fp32", "w8a8", "w8a16"], "sudoku_attn": ["fp32", "w8a8", "w8a16"],
         "maze": ["fp32", "w8a8", "w8a16", "w8a8rope16"]}
PNAME = {"fp32": "fp32 (fp16 on HTP)", "w8a8": "w8a8", "w8a16": "w8a16", "w8a8rope16": "w8a8rope16"}
HOST = {"sudoku_mlp": "host sim of compiled QDQ graph, test 0-99", "sudoku_attn": "host sim of compiled QDQ graph, test 0-99",
        "maze": "host sim of compiled QDQ graph, test 0-49"}
GEN_ORDER = ["v65", "v66", "v68", "v69", "v73", "v75", "v79", "v81", "tflite"]
PASS, FAIL = 0.95, 0.5
# controller (E5_PHONE_CTL.md): cheap graph, precise graph, finishing cycles
CTL = {"sudoku_mlp": ("w8a8", "w8a16"), "sudoku_attn": ("w8a8", "w8a16"), "maze": ("w8a8rope16", "fp32")}
FIN = 6


def load():
    prof = json.load(open(DEV / "latency" / "aihub_profiles_sep29.json"))
    devs = json.load(open(DEV / "aihub_devices_sep29.json"))["devices"]
    acc = json.load(open(DEV / "device_faithful_acc.json"))
    meta = json.load(open(DEV / "breadth_chain_meta.json"))["chains"]
    ctl = json.load(open(DEV / "ctl" / "e5_phone_ctl.json"))
    return prof, devs, acc, meta, ctl


def device_info(name, devs, hub_device=None):
    """chip label, Hexagon generation, OS string, format, fp16-HTP flag."""
    d = devs.get(name)
    if d is None and hub_device:      # attributes the profile job itself carries (devices the hub no longer lists)
        attrs = hub_device.split("attributes=[")[-1].rstrip("])").replace("'", "").split(", ")
        os_ = hub_device.split("os='")[1].split("'")[0] if "os='" in hub_device else "?"
        a = dict(x.split(":", 1) for x in attrs if ":" in x)
        chips = [x.split(":", 1)[1] for x in attrs if x.startswith("chipset:")]
        d = {"chipset": chips, "hexagon": a.get("hexagon"), "os": os_, "os_kind": a.get("os", "?"),
             "htp_fp16": a.get("htp-supports-fp16"), "format": a.get("format", "?")}
    if d is None:
        c = CHIP.get(name)
        return (c[0] if c else "?", c[1] if c else "?", "?", "?", c[2] if c else None)
    chip = (d["chipset"] or ["?"])[0].replace("qualcomm-", "").replace("samsung-", "").replace("google-", "")
    hexg = d.get("hexagon") or ("tflite" if not any("qualcomm" in c for c in d["chipset"] or []) else "?")
    if hexg == "?" and name in CHIP:
        hexg = CHIP[name][1]
    return chip, hexg, f"{d.get('os_kind', '?')} {d.get('os', '?')}", d.get("format", "?"), d.get("htp_fp16")


def qairt_of(r):
    """converter version of the profiled DLC's compile job: the pinned version read back from the hub where the compile was
    pinned (Sep 29), else by the compile job's submission date (DIAG_ACCT_SPLIT.md section 1: Sep 24-25 = 2.45, Sep 26 = 2.50)."""
    v = r.get("compile_qairt_sdk_version") or ""
    if r.get("compile_qairt_pin"):
        return f"{v[:4]} (pinned)"
    if r.get("from_compile") in Q250 or v.startswith("2.50"):
        return "2.50 (set-aside compile)"
    sub = (r.get("compile_submitted") or "")[:10]
    return "2.45 (hub default, compiled Sep 24-25)" if sub in ("2026-09-24", "2026-09-25") else f"? (compiled {sub})"


def graph_of(r):
    if r.get("graph"):
        return f"{r['graph']} H{r['H']}"
    return "stepc H1"                       # the ctl profiles are the stepc_H1 qdq graphs (aihub_acct.source_path)


def verdict(ratio):
    return "PASS" if ratio >= PASS else ("FAIL" if ratio < FAIL else "WARN")


def accuracy_index(acc, meta, devs):
    """(task, prec, device) -> list of {acct, exact8, cell8, verdict} from completed chains; failure text otherwise."""
    names = sorted(set(list(devs) + list(CHIP) + ["Snapdragon 8 Elite QRD", "Snapdragon 8 Elite Gen 5 QRD"]), key=len, reverse=True)
    host = {}
    for r in acc:
        if r["prec"] == "fp32" and r["source"] == HOST.get(r["task"]) and r["step"] == 8:
            host[r["task"]] = r["exact"]
    out = collections.defaultdict(list)
    for r in acc:
        if not r["source"].startswith("ON DEVICE") or r["step"] != 8:
            continue
        tag = r["source"][len("ON DEVICE "):]
        dev = next((n for n in names if tag.startswith(n.replace(" ", "_"))), None)
        if dev is None:
            continue
        acct = tag[len(dev.replace(" ", "_")):].strip("_") or "main"
        h = host.get(r["task"])
        out[(r["task"], r["prec"], dev)].append({"acct": acct, "exact8": r["exact"], "cell8": r["cell"],
                                                 "verdict": verdict(r["exact"] / h) if h else "?"})
    failed = {}
    for m in meta:
        if m["set_aside"] or not m.get("failed"):
            continue
        f = m["failed"]
        cls = ("class A: MODEL_GRAPH_ERROR at composition" if "composeGraphsFromDlc" in f else
               "class B: TFLite GPU delegate node 410" if "TfLiteGpuDelegate" in f else
               "class B: TFLite segfault" if "segmentation" in f else f[:60])
        failed.setdefault((m["task"], m["prec"], m["device"]), cls)
    return out, failed, host


def acc_cell(task, prec, dev, accidx, failed):
    rows = accidx.get((task, prec, dev))
    if rows:
        # one entry per account; the series are identical inside a generation, so the exact values agree
        ex = sorted({(round(100 * r["exact8"]), r["verdict"]) for r in rows})
        return " / ".join(f"{e} {v}" for e, v in ex) + f" (n={len(rows)} chain{'s' if len(rows) > 1 else ''})"
    if (task, prec, dev) in failed:
        return "no chain: " + failed[(task, prec, dev)]
    return "no chain on this device"


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def main():
    prof, devs, acc, meta, ctl = load()
    accidx, failed, host = accuracy_index(acc, meta, devs)
    ok = [r for r in prof if r.get("status") == "SUCCESS"]
    info = {}
    for r in prof:
        if r["device"] not in info:
            info[r["device"]] = device_info(r["device"], devs, r.get("hub_device"))
    gen_of = {d: i[1] for d, i in info.items()}
    order = lambda d: (GEN_ORDER.index(gen_of[d]) if gen_of[d] in GEN_ORDER else 99, d)  # noqa: E731
    out = {"sources": {"profiles": "results/devices/latency/aihub_profiles_sep29.json",
                       "devices": "results/devices/aihub_devices_sep29.json",
                       "accuracy": "results/devices/device_faithful_acc.json (step 8, ON DEVICE rows) + breadth_chain_meta.json (failed chains)",
                       "controller": "results/devices/ctl/e5_phone_ctl.json"},
           "host_fp32_exact8": host, "table1": {}, "table2": {}, "table3": {}, "flatness": {}}
    md = []

    # ---------------------------------------------------------------- table 1: per task and graph, one row per profile job
    md.append("## 1. Per-step latency of every profiled graph on every device\n")
    md.append("One row per profile job. `graph`: `step H1` = the logits-only H=1 step graph of the Sep 24-25 fan-out, `stepc H1` = the "
              "deployable carry-in / carry-out step graph (the graph the chains run; on the S24 both were profiled, and every "
              "controller profile is a stepc graph). ms = hub estimated inference time; median / first = of the 100 timed "
              "iterations; mem = estimated inference peak memory; units = layers per compute unit. Accuracy = exact % at 8 "
              "device steps of the chain of the same graph family on the same device (PASS >= 0.95 x host fp32, FAIL < 0.5, "
              "WARN between; host fp32 step 8: Sudoku-MLP 58, Sudoku-attn 51, Maze 88), n = chains on that device.\n")
    for task in TASKS:
        for prec in PRECS[task]:
            rows = [r for r in ok if r["task"] == task and r["prec"] == prec and (r.get("graph") in ("step", "stepc", None))
                    and (r.get("H") in (1, None))]
            if not rows:
                continue
            rows.sort(key=lambda r: (order(r["device"]), graph_of(r), r.get("date") or ""))
            md.append(f"### {TNAME[task]} {PNAME[prec]}\n")
            md.append("| device | chip | Hexagon | OS | runtime | graph | QAIRT | ms/step | median | first | mem MB | units | job | accuracy (step 8) |")
            md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
            t1 = []
            for r in rows:
                chip, hexg, os_, fmt_, _ = info[r["device"]]
                a = acc_cell(task, prec, r["device"], accidx, failed)
                units = ",".join(f"{k}:{v}" for k, v in sorted(r["units"].items()))
                md.append(f"| {r['device']} | {chip} | {hexg} | {os_} | {r['runtime']} | {graph_of(r)} | {qairt_of(r)} | "
                          f"{fmt(r['ms'], 3)} | {fmt(r['median_ms'], 3)} | {fmt(r['first_ms'], 3)} | {fmt(r['peak_mb'], 0)} | {units} | "
                          f"{r['job_id']} | {a} |")
                t1.append({"device": r["device"], "chip": chip, "hexagon": hexg, "os": os_, "runtime": r["runtime"],
                           "graph": graph_of(r), "qairt": qairt_of(r), "ms": r["ms"], "median_ms": r["median_ms"],
                           "first_ms": r["first_ms"], "peak_mb": r["peak_mb"], "units": r["units"], "job_id": r["job_id"],
                           "from_compile": r.get("from_compile"), "account": r["account"], "accuracy": a})
            # failed profiles of the same graph (the hub refused to run it)
            bad = [r for r in prof if r["task"] == task and r["prec"] == prec and r.get("status") == "FAILED"
                   and r.get("graph") in ("step", "stepc", None) and r.get("H") in (1, None)]
            for r in sorted(bad, key=lambda r: order(r["device"])):
                chip, hexg, os_, _, _ = info[r["device"]]
                e = r.get("error", "")
                short = ("MODEL_GRAPH_ERROR" if "composeGraphsFromDlc" in e else "segfault" if "segmentation" in e else
                         "INVALID_CONFIG" if "INVALID_CONFIG" in e else "access violation" if "access violation" in e else e[:50])
                md.append(f"| {r['device']} | {chip} | {hexg} | {os_} | {r['runtime']} | {graph_of(r)} | {qairt_of(r)} | "
                          f"profile FAILED: {short} | | | | | {r['job_id']} | {acc_cell(task, prec, r['device'], accidx, failed)} |")
                t1.append({"device": r["device"], "chip": chip, "hexagon": hexg, "graph": graph_of(r), "job_id": r["job_id"],
                           "failed": e[:200], "accuracy": acc_cell(task, prec, r["device"], accidx, failed)})
            md.append("")
            out["table1"][f"{task}/{prec}"] = t1
    md.append("Source: `results/devices/latency/aihub_profiles_sep29.json` (job ids in the table; main-account rows = "
              "`sk-p4d:~/aihub_artifacts/h1/<task>/jobs.json` kind=profile, acct rows = `~/aihub_artifacts/h1/acct_<acct>/jobs.json` "
              "kind=profile and the `profile` block of `results/devices/ctl/<task>/infer/ctl_*.json`); accuracy from "
              "`device_faithful_acc.json` step 8 and `breadth_chain_meta.json`; `tools/e5_latency_table.py` table 1.\n")

    # ---------------------------------------------------------------- table 2: by generation, fp16 vs 8-bit (step H1 fan-out only)
    md.append("## 2. 8-bit against fp16 by Hexagon generation\n")
    md.append("Same graph family on every row: the `step H1` graphs of the Sep 24-25 fan-out (one DLC per precision, profiled on "
              "every device; QAIRT 2.45). ms = hub estimate. Speed-up = fp16 ms / w8a8 ms (> 1 means the 8-bit graph is faster). "
              "The S24 / S25 `stepc` rows (pinned 2.45, Sep 29) are appended for the two controller phones. Accuracy of the "
              "8-bit graph on that device as in table 1.\n")
    step = collections.defaultdict(dict)   # (task, device) -> prec -> row
    for r in ok:
        if r.get("graph") == "step" and r.get("H") == 1 and r["prec"] in ("fp32", "w8a8", "w8a16"):
            step[(r["task"], r["device"])][r["prec"]] = r
    for task in TASKS:
        md.append(f"### {TNAME[task]}\n")
        md.append("| Hexagon | device | chip | runtime | fp16 ms | w8a8 ms | w8a16 ms | fp16 / w8a8 | fp16 / w8a16 | 8-bit faster? | w8a8 accuracy | w8a16 accuracy |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        t2, by_gen, by_gen16 = [], collections.defaultdict(list), collections.defaultdict(list)
        for (t, dev), pr in sorted(step.items(), key=lambda kv: order(kv[0][1])):
            if t != task:
                continue
            chip, hexg, _, _, _ = info[dev]
            f, a8, a16 = (pr.get(p, {}).get("ms") for p in ("fp32", "w8a8", "w8a16"))
            s8 = f / a8 if f and a8 else None
            s16 = f / a16 if f and a16 else None
            faster = "-" if s8 is None else ("yes" if s8 > 1.10 else "no" if s8 < 0.90 else "flat (within 10 %)")
            row = {"hexagon": hexg, "device": dev, "chip": chip, "runtime": pr[next(iter(pr))]["runtime"], "fp16_ms": f,
                   "w8a8_ms": a8, "w8a16_ms": a16, "speedup_w8a8": s8, "speedup_w8a16": s16,
                   "w8a8_acc": acc_cell(task, "w8a8", dev, accidx, failed), "w8a16_acc": acc_cell(task, "w8a16", dev, accidx, failed),
                   "jobs": {p: pr[p]["job_id"] for p in pr}}
            t2.append(row)
            if s8:
                by_gen[hexg].append(s8)
            if s16:
                by_gen16[hexg].append(s16)
            md.append(f"| {hexg} | {dev} | {chip} | {row['runtime']} | {fmt(f)} | {fmt(a8)} | {fmt(a16)} | {fmt(s8)} | {fmt(s16)} | "
                      f"{faster} | {row['w8a8_acc']} | {row['w8a16_acc']} |")
        # controller stepc rows on the pinned compile
        for dev in ("Samsung Galaxy S24", "Samsung Galaxy S25"):
            pr = {r["prec"]: r for r in ok if r["account"] != "main" and r["task"] == task and r["device"] == dev
                  and r.get("from_compile") not in Q250}
            if not pr:
                continue
            chip, hexg, _, _, _ = info[dev]
            f, a8, a16 = (pr.get(p, {}).get("ms") for p in ("fp32", "w8a8", "w8a16"))
            s8 = f / a8 if f and a8 else None
            md.append(f"| {hexg} | {dev} (stepc, pinned 2.45) | {chip} | qnn_dlc | {fmt(f)} | {fmt(a8)} | {fmt(a16)} | {fmt(s8)} | "
                      f"{fmt(f / a16) if f and a16 else '-'} | {'-' if s8 is None else ('yes' if s8 > 1.05 else 'no' if s8 < 0.95 else 'flat')} | "
                      f"{acc_cell(task, 'w8a8', dev, accidx, failed)} | {acc_cell(task, 'w8a16', dev, accidx, failed)} |")
            t2.append({"hexagon": hexg, "device": dev + " (stepc pinned)", "fp16_ms": f, "w8a8_ms": a8, "w8a16_ms": a16,
                       "speedup_w8a8": s8, "jobs": {p: pr[p]["job_id"] for p in pr}})
        md.append("")
        md.append("| Hexagon generation | devices with both graphs | fp16 / w8a8 (median, range) | fp16 / w8a16 (median, range) | 8-bit faster on the generation? |")
        md.append("|---|---|---|---|---|")
        gsum = {}
        for g in GEN_ORDER:
            if g in by_gen:
                v, v16 = by_gen[g], by_gen16.get(g, [])
                med = statistics.median(v)
                gsum[g] = {"n": len(v), "median": med, "min": min(v), "max": max(v),
                           "w8a16": {"n": len(v16), "median": statistics.median(v16), "min": min(v16), "max": max(v16)} if v16 else None}
                r16 = f"{statistics.median(v16):.2f} ({min(v16):.2f}-{max(v16):.2f}, n={len(v16)})" if v16 else "-"
                md.append(f"| {g} | {len(v)} | {med:.2f} ({min(v):.2f}-{max(v):.2f}) | {r16} | "
                          f"{'yes' if min(v) > 1.10 else 'no' if max(v) < 0.90 else 'flat within 10 %' if all(0.90 <= x <= 1.10 for x in v) else 'mixed'} |")
        md.append("")
        out["table2"][task] = {"rows": t2, "by_generation": gsum}
    md.append("Source: table 1 rows with graph `step H1` (fan-out) and the acct2 / acct3 stepc rows; `tools/e5_latency_table.py` table 2.\n")

    # ---------------------------------------------------------------- flatness: is the per-step time independent of the graph?
    md.append("### 2b. Spread of the per-step time across the three precisions (dispatch-bound test)\n")
    md.append("max / min of the fp16, w8a8 and w8a16 `step H1` times on the same device, per task. A task whose time is flat "
              "across the three graphs (< 1.10) on a device is not compute-bound there: halving the weight bytes does not move "
              "the step, so the step is dominated by per-inference overhead (dispatch, I/O, the 42-copy graph walk). Verdict "
              "lists the tasks that are flat on that device.\n")
    md.append("| device | Hexagon | Sudoku-MLP max/min | Sudoku-attn max/min | Maze max/min | verdict |")
    md.append("|---|---|---|---|---|---|")
    for dev in sorted({d for _, d in step}, key=order):
        spreads = {}
        for task in TASKS:
            v = [r["ms"] for r in step.get((task, dev), {}).values()]
            spreads[task] = max(v) / min(v) if len(v) >= 2 else None
        flat = [TNAME[t] for t, v in spreads.items() if v is not None and v < 1.10]
        vs = [s for s in spreads.values() if s]
        verdict_ = "-" if not vs else ("flat on " + ", ".join(flat) + " (dispatch-bound)" if flat else "graph-dependent on every task")
        out["flatness"][dev] = {**spreads, "verdict": verdict_}
        md.append(f"| {dev} | {info[dev][1]} | {fmt(spreads['sudoku_mlp'])} | {fmt(spreads['sudoku_attn'])} | {fmt(spreads['maze'])} | {verdict_} |")
    md.append("\nSource: same rows as table 2.\n")

    # ---------------------------------------------------------------- table 3: controller cost per puzzle
    md.append("## 3. Controller cost per puzzle (ms) on every device with the two graphs profiled\n")
    steps = {t: {"S24": ctl[f"S24/{t}"]["cheap_steps_mean"], "S25": ctl[f"S25/{t}"]["cheap_steps_mean"]} for t in TASKS}
    rope = {k: ctl[f"{k}/maze"]["cost"]["cheap_ms"] for k in ("S24", "S25")}
    md.append("cost = cheap ms x mean cheap cycles + precise ms x 6 finishing cycles (E5_PHONE_CTL.md section 2). The cycle counts are "
              "the measured S24 / S25 controller counts: " + "; ".join(f"{TNAME[t]} {steps[t]['S24']:.1f} (S24) / {steps[t]['S25']:.1f} (S25)" for t in TASKS)
              + ". Sudoku: cheap = w8a8, precise = w8a16; Maze: cheap = w8a8rope16, precise = fp32 (fp16 on the HTP). Only the S24 "
              "and S25 ran the controller (rows marked measured, profiled on the pinned stepc graphs); every other row is a "
              "**projection** from that device's `step H1` fan-out profiles with the S24 cycle counts (S25 counts in brackets), "
              "and for Maze the w8a8 step profile stands in for the rope16 graph (no rope16 profile exists off the two phones; on "
              f"the S24 the pinned rope16 stepc profile reads {rope['S24']:.2f} ms against {step[('maze', 'Samsung Galaxy S24')]['w8a8']['ms']:.2f} ms for the "
              f"w8a8 step graph, on the S25 {rope['S25']:.2f} against {step[('maze', 'Samsung Galaxy S25')]['w8a8']['ms']:.2f}). A projection says nothing "
              "about accuracy: the accuracy column is the fixed-8 verdict of the cheap graph on that device, and where the cheap "
              "graph FAILs (Maze w8a8 on v73 / v75 / v79, rope16 on v79) the controller has no cheap phase to exit from.\n")
    for task in TASKS:
        cheap, precise = CTL[task]
        md.append(f"### {TNAME[task]} (cheap {cheap}, precise {precise})\n")
        md.append("| Hexagon | device | cheap ms | precise ms | exit only | exit + finishing | fixed 8 cheap | precise fixed 8 | kind | cheap-graph accuracy |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        t3 = []
        for dev in sorted({d for _, d in step}, key=order):
            pr = step.get((task, dev), {})
            cp = "w8a8" if cheap == "w8a8rope16" else cheap
            c, p = pr.get(cp, {}).get("ms"), pr.get(precise, {}).get("ms")
            if c is None or p is None:
                continue
            s24, s25 = steps[task]["S24"], steps[task]["S25"]
            e24, e25 = c * s24, c * s25
            row = {"device": dev, "hexagon": info[dev][1], "cheap_ms": c, "precise_ms": p, "cheap_graph_profiled": cp + " step H1",
                   "exit_only_ms_s24_counts": e24, "exit_only_ms_s25_counts": e25, "exit_finish_ms_s24_counts": e24 + FIN * p,
                   "exit_finish_ms_s25_counts": e25 + FIN * p, "fixed8_cheap_ms": 8 * c, "precise_fixed8_ms": 8 * p, "kind": "projection",
                   "cheap_acc": acc_cell(task, cheap, dev, accidx, failed)}
            t3.append(row)
            md.append(f"| {info[dev][1]} | {dev} | {c:.2f} ({cp} step) | {p:.2f} | {e24:.1f} [{e25:.1f}] | {e24 + FIN * p:.1f} [{e25 + FIN * p:.1f}] | "
                      f"{8 * c:.1f} | {8 * p:.1f} | projection | {row['cheap_acc']} |")
        for dev, key in (("Samsung Galaxy S24", "S24"), ("Samsung Galaxy S25", "S25")):
            c_ = ctl[f"{key}/{task}"]["cost"]
            row = {"device": dev, "hexagon": info[dev][1], "cheap_ms": c_["cheap_ms"], "precise_ms": c_["precise_ms"],
                   "exit_only_ms": c_["exit_only_ms"], "exit_finish_ms": c_["exit_finish_ms"], "fixed8_cheap_ms": c_["fixed8_ms"],
                   "precise_fixed8_ms": c_["precise_fixed8_ms"], "kind": "measured (pinned stepc profiles, own cycle counts)",
                   "cheap_acc": f"controller exit-only {100 * ctl[f'{key}/{task}']['exact_exit_only']:.0f}, j=2 {100 * ctl[f'{key}/{task}']['exact_finish_all_j2']:.0f}"}
            t3.append(row)
            md.append(f"| {info[dev][1]} | {dev} | {c_['cheap_ms']:.2f} ({cheap} stepc) | {c_['precise_ms']:.2f} | {c_['exit_only_ms']:.1f} | "
                      f"{c_['exit_finish_ms']:.1f} | {c_['fixed8_ms']:.1f} | {c_['precise_fixed8_ms']:.1f} | measured | {row['cheap_acc']} |")
        md.append("")
        out["table3"][task] = t3
    md.append("Source: table 2 rows (projections), `results/devices/ctl/e5_phone_ctl.json` `cost` and `cheap_steps_mean` "
              "(measured rows); `tools/e5_latency_table.py` table 3.\n")

    (DEV / "latency" / "e5_latency_table.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
