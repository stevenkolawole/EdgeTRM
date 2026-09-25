#!/usr/bin/env python3
"""X3 device read-outs: the H=1 device table, the granularity compile matrix, per-answer latency,
accuracy under a wall-clock deadline, and energy per solved instance.

Inputs (pulled into the repo):
  results/devices/aihub_h1/status.json      one row per AI Hub job (aihub_h1.py --stage status)
  results/devices/aihub_h1/<task>_jobs.json the job records (quantize/compile/profile/inference ids)
  results/breadth/*.json, results/devices/chipq/*.json   accuracy vs (H, n) per quantizer
Outputs:
  results/devices/device_table.json / .md   step latency, memory, per-answer latency (H=1 and H=3)
  results/devices/gran_matrix.json / .md    device x granularity x bit-width compile/run matrix
  results/devices/deadline.json / .md       best accuracy reachable within a deadline, per device

    python device_table.py [--tables all|latency|gran|deadline]
"""
import argparse
import collections
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEV = HERE / "results" / "devices"
AIH = DEV / "aihub_h1"

# Supervision steps per answer at H=1 (accuracy depends on H x n only; see X3.md):
# ARC and Maze saturate at H x n = 8; Sudoku keeps improving, 16 is the harness's deepest H=1 cell.
N_H1 = {"arc": 8, "maze": 8, "sudoku_mlp": 16, "sudoku_attn": 16}
N_OFFICIAL = 16          # official TRM evaluation: H=3, 16 supervision steps
ACC_TASK = {"maze": "maze_attn", "arc": "arc_attn", "sudoku_mlp": "sudoku_mlp", "sudoku_attn": "sudoku_attn"}
# device precision -> harness quantizer with the same weight/activation format
HARNESS_Q = {"fp32": "fp32", "w8a8": "w8c+a8", "w4a8": "w4c+a8", "w8a16": "w8c+a16", "w4a16": "w4c+a16",
             "w4a8_t": "w4t+a8", "w8a8_t": "w8t+a8", "w4a8_g32": "w4g32+a8", "w4a8_g128": "w4g128+a8",
             "w4a16_g32": "w4g32+a16", "w4a8_l64": "w4l64+a8", "w4a16_l64": "w4l64+a16", "w4a8_l32": "w4l32+a8"}
GRAN_COLS = ["fp32", "w8a8", "w8a8_t", "w8a16", "w4a8", "w4a8_t", "w4a8_g32", "w4a8_g128", "w4a8_l32", "w4a8_l64",
             "w4a16", "w4a16_g32", "w4a16_l64"]
CHIP = {  # device -> (chipset, hexagon, fp16 on HTP)
    "Samsung Galaxy S21": ("SD 888", "v68", False), "Samsung Galaxy A73 5G": ("SD 778G", "v68", False),
    "Dragonwing RB3 Gen 2 Vision Kit": ("QCS6490", "v68", False), "SA8295P ADP": ("SA8295P", "v68", True),
    "Samsung Galaxy S22 5G": ("8 Gen 1", "v69", True), "Samsung Galaxy S23": ("8 Gen 2", "v73", True),
    "Snapdragon 7 Gen 4 QRD": ("7 Gen 4", "v73", False), "Dragonwing Q-6690 MTP": ("QCM6690", "v73", False),
    "QCS8550 (Proxy)": ("QCS8550", "v73", True), "Dragonwing IQ-9075 EVK": ("QCS9075", "v73", True),
    "SA8775P ADP": ("SA8775P", "v73", True), "Snapdragon X Elite CRD": ("X Elite", "v73", True),
    "Snapdragon X Plus 8-Core CRD": ("X Plus", "v73", True), "Samsung Galaxy S24": ("8 Gen 3", "v75", True),
    "SA7255P ADP": ("SA7255P", "v75", True), "Arduino VENTUNO Q": ("QCS8275", "v75", True),
    "Samsung Galaxy S25": ("8 Elite (Galaxy)", "v79", True), "Snapdragon 8 Elite QRD": ("8 Elite", "v79", True),
    "Samsung Galaxy S26": ("8 Elite Gen 5 (Galaxy)", "v81", True),
    "Snapdragon 8 Elite Gen 5 QRD": ("8 Elite Gen 5", "v81", True),
    "Snapdragon X2 Elite CRD": ("X2 Elite", "v81", True),
    "Google Pixel 3": ("SD 845", "v65*", False), "Google Pixel 3a": ("SD 670", "v65*", False),
    "Google Pixel 4": ("SD 855", "v66*", False), "Google Pixel 5": ("SD 765G", "v66*", False),
    "Samsung Galaxy Tab S7": ("SD 865+", "v66*", False), "Xiaomi Redmi Note 10 5G": ("SD 678", "v66*", False),
    "Google Pixel 6": ("Tensor G1", "-", False), "Google Pixel 7": ("Tensor G2", "-", False),
    "Google Pixel 8": ("Tensor G3", "-", False), "Google Pixel 9": ("Tensor G4", "-", False),
    "Google Pixel 10": ("Tensor G5", "-", False), "Samsung Galaxy Note 20 (Intl)": ("Exynos 990", "-", False),
    "Samsung Galaxy A53 5G": ("Exynos 1280", "-", False), "Samsung Galaxy A14 5G": ("Exynos 1330", "-", False),
}
DEV_ORDER = list(CHIP)


def load_status():
    rows = json.loads((AIH / "status.json").read_text())
    comp, prof = {}, {}
    for r in rows:
        key = (r["task"], r["graph"], r["H"], r["prec"], r.get("device"))
        if r["kind"] == "compile":
            comp.setdefault(key, []).append(r)
        elif r["kind"] == "profile":
            prof.setdefault(key, []).append(r)
    return rows, comp, prof


def cell(comp, prof, key):
    """Status of one (task, graph, H, prec, device): the latest successful profile, else the reason."""
    ps = [p for p in prof.get(key, []) if p.get("status") == "SUCCESS" and p.get("ms") is not None]
    if ps:
        p = ps[-1]
        return {"status": "RUN", "ms": p["ms"], "peak_mb": p.get("peak_mb"), "load_peak_mb": p.get("load_peak_mb"),
                "units": p.get("units"), "profile_job": p["job_id"], "compile_job": p.get("from_compile"),
                "runtime": p.get("runtime")}
    pf = [p for p in prof.get(key, []) if p.get("status") == "FAILED"]
    cs = comp.get(key, [])
    if pf:
        return {"status": "PROFILE_FAILED", "error": pf[-1].get("error", "")[:300], "profile_job": pf[-1]["job_id"],
                "compile_job": pf[-1].get("from_compile"), "log_errors": pf[-1].get("log_errors", [])}
    if any(c.get("status") == "FAILED" for c in cs):
        c = [c for c in cs if c.get("status") == "FAILED"][-1]
        return {"status": "COMPILE_FAILED", "error": c.get("error", "")[:300], "compile_job": c["job_id"]}
    if any(c.get("status") == "SUCCESS" for c in cs):
        return {"status": "COMPILED_PROFILE_PENDING"}
    if cs:
        return {"status": "PENDING"}
    return None


def short_reason(err, log_errors=()):
    e = ((err or "") + " " + " ".join(log_errors or [])).lower()
    if "doesn't support fp16" in e:
        op = next((l.split("validate op ")[1].split(" with")[0] for l in log_errors or [] if "validate op " in l), "")
        return f"HTP has no FP16 (op {op})" if op else "HTP has no FP16"
    if "floating-point type which is not supported" in e:
        return "float tensor on integer-only HTP"
    if "block" in e and ("not supported" in e or "unsupported" in e):
        return "block quantization unsupported"
    if "int4" in e and "tf-lite" in e.replace("tflite", "tf-lite"):
        return "no INT4 in TF-Lite"
    if "memory" in e and "exceed" in e:
        return "memory limit"
    if "segmentation" in e:
        return "runtime segfault"
    return (err or "")[:80]


# ----------------------------------------------------------------------------- accuracy
def load_acc():
    """acc[(task, quantizer)][H*n] = exact accuracy (H*n = supervision cycles; H=1 rows preferred)."""
    acc = collections.defaultdict(dict)
    files = glob.glob(str(HERE / "results" / "breadth" / "*.json")) + glob.glob(str(DEV / "chipq" / "*.json"))
    for f in files:
        name = Path(f).name
        if name.startswith(("variant_", "noise_", "anoise", "residual", "smoke", "carry_")):
            continue
        try:
            d = json.loads(Path(f).read_text())
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or "records" not in d:
            continue
        for r in d["records"]:
            if not isinstance(r, dict) or "H_cycles" not in r or r.get("variant"):
                continue
            q = r["quant"]["spec"] if isinstance(r.get("quant"), dict) else None
            a = r.get("pexact", r.get("arc_pass2"))
            if q is None or a is None or r.get("H_cycles") != 1:
                continue
            acc[(r["task"], q)][r["nsup"]] = a
    return acc


# ----------------------------------------------------------------------------- tables
def latency_table(comp, prof):
    out, md = [], ["| Task | Device | Chip | HTP | Prec | H=1 step ms | H=3 step ms | ratio | peak MB (H=1) | "
                   "n (H=1) | answer ms H=1 | answer ms H=3 n=16 | speed-up |",
                   "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in ("sudoku_mlp", "sudoku_attn", "maze", "arc"):
        for dev in DEV_ORDER:
            for prec in ("fp32", "w8a8", "w4a8", "w8a16", "w4a16"):
                c1 = cell(comp, prof, (task, "step", 1, prec, dev))
                c3 = cell(comp, prof, (task, "step", 3, prec, dev))
                if c1 is None and c3 is None:
                    continue
                n = N_H1[task]
                row = {"task": task, "device": dev, "chip": CHIP[dev][0], "hexagon": CHIP[dev][1], "prec": prec,
                       "H1": c1, "H3": c3, "n_H1": n}
                if c1 and c1["status"] == "RUN":
                    row["answer_ms_H1"] = n * c1["ms"]
                if c3 and c3["status"] == "RUN":
                    row["answer_ms_H3_n16"] = N_OFFICIAL * c3["ms"]
                if c1 and c3 and c1["status"] == c3["status"] == "RUN":
                    row["step_ratio_H1_over_H3"] = c1["ms"] / c3["ms"]
                    row["answer_speedup"] = row["answer_ms_H3_n16"] / row["answer_ms_H1"]
                out.append(row)

                def f(c, k="ms"):
                    if not c:
                        return "-"
                    if c["status"] == "RUN":
                        return f"{c[k]:.2f}" if c.get(k) is not None else "-"
                    return {"COMPILE_FAILED": "CF", "PROFILE_FAILED": "PF"}.get(c["status"], "…")
                md.append(f"| {task} | {dev} | {CHIP[dev][0]} | {CHIP[dev][1]} | {prec} | {f(c1)} | {f(c3)} | "
                          f"{row.get('step_ratio_H1_over_H3', float('nan')):.2f} | {f(c1, 'peak_mb')} | {n} | "
                          f"{row.get('answer_ms_H1', float('nan')):.1f} | {row.get('answer_ms_H3_n16', float('nan')):.1f} | "
                          f"{row.get('answer_speedup', float('nan')):.1f} |")
    (DEV / "device_table.json").write_text(json.dumps(out, indent=1))
    (DEV / "device_table.md").write_text(
        "# H=1 vs H=3 step latency on AI Hub devices (qnn_dlc / tflite; CF = compile failed, "
        "PF = profile failed, … = pending)\n\n" + "\n".join(md) + "\n")
    return out


def gran_matrix(comp, prof):
    out = []
    md = ["| Task | Device | HTP | " + " | ".join(GRAN_COLS) + " |", "|---|---|---|" + "---|" * len(GRAN_COLS)]
    reasons = collections.Counter()
    for task in ("sudoku_mlp", "sudoku_attn", "maze", "arc"):
        for dev in DEV_ORDER:
            cells = {}
            for prec in GRAN_COLS:
                c = cell(comp, prof, (task, "step", 1, prec, dev))
                if c:
                    cells[prec] = c
                    if c["status"] in ("COMPILE_FAILED", "PROFILE_FAILED"):
                        reasons[(prec, CHIP[dev][1], short_reason(c.get("error"), c.get("log_errors")))] += 1
            if not cells:
                continue
            out.append({"task": task, "device": dev, "chip": CHIP[dev][0], "hexagon": CHIP[dev][1], "cells": cells})

            def f(c):
                if c is None:
                    return ""
                if c["status"] == "RUN":
                    return f"{c['ms']:.1f}"
                return {"COMPILE_FAILED": "CF", "PROFILE_FAILED": "PF"}.get(c["status"], "…")
            md.append(f"| {task} | {dev} | {CHIP[dev][1]} | " + " | ".join(f(cells.get(p)) for p in GRAN_COLS) + " |")
    md += ["", "Failure reasons (precision, HTP version, reason: count):", ""]
    md += [f"- {k[0]}, {k[1]}: {k[2]} ({v})" for k, v in sorted(reasons.items())]
    (DEV / "gran_matrix.json").write_text(json.dumps(out, indent=1))
    (DEV / "gran_matrix.md").write_text("# Device x granularity x bit-width (H=1 step; ms per step; CF compile "
                                        "failed; PF profile failed)\n\n" + "\n".join(md) + "\n")
    return out


def deadline(comp, prof, acc, deadlines_ms=(5, 10, 20, 50, 100, 200, 500, 1000, 2000)):
    """Best exact accuracy within a wall-clock deadline T, per device. Each option pairs a latency and an
    accuracy of the SAME graph: the deployable carry step (`stepc`, H=1) profiled on that device, with the
    accuracy of that stepc graph (on that device's chain if one exists, else the host simulation of the
    compiled graph), at n steps; time = n x step latency. (Audit Sep 25: the earlier version paired the
    logits-only `step` graph's latency with harness accuracies.) `acc` is unused and kept for the call."""
    df, _ = device_faithful()
    out = []
    for task in ("sudoku_mlp", "sudoku_attn", "maze", "arc"):
        for dev in DEV_ORDER:
            opts = []
            for (t, prec), steps in df.items():
                if t != task:
                    continue
                c = cell(comp, prof, (task, "stepc", 1, prec, dev))
                if not c or c["status"] != "RUN":
                    continue
                for n, (a, cnt, src) in steps.items():
                    if src.startswith("ON DEVICE") and dev.replace(" ", "_") not in src:
                        continue      # another device's chain is not this device's accuracy
                    opts.append({"prec": prec, "n": n, "ms": n * c["ms"], "acc": a, "acc_source": src,
                                 "latency_graph": "stepc", "profile_job": c.get("profile_job")})
            if not opts:
                continue
            best = {}
            for T in deadlines_ms:
                fit = [o for o in opts if o["ms"] <= T]
                best[T] = max(fit, key=lambda o: (o["acc"], -o["ms"])) if fit else None
            out.append({"task": task, "device": dev, "chip": CHIP[dev][0], "best_within": best, "options": opts})
    (DEV / "deadline.json").write_text(json.dumps(out, indent=1))
    md = ["| Task | Device | " + " | ".join(f"{T} ms" for T in deadlines_ms) + " |",
          "|---|---|" + "---|" * len(deadlines_ms)]
    for r in out:
        cells = []
        for T in deadlines_ms:
            b = r["best_within"][T]
            cells.append("-" if b is None else f"{b['acc']*100:.1f} ({b['prec']}, n={b['n']})")
        md.append(f"| {r['task']} | {r['device']} | " + " | ".join(cells) + " |")
    (DEV / "deadline.md").write_text("# Best exact accuracy within a wall-clock deadline (stepc graph: profiled step "
                                     "latency x accuracy of the same graph)\n\n" + "\n".join(md) + "\n")
    return out


def runtime_of(source):
    if source.startswith("ON DEVICE"):
        return "QNN HTP on " + source[len("ON DEVICE "):].replace("_", " ")
    return "ONNX Runtime CPU, fused integer QDQ kernels (host sim)"


def device_faithful():
    """{(task, prec): {step: (exact, count, source)}} from results/devices/device_faithful_acc.json, the
    accuracy of the very graph that is timed. If any on-device chain exists for (task, prec), ONLY
    on-device rows are used for that pair (steps it did not run stay empty: an ORT number describes the
    host runtime, not the phone); otherwise the host simulation with the largest sample."""
    f = DEV / "device_faithful_acc.json"
    rows = json.loads(f.read_text()) if f.exists() else []
    has_dev = {(r["task"], r["prec"]) for r in rows if r["source"].startswith("ON DEVICE")}
    out = collections.defaultdict(dict)
    for r in rows:
        key = (r["task"], r["prec"])
        if key in has_dev and not r["source"].startswith("ON DEVICE"):
            continue
        cur = out[key].get(r["step"])
        if cur is None or r["count"] > cur[1]:
            out[key][r["step"]] = (r["exact"], r["count"], r["source"])
    return out, has_dev


def energy_table(acc):
    """Energy per answer and per solved instance, E_solved = E_step x n / accuracy(n).
    Accuracy is always that of the graph whose cost is counted: for the analytical estimate of an
    AI Hub format, the host simulation (or on-device run) of that compiled graph at H=1, n steps
    (fp16 rows use the compiled fp32 graph, which HTP executes in fp16); rows without such an
    accuracy get no mJ/solved. The measured L40S rows time the fp32 torch model, whose accuracy is
    the harness fp32 cell. Every row names its accuracy source."""
    rows = []
    est = json.loads((DEV / "energy_estimate.json").read_text()) if (DEV / "energy_estimate.json").exists() else []
    df, has_dev = device_faithful()
    for e in est:
        if e["H"] != 1:
            continue
        prec = "fp32" if e["format"] in ("fp32", "fp16") else e["format"]
        steps = df.get((e["task"], prec), {})
        for n in (1, 2, 4, 8, 16):
            for bound in ("weights_on_chip", "weights_dram"):
                ms = e[f"mj_per_step_{bound}"]
                a = steps.get(n)
                row = {"energy_source": "estimate (Horowitz 45 nm)", "task": e["task"], "format": e["format"],
                       "bound": bound, "n": n, "mj_per_answer": ms * n}
                if a is None:
                    why = ("on-device chain ran fewer steps; host runtime not substituted"
                           if (e["task"], prec) in has_dev else "no device-faithful accuracy for this graph yet")
                    row.update(acc=None, acc_source=why, runtime="-", mj_per_solved=None)
                else:
                    row.update(acc=a[0], acc_n=a[1], acc_source=a[2], runtime=runtime_of(a[2]),
                               mj_per_solved=(ms * n / a[0]) if a[0] > 0 else float("inf"))
                rows.append(row)
    meas = DEV / "energy_l40s.json"
    if meas.exists():
        m = json.loads(meas.read_text())
        for r in m["runs"]:
            task = {"maze_attn": "maze"}.get(r["task"], r["task"])
            if r["dtype"] != "float32":
                continue
            for n, a in sorted(acc.get((ACC_TASK[task], "fp32"), {}).items()):
                for kind in ("total", "incr"):
                    e = r["median"][f"e_{kind}_mj_per_puzzle_step"]
                    rows.append({"energy_source": f"measured NVML {m['gpu']} batch {r['batch']} {r['dtype']}",
                                 "task": task, "format": r["dtype"], "bound": kind, "n": n, "acc": a,
                                 "acc_source": "harness fp32 (the timed torch graph), n=1000",
                                 "runtime": "PyTorch CUDA (L40S)", "mj_per_answer": e * n,
                                 "mj_per_solved": e * n / a if a > 0 else float("inf")})
    (DEV / "energy_solved.json").write_text(json.dumps(rows, indent=1))
    md = ["| energy source | task | format | bound | n | exact | accuracy runtime | accuracy source | mJ/answer | mJ/solved |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        a = "-" if r["acc"] is None else f"{r['acc']*100:.1f}"
        sol = "-" if r["mj_per_solved"] is None else ("inf (0 solved)" if r["mj_per_solved"] == float("inf")
                                                       else f"{r['mj_per_solved']:.2f}")
        md.append(f"| {r['energy_source']} | {r['task']} | {r['format']} | {r['bound']} | {r['n']} | {a} | {r.get('runtime', '-')} | "
                  f"{r['acc_source']}{'' if r.get('acc_n') is None else ' (n=%d)' % r['acc_n']} | "
                  f"{r['mj_per_answer']:.2f} | {sol} |")
    (DEV / "energy_solved.md").write_text(
        "# Energy per solved instance (E_step x n / accuracy of the same graph)\n\n"
        "Accuracy is taken from the graph whose cost is counted (device-faithful): host simulation of the AI Hub "
        "compiled QDQ graph at H=1 (or the on-device chain when it exists), never the harness quantizer. "
        "Regenerated Sep 24 after the harness-accuracy version credited Sudoku-MLP w8a8/w4a8 with 61% while the "
        "compiled graphs solve 0%.\n\n" + "\n".join(md) + "\n")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="all")
    a = ap.parse_args()
    rows, comp, prof = load_status() if (AIH / "status.json").exists() else ([], {}, {})
    print(f"{len(rows)} job rows")
    if a.tables in ("all", "latency"):
        t = latency_table(comp, prof)
        print("latency rows", len(t))
    if a.tables in ("all", "gran"):
        g = gran_matrix(comp, prof)
        print("gran rows", len(g))
    if a.tables in ("all", "deadline"):
        d = deadline(comp, prof, load_acc())
        print("deadline rows", len(d))
    if a.tables in ("all", "energy"):
        e = energy_table(load_acc())
        print("energy rows", len(e))


if __name__ == "__main__":
    main()
