#!/usr/bin/env python3
"""AI Hub device-breadth read-out on the pinned (QAIRT 2.45) records: per-graph device tables, the fault table,
the two-step gate proxy and the Maze RoPE fault by Hexagon generation. Rewrite of the READOUT_SEP29 section-1
scratchpad script (s1_devices.py) against the Sep 29 rerun; prints Markdown for reports/edgetrm_exp/E3_FAULT_TABLE.md.

Inputs (all in EdgeTRM/results/devices/):
  device_faithful_acc.json   rows {task, prec, step, exact, cell, count, source} from device_acc.py on p4d
                             (dev_*.json.qairt250 records are not globbed, so no QAIRT 2.50 device row is present)
  breadth_chain_meta.json    one entry per chain record on p4d: compile job(s), QAIRT version, guard verdict, failure text
  aihub_devices_sep29.json   AI Hub device attributes (chipset, hexagon:vNN, htp-supports-fp16, os) listed Sep 29

Host reference: the host simulation of the compiled fp32 graph (ORT), as release_check.device_probe uses the fp32
model as the reference; the same-precision host simulation is shown as a second reference row where it exists.
Gate proxy (record-level; the coded gate needs the per-puzzle npz on p4d): probe = cell s2 / host-fp32 cell s2,
verdict = exact s8 / host-fp32 exact s8, both cut at PASS >= 0.95 and FAIL < 0.5 (release_check THRESHOLDS
written_pass / device_written_pass), WARN between.
"""
import argparse
import collections
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent
DEV = HERE / "results" / "devices"
GRAPHS = [("sudoku_mlp", "fp32"), ("sudoku_mlp", "w8a8"), ("sudoku_mlp", "w8a16"), ("sudoku_attn", "fp32"),
          ("sudoku_attn", "w8a8"), ("sudoku_attn", "w8a16"), ("maze", "fp32"), ("maze", "w8a8"), ("maze", "w8a8rope16")]
BREADTH = {("sudoku_mlp", "fp32"), ("sudoku_mlp", "w8a8"), ("sudoku_mlp", "w8a16"), ("sudoku_attn", "w8a8"),
           ("maze", "fp32"), ("maze", "w8a8"), ("maze", "w8a8rope16")}
HOST = {"sudoku_mlp": "host sim of compiled QDQ graph, test 0-99", "sudoku_attn": "host sim of compiled QDQ graph, test 0-99",
        "maze": "host sim of compiled QDQ graph, test 0-49"}
# devices no longer hosted on Sep 29 (dropped from the hub list); silicon from the device name and DIAG_ACCT_SPLIT.md
UNHOSTED = {"Snapdragon 8 Elite QRD": {"chipset": ["qualcomm-snapdragon-8-elite (SM8750, same silicon as the S25)"], "hexagon": "v79",
                                       "os": "15", "os_kind": "android", "htp_fp16": "true", "note": "unhosted since ~Sep 28"},
            "Snapdragon 8 Elite Gen 5 QRD": {"chipset": ["qualcomm-snapdragon-8-elite-gen-5 (by name; attributes not listed)"], "hexagon": "?",
                                             "os": "?", "os_kind": "android", "htp_fp16": "?", "note": "unhosted since Sep 29; Hexagon generation not read from the hub"}}
PASS, FAIL = 0.95, 0.5


def verdict(ratio):
    return "PASS" if ratio >= PASS else ("FAIL" if ratio < FAIL else "WARN")


def load():
    rec = json.load(open(DEV / "device_faithful_acc.json"))
    meta = json.load(open(DEV / "breadth_chain_meta.json"))["chains"]
    devs = json.load(open(DEV / "aihub_devices_sep29.json"))["devices"]
    devs = {**{k: dict(v) for k, v in devs.items()}, **UNHOSTED}
    tab = collections.defaultdict(dict)   # (task, prec, source) -> step -> row
    for r in rec:
        tab[(r["task"], r["prec"], r["source"])][r["step"]] = r
    return tab, {(m["task"], m["prec"], m["source"]): m for m in meta if not m["set_aside"]}, devs, meta


def short(v):
    return "2.45" if v.startswith("2.45") else ("2.50" if v.startswith("2.50") else v)


def dev_label(m):
    return f"{m['device']} ({m['acct']})"


def pct(x):
    return f"{100 * x:.0f}" if x is not None else "-"


def pct1(x):
    return f"{100 * x:.1f}" if x is not None else "-"


def steps(rows):
    return [rows.get(s, {}) for s in (1, 2, 8)]


def section_tables(tab, meta, devs):
    out = []
    for task, prec in GRAPHS:
        host = tab.get((task, "fp32", HOST[task]))
        hsame = tab.get((task, prec, HOST[task])) if prec != "fp32" else None
        chains = sorted([k for k in tab if k[0] == task and k[1] == prec and k[2].startswith("ON DEVICE")],
                        key=lambda k: (meta.get(k, {}).get("acct", ""), k[2]))
        if not chains:
            continue
        out.append(f"\n#### {task} {prec}\n")
        out.append("| device | chip | Hexagon | OS | runtime | acct | QAIRT | guard | exact s1 | s2 | s8 | cell s1 | s2 | s8 |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        h = steps(host)
        out.append(f"| host fp32 graph (ORT) | - | - | - | onnxruntime | - | - | - | {pct(h[0].get('exact'))} | {pct(h[1].get('exact'))} | {pct(h[2].get('exact'))} | {pct1(h[0].get('cell'))} | {pct1(h[1].get('cell'))} | {pct1(h[2].get('cell'))} |")
        if hsame:
            h = steps(hsame)
            out.append(f"| host {prec} graph (ORT) | - | - | - | onnxruntime | - | - | - | {pct(h[0].get('exact'))} | {pct(h[1].get('exact'))} | {pct(h[2].get('exact'))} | {pct1(h[0].get('cell'))} | {pct1(h[1].get('cell'))} | {pct1(h[2].get('cell'))} |")
        elif prec != "fp32":
            out.append(f"| host {prec} graph (ORT) | no record | | | | | | | | | | | | |")
        for k in chains:
            m = meta.get(k)
            if m is None:
                out.append(f"| {k[2]} | (no chain meta) | | | | | | | | | | | | |")
                continue
            d = devs.get(m["device"], {})
            s = steps(tab[k])
            q = " / ".join(sorted({short(v) for v in m["qairt"]})) or "?"
            g = m["guard"] or ("n/a (float graph)" if prec == "fp32" else "pre-guard record")
            chip = (d.get("chipset") or ["?"])[0].replace("qualcomm-", "")
            out.append(f"| {dev_label(m)} | {chip} | {d.get('hexagon', '?')} | {d.get('os_kind', '?')} {d.get('os', '?')} | {m['runtime']} | {m['acct']} | {q} | {g} | "
                       f"{pct(s[0].get('exact'))} | {pct(s[1].get('exact'))} | {pct(s[2].get('exact'))} | {pct1(s[0].get('cell'))} | {pct1(s[1].get('cell'))} | {pct1(s[2].get('cell'))} |")
    return "\n".join(out)


def gate(tab, meta, devs, only_245=True):
    rows = []
    for k, m in meta.items():
        task, prec, src = k
        if (task, prec) not in BREADTH and (task, prec) not in {("sudoku_attn", "fp32"), ("sudoku_attn", "w8a16")}:
            continue
        if k not in tab or 2 not in tab[k] or 8 not in tab[k]:
            continue
        if only_245 and not all(v.startswith("2.45") for v in m["qairt"]):
            continue
        host = tab[(task, "fp32", HOST[task])]
        probe = tab[k][2]["cell"] / host[2]["cell"]
        s8 = tab[k][8]["exact"] / host[8]["exact"]
        rows.append(dict(task=task, prec=prec, device=dev_label(m), hexagon=devs.get(m["device"], {}).get("hexagon", "?"),
                         probe=probe, s8=s8, pv=verdict(probe), vv=verdict(s8), qairt=" / ".join(sorted({short(v) for v in m["qairt"]}))))
    return rows


def rope_table(tab, meta, devs):
    out = ["| device | Hexagon | chip | acct | QAIRT | z_L carry (guard) | maze w8a8 exact s8 / cell s8 | maze w8a8rope16 exact s8 / cell s8 |", "|---|---|---|---|---|---|---|---|"]
    by_dev = collections.defaultdict(dict)
    for k, m in meta.items():
        if k[0] == "maze" and k[1] in ("w8a8", "w8a8rope16") and k in tab and 8 in tab[k]:
            by_dev[(m["device"], m["acct"])][k[1]] = (tab[k][8]["exact"], tab[k][8]["cell"], m)
    order = sorted(by_dev, key=lambda x: (str(devs.get(x[0], {}).get("hexagon", "?")), x[0], x[1]))
    for dev, acct in order:
        cells = by_dev[(dev, acct)]
        d = devs.get(dev, {})
        m = cells.get("w8a8rope16", cells.get("w8a8"))[2]
        q = " / ".join(sorted({short(v) for v in m["qairt"]}))
        gl = m.get("guard_zL_dlc_dtype") or ("float16 (2.45 drops the output Q/DQ; DIAG_ACCT_SPLIT.md)" if "w8a8" in cells and not cells.get("w8a8rope16") else "-")
        if "w8a8rope16" in cells:
            mr = cells["w8a8rope16"][2]
            gl = mr.get("guard_zL_dlc_dtype") or "uint8 (jgzlzmjz5, graph encoding; DIAG_ACCT_SPLIT.md)"
        w = cells.get("w8a8"); r = cells.get("w8a8rope16")
        out.append(f"| {dev} | {d.get('hexagon', '?')} | {(d.get('chipset') or ['?'])[0].replace('qualcomm-', '')} | {acct} | {q} | {gl} | "
                   f"{pct(w[0]) + ' / ' + pct1(w[1]) if w else '-'} | {pct(r[0]) + ' / ' + pct1(r[1]) if r else '-'} |")
    return "\n".join(out)


def fault_rows(meta_all, devs):
    """every chain record that failed, with its error text, grouped by device and class"""
    rows = []
    for m in meta_all:
        if m["failed"] and (m["task"], m["prec"]) in BREADTH:
            f = m["failed"]
            cls = ("MODEL_GRAPH_ERROR" if "MODEL_GRAPH_ERROR" in f else "GPU delegate" if "TfLiteGpuDelegateV2" in f else
                   "segfault" if "segmentation fault" in f else "memory" if "memory usage" in f else "INVALID_CONFIG" if "INVALID_CONFIG" in f else "other")
            rows.append(dict(task=m["task"], prec=m["prec"], device=m["device"], acct=m["acct"], set_aside=m["set_aside"],
                             qairt=" / ".join(sorted({short(v) for v in m["qairt"]})), cls=cls, guard=m["guard"], text=f[:160]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-versions", action="store_true", help="gate view over every device row (adds the acct2 fp32 rows compiled on 2.50)")
    a = ap.parse_args()
    tab, meta, devs, meta_all = load()
    print("## Device tables\n")
    print(section_tables(tab, meta, devs))
    print("\n## Gate proxy" + (" (every row)" if a.all_versions else " (QAIRT 2.45 rows only)") + "\n")
    rows = gate(tab, meta, devs, only_245=not a.all_versions)
    conf = collections.Counter((r["pv"], r["vv"]) for r in rows)
    print("| probe (cell s2) | step-8 PASS | step-8 WARN | step-8 FAIL | total |\n|---|---|---|---|---|")
    for pv in ("PASS", "WARN", "FAIL"):
        print(f"| {pv} | " + " | ".join(str(conf[(pv, vv)]) for vv in ("PASS", "WARN", "FAIL")) + f" | {sum(conf[(pv, vv)] for vv in ('PASS', 'WARN', 'FAIL'))} |")
    print(f"| total | " + " | ".join(str(sum(conf[(pv, vv)] for pv in ('PASS', 'WARN', 'FAIL'))) for vv in ("PASS", "WARN", "FAIL")) + f" | {len(rows)} |")
    print("\nmisses (probe verdict != step-8 verdict):\n")
    print("| task | graph | device | Hexagon | QAIRT | probe ratio | probe | s8 ratio | s8 |\n|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["task"], r["prec"], r["device"])):
        if r["pv"] != r["vv"]:
            print(f"| {r['task']} | {r['prec']} | {r['device']} | {r['hexagon']} | {r['qairt']} | {r['probe']:.2f} | {r['pv']} | {r['s8']:.2f} | {r['vv']} |")
    print("\nprobe-PASS chains that FAIL at step 8:", sum(1 for r in rows if r["pv"] == "PASS" and r["vv"] == "FAIL"))
    print("step-8 FAIL chains flagged (WARN or FAIL) at step 2:", sum(1 for r in rows if r["vv"] == "FAIL" and r["pv"] != "PASS"), "/", sum(1 for r in rows if r["vv"] == "FAIL"))
    print("\n## Maze quantized graphs by Hexagon generation\n")
    print(rope_table(tab, meta, devs))
    print("\n## Failed chain records\n")
    fr = fault_rows(meta_all, devs)
    print("| device | Hexagon | htp fp16 | acct | graph | QAIRT | set aside | class | text |\n|---|---|---|---|---|---|---|---|---|")
    for r in sorted(fr, key=lambda r: (r["device"], r["set_aside"], r["task"], r["prec"])):
        d = devs.get(r["device"], {})
        print(f"| {r['device']} | {d.get('hexagon', '?')} | {d.get('htp_fp16', '-')} | {r['acct']} | {r['task']} {r['prec']} | {r['qairt']} | {int(r['set_aside'])} | {r['cls']} | {r['text']} |")
    print("\nclass counts (current records / set-aside 2.50 stubs):", collections.Counter((r["cls"], r["set_aside"]) for r in fr))
    print("\n## Guard verdicts on current records\n", collections.Counter((m["guard"], m["guard_zL_dlc_dtype"]) for m in meta.values() if (m["task"], m["prec"]) in BREADTH))


if __name__ == "__main__":
    main()
