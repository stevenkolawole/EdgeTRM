#!/usr/bin/env python3
"""E4: planner transfer across chips (reports/edgetrm_exp/E4_PLANNER_TRANSFER.md), acct2 + acct3, QAIRT pinned at
2.45 (aihub_acct.QAIRT), encoding guard on every chain. Tokens are never read or printed here:
qai_hub.Client(profile=...) loads them. Runs on p4d from ~/x3.

The two Sudoku-MLP plans of the paper table (qdq_w8a16/variant_m3_greedy_{40,60}_legal.onnx: 40 / 60 activation
tensors at 16 bit in reverse-greedy delta_T order, both Hexagon legality rules applied) are rebuilt with the RMSNorm
pattern broken at export (aihub_e3_followup.break_rmsnorm: ReduceMean -> ReduceSum * 1/N, both quantized, N from
shape inference: 97 for the 14 token-mixing norms, 512 for the 14 channel norms, as in the E3 nofuse2 graph) so the qairt-converter fuses nothing and nothing falls back to float:
the DLCs stay all-integer and compose on the no-fp16 parts. Prec tags m3g40_nofuse2 / m3g60_nofuse2. The Maze
"plan" is the all-integer w8a8 nofuse graph (no device-legal Maze plan exists), prec w8a8_nofuse2, and the
all-integer vendor default (sudoku_mlp w8a8_nofuse2) is probed on the S25.

    python aihub_e4_planner.py build                       # the two nofuse plan graphs (no hub)
    python aihub_e4_planner.py hostref sudoku_mlp:m3g60_nofuse2 [--threads 12]   # reference host column, one per process
    python aihub_e4_planner.py compile [acct2|acct3] [task:prec ...]
    python aihub_e4_planner.py run --steps 2 [--only tag ...]  # every chain in CHAINS, gate first
    python aihub_e4_planner.py run --steps 8 --only sudoku_mlp:m3g60_nofuse2:S25@acct2 ...
    python aihub_e4_planner.py status
    python aihub_e4_planner.py table > table.md            # per-device table for the report
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_e3_followup as E  # noqa: E402  (installs its source_path override on aihub_acct; SRC / CHAINS are extended here)
# Sep 29 (E3_GATE_PHI_STEP2): chains run through aihub_acct.Chain, which now keeps every step's carry (zH_steps / zL_steps).
import aihub_infer as I  # noqa: E402

S23, S25, G7 = E.S23, E.S25, E.G7
XE, IQ = "Snapdragon X Elite CRD", "Dragonwing IQ-9075 EVK"
E.SHORT.update({XE: "XElite", IQ: "IQ9075"})
HEX = {S23: "v73", S25: "v79", XE: "v73", IQ: "v73", G7: "v73 (no fp16 HTP)"}
PLANS = {"m3g40_nofuse2": ("qdq_w8a16", "variant_m3_greedy_40_legal.onnx", "variant_m3_greedy_40_legal_nofuse2.onnx", 40),
         "m3g60_nofuse2": ("qdq_w8a16", "variant_m3_greedy_60_legal.onnx", "variant_m3_greedy_60_legal_nofuse2.onnx", 60)}
for p, (base, _, out, _) in PLANS.items():
    E.SRC[("sudoku_mlp", p)] = (base, out)
# (acct, task, prec, device): the Sudoku plans on all five parts, the Maze all-integer graph on the three v73 parts
# without a row (S23 / S25 rows exist from E3_FOLLOWUP), the all-integer vendor default on the S25
CHAINS = [(a, "sudoku_mlp", p, d) for p in PLANS for a, d in
          (("acct2", S23), ("acct2", S25), ("acct3", XE), ("acct3", IQ), ("acct2", G7))] \
    + [("acct3", "maze", "w8a8_nofuse2", XE), ("acct3", "maze", "w8a8_nofuse2", IQ), ("acct2", "maze", "w8a8_nofuse2", G7)] \
    + [("acct2", "sudoku_mlp", "w8a8_nofuse2", S25)]
E.CHAINS[:] = CHAINS


def build():
    import onnx
    from onnx import numpy_helper
    d = I.idir("sudoku_mlp")
    for p, (base, src, out, k) in PLANS.items():
        o = d / base / out
        if o.exists():
            print(p, "exists:", o, flush=True)
            continue
        m = onnx.load(str(d / base / src), load_external_data=False)
        n = E.break_rmsnorm(m)
        onnx.save(m, str(o))
        Ns = sorted({int(round(1.0 / float(numpy_helper.to_array(i)))) for i in m.graph.initializer if i.name.endswith("_invN")})
        print(p, "rewrote", n, "ReduceMean nodes; N =", Ns, "->", o, flush=True)
        assert Ns == [97, 512], Ns          # 14 token-mixing norms over 97, 14 channel norms over 512 (= the E3 nofuse2 graph)
    for p in PLANS:
        print(p, "graph output encodings:", json.dumps(E.X.out_enc("sudoku_mlp", p)), flush=True)


def load(task, name):
    f = I.idir(task) / f"{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def host_ref(task, prec):
    """reference series [(exact, cell)] of the graph's host run, from whichever record exists."""
    d = I.idir(task)
    for f in (d / f"host_{prec}_ref.json", d / f"host_{prec}_ref.16.json"):
        if f.exists():
            return [(s["exact"], s["cell"]) for s in json.loads(f.read_text())["steps"]]
    m = {"m3g40_nofuse2": "m3_greedy_40_legal", "m3g60_nofuse2": "m3_greedy_60_legal"}.get(prec)
    if m and (d / "ablate" / f"eval_{m}_0_100_ref.json").exists():
        r = json.loads((d / "ablate" / f"eval_{m}_0_100_ref.json").read_text())
        return list(zip(r["exact_per_step"], r["cell_per_step"]))
    return None


def dev_row(task, prec, dev, acct=None):
    d = I.idir(task)
    fs = sorted(d.glob(f"dev_{prec}_{dev.replace(' ', '_')}_{acct or '*'}.json"))
    fs = [f for f in fs if f.stem[4 + len(prec) + 1:].startswith(dev.replace(" ", "_"))]
    if not fs and acct is None:
        fs = [f for f in [d / f"dev_{prec}_{dev.replace(' ', '_')}.json"] if f.exists()]
    if not fs:
        return None
    m = json.loads(fs[-1].read_text())
    m["_file"] = fs[-1].name
    return m


def fmt(series, k=8):
    return " ".join(f"{100 * e:.0f}/{100 * c:.1f}" for e, c in series[:k])


def verdicts(dev, host):
    """probe = device / host cell at step 2; verdict = device / host exact at step 8 (E3 cuts: PASS >= 0.95, FAIL < 0.5)."""
    if not dev or not host or len(dev) < 2:
        return "-", "-"
    pr = E.ratio(dev[1][1], host[1][1])
    pv = f"{pr:.2f} {E.verdict(pr)}" if pr is not None else "-"
    if len(dev) < 8:
        return pv, f"{len(dev)} steps"
    vr = E.ratio(dev[7][0], host[7][0])
    return pv, (f"{vr:.2f} {E.verdict(vr)}" if vr is not None else "n/a (host 0)")


def table():
    for dev in (S23, S25, XE, IQ, G7):
        print(f"\n### {dev} ({HEX[dev]})\n")
        print("| row | graph / DLC | steps | exact / cell per step | probe (cell s2 / host) | verdict (exact s8 / host) |\n|---|---|---|---|---|---|")
        for task in ("sudoku_mlp", "maze"):
            n = 100 if task == "sudoku_mlp" else 50
            fp = dev_row(task, "fp32", dev)
            if fp and fp.get("steps"):
                print(f"| {task} device fp32 (fp16 HTP) | stepc_H1 fp32 | {len(fp['steps'])} | {fmt([(s['exact'], s['cell']) for s in fp['steps']])} | | |")
            elif fp:
                print(f"| {task} device fp32 | stepc_H1 fp32 | 0 | refused: {(fp.get('failed') or '')[:60]} | | |")
            for prec, label in (("w8a8", "vendor default, fused DLC (RMSNorm fp16 islands, fp16 carry)"),
                                ("w8a8_nofuse2", "vendor default, all-integer" if task == "sudoku_mlp" else "all-integer w8a8 (the Maze 'plan')"),
                                ("m3g40_nofuse2", "planner legal@40, all-integer"), ("m3g60_nofuse2", "planner legal@60, all-integer")):
                if task == "maze" and prec.startswith("m3"):
                    continue
                m = dev_row(task, prec, dev)
                h = host_ref(task, prec if prec != "w8a8" else "w8a8")
                if m is None:
                    continue
                st = [(s["exact"], s["cell"]) for s in m.get("steps", [])]
                if m.get("failed") and not st:
                    print(f"| {task} {label} | {prec} | 0 | refused: {m['failed'][:70]} | - | - |")
                    continue
                pv, vv = verdicts(st, h)
                conv = (m.get("converter") or {}).get("converter") or "unpinned qairt path (Sep 25)"
                print(f"| {task} {label} | {prec} ({m['_file'].split('_')[-1][:-5]}, {conv}, guard {(m.get('guard') or {}).get('verdict', '-')}) | {len(st)} | {fmt(st)} | {pv} | {vv} |")
            for prec in ("w8a8_nofuse2", "m3g40_nofuse2", "m3g60_nofuse2"):
                if task == "maze" and prec.startswith("m3"):
                    continue
                h = host_ref(task, prec)
                if h:
                    print(f"| {task} host reference ({prec}) | ORT reference QDQ | {len(h)} | {fmt(h)} | | |")
            print(f"| | | | (exact % / cell %, n = {n}) | | |")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "hostref", "compile", "run", "status", "table"])
    ap.add_argument("names", nargs="*")
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--slots", type=int, default=0)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=[])
    a = ap.parse_args()
    {"build": build, "hostref": lambda: E.hostref(a.names, a.threads, True), "compile": lambda: E.compile_(a.names),
     "run": lambda: E.run(a.only, a.steps, a.slots), "status": E.status, "table": table}[a.cmd]()
