#!/usr/bin/env python3
"""M5 overthinking control (coordinator, reports/edgetrm_lit/08_angles_sep25.md Q1).

Literature: overthinking exists at full precision (Kaya ICML'19: destructive overthinking for 50% of fp CNN
misclassifications; Lotfi 2606.00206: 26% of BF16 errors are answer-reached-then-abandoned). The claim to test is
therefore not existence but (a) compression raises it past the basin edge as a function of fidelity, and (b) the
model's own halting head recovers it where label-free / confidence exits do not (Mehra 2212.01562: oracle exit
>= 10 points, practical exits ~1 under input shift; BitCal-TTS 2605.05561: confidence stops halt too early under NF4).

Per cell (model x quantizer / noise draw), from the per-step records (analyze_m5 loaders, K default-H steps):
  V_abs      share of instances right at some step <= K and wrong at step K ("visit then lose")
  V_rel      V_abs / (1 - acc_K): share of the final errors that were right earlier (Kaya's / Lotfi's quantity)
  oracle     accuracy of the first-correct-step exit (the ceiling of any exit rule)
  head       accuracy of the fp32-chosen halting-head rule (and native), rescues and premature halts vs step K
  label-free exits tuned on fp32 and frozen: max-softmax confidence (msp, BitCal / Kaya's rule), answer
             stability (argmax unchanged 2 steps, PABEE-style, untuned), convergence residual
  recovery   (acc_rule - acc_K) / (oracle - acc_K): share of the oracle gap a rule recovers (Mehra's framing)
  phi        final-state fidelity cos(z_H, z_H fp32): X8 detector records (same Babel trajectories) where they
             exist, else the p4d breadth records at the same H and 16 steps (deterministic quantizers only)
Also MDEQ (X6: Broyden per-budget snapshots, phi = cosine to the fp32 equilibrium at the same budget) and the
deep-thinking mazes (M5 records; phi from the p4d dt_extrap records, quantizers only).

    python m5_overthink.py            # table + results/breadth/m5_overthink.json + figures/m5_overthink.pdf
"""
import glob
import json
import os
from pathlib import Path

import numpy as np

import analyze_m5 as A

RB = A.RB
DET = Path(os.environ.get("M5_DET_DIR", RB / "babel" / "x8"))
BREADTH = {"sudoku_mlp": "sudoku_mlp_all.json", "sudoku_attn": "sudoku_attn_all.json", "hrm_sudoku": "hrm_sudoku_all.json",
           "maze_attn": "maze_attn_n816.json", "hrm_maze": "hrm_maze_n816.json"}


def phi_table(model):
    """spec -> fidelity_H at the default H, 16 steps: X8 det records first, then breadth records."""
    out = {}
    for f in sorted(glob.glob(str(DET / f"det_{model}__*.json"))):
        for r in json.load(open(f))["records"]:
            if "fid_H" in r["final"]:
                out.setdefault(r["spec"], (float(np.mean(r["final"]["fid_H"])), "X8 det (Babel)"))
    bf = RB / BREADTH[model]
    if bf.exists():
        for r in json.load(open(bf))["records"]:
            q = r["quant"] if isinstance(r["quant"], dict) else eval(r["quant"])
            if str(r["H_cycles"]) == str(A.HDEF[model]) and str(r["nsup"]) == "16":
                out.setdefault(q["spec"], (float(r["fidelity_H"]), "breadth (p4d)"))
    return out


def cell_metrics(c, ref, sel, head=True):
    n, K = c.c.shape
    fin = c.c[:, -1]
    anyc = c.c.any(1)
    m = {"n": n, "K": K, "acc": 100 * fin.mean(), "oracle": 100 * anyc.mean(), "V_abs": 100 * (anyc & ~fin).mean()}
    m["V_rel"] = 100 * (anyc & ~fin).sum() / max((~fin).sum(), 1)
    gap = m["oracle"] - m["acc"]
    rules = {"head (fp32-chosen)": ("sel", sel), "head native": ("r", ("native", None)), "stability 2": ("r", ("stab", 2))}
    if not head:
        rules = {"head (fp32-chosen)": ("sel", sel), "stability 2": ("r", ("stab", 2))}
    for nm, (kind, rule) in rules.items():
        t = A.exit_steps(c, *rule)
        if t is None:
            continue
        ok, st = A.score(c, t)
        m[nm] = {"acc": 100 * ok.mean(), "steps": float(st.mean()), "rescued": 100 * (ok & ~fin).mean(),
                 "premature": 100 * (~ok & fin).mean(), "recovery": (100 * ok.mean() - m["acc"]) / gap if gap > 0.5 else None}
    for nm, rule in (("max-softmax (tuned fp32)", "msp"), ("residual (tuned fp32)", "res")):
        if ref is None or (rule == "msp" and (c.msp is None or ref.msp is None)) or (rule == "res" and (c.res is None or ref.res is None)):
            continue
        t, prm = A.crossfit(ref, c, rule, 0.5)
        if t is None:
            continue
        ok, st = A.score(c, t)
        m[nm] = {"acc": 100 * ok.mean(), "steps": float(st.mean()), "rescued": 100 * (ok & ~fin).mean(),
                 "premature": 100 * (~ok & fin).mean(), "recovery": (100 * ok.mean() - m["acc"]) / gap if gap > 0.5 else None,
                 "params": prm}
    return m


def trm_hrm(models, K=16):
    rows = []
    for model in models:
        cells = A.cells_for(model, K)
        if "fp32" not in cells:
            continue
        ref = cells["fp32"]
        sel = A.select_rule(ref)
        phis = phi_table(model)
        for spec in sorted(cells, key=A.spec_order):
            if spec.endswith("|h32") or spec.startswith("w5") or spec.startswith("w6"):
                continue
            c = cells[spec]
            m = cell_metrics(c, ref, sel)
            ph = phis.get(spec) or (phis.get(spec.split("@s")[0]) if "@s" not in spec else None) or ((1.0, "fp32") if spec == "fp32" else None)
            m.update({"family": model, "spec": spec, "phi": ph[0] if ph else None, "phi_src": ph[1] if ph else None,
                      "src": c.src, "sel_rule": A.rule_name(*sel)})
            rows.append(m)
    return rows


def dt_rows():
    rows = []
    for size in (13, 25, 33):
        cells = A.load_dt(size)
        if "fp32" not in cells:
            continue
        phis = {}
        f = RB / "p4d" / f"dt_extrap_mazes_{size}.json"
        T = cells["fp32"].K
        if f.exists():
            for r in json.load(open(f))["records"]:
                if r["iters"] == T or (r["iters"] == 300 and T == 400) or (r["iters"] == 200 and T == 200):
                    phis.setdefault(r["quant"]["spec"], float(r["fidelity"]))
        for spec, c in sorted(cells.items(), key=lambda kv: A.spec_order(kv[0])):
            m = cell_metrics(c, None, ("stab", 10), head=False)
            m["stability 10"] = m.pop("head (fp32-chosen)")
            ph = phis.get(spec) if "@s" not in spec else None
            m.update({"family": f"dt_maze{size}", "spec": spec, "phi": 1.0 if spec == "fp32" else ph, "phi_src": "dt_extrap (p4d)"})
            rows.append(m)
    return rows


def mdeq_rows():
    rows = []
    R = RB.parent / "transfer" / "babel"
    for prefix, name in (("deq_small", "mdeq_small"), ("deq_xl", "mdeq_xl")):
        seen = set()
        for f in sorted(glob.glob(str(R / f"{prefix}_*.npz"))):
            d = np.load(f)
            B = list(d["budgets"])
            j26 = B.index(26)
            seed = Path(f).stem.split("_")[-1]
            for k in d.files:
                p = k.split("|")
                if len(p) != 3 or p[2] != "c1" or p[1] != "broyden":
                    continue
                spec = p[0] if (p[0] == "fp32" or "noise" not in p[0]) else f"{p[0]}@{seed}"
                if spec in seen:
                    continue
                seen.add(spec)
                c = d[k][:j26 + 1].T.astype(bool)                      # budgets 1..26 (the default)
                fin = c[:, -1]
                anyc = c.any(1)
                cos = d[f"{p[0]}|broyden|cos"][j26].mean() if f"{p[0]}|broyden|cos" in d.files else None
                rows.append({"family": name, "spec": spec, "n": c.shape[0], "K": 26, "acc": 100 * fin.mean(),
                             "oracle": 100 * anyc.mean(), "V_abs": 100 * (anyc & ~fin).mean(),
                             "V_rel": 100 * (anyc & ~fin).sum() / max((~fin).sum(), 1),
                             "phi": float(cos) if cos is not None else None, "phi_src": "X6 cos to fp32 equilibrium"})
    return rows


def main():
    rows = trm_hrm(["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze_attn", "hrm_maze"])
    rows += dt_rows() + mdeq_rows()
    fmt = lambda x: f"{x:6.1f}" if isinstance(x, (int, float)) and x is not None else "     -"  # noqa: E731
    fam = None
    for r in rows:
        if r["family"] != fam:
            fam = r["family"]
            print(f"\n=== {fam}: acc@K | V_abs | V_rel (share of final errors right earlier) | oracle | head native acc, recovery, "
                  f"rescued/premature | max-softmax acc, recovery, premature | stability acc, recovery | phi")
        h = r.get("head native") or r.get("stability 10") or {}
        ms = r.get("max-softmax (tuned fp32)") or {}
        st = r.get("stability 2") or {}
        rec = lambda d: f"{d['recovery']:+5.2f}" if d.get("recovery") is not None else "    -"  # noqa: E731
        print(f"{r['spec']:15s} {r['acc']:5.1f} | {r['V_abs']:5.1f} | {r['V_rel']:5.1f} | {r['oracle']:5.1f} | "
              + (f"{h['acc']:5.1f} {rec(h)} {h['rescued']:4.1f}/{h['premature']:4.1f}" if h else " " * 22) + " | "
              + (f"{ms['acc']:5.1f} {rec(ms)} {ms['premature']:4.1f}" if ms else " " * 16) + " | "
              + (f"{st['acc']:5.1f} {rec(st)}" if st else " " * 11) + f" | {fmt(r.get('phi'))}")
    json.dump(rows, open(RB / "m5_overthink.json", "w"), indent=1, default=float)
    print("\nwrote", RB / "m5_overthink.json")
    plot(rows)


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fams = ["sudoku_mlp", "sudoku_attn", "hrm_sudoku", "maze_attn", "hrm_maze", "dt_maze33", "mdeq_small", "mdeq_xl"]
    cols = dict(zip(fams, ["C3", "C0", "C2", "C1", "C4", "C5", "C7", "C8"]))
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for r in rows:
        if r["family"] not in cols or r.get("phi") is None:
            continue
        x = r["phi"]
        ax[0].scatter(x, r["V_rel"], s=14, color=cols[r["family"]], alpha=0.8)
        ax[1].scatter(x, r["V_abs"], s=14, color=cols[r["family"]], alpha=0.8)
        h = r.get("head (fp32-chosen)") or r.get("stability 10")
        if h and h.get("recovery") is not None and r["oracle"] - r["acc"] >= 2:
            ax[2].scatter(x, h["recovery"], s=16, color=cols[r["family"]], marker="o")
            ms = r.get("max-softmax (tuned fp32)")
            if ms and ms.get("recovery") is not None:
                ax[2].scatter(x, ms["recovery"], s=16, color=cols[r["family"]], marker="x")
    ax[0].set_ylabel("final errors that were right earlier (%)")
    ax[1].set_ylabel("visit-then-lose, share of all instances (%)")
    ax[2].set_ylabel("share of oracle gap recovered (o head/stability, x max-softmax)")
    for a in ax:
        a.set_xlabel("fidelity phi (cos to fp32 final state)")
        a.grid(alpha=0.3)
    for f in fams:
        ax[0].scatter([], [], color=cols[f], label=f)
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    out = RB.parent.parent / "figures" / "m5_overthink.pdf"
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
