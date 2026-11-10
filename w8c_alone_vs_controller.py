#!/usr/bin/env python3
"""Reviewer question: what does the w8c model reach BY ITSELF under the same halting-head rule as Table 3?

Same 20 Sudoku compressed cells, dev rows 0-999 (selection) and held-out rows 1000-1999 (every number), halting-head
rule frozen on the fp32 dev trajectory (C.rules_for), budget 64 in compressed-loop units. A w8c step costs r = 8/b_eff
compressed-loop units (an fp32 step 32/b_eff). Configs: fixed depth D0, or head stop capped at D0, D0 on the Table 3 grid.
Per budget the config with the best dev accuracy at dev cost <= budget (ties: lower cost), scored held out.
Existing rows (fixed / stop / controller E+P w8c) are re-derived with the repo pipeline (C.Split + AD.select).
Records: results/m5/ctl/p4d (existing), results/m5/ctl/p4d_w8c (w8c det records from exit_sweep.py --specs w8c).

    python w8c_alone_vs_controller.py [--budget 64]
"""
import argparse, json, os
from pathlib import Path
import numpy as np
import analyze_m5 as A, m5_addons as AD, m5_ctl_ho as C
from m5_controller import fire_mask, first

W8 = Path(os.environ.get("W8C_DIR", A.HERE / "results" / "m5" / "ctl" / "p4d_w8c"))
D0S = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)


def alone(det_dev, det_ho, rule, hrm, r, bud):
    """det_*: (correct, qh, qc) at default-H steps. returns {mode: dict}"""
    out = {}
    T = min(det_dev[0].shape[1], det_ho[0].shape[1])
    def ev(d, mode, D0):
        c, q, qc = d
        n = len(c)
        if mode == "fixed":
            return c[:, D0 - 1], np.full(n, r * D0)
        td = first(fire_mask(q, qc if hrm else None, rule, hrm))
        t = np.where((td >= 0) & (td < D0), td, D0 - 1)
        return c[np.arange(n), t], r * (t + 1.0)
    for mode in ("fixed", "stop"):
        best = None
        for D0 in [d for d in D0S if d <= T]:
            a, c = ev(det_dev, mode, D0)
            if c.mean() <= bud + 1e-9:
                k = (a.mean(), -c.mean())
                if best is None or k > best[0]:
                    best = (k, D0)
        if best is None:
            out[mode] = None
            continue
        a, c = ev(det_ho, mode, best[1])
        out[mode] = {"D0": best[1], "acc_i": a, "cost": float(c.mean()), "dev_acc": 100 * best[0][0]}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--budget", type=int, default=64); ap.add_argument("--json", default=str(A.HERE / "results" / "m5" / "w8c_alone.json"))
    a = ap.parse_args(); bud = a.budget
    rng = np.random.default_rng(0)
    rows = []
    for fam in C.families():
        if fam["kind"] == "maze":
            continue
        task, sub = fam["task"], fam.get("sub", fam["task"])
        fp = C.cell_splits(task, "fp32", fam)
        rule, rrule = C.rules_for(fp[0][0], task)
        hrm = task.startswith("hrm")
        for spec in fam["cells"]:
            if spec == "fp32" or "noise" in spec:
                continue
            be = AD.b_eff(spec)
            res = {}
            for pol, rp in (("fp32", 32.0 / be), ("w8c", 8.0 / be)):
                C.POL = pol
                try:
                    det, m2, pc, h = C.cell_splits(task, spec, fam)
                finally:
                    C.POL = "fp32"
                dev, ho = [C.Split(*det[k], *m2[k], pc[k], h, rule, rrule) for k in (0, 1)]
                for m in ("fixed", "E", "E+P"):
                    if pol == "w8c" and m != "E+P":
                        continue
                    res[(m, pol)] = AD.select(dev, ho, dev.configs(m), rp, rp).get(bud)
            dw = [C.load_det(W8 / f"det_o{o}" / sub / "w8c.npz") for o in (0, 1000)]
            d32 = [C.load_det(C.ROOT / f"det_o{o}" / sub / "fp32.npz") for o in (0, 1000)]
            w = alone(dw[0], dw[1], rule, hrm, 8.0 / be, bud)
            f = alone(d32[0], d32[1], rule, hrm, 32.0 / be, bud)
            rows.append(dict(model=fam["name"], spec=spec, be=be, res=res, w8=w, fp=f))
    def m(x): return 100 * x.mean()
    print(f"budget {bud} compressed-loop units; held-out rows 1000-1999; cost = compressed-loop units; bits = bits read per weight per puzzle (cost x b_eff)")
    print(f"{'model':11s} {'spec':7s} {'be':>4s} | fixed acc@cost | stop acc@cost | ctl(E+P w8c) acc@cost | w8c fixed acc@cost(D0) | w8c stop acc@cost(D0) | fp32 stop acc@cost | bits: fixed stop ctl w8cfix w8cstop")
    agg = {k: [] for k in ("fixed", "stop", "ctl", "w8fix", "w8stop", "fp32fix", "fp32stop")}
    for x in rows:
        R = x["res"]; be = x["be"]
        e = {"fixed": R[("fixed", "fp32")], "stop": R[("E", "fp32")], "ctl": R[("E+P", "w8c")]}
        v = {"fixed": (e["fixed"]["acc_i"], e["fixed"]["cost"]), "stop": (e["stop"]["acc_i"], e["stop"]["cost"]), "ctl": (e["ctl"]["acc_i"], e["ctl"]["cost"]),
             "w8fix": (x["w8"]["fixed"]["acc_i"], x["w8"]["fixed"]["cost"]), "w8stop": (x["w8"]["stop"]["acc_i"], x["w8"]["stop"]["cost"]),
             "fp32fix": (x["fp"]["fixed"]["acc_i"], x["fp"]["fixed"]["cost"]) if x["fp"]["fixed"] else None,
             "fp32stop": (x["fp"]["stop"]["acc_i"], x["fp"]["stop"]["cost"]) if x["fp"]["stop"] else None}
        x["v"] = v
        for k in agg:
            agg[k].append(v[k])
        f = lambda t: f"{m(t[0]):5.1f}@{t[1]:5.1f}" if t else "   n/a   "
        print(f"{x['model']:11s} {x['spec']:7s} {be:4.1f} | {f(v['fixed'])} | {f(v['stop'])} | {f(v['ctl'])} | {f(v['w8fix'])}({x['w8']['fixed']['D0']}) | {f(v['w8stop'])}({x['w8']['stop']['D0']}) | {f(v['fp32stop'])} | "
              + " ".join(f"{t[1]*be:6.0f}" for t in (v['fixed'], v['stop'], v['ctl'], v['w8fix'], v['w8stop'])))
    print(f"\nMEANS over {len(rows)} models (acc, cost units, bits/weight/puzzle):")
    summary = {}
    for k, l in agg.items():
        l2 = [t for t in l if t]
        acc = np.mean([m(t[0]) for t in l2]); cost = np.mean([t[1] for t in l2])
        bits = np.mean([t[1] * x["be"] for t, x in zip(l, rows) if t])
        summary[k] = dict(n=len(l2), acc=acc, cost=cost, bits=bits)
        print(f"  {k:9s} n={len(l2):2d} acc {acc:5.1f} cost {cost:5.1f} bits {bits:7.1f}")
    def pooled(k1, k2):
        ds = [t1[0].astype(float) - t2[0] for t1, t2 in zip(agg[k1], agg[k2])]
        bs = np.mean([d[rng.integers(0, len(d), (2000, len(d)))].mean(1) for d in ds], 0)
        return 100 * np.mean([d.mean() for d in ds]), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
    diffs = {}
    for k1, k2 in (("ctl", "w8stop"), ("ctl", "w8fix"), ("w8stop", "w8fix"), ("ctl", "fixed"), ("w8stop", "fixed")):
        diffs[f"{k1}-{k2}"] = pooled(k1, k2); print(f"  {k1}-{k2}: {diffs[f'{k1}-{k2}'][0]:+.2f} [{diffs[f'{k1}-{k2}'][1]:+.2f},{diffs[f'{k1}-{k2}'][2]:+.2f}] (pooled paired bootstrap)")
    print("  stored bits/weight: controller = b_eff + 8, mean", np.mean([x["be"] + 8 for x in rows]), "; w8c alone = 8; compressed alone = b_eff mean", np.mean([x["be"] for x in rows]), "; fp32 = 32")
    json.dump(dict(budget=bud, summary=summary, diffs=diffs, per_model=[dict(model=x["model"], spec=x["spec"], b_eff=x["be"],
        **{k: ({"acc": m(t[0]), "cost": t[1], "bits": t[1] * x["be"]} if t else None) for k, t in x["v"].items()},
        w8_D0={k: x["w8"][k]["D0"] for k in ("fixed", "stop")}) for x in rows]), open(a.json, "w"), indent=1, default=float)
    print("wrote", a.json)


if __name__ == "__main__":
    main()
