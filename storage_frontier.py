#!/usr/bin/env python3
"""Storage frontier of the controller with 4-bit finishing (Yfef Q1 follow-up).

Records: results/m5/ctl/p4d (Table 3), p4d_w8c (w8c det), p4d_w4 (w4c / w4g32 polish records, det records of extra single formats).
Same pipeline as Table 3 / w8c_alone_vs_controller.py: halting rule frozen on the fp32 dev trajectory, configuration chosen on dev
rows 0-999, scored on held-out rows 1000-1999. Cost unit = one step of the cheap model; a finishing step costs r = b_fin / b_eff(cheap).
Traffic = bits read per weight per puzzle (cost x b_eff(cheap)); a single-format step costs b_eff(format) bits. Stored bits per weight:
single format = b_eff(format); controller = b_eff(cheap) + b(finishing copy) (w4c 4, w4g32 4.5, w8c 8).
  Part 1: Table 3 analogue at budgets 16/24/32/64 cheap-loop units (20 cells). Part 2: frontier per stored cap S and traffic cap Tb (bits).
    python storage_frontier.py [--out results/m5/storage_frontier]
"""
import argparse, json, os, shutil, sys, tempfile
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
RES = H / "results" / "m5" / "ctl"
MERGED = Path(tempfile.mkdtemp(prefix="m5merge_"))
for src in ("p4d", "p4d_w8c", "p4d_w4"):                        # symlink tree: first source wins on a path clash
    for f in (RES / src).rglob("*"):
        if f.is_file() and not f.name.endswith((".log",)):
            dst = MERGED / f.relative_to(RES / src)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True); os.symlink(f, dst)
os.environ["M5_CTL_DIR"] = str(MERGED)
sys.path.insert(0, str(H))
import analyze_m5 as A, m5_addons as AD, m5_ctl_ho as C
from m5_controller import fire_mask, first

D0S = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
FIN = {"w4c": 4.0, "w4g32": 4.5, "w8c": 8.0}
SINGLES = ("w2g32", "w3c", "w3g32", "w4t", "w4c", "w4g32", "w6c", "w8c")
NB = 2000
rng = np.random.default_rng(0)


def alone(dd, dh, rule, hrm, r, bud):
    """single format, halting-head stop or fixed depth, D0 chosen on dev under traffic <= bud (units of r)."""
    out = {}
    T = min(dd[0].shape[1], dh[0].shape[1])
    def ev(d, mode, D0):
        c, q, qc = d; n = len(c)
        if mode == "fixed":
            return c[:, D0 - 1], np.full(n, r * D0)
        td = first(fire_mask(q, qc if hrm else None, rule, hrm))
        t = np.where((td >= 0) & (td < D0), td, D0 - 1)
        return c[np.arange(n), t], r * (t + 1.0)
    for mode in ("fixed", "stop"):
        best = None
        for D0 in [d for d in D0S if d <= T]:
            a, c = ev(dd, mode, D0)
            if c.mean() <= bud + 1e-9:
                k = (a.mean(), -c.mean())
                if best is None or k > best[0]:
                    best = (k, D0)
        if best is not None:
            a, c = ev(dh, mode, best[1])
            out[mode] = dict(dev=best[0][0], acc=a.astype(float), cost=c, D0=best[1])
    return out


def pooled(ds):
    """ds: list of per-model per-puzzle difference arrays -> mean over models, 95% interval of the pooled mean (paired bootstrap per model)."""
    bs = np.mean([d[rng.integers(0, len(d), (NB, len(d)))].mean(1) for d in ds], 0)
    return 100 * np.mean([d.mean() for d in ds]), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


def load_model(fam):
    task, sub = fam["task"], fam.get("sub", fam["task"])
    hrm = task.startswith("hrm")
    fp = C.cell_splits(task, "fp32", fam)
    rule, rrule = C.rules_for(fp[0][0], task)
    singles = {}
    for s in SINGLES:
        d = [C.load_det(C.ROOT / f"det_o{o}" / sub / f"{s}.npz") for o in (0, 1000)]
        if all(x is not None for x in d):
            singles[s] = d
    ctl = {}                                                   # (cheap, fin) -> (dev Split, ho Split)
    for cheap in fam["cells"]:
        if cheap == "fp32" or "noise" in cheap:
            continue
        for fin in FIN:
            if fin == cheap:                                    # finishing with the same format is the cheap model itself
                continue
            C.POL = fin
            try:
                cs = C.cell_splits(task, cheap, fam)
            finally:
                C.POL = "fp32"
            if cs is None:
                continue
            det, m2, pc, h = cs
            ctl[(cheap, fin)] = [C.Split(*det[k], *m2[k], pc[k], h, rule, rrule) for k in (0, 1)]
    return dict(fam=fam, hrm=hrm, rule=rule, singles=singles, ctl=ctl)


def ctl_eval(split_pair, cheap, fin, only_polish=True):
    """all E+P configs on dev and held out: list of (cfg, dev acc, dev cost units, ho acc_i, ho cost units)."""
    dev, ho = split_pair
    r = FIN[fin] / AD.b_eff(cheap)
    ed, eh = AD.honest(dev, r), AD.honest(ho, r)
    out = []
    for cfg in dev.configs("E+P"):
        if only_polish and not (cfg[0] == "m1" or (cfg[0] == "ctl" and (cfg[3] or cfg[4]))):
            continue
        a, c, _ = ed(cfg); ah, ch, _ = eh(cfg)
        out.append((cfg, float(a.mean()), float(c.mean()), ah.astype(float), ch))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(H / "results/m5/storage_frontier")); a = ap.parse_args()
    M = [load_model(f) for f in C.families() if f["kind"] == "sudoku"]
    log = []
    def P(*x):
        s = " ".join(str(y) for y in x); print(s); log.append(s)
    P(f"models {len(M)}; singles available per model:", {m['fam']['name']: sorted(m['singles']) for m in M})
    P("controller (cheap,finish) pairs per model:", {m['fam']['name']: sorted(f"{c}+{f}" for c, f in m['ctl']) for m in M})
    # ---------------- Part 1: Table 3 analogue (cells, budgets in cheap-loop units)
    res = {"part1": {}, "part2": {}}
    for bud in (16, 24, 32, 64):
        rows = []
        for m in M:
            fam = m["fam"]; task = fam["task"]; fp = C.cell_splits(task, "fp32", fam)
            for cheap in fam["cells"]:
                if cheap == "fp32" or "noise" in cheap:
                    continue
                be = AD.b_eff(cheap); cap = bud * be
                r = dict(model=fam["name"], cheap=cheap, be=be)
                # fixed / stop of the cheap model, controller with each finisher (E+P), singles alone (stop) at the same traffic
                dev0, ho0 = None, None
                for fin in FIN:
                    if (cheap, fin) in m["ctl"]:
                        dv, hv = m["ctl"][(cheap, fin)]
                        sel = AD.select(dv, hv, dv.configs("E+P"), FIN[fin] / be, FIN[fin] / be).get(bud)
                        r[f"ctl_{fin}"] = (sel["acc_i"].astype(float), sel["cost"] * be) if sel else None
                        if dev0 is None:
                            dev0, ho0 = dv, hv
                if dev0 is not None:
                    sel = AD.select(dev0, ho0, dev0.configs("fixed"), 1, 1).get(bud); r["fixed"] = (sel["acc_i"].astype(float), sel["cost"] * be)
                    sel = AD.select(dev0, ho0, dev0.configs("E"), 1, 1).get(bud); r["stop"] = (sel["acc_i"].astype(float), sel["cost"] * be)
                else:                                              # cheap w4c: no finisher except w8c exists for it
                    pass
                for s, d in m["singles"].items():
                    al = alone(d[0], d[1], m["rule"], m["hrm"], AD.b_eff(s), cap)
                    if "stop" in al:
                        r[f"alone_{s}"] = (al["stop"]["acc"], float(al["stop"]["cost"].mean()))
                    if "fixed" in al and s in ("w8c", "w4c"):
                        r[f"alonefix_{s}"] = (al["fixed"]["acc"], float(al["fixed"]["cost"].mean()))
                rows.append(r)
        # same-format finishing = no finisher: controller(w4c) of a w4c cell is its stop row, stored b_eff
        for r in rows:
            if r["cheap"] == "w4c" and "ctl_w4c" not in r and "stop" in r:
                r["ctl_w4c"] = r["stop"]
        keys = ["fixed", "stop", "ctl_w8c", "ctl_w4c", "ctl_w4g32", "alone_w8c", "alone_w6c", "alone_w4g32", "alone_w4c", "alonefix_w8c", "alonefix_w4c"]
        stored = {"fixed": lambda r: r["be"], "stop": lambda r: r["be"], "ctl_w8c": lambda r: r["be"] + 8, "ctl_w4c": lambda r: r["be"] + (0 if r["cheap"] == "w4c" else 4),
                  "ctl_w4g32": lambda r: r["be"] + 4.5, "alone_w8c": lambda r: 8, "alone_w6c": lambda r: 6, "alone_w4g32": lambda r: 4.5, "alone_w4c": lambda r: 4,
                  "alonefix_w8c": lambda r: 8, "alonefix_w4c": lambda r: 4}
        P(f"\n=== PART 1, budget {bud} cheap-loop units (traffic cap = {bud} x b_eff(cheap) bits), held-out rows 1000-1999; mean over cells of acc | traffic (bits/weight/puzzle) | stored bits/weight")
        tab = {}
        for k in keys:
            rs = [r for r in rows if r.get(k)]
            if not rs: continue
            acc = np.mean([100 * r[k][0].mean() for r in rs]); tr = np.mean([r[k][1] for r in rs]); st = np.mean([stored[k](r) for r in rs])
            tab[k] = dict(n=len(rs), acc=acc, traffic=tr, stored=st)
            P(f"  {k:13s} n={len(rs):2d}  acc {acc:5.1f}  traffic {tr:6.1f}  stored {st:5.2f}")
        # paired differences on common cells
        diffs = {}
        for k1, k2 in (("ctl_w4c", "alone_w4c"), ("ctl_w4c", "alone_w4g32"), ("ctl_w4c", "ctl_w8c"), ("ctl_w4c", "stop"), ("ctl_w4g32", "alone_w4g32"), ("ctl_w4g32", "ctl_w8c"), ("ctl_w4c", "alone_w8c"), ("alone_w4c", "alone_w8c")):
            rs = [r for r in rows if r.get(k1) and r.get(k2)]
            if rs:
                d = pooled([r[k1][0].astype(float) - r[k2][0] for r in rs]); diffs[f"{k1}-{k2}"] = (len(rs),) + d
                P(f"  {k1}-{k2}: {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}] (n={len(rs)} cells)")
        res["part1"][bud] = dict(table=tab, diffs=diffs, per_cell={f"{r['model']}|{r['cheap']}": {k: (100 * r[k][0].mean(), r[k][1]) for k in keys if r.get(k)} for r in rows})
    # ---------------- Part 2: frontier per model
    P("\n=== PART 2: frontier. per model: best single-format config (format with b_eff <= S, stop or fixed, D0 on dev under traffic <= Tb) vs best controller")
    P("    (cheap format + finishing copy with b_eff(cheap)+b(fin) <= S; configs with at least one finishing step; chosen on dev under traffic <= Tb)")
    for S in (6.5, 8.0, 12.0):
        for Tb in (64, 96, 128, 256):
            dl, rows = [], []
            for m in M:
                best_s = None
                for s, d in m["singles"].items():
                    if AD.b_eff(s) > S + 1e-9: continue
                    al = alone(d[0], d[1], m["rule"], m["hrm"], AD.b_eff(s), Tb)
                    for mode, v in al.items():
                        k = (v["dev"], -v["cost"].mean())
                        if best_s is None or k > best_s[0]:
                            best_s = (k, s, mode, v)
                best_c = None
                for (cheap, fin), pair in m["ctl"].items():
                    be = AD.b_eff(cheap)
                    if be + FIN[fin] > S + 1e-9: continue
                    for cfg, da, dc, ha, hc in ctl_eval(pair, cheap, fin):
                        if dc * be <= Tb + 1e-9:
                            k = (da, -dc * be)
                            if best_c is None or k > best_c[0]:
                                best_c = (k, f"{cheap}+{fin}", cfg, ha, hc * be)
                row = dict(model=m["fam"]["name"], single=(best_s[1], best_s[2], float(best_s[3]["acc"].mean()), float(best_s[3]["cost"].mean())) if best_s else None,
                           ctl=(best_c[1], best_c[2], float(best_c[3].mean()), float(best_c[4].mean())) if best_c else None)
                if best_s and best_c:
                    dl.append(best_c[3] - best_s[3]["acc"])
                rows.append((row, best_s, best_c))
            ns = sum(1 for _, s, _ in rows if s); nc = sum(1 for _, _, c in rows if c)
            P(f"\n  S <= {S}, Tb <= {Tb} bits: singles exist on {ns}/{len(M)} models, controllers on {nc}/{len(M)}")
            both = [(r, s, c) for r, s, c in rows if s and c]
            if both:
                d = pooled([c[3] - s[3]["acc"] for _, s, c in both])
                ms = np.mean([100 * s[3]["acc"].mean() for _, s, _ in both]); mc = np.mean([100 * c[3].mean() for _, _, c in both])
                ts = np.mean([s[3]["cost"].mean() for _, s, _ in both]); tc = np.mean([c[4].mean() for _, _, c in both])
                win = sum(1 for _, s, c in both if (c[3] - s[3]["acc"]).mean() > 0)
                P(f"    over {len(both)} models with both: best single {ms:5.1f} @ {ts:5.1f} bits; best controller {mc:5.1f} @ {tc:5.1f} bits; controller - single {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}]; controller ahead on {win}/{len(both)} models")
            for r, s, c in rows:
                P(f"      {r['model']:10s} single {r['single'][0] if r['single'] else '-':6s}{'/' + r['single'][1] if r['single'] else '':6s} {100*r['single'][2] if r['single'] else 0:5.1f}@{r['single'][3] if r['single'] else 0:5.1f}"
                  f" | ctl {r['ctl'][0] if r['ctl'] else '-':11s} {100*r['ctl'][2] if r['ctl'] else 0:5.1f}@{r['ctl'][3] if r['ctl'] else 0:5.1f}")
            res["part2"][f"S{S}_T{Tb}"] = dict(n_both=len(both), diff=(d if both else None), single_mean=ms if both else None, ctl_mean=mc if both else None,
                                               rows=[r for r, _, _ in rows])
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    open(a.out + ".txt", "w").write("\n".join(log) + "\n")
    shutil.rmtree(MERGED, ignore_errors=True)


if __name__ == "__main__":
    main()
