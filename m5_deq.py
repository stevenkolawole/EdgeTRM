#!/usr/bin/env python3
"""M5 on deep equilibrium models (X6 records): stop the solver early, label-free.

MDEQ-Small / MDEQ-XL ImageNet (per image: top-1 correctness and relative residual ||f(z)-z||/||f(z)|| at solver
budgets 1,2,4,8,12,16,26,40,60,100; Broyden and Picard) and DEQ-Transformer WT103 (per 150-token segment: NLL
and residual at solver caps 3,5,10,20,30,60; Anderson and Picard). Records: EdgeTRM/results/transfer/babel/*.npz.
Rules (all chosen on fp32 of the same model and solver, then frozen for every perturbed cell):
  fixed b        the solver budget b (the models' defaults: MDEQ 26 Broyden steps, DEQ-T its training cap)
  depth*         the smallest fixed budget within the loss budget of fp32's best budget
  res[eps]       per image / segment, stop at the first recorded budget whose residual is below eps
                 (eps tuned on fp32: fewest mean steps with accuracy within 0.5 point / ppl within 1%)
Cross-fit on image / segment parity as in analyze_m5. Reported against the default budget and the largest one.

    python m5_deq.py
"""
import glob
import json
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results" / "transfer" / "babel"


def load_mdeq(prefix):
    cells = {}
    for f in sorted(glob.glob(str(R / f"{prefix}_*.npz"))):
        d = np.load(f)
        B = d["budgets"]
        for k in d.files:
            p = k.split("|")
            if len(p) == 3 and p[2] == "c1":
                spec, solver = p[0], p[1]
                seed = Path(f).stem.split("_")[-1]
                key = (spec if spec == "fp32" else f"{spec}@{seed}" if "noise" in spec else spec, solver)
                if key in cells:
                    continue
                cells[key] = {"c": d[k].T.astype(bool), "res": d[f"{spec}|{solver}|res"].T.astype(np.float32), "B": B}
    return cells


def first_below(res, eps):
    m = res < eps
    return np.where(m.any(1), m.argmax(1), res.shape[1] - 1)


def mdeq_eval(cells, solver, loss=0.5):
    ref = cells[("fp32", solver)]
    B = ref["B"]
    n = ref["c"].shape[0]
    halves = [np.arange(0, n, 2), np.arange(1, n, 2)]
    accB = ref["c"].mean(0) * 100
    target = accB.max() - loss
    d_star = int(np.nonzero(accB >= target - 1e-9)[0][0])
    eps_grid = [10 ** e for e in np.arange(-6, 0.01, 0.25)]

    def tune(h):
        c, r = ref["c"][halves[h]], ref["res"][halves[h]]
        best = None
        tgt = c.mean(0).max() * 100 - loss
        for eps in eps_grid:
            t = first_below(r, eps)
            acc = c[np.arange(len(t)), t].mean() * 100
            cost = B[t].mean()
            if acc >= tgt - 1e-9 and (best is None or cost < best[1]):
                best = (eps, cost)
        return best[0] if best else 0.0                      # no threshold meets the budget: never stop early

    eps = [tune(0), tune(1)]
    rows = []
    for (spec, sv), cell in sorted(cells.items(), key=lambda kv: (kv[0][0] != "fp32", kv[0][0])):
        if sv != solver:
            continue
        c, r = cell["c"], cell["res"]
        t = np.empty(n, int)
        for a, b in ((0, 1), (1, 0)):
            t[halves[b]] = first_below(r[halves[b]], eps[a])
        acc_res = c[np.arange(n), t].mean() * 100
        rows.append({"spec": spec, "acc_default": c[:, list(B).index(26)].mean() * 100 if 26 in B else None,
                     "acc_max": c[:, -1].mean() * 100, "acc_best_budget": c.mean(0).max() * 100,
                     "best_budget": int(B[c.mean(0).argmax()]),
                     "acc_depth": c[:, d_star].mean() * 100, "depth": int(B[d_star]),
                     "acc_res": acc_res, "steps_res": float(B[t].mean()), "eps": eps})
    return rows


def load_deqlm(prefix):
    cells = {}
    for f in sorted(glob.glob(str(R / f"{prefix}_*.npz")) + sorted(glob.glob(str(R / f"{prefix}.npz")))):
        d = np.load(f)
        seed = Path(f).stem.split("_")[-1]
        for k in d.files:
            p = k.split("|")
            if len(p) == 4 and p[3] == "seg_nll":
                spec, solver, cap = p[0], p[1], int(p[2])
                key = (spec if spec == "fp32" or "noise" not in spec else f"{spec}@{seed}", solver)
                cells.setdefault(key, {})[cap] = (d[k].astype(np.float64), d[f"{spec}|{solver}|{cap}|seg_res"].astype(np.float64))
    return cells


def deqlm_eval(cells, solver, loss_rel=0.01):
    ref = cells[("fp32", solver)]
    caps = sorted(ref)
    n = len(ref[caps[0]][0])
    halves = [np.arange(0, n, 2), np.arange(1, n, 2)]
    ppl = lambda nll: float(np.exp(nll.mean()))
    eps_grid = [10 ** e for e in np.arange(-5, 0.01, 0.25)]

    def arrays(cell):
        cs = [c for c in caps if c in cell]
        return cs, np.stack([cell[c][0] for c in cs], 1), np.stack([cell[c][1] for c in cs], 1)

    cs0, N0, Rz0 = arrays(ref)

    def tune(h):
        N, Rz = N0[halves[h]], Rz0[halves[h]]
        tgt = min(ppl(N[:, j]) for j in range(len(cs0))) * (1 + loss_rel)
        best = None
        for eps in eps_grid:
            t = first_below(Rz, eps)
            p = ppl(N[np.arange(len(t)), t])
            cost = np.array(cs0)[t].mean()
            if p <= tgt and (best is None or cost < best[1]):
                best = (eps, cost)
        return best[0] if best else eps_grid[-1]

    eps = [tune(0), tune(1)]
    pp0 = [ppl(N0[:, j]) for j in range(len(cs0))]
    d_star = int(np.nonzero(np.array(pp0) <= min(pp0) * (1 + loss_rel))[0][0])
    rows = []
    for (spec, sv), cell in sorted(cells.items(), key=lambda kv: (kv[0][0] != "fp32", kv[0][0])):
        if sv != solver:
            continue
        cs, N, Rz = arrays(cell)
        if cs != cs0:
            continue
        t = np.empty(n, int)
        for a, b in ((0, 1), (1, 0)):
            t[halves[b]] = first_below(Rz[halves[b]], eps[a])
        rows.append({"spec": spec, "ppl_by_cap": {c: ppl(N[:, j]) for j, c in enumerate(cs)},
                     "ppl_depth": ppl(N[:, d_star]), "depth": cs[d_star],
                     "ppl_res": ppl(N[np.arange(n), t]), "steps_res": float(np.array(cs)[t].mean()), "eps": eps})
    return rows


def main():
    out = {}
    for prefix, name in (("deq_small", "MDEQ-Small ImageNet (n=5000)"), ("deq_xl", "MDEQ-XL ImageNet (n=2000)")):
        cells = load_mdeq(prefix)
        for solver in ("broyden", "picard"):
            if ("fp32", solver) not in cells:
                continue
            rows = mdeq_eval(cells, solver)
            out[f"{prefix}|{solver}"] = rows
            print(f"\n=== {name}, {solver}: top-1 %. default = 26 steps; depth* = fixed budget chosen on fp32 ({rows[0]['depth']}); "
                  f"res = residual stop tuned on fp32 (eps {rows[0]['eps'][0]:.0e}/{rows[0]['eps'][1]:.0e})")
            print(f"{'spec':16s} {'@26':>6s} {'@100':>6s} {'best b':>10s} | {'depth*':>8s} | {'res stop':>8s} {'steps':>6s}")
            for r in rows:
                print(f"{r['spec']:16s} {r['acc_default'] or float('nan'):6.1f} {r['acc_max']:6.1f} {r['acc_best_budget']:6.1f}@{r['best_budget']:<3d} | "
                      f"{r['acc_depth']:8.1f} | {r['acc_res']:8.1f} {r['steps_res']:6.1f}")
    for prefix, name in (("deqlm", "DEQ-Transformer WT103 (103 segments)"),):
        cells = load_deqlm(prefix)
        for solver in ("anderson", "picard"):
            if ("fp32", solver) not in cells:
                continue
            rows = deqlm_eval(cells, solver)
            out[f"{prefix}|{solver}"] = rows
            caps = sorted(rows[0]["ppl_by_cap"])
            print(f"\n=== {name}, {solver}: perplexity by solver cap {caps}; depth* = {rows[0]['depth']}; res stop tuned on fp32")
            for r in rows:
                pc = " ".join(f"{r['ppl_by_cap'][c]:8.1f}" for c in caps)
                print(f"{r['spec']:16s} {pc} | depth* {r['ppl_depth']:8.1f} | res stop {r['ppl_res']:8.1f} at {r['steps_res']:5.1f}")
    json.dump(out, open(Path(__file__).resolve().parent / "results" / "breadth" / "m5_deq.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
