#!/usr/bin/env python3
"""Does the label-free push-vs-room law, used as a format selector, pick a format that survives? (Reviewer: "just pick w4c".)

Selector: among a model's single formats with a measured push <= room, take the fewest stored bits (tie: lower traffic).
Push = median one-step push at T=16, rows 0-255, from results/m5/push_gptq_vs_rtn (RTN cells) and, where that file lacks the
format, from the contraction records results/breadth/babel/x2/contraction_*.json (delta_T median, same definition: it reproduces
the push file where both exist). A format without a measured push is never filled in. A pick is "determined" only when every
format without a measured push has more stored bits than the pick (it could not have won); only determined models enter the pooled
means. Accuracy: held-out rows 1000-1999, halting-head stop, D0 chosen on dev under traffic <= budget x b_eff(cheap) bits, with the
cheap format of the model's lowest-bit controller cell (the same pipeline as storage_frontier.py). Collapse: accuracy at 16 steps
< 0.5 x fp32 (gptq_push_rule.py).   python law_selects_format.py [--out results/m5/law_selects_format]
"""
import argparse, glob, json, shutil, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import storage_frontier as SF
import gptq_push_rule as GP
import m5_addons as AD, m5_ctl_ho as C

BUDS = (16, 24, 64)
ROOMS = {"pooled 0.344": lambda task, mr: 0.344, "refit 0.327": lambda task, mr: 0.327, "per-mixer": lambda task, mr: mr.get(task, (None,))[0]}
SKIP = ("fp32", "noise", "anoise")


def measured_push():
    """{(task, fmt): (push, source)}; the RTN push file first, contraction delta_T for formats it lacks."""
    out = {}
    for task in ("sudoku_mlp", "sudoku_attn"):
        for k, v in json.load(open(H / f"results/m5/push_gptq_vs_rtn/{task}.json"))["cells"].items():
            if not v["gptq"]:
                out[(task, k)] = (v["push"]["16"]["median"], "push_gptq_vs_rtn")
    for f in sorted(glob.glob(str(H / "results/breadth/babel/x2/contraction_*.json"))):
        d = json.load(open(f))
        if not d["task"].endswith("sudoku") and not d["task"].startswith("sudoku") or d["offset"] != 0 or d["T"] != 16:
            continue
        for r in d["records"]:
            P = r["per_input"]
            dt = P.get("deltaT")
            if dt is None and "delta" in P:
                dt = np.asarray(P["delta"], float)[:, 16]
            sp = r["spec"]
            if dt is None or sp == "fp32" or sp.startswith("noise") or "@" in sp or (d["task"], sp) in out:
                continue
            out[(d["task"], sp)] = (float(np.median(dt)), "contraction/" + Path(f).name)
    for f in sorted(glob.glob(str(H / "results/m5/push_missing_formats/*.json"))):
        name = Path(f).stem
        for k, v in json.load(open(f))["cells"].items():
            new = v["push"]["16"]["median"]
            if (name, k) in out:
                print(f"  both sources {name} {k}: old {out[(name, k)][0]:.3f} (used), new {new:.3f}")
            else:
                out[(name, k)] = (new, "push_missing_formats/" + Path(f).name)
    return out


def select(push, bits, traffic, room):
    ok = [f for f in push if push[f] <= room]
    return min(ok, key=lambda f: (bits[f], traffic[f])) if ok else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(H / "results/m5/law_selects_format")); a = ap.parse_args()
    PU, mixer = measured_push(), GP.mixer_rooms()
    M = [SF.load_model(f) for f in C.families() if f["kind"] == "sudoku"]
    log = []
    def P(*x):
        s = " ".join(str(y) for y in x); print(s); log.append(s)
    P("room per mixer (median implied room, n):", mixer)
    res = {"coverage": {}, "models": {}, "pooled": {}}
    P("\n=== COVERAGE: formats with a measured push (push) / without (-)")
    cells = {}
    for m in M:
        fam = m["fam"]; task = fam["task"]
        cheap = min((c for c in fam["cells"] if not c.startswith(SKIP)), key=AD.b_eff)
        be = AD.b_eff(cheap)
        push = {f: PU[(fam["name"], f)][0] for f in m["singles"] if (fam["name"], f) in PU}
        miss = [f for f in m["singles"] if f not in push]
        res["coverage"][fam["name"]] = dict(measured={f: dict(push=push[f], source=PU[(fam["name"], f)][1]) for f in push}, missing=miss)
        P(f"  {fam['name']:12s} measured {len(push)}/{len(m['singles'])}: " + " ".join(f"{f}={push[f]:.3f}" for f in sorted(push, key=AD.b_eff)) + f" | missing: {','.join(miss)}")
        al = {b: {f: SF.alone(d[0], d[1], m["rule"], m["hrm"], AD.b_eff(f), b * be).get("stop") for f, d in m["singles"].items()} for b in BUDS}
        ctl = {}
        for b in BUDS:
            if (cheap, "w8c") in m["ctl"]:
                dv, hv = m["ctl"][(cheap, "w8c")]
                s = AD.select(dv, hv, dv.configs("E+P"), 8.0 / be, 8.0 / be).get(b)
                ctl[b] = (s["acc_i"].astype(float), float(np.mean(s["cost"] * be))) if s else None
        cells[fam["name"]] = (m, cheap, be, push, miss, al, ctl)
    def arm(b, al, f):
        v = al[b].get(f)
        return None if v is None else (v["acc"], float(v["cost"].mean()), AD.b_eff(f))
    for rname, rf in ROOMS.items():
        P(f"\n=== ROOM {rname}: label-free pick, held-out rows 1000-1999 (acc | traffic bits/weight/puzzle | stored bits); cap = budget x b_eff(cheap of the controller cell)")
        diff = {k: {b: [] for b in BUDS} for k in ("w8c_alone", "controller")}
        ncol = nund = ndet = 0
        res["models"][rname] = {}
        for name, (m, cheap, be, push, miss, al, ctl) in cells.items():
            task = m["fam"]["task"]; room = rf(task, mixer)
            if not push or room is None:
                P(f"  {name:11s} no pick: " + ("no measured push for any format" if not push else "no room defined for this mixer")); res["models"][rname][name] = dict(pick=None); continue
            bits = {f: AD.b_eff(f) for f in push}
            trf = {f: (arm(16, al, f) or (0, 1e9))[1] for f in push}
            pick = select(push, bits, trf, room)
            if pick is None:
                P(f"  {name:11s} no measured format has push <= {room}"); res["models"][rname][name] = dict(pick=None); continue
            det = all(AD.b_eff(f) > bits[pick] for f in miss)
            sub = m["fam"].get("sub", task)
            fp, ac = GP.load_acc(sub, "fp32", slice(0, 2000)), GP.load_acc(sub, pick, slice(0, 2000))
            ret = ac / fp if ac is not None else float("nan"); col = bool(ret < 0.5); ncol += det and col
            P(f"  {name:11s} cheap {cheap} (cap {BUDS[0]}/{BUDS[1]}/{BUDS[2]} x {be}): pick {pick} (push {push[pick]:.3f} <= room {room:.3f}, retained {ret:.2f}{', COLLAPSES' if col else ''}); "
              + ("determined" if det else f"UNDETERMINED: unmeasured {','.join(f for f in miss if AD.b_eff(f) <= bits[pick])} could be cheaper"))
            rec = dict(pick=pick, push=push[pick], room=room, retained=ret, collapses=bool(col), determined=det, budgets={})
            for b in BUDS:
                dev_best = max(((v["dev"], f) for f, v in al[b].items() if v), key=lambda t: t[0])[1]
                rows = {"law pick": arm(b, al, pick), "w8c alone": arm(b, al, "w8c"),
                        "controller+w8c": None if not ctl.get(b) else (ctl[b][0], ctl[b][1], be + 8.0), f"dev-best ({dev_best})": arm(b, al, dev_best)}
                P(f"      b={b:2d}: " + " | ".join(f"{k} {100*v[0].mean():5.1f} {v[1]:5.1f} {v[2]:4.1f}" for k, v in rows.items() if v))
                rec["budgets"][b] = {k: dict(acc=100 * v[0].mean(), traffic=v[1], stored=v[2]) for k, v in rows.items() if v}
                if det and rows["law pick"]:
                    if rows["w8c alone"]: diff["w8c_alone"][b].append(rows["law pick"][0] - rows["w8c alone"][0])
                    if rows["controller+w8c"]: diff["controller"][b].append(rows["law pick"][0] - rows["controller+w8c"][0])
            ndet += det
            res["models"][rname][name] = rec
        P(f"  determined picks {ndet}; of these the law picked a collapsing format on {ncol}")
        pooled = {"n_determined": ndet, "n_collapse": int(ncol)}
        for k, lab in (("w8c_alone", "law pick - w8c alone"), ("controller", "law pick - controller+w8c")):
            for b in BUDS:
                if diff[k][b]:
                    d = SF.pooled(diff[k][b]); pooled[f"{k}_b{b}"] = dict(n=len(diff[k][b]), mean=d[0], lo=d[1], hi=d[2])
                    P(f"  pooled b={b:2d} {lab}: {d[0]:+.2f} [{d[1]:+.2f},{d[2]:+.2f}] (n={len(diff[k][b])} models)")
        res["pooled"][rname] = pooled
    json.dump(res, open(a.out + ".json", "w"), indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
    open(a.out + ".txt", "w").write("\n".join(log) + "\n")
    shutil.rmtree(SF.MERGED, ignore_errors=True)


if __name__ == "__main__":
    main()
