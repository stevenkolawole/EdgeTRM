#!/usr/bin/env python3
"""Read-out for the looped-LM runs of looped_decode.py (EdgeTRM X4: Q6 decoding
rescue and the looped-LM transfer protocol).  Reads only written records
(<tag>/<decode>.jsonl and <tag>/tf2.json | tf.json) under one or more result
roots; nothing is taken from logs.

    python analyze_looped.py                       # all sections, default roots
    python analyze_looped.py q6 --roots results/looped/p4d/q6
    python analyze_looped.py tf | div | len | kl | sweep

Sections
  q6     accuracy / degenerate / loop / capped / mean length per (model, loops,
         quant) x decode, Wilson 95% intervals, sampling seeds as mean [min, max],
         and the fraction of the bf16-greedy gap each control closes
  tf     teacher-forced metrics beside generation: gold NLL, top-1 agreement with
         bf16, KL(bf16||q) (top-64 + tail bin), latent fidelity, likelihood-scored
         answer accuracy (tf_ans_acc) vs greedy generation accuracy (the protocol gap)
  div    first divergence of the quantized greedy output from the bf16 greedy output
         (token index when ids are stored, else character index), and whether
         problems that diverge late still fail
  len    accuracy binned by output length (and cap hits) per decode
  kl     per-problem teacher-forced KL / NLL increase vs generation failure on the
         problems bf16 greedy solves (AUROC, accuracy by KL quartile)
  sweep  loops x quantizer table for greedy (accuracy, NLL, KL, fidelity)
"""
import argparse
import json
import math
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ROOTS = [HERE / "results/looped/p4d/q6", HERE / "results/looped/babel/q6",
                 HERE / "results/looped/p4d/sweep", HERE / "results/looped/babel/sweep"]
TAG = re.compile(r"(?P<model>.+)_s(?P<steps>\d+)_(?P<quant>[a-z0-9]+)_(?P<scope>core|all)$")
DEC_ORDER = ["greedy", "rep1.1", "rep1.3", "nrng3", "topk20t0.7", "mark5", "loopstop", "loopban12", "dry"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (100 * (c - h), 100 * (c + h))


def read_jsonl(p):
    recs = {}
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        recs[r["idx"]] = r
    return recs


def load(roots):
    """cells[(model, steps, quant, scope)] = {"gen": {decode: {idx: rec}}, "tf": tf dict or None, "root": root}"""
    cells = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            m = TAG.match(d.name)
            if not d.is_dir() or not m:
                continue
            key = (m["model"], int(m["steps"]), m["quant"], m["scope"])
            c = cells.setdefault(key, {"gen": {}, "tf": None, "tf_v": None, "root": str(root)})
            for f in sorted(d.glob("*.jsonl")):
                recs = read_jsonl(f)
                side = d / f"{f.stem}.ids.json"          # looped_add_ids.py sidecar for v1 records
                if side.exists():
                    ids = json.load(open(side))["ids"]
                    for i, r in recs.items():
                        if "ids" not in r and str(i) in ids:
                            r["ids"], r["ids_reencoded"] = ids[str(i)], True
                c["gen"][f.stem] = {**c["gen"].get(f.stem, {}), **recs}
            for name, v in (("tf2.json", 2), ("tf.json", 1)):
                p = d / name
                if p.exists() and (c["tf"] is None or (c["tf_v"] or 0) < v):
                    try:
                        c["tf"], c["tf_v"] = json.load(open(p)), v
                    except json.JSONDecodeError:
                        pass
    return cells


def complete(recs, n):
    return len(recs) >= n


def acc(recs):
    return 100 * sum(r["correct"] for r in recs.values()) / len(recs) if recs else float("nan")


def rate(recs, k):
    return 100 * sum(bool(r[k]) for r in recs.values()) / len(recs) if recs else float("nan")


def loop_rate(recs):
    return 100 * sum(r["tail_period"] > 0 for r in recs.values()) / len(recs) if recs else float("nan")


def groups(cells):
    g = defaultdict(dict)
    for (model, steps, quant, scope), c in cells.items():
        g[(model, steps, scope)][quant] = c
    return g


def dec_family(d):
    return re.sub(r"s\d+$", "", d) if d.startswith("topk") else d


def fmt(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# ----------------------------------------------------------------------------- sections
def sec_q6(cells, nmin):
    print("## Q6: decoding controls on compressed looped LMs (GSM8K; % of n; 95% Wilson CI)\n")
    for (model, steps, scope), qs in sorted(groups(cells).items()):
        decs = sorted({dec_family(d) for c in qs.values() for d in c["gen"]},
                      key=lambda d: DEC_ORDER.index(d) if d in DEC_ORDER else 99)
        if not decs:
            continue
        print(f"### {model}, {steps} loops, scope {scope}\n")
        print("| quant | decode | n | acc | 95% CI | degenerate | loop | capped | mean len | gap closed |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        bf = qs.get("bf16", {}).get("gen", {}).get("greedy")
        for quant in sorted(qs, key=lambda q: (q != "bf16", q)):
            gen = qs[quant]["gen"]
            g0 = gen.get("greedy")
            for dfam in decs:
                runs = [d for d in gen if dec_family(d) == dfam]
                if not runs:
                    continue
                runs_ok = [d for d in runs if complete(gen[d], nmin)]
                partial = "" if len(runs_ok) == len(runs) else " (incomplete)"
                if dfam.startswith("topk"):
                    accs = [acc(gen[d]) for d in runs]
                    n = min(len(gen[d]) for d in runs)
                    a = st.mean(accs)
                    ci = f"seeds {', '.join(fmt(x) for x in accs)}"
                    deg = st.mean(rate(gen[d], "degenerate") for d in runs)
                    lp = st.mean(loop_rate(gen[d]) for d in runs)
                    cap = st.mean(rate(gen[d], "capped") for d in runs)
                    ln = st.mean(st.mean(r["n_gen"] for r in gen[d].values()) for d in runs)
                    label = f"{dfam} x{len(runs)}{partial}"
                else:
                    recs = gen[runs[0]]
                    n = len(recs)
                    a = acc(recs)
                    lo, hi = wilson(sum(r["correct"] for r in recs.values()), n)
                    ci = f"{fmt(lo)}–{fmt(hi)}"
                    deg, lp, cap = rate(recs, "degenerate"), loop_rate(recs), rate(recs, "capped")
                    ln = st.mean(r["n_gen"] for r in recs.values())
                    label = dfam + partial
                    if dfam.startswith("pfx"):          # leak check: gold answer already inside the forced prefix
                        leak = sum(bool(re.search(r"(?<![\d.])" + re.escape(r["gold"]) + r"(?![\d])", r.get("prefix_text", "")))
                                   for r in recs.values())
                        label += f" (gold in prefix: {leak})"
                gap = ""
                if quant != "bf16" and bf and g0 and dfam != "greedy":
                    den = acc(bf) - acc(g0)
                    if abs(den) > 1e-9:
                        gap = fmt(100 * (a - acc(g0)) / den, 0) + "%"
                print(f"| {quant} | {label} | {n} | {fmt(a)} | {ci} | {fmt(deg)} | {fmt(lp)} | {fmt(cap)} | {fmt(ln, 0)} | {gap} |")
        print()


def sec_tf(cells, nmin):
    print("## Teacher-forced metrics vs generation (the protocol gap)\n")
    print("| model | loops | quant | n | gold NLL | tf top-1 | agree w/ bf16 | KL(bf16‖q) | fidelity | "
          "likelihood ans-acc | greedy gen acc | gap (lik − gen) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (model, steps, quant, scope), c in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] != "bf16", kv[0][2])):
        tf = c["tf"]
        if not tf:
            continue
        s = tf["summary"]
        g = c["gen"].get("greedy")
        ga = acc(g) if g else None
        la = 100 * s["tf_ans_acc"] if "tf_ans_acc" in s else None
        gap = (la - ga) if (la is not None and ga is not None) else None
        print(f"| {model} | {steps} | {quant} | {len(tf['records'])} | {fmt(s['nll'], 3)} | {fmt(s['tf_acc'], 3)} | "
              f"{fmt(s['tf_agree'], 3)} | {fmt(s.get('kl'), 3)} | {fmt(s['fidelity'], 3)} | {fmt(la)} | {fmt(ga)} | "
              f"{fmt(gap)} |")
    print("\n(tf v1 files have no KL or likelihood answer accuracy; v2 = tf2.json.)\n")


def first_div(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None      # None = identical


def sec_div(cells, nmin):
    print("## First divergence of quantized greedy output from bf16 greedy\n")
    print("| model | loops | quant | n | identical | unit | median first-div | IQR | acc if div < 32 | "
          "acc if div >= 32 | acc if identical |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (model, steps, scope), qs in sorted(groups(cells).items()):
        bf = qs.get("bf16", {}).get("gen", {}).get("greedy")
        if not bf:
            continue
        for quant, c in sorted(qs.items()):
            if quant == "bf16" or "greedy" not in c["gen"]:
                continue
            g = c["gen"]["greedy"]
            idx = sorted(set(g) & set(bf))
            use_ids = all("ids" in g[i] and "ids" in bf[i] for i in idx)
            reenc = use_ids and any(g[i].get("ids_reencoded") or bf[i].get("ids_reencoded") for i in idx)
            unit = ("token (re-encoded)" if reenc else "token") if use_ids else "char"
            divs, ident, early, late = [], [], [], []
            for i in idx:
                a = g[i]["ids"] if use_ids else g[i]["text"]
                b = bf[i]["ids"] if use_ids else bf[i]["text"]
                d = first_div(a, b)
                if d is None:
                    ident.append(g[i]["correct"])
                    continue
                divs.append(d)
                (early if d < (32 if use_ids else 128) else late).append(g[i]["correct"])
            if not idx:
                continue
            q = st.quantiles(divs, n=4) if len(divs) >= 4 else [float("nan")] * 3
            m = lambda xs: fmt(100 * sum(xs) / len(xs)) + f" ({len(xs)})" if xs else "–"  # noqa: E731
            print(f"| {model} | {steps} | {quant} | {len(idx)} | {len(ident)} | {unit} | "
                  f"{fmt(st.median(divs), 0) if divs else '–'} | {fmt(q[0], 0)}–{fmt(q[2], 0)} | {m(early)} | {m(late)} | "
                  f"{m(ident)} |")
    print("\n(char unit: v1 records without token ids; the early/late split is 32 tokens or 128 characters.)\n")


def sec_len(cells, nmin):
    print("## Accuracy by output length (greedy and controls)\n")
    bins = [(0, 128), (128, 256), (256, 384), (384, 512)]
    print("| model | loops | quant | decode | " + " | ".join(f"len {a}-{b - 1}" for a, b in bins) + " | capped |")
    print("|---|---|---|---|" + "---|" * (len(bins) + 1))
    for (model, steps, quant, scope), c in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] != "bf16", kv[0][2])):
        for dec in sorted(c["gen"], key=lambda d: DEC_ORDER.index(dec_family(d)) if dec_family(d) in DEC_ORDER else 99):
            recs = list(c["gen"][dec].values())
            cells_ = []
            for a, b in bins:
                xs = [r["correct"] for r in recs if a <= r["n_gen"] < b and not r["capped"]]
                cells_.append(f"{fmt(100 * sum(xs) / len(xs))} ({len(xs)})" if xs else "–")
            cp = [r["correct"] for r in recs if r["capped"]]
            cells_.append(f"{fmt(100 * sum(cp) / len(cp))} ({len(cp)})" if cp else "–")
            print(f"| {model} | {steps} | {quant} | {dec} | " + " | ".join(cells_) + " |")
    print()


def auroc(pos, neg):
    """P(score_pos > score_neg), ties half."""
    if not pos or not neg:
        return float("nan")
    w = 0.0
    for p in pos:
        for q in neg:
            w += 1.0 if p > q else (0.5 if p == q else 0.0)
    return w / (len(pos) * len(neg))


def sec_kl(cells, nmin):
    print("## Per-problem teacher-forced damage vs generation failure (problems bf16 greedy solves)\n")
    print("| model | loops | quant | n | fail rate | AUROC KL→fail | AUROC ΔNLL→fail | AUROC (1−agree)→fail | "
          "acc by KL quartile (low→high) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (model, steps, scope), qs in sorted(groups(cells).items()):
        bf = qs.get("bf16")
        if not bf or "greedy" not in bf["gen"] or not bf["tf"]:
            continue
        bfr = {r["idx"]: r for r in bf["tf"]["records"]}
        solved = {i for i, r in bf["gen"]["greedy"].items() if r["correct"]}
        for quant, c in sorted(qs.items()):
            if quant == "bf16" or not c["tf"] or "greedy" not in c["gen"]:
                continue
            tfr = {r["idx"]: r for r in c["tf"]["records"]}
            g = c["gen"]["greedy"]
            idx = sorted(i for i in solved if i in g and i in tfr and i in bfr)
            if not idx:
                continue
            fail = {i: 1 - g[i]["correct"] for i in idx}
            has_kl = all("kl" in tfr[i] for i in idx)
            dn = {i: tfr[i]["nll"] - bfr[i]["nll"] for i in idx}
            ag = {i: 1 - tfr[i]["tf_agree"] for i in idx}
            pos = [i for i in idx if fail[i]]
            neg = [i for i in idx if not fail[i]]
            a_kl = auroc([tfr[i]["kl"] for i in pos], [tfr[i]["kl"] for i in neg]) if has_kl else float("nan")
            a_dn = auroc([dn[i] for i in pos], [dn[i] for i in neg])
            a_ag = auroc([ag[i] for i in pos], [ag[i] for i in neg])
            key = (lambda i: tfr[i]["kl"]) if has_kl else (lambda i: dn[i])
            srt = sorted(idx, key=key)
            qn = [srt[k * len(srt) // 4:(k + 1) * len(srt) // 4] for k in range(4)]
            qa = " / ".join(fmt(100 * sum(1 - fail[i] for i in b) / len(b), 0) if b else "–" for b in qn)
            print(f"| {model} | {steps} | {quant} | {len(idx)} | {fmt(100 * len(pos) / len(idx))} | {fmt(a_kl, 2)} | "
                  f"{fmt(a_dn, 2)} | {fmt(a_ag, 2)} | {qa}{'' if has_kl else ' (by ΔNLL)'} |")
    print()


def sec_sweep(cells, nmin):
    print("## Loops x quantizer (greedy accuracy % [n]; gold NLL; KL; fidelity)\n")
    by_model = defaultdict(dict)
    for (model, steps, quant, scope), c in cells.items():
        by_model[(model, scope)][(steps, quant)] = c
    for (model, scope), d in sorted(by_model.items()):
        quants = sorted({q for _, q in d}, key=lambda q: (q != "bf16", q))
        steps = sorted({s for s, _ in d})
        print(f"### {model} (scope {scope})\n")
        print("| loops | " + " | ".join(quants) + " |")
        print("|---|" + "---|" * len(quants))
        for s in steps:
            row = []
            for q in quants:
                c = d.get((s, q))
                if not c:
                    row.append("–")
                    continue
                g = c["gen"].get("greedy")
                t = c["tf"]["summary"] if c["tf"] else {}
                txt = (f"{fmt(acc(g))} [{len(g)}]" if g else "–") + (f" nll {fmt(t.get('nll'), 2)}" if t else "")
                if t and q != "bf16":
                    txt += f" KL {fmt(t.get('kl'), 2)} φ {fmt(t.get('fidelity'), 2)}"
                row.append(txt)
            print(f"| {s} | " + " | ".join(row) + " |")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["q6", "tf", "div", "len", "kl", "sweep"])
    ap.add_argument("--roots", nargs="*", default=None)
    ap.add_argument("--nmin", type=int, default=200, help="records needed to call a run complete")
    a = ap.parse_args()
    cells = load(a.roots or DEFAULT_ROOTS)
    if not cells:
        print("no results under", a.roots or DEFAULT_ROOTS)
        return
    fns = {"q6": sec_q6, "tf": sec_tf, "div": sec_div, "len": sec_len, "kl": sec_kl, "sweep": sec_sweep}
    for s in a.sections:
        fns[s](cells, a.nmin)


if __name__ == "__main__":
    main()
