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
  pool   deduplicated pool (one row per model x quantizer x loops, accuracy and teacher-forced record from
         one directory), per-model and pooled AUROC, best fidelity cut per model with a bootstrap CI.
         Default roots POOL_ROOTS; `python analyze_looped.py pool` is the documented command.
  boundary  every LM cell with generation (Sep 13-15 Lambda grid + this harness): latent fidelity vs
         teacher-forced NLL increase vs retained accuracy, and AUROC of each for predicting collapse
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


LAMBDA = HERE / "results/breadth/lambda"
LAMBDA_FILES = [("huginn", ["huginn_gsm8k_s4.json", "huginn_gsm8k_s8.json", "huginn_gsm8k_s16.json", "huginn_gsm8k_s32.json",
                            "huginn_gsm8k_s64_n50.json"], "huginn_gsm8k_s4_8_16_32_64_nogen.json"),
                ("ouro14", ["ouro14_gsm8k_s1_2.json", "ouro14_gsm8k_s3_4_n100.json", "ouro14_gsm8k_s6_n100.json",
                            "ouro14_gsm8k_s8_n100.json"], "ouro14_gsm8k_s1_2_3_4_6_8_nogen.json"),
                ("ouro26", ["ouro26_gsm8k_s1_2.json", "ouro26_gsm8k_s3_4_n100.json", "ouro26_gsm8k_s6_n100.json",
                            "ouro26_gsm8k_s8_n100.json"], "ouro26_gsm8k_s1_2_3_4_6_8_nogen.json")]


def boundary_cells(cells):
    """Every (model, loops, quant) with a greedy accuracy and a teacher-forced record, from the Sep 13-15
    Lambda grid (Table E) and from this harness (source 'x4'); bf16 at the same loops as reference."""
    rows = []
    for model, gen_files, nogen in LAMBDA_FILES:
        gen, ng = {}, {}
        for f in gen_files:
            p = LAMBDA / f
            if p.exists():
                for r in json.load(open(p))["records"]:
                    if r.get("acc") is not None:
                        gen.setdefault((r["steps"], r["quant"]["spec"]), r)
        p = LAMBDA / nogen
        if p.exists():
            ng = {(r["steps"], r["quant"]["spec"]): r for r in json.load(open(p))["records"]}
        for (st_, q), g in gen.items():
            b, t, tb = gen.get((st_, "bf16")), ng.get((st_, q)), ng.get((st_, "bf16"))
            if q == "bf16" or not b or not t or not tb:
                continue
            rows.append({"src": "lambda", "model": model, "loops": st_, "quant": q, "n": g["n"], "acc": 100 * g["acc"],
                         "acc_bf16": 100 * b["acc"], "nll": t["nll"], "nll_bf16": tb["nll"], "fid": t["fidelity"],
                         "agree": None, "kl": None})
    for (model, steps, quant, scope), c in cells.items():
        if quant == "bf16" or not c["tf"] or "greedy" not in c["gen"]:
            continue
        b = cells.get((model, steps, "bf16", scope))
        if not b or not b["tf"] or "greedy" not in b["gen"]:
            continue
        t, tb = c["tf"]["summary"], b["tf"]["summary"]
        rows.append({"src": "x4", "model": model, "loops": steps, "quant": quant, "n": len(c["gen"]["greedy"]),
                     "acc": acc(c["gen"]["greedy"]), "acc_bf16": acc(b["gen"]["greedy"]), "nll": t["nll"],
                     "nll_bf16": tb["nll"], "fid": t["fidelity"], "agree": t.get("tf_agree"), "kl": t.get("kl")})
    return rows


def sec_boundary(cells, nmin):
    print("## Do generators follow the same boundary? latent fidelity vs teacher-forced loss vs retained accuracy\n")
    print("retained = greedy accuracy / bf16 greedy accuracy at the same loop count; collapse = retained < 50%.\n")
    rows = boundary_cells(cells)
    rows.sort(key=lambda r: (-r["fid"]))
    print("| src | model | loops | quant | n | acc (bf16) | retained | fidelity | ΔNLL (nats) | ΔNLL % | agree | KL |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        ret = 100 * r["acc"] / r["acc_bf16"] if r["acc_bf16"] > 0 else float("nan")
        r["ret"] = ret
        dn = r["nll"] - r["nll_bf16"]
        print(f"| {r['src']} | {r['model']} | {r['loops']} | {r['quant']} | {r['n']} | {fmt(r['acc'])} ({fmt(r['acc_bf16'])}) | "
              f"{fmt(ret, 0)}% | {fmt(r['fid'], 3)} | {dn:+.3f} | {100 * dn / r['nll_bf16']:+.0f}% | "
              f"{fmt(r['agree'], 3) if r['agree'] is not None else '–'} | {fmt(r['kl'], 3) if r['kl'] is not None else '–'} |")
    ok = [r for r in rows if r["acc_bf16"] >= 10 and not math.isnan(r["ret"])]
    col = [r for r in ok if r["ret"] < 50]
    held = [r for r in ok if r["ret"] >= 50]
    if col and held:
        print(f"\nCells with bf16 >= 10%: {len(held)} hold, {len(col)} collapse.")
        print(f"- fidelity: collapsed max {max(r['fid'] for r in col):.3f}, held min {min(r['fid'] for r in held):.3f}; "
              f"AUROC(1-fid -> collapse) {auroc([1 - r['fid'] for r in col], [1 - r['fid'] for r in held]):.3f}")
        rel = lambda r: (r["nll"] - r["nll_bf16"]) / r["nll_bf16"]  # noqa: E731
        print(f"- relative ΔNLL: collapsed min {100 * min(rel(r) for r in col):+.1f}%, held max {100 * max(rel(r) for r in held):+.1f}%; "
              f"AUROC(ΔNLL% -> collapse) {auroc([rel(r) for r in col], [rel(r) for r in held]):.3f}")
        dn = lambda r: r["nll"] - r["nll_bf16"]  # noqa: E731
        print(f"- absolute ΔNLL: collapsed min {min(dn(r) for r in col):+.3f}, held max {max(dn(r) for r in held):+.3f}; "
              f"AUROC(ΔNLL -> collapse) {auroc([dn(r) for r in col], [dn(r) for r in held]):.3f}")
        print("  (destroyed cells with fidelity < 0.5 are trivially separated by every metric; the informative "
              "comparison is among cells with fidelity >= 0.5:)")
        col2 = [r for r in col if r["fid"] >= 0.5]
        held2 = [r for r in held if r["fid"] >= 0.5]
        if col2 and held2:
            print(f"  fidelity AUROC {auroc([1 - r['fid'] for r in col2], [1 - r['fid'] for r in held2]):.3f}, "
                  f"ΔNLL% AUROC {auroc([rel(r) for r in col2], [rel(r) for r in held2]):.3f}, "
                  f"ΔNLL AUROC {auroc([dn(r) for r in col2], [dn(r) for r in held2]):.3f} "
                  f"({len(col2)} collapsed, {len(held2)} held)")
    print()


POOL_ROOTS = ["results/looped/p4d/q6", "results/looped/p4d/olmo", "results/looped/p4d/olmo200",
              "results/looped/p4d/loops", "results/looped/p4d/n500", "results/looped/p4d/olmo500",
              "results/looped/babel/q6", "results/looped/p5/n500", "results/looped/p5/band"]
PLATFORM_RANK = {"p5": 0, "p4d": 1, "babel": 2, "lambda": 3}


def pool_rows(roots, cut=50.0):
    """One row per (model, quant, loops): greedy accuracy and teacher-forced record from the SAME result
    directory (one platform, one n), retained against that directory's own bf16 greedy at the same loops.
    Duplicates across directories/platforms are resolved by largest n, then platform (p5 > p4d > Babel >
    Lambda).  Lambda cells (Sep 13-15 grid) are included only where no newer record of that config exists."""
    cand = []
    for root in roots:
        root = HERE / root if not Path(root).is_absolute() else Path(root)
        if not root.exists():
            continue
        plat = root.parent.name
        cells = load([root])
        for (model, steps, quant, scope), c in cells.items():
            if quant == "bf16" or scope != "core" or not c["tf"] or "greedy" not in c["gen"]:
                continue
            b = cells.get((model, steps, "bf16", scope))
            if not b or "greedy" not in b["gen"] or not b["tf"]:
                continue
            g, gb = c["gen"]["greedy"], b["gen"]["greedy"]
            idx = sorted(set(g) & set(gb))
            if not idx:
                continue
            t, tb = c["tf"]["summary"], b["tf"]["summary"]
            a = 100 * sum(g[i]["correct"] for i in idx) / len(idx)
            ab = 100 * sum(gb[i]["correct"] for i in idx) / len(idx)
            cand.append({"platform": plat, "dir": root.name, "model": model, "loops": steps, "quant": quant,
                         "n_gen": len(idx), "n_tf": len(c["tf"]["records"]), "acc": a, "acc_bf16": ab,
                         "fid": t["fidelity"], "nll": t["nll"], "nll_bf16": tb["nll"], "kl": t.get("kl")})
    for model, gen_files, nogen in LAMBDA_FILES:
        gen, ng = {}, {}
        for f in gen_files:
            p = LAMBDA / f
            if p.exists():
                for r in json.load(open(p))["records"]:
                    if r.get("acc") is not None:
                        gen.setdefault((r["steps"], r["quant"]["spec"]), r)
        p = LAMBDA / nogen
        if p.exists():
            ng = {(r["steps"], r["quant"]["spec"]): r for r in json.load(open(p))["records"]}
        for (st_, q), g in gen.items():
            b, t, tb = gen.get((st_, "bf16")), ng.get((st_, q)), ng.get((st_, "bf16"))
            if q == "bf16" or not b or not t or not tb or g["n"] != b["n"]:
                continue
            cand.append({"platform": "lambda", "dir": "lambda", "model": model, "loops": st_, "quant": q,
                         "n_gen": g["n"], "n_tf": t.get("n", g["n"]), "acc": 100 * g["acc"], "acc_bf16": 100 * b["acc"],
                         "fid": t["fidelity"], "nll": t["nll"], "nll_bf16": tb["nll"], "kl": None})
    best = {}
    for r in cand:
        k = (r["model"], r["quant"], r["loops"])
        key = (-min(r["n_gen"], r["n_tf"]), PLATFORM_RANK.get(r["platform"], 9))
        if k not in best or key < best[k][0]:
            best[k] = (key, r)
    rows = [v[1] for v in best.values()]
    for r in rows:
        r["ret"] = 100 * r["acc"] / r["acc_bf16"] if r["acc_bf16"] > 0 else float("nan")
        r["collapse"] = r["ret"] < cut
    return [r for r in rows if r["acc_bf16"] >= 10 and not math.isnan(r["ret"])], len(cand)


def best_threshold(rows):
    """Fidelity cut t (collapse predicted iff fid < t) minimising misclassified cells; returns (t, errors),
    t = midpoint of the optimal interval between adjacent sorted fidelities."""
    xs = sorted(set(r["fid"] for r in rows))
    cuts = [xs[0] - 1e-3] + [(a + b) / 2 for a, b in zip(xs, xs[1:])] + [xs[-1] + 1e-3]
    scored = [(sum((r["fid"] < t) != r["collapse"] for r in rows), t) for t in cuts]
    e = min(x[0] for x in scored)
    ts = [t for x, t in scored if x == e]
    return (ts[0] + ts[-1]) / 2, e


def sec_pool(cells, nmin, roots=None, cut=50.0, boots=2000, seed=0):
    import random
    rows, ncand = pool_rows(roots or POOL_ROOTS, cut)
    print(f"## Looped-LM pool: one row per model x quantizer x loops (collapse = retained < {cut:.0f}%)\n")
    print(f"{ncand} candidate records -> {len(rows)} unique cells with bf16 >= 10%. Command: "
          f"`python EdgeTRM/analyze_looped.py pool` (roots: {', '.join(roots or POOL_ROOTS)} + Lambda grid).\n")
    print("| model | platform | loops | quant | n | acc (bf16) | retained | fidelity | ΔNLL % | collapse |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["model"], -r["fid"])):
        print(f"| {r['model']} | {r['platform']} | {r['loops']} | {r['quant']} | {min(r['n_gen'], r['n_tf'])} | "
              f"{fmt(r['acc'])} ({fmt(r['acc_bf16'])}) | {fmt(r['ret'], 1)}% | {r['fid']:.3f} | "
              f"{100 * (r['nll'] - r['nll_bf16']) / r['nll_bf16']:+.0f}% | {'yes' if r['collapse'] else ''} |")
    print("\n| model | cells (held/collapsed) | collapsed max φ | held min φ | consistent cut interval (width) | "
          "cells in 0.75-0.90 | AUROC(1-φ) | AUROC(ΔNLL%) | best φ cut (errors) | bootstrap 95% CI of cut |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rng = random.Random(seed)
    groups_ = defaultdict(list)
    for r in rows:
        groups_[r["model"]].append(r)
    groups_["POOLED"] = rows
    inter = []
    for m, rs in groups_.items():
        col = [r for r in rs if r["collapse"]]
        held = [r for r in rs if not r["collapse"]]
        rel = lambda r: (r["nll"] - r["nll_bf16"]) / r["nll_bf16"]  # noqa: E731
        a_f = auroc([1 - r["fid"] for r in col], [1 - r["fid"] for r in held])
        a_n = auroc([rel(r) for r in col], [rel(r) for r in held])
        nband = sum(0.75 <= r["fid"] <= 0.90 for r in rs)
        lo_, hi_ = max((r["fid"] for r in col), default=float("nan")), min((r["fid"] for r in held), default=float("nan"))
        if m != "POOLED" and lo_ < hi_:
            inter.append((lo_, hi_))
        if col and held:
            t, e = best_threshold(rs)
            bs = []
            for _ in range(boots):
                smp = [rng.choice(rs) for _ in rs]
                if any(r["collapse"] for r in smp) and not all(r["collapse"] for r in smp):
                    bs.append(best_threshold(smp)[0])
            bs.sort()
            ci = f"{bs[int(0.025 * len(bs))]:.3f}–{bs[int(0.975 * len(bs)) - 1]:.3f}" if bs else "–"
            tt = f"{t:.3f} ({e})"
        else:
            tt, ci = "–", "–"
        print(f"| {m} | {len(held)}/{len(col)} | {max((r['fid'] for r in col), default=float('nan')):.3f} | "
              f"{min((r['fid'] for r in held), default=float('nan')):.3f} | "
              f"{'[%.3f, %.3f] (%.3f)' % (lo_, hi_, hi_ - lo_) if lo_ < hi_ else 'overlap'} | {nband} | "
              f"{fmt(a_f, 3)} | {fmt(a_n, 3)} | {tt} | {ci} |")
    if inter:
        lo_all, hi_all = max(a for a, _ in inter), min(b for _, b in inter)
        print(f"\nIntersection of the per-model consistent cut intervals: "
              + (f"[{lo_all:.3f}, {hi_all:.3f}] -> one cut fits every model" if lo_all < hi_all
                 else f"empty ({lo_all:.3f} > {hi_all:.3f}) -> no single cut fits every model"))
    print("\nBootstrap resamples cells within the model (2000 draws); the cut is the midpoint of the error-minimising"
          " interval, so a model without cells on both sides near the boundary has a wide or undefined CI.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sections", nargs="*", default=["boundary", "q6", "tf", "div", "len", "kl", "sweep"])
    ap.add_argument("--roots", nargs="*", default=None)
    ap.add_argument("--nmin", type=int, default=200, help="records needed to call a run complete")
    a = ap.parse_args()
    cells = load(a.roots or DEFAULT_ROOTS)
    if not cells:
        print("no results under", a.roots or DEFAULT_ROOTS)
        return
    fns = {"q6": sec_q6, "tf": sec_tf, "div": sec_div, "len": sec_len, "kl": sec_kl, "sweep": sec_sweep,
           "boundary": sec_boundary, "pool": sec_pool}
    for s in a.sections:
        fns[s](cells, a.nmin)


if __name__ == "__main__":
    main()
