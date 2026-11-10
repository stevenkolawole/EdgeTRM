#!/usr/bin/env python3
"""Second-route re-derivation of the n=500 looped-LM accuracies from greedy.jsonl (no analyze_looped import).
Accuracy = mean of the per-problem `correct`; also recomputed from gold vs pred (numeric equality) as a cross-check. Wilson 95%
half-width; retained = compressed / bf16 of the same family. Root: results/looped/p4d (HEAD export when the working tree lacks the files).
    python looped_n500_check.py ROOT
"""
import json, math, sys, glob, re, os
root = sys.argv[1]
def wil(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * c - 100 * h, 100 * c + 100 * h, 100 * h
def readrows(path):
    """one record per line; a line holding a truncated record followed by a complete one (a write race) yields the complete one"""
    out, bad = [], 0
    for l in open(path).read().split("\n"):
        if not l.strip():
            continue
        try:
            out.append(json.loads(l))
        except Exception:
            bad += 1
            j = l.find('{"idx"', 5)
            if j > 0:
                try: out.append(json.loads(l[j:]))
                except Exception: pass
    readrows.bad = bad
    d = {}
    for r in out:
        d[r["idx"]] = r                       # a duplicated idx (resumed run) keeps the last record
    readrows.dups = len(out) - len(d)
    return list(d.values())
def num(s):
    try: return float(str(s).replace(",", "").strip())
    except Exception: return None
for sub, pat in (("huginn_p4d_n500", "huginn_s32_*_core"), ("olmo_p4d_n500", "olmo_rec32_s32_*_core"), ("n500", "ouro*_core")):
    cells = sorted(glob.glob(f"{root}/{sub}/{pat}"))
    res = {}
    for c in cells:
        rows = readrows(f"{c}/greedy.jsonl")
        k = sum(int(r["correct"]) for r in rows); n = len(rows)
        k2 = sum(1 for r in rows if num(r["pred"]) is not None and num(r["gold"]) is not None and abs(num(r["pred"]) - num(r["gold"])) < 1e-6)
        idx = sorted(r["idx"] for r in rows)
        res[os.path.basename(c)] = (k, n, k2, (idx == list(range(n)), readrows.bad, readrows.dups))
    print(f"== {sub}")
    fam = {}
    for name, (k, n, k2, ok) in res.items():
        m = re.match(r"(.*?)_(bf16|w\w+)_core", name); f, q = m.group(1), m.group(2)
        fam.setdefault(f, {})[q] = (k, n, k2, ok)
    for f, qs in fam.items():
        b = qs.get("bf16")
        for q, (k, n, k2, ok) in sorted(qs.items()):
            lo, hi, h = wil(k, n)
            ret = f"{100 * (k / n) / (b[0] / b[1]):.0f}%" if b and b[0] else "-"
            print(f"  {f:20s} {q:6s} n={n} correct {k} ({100 * k / n:.1f} +-{h:.1f}) retained {ret}  recount-from-gold {k2} {'MATCH' if k2 == k else 'DIFF'} (idx==0..n-1, corrupt lines, duplicate idx) {ok}")
