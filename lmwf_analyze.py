"""LMWF read-out: fresh per-loop weight noise on looped LMs at full size (Babel L40S, GSM8K test n=1319, batch 4 in
every cell). Paired comparison of each wf arm against the same-platform, same-batch bf16 cell over the same problems
(the s4_wf.py logic of READOUT_SEP29 section 4, generalised to n problems and a results root).

    python EdgeTRM/lmwf_analyze.py --root EdgeTRM/results/looped/babel/lmwf [--n 1319] [--arms wf0.05,wf0.1]

Layout under --root (as written by the 20_LMWF_*.sbatch jobs into babel:~/lmwf/results/):
    <model>_<arm>/<model>_s32_<arm>_core/{greedy.jsonl,tf2.json,summary.json}
Per-problem correctness comes from greedy.jsonl, per-problem teacher-forced quantities from tf2.json["records"];
intervals are paired bootstrap (4000 resamples, seed 0) over the problems present in BOTH cells. Prediction (a):
wf0.05 - bf16 > 0 with the interval excluding zero on both models; (b): wf0.05 inside the interval, wf0.1 below.
"""
import argparse
import json
import os

import numpy as np

rng = np.random.default_rng(0)


def boot(d, B=4000):
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    m = d[idx].mean(1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load(d):
    g = [json.loads(l) for l in open(d + "/greedy.jsonl") if l.strip()]
    g = {r["idx"]: r for r in g}
    t = json.load(open(d + "/tf2.json"))
    rec = {r["idx"]: r for r in t["records"]}
    s = json.load(open(d + "/summary.json"))
    return g, rec, s


def table(name, ref, arms, n):
    gr, tr, sr = load(ref)
    ids = sorted(i for i in gr if i < n)
    print(f"\n### {name}\n")
    print(f"reference: {ref} (n={sr['meta']['n']}, batch {sr['meta']['batch']}, {sr['meta']['gpu']}, transformers "
          f"{sr['meta']['transformers']}, script {sr['meta']['script_sha1']}); {len(ids)} problems decoded; greedy acc "
          f"on them = {100 * np.mean([gr[i]['correct'] for i in ids]):.1f}, summary acc {sr['gen']['greedy']['acc']}")
    if len(ids) < n:
        print(f"WARNING: reference incomplete ({len(ids)}/{n} rows); intervals below are over the common subset")
    print("\n| arm | batch | n paired | greedy acc | acc - bf16 [paired 95%] | wins/losses vs bf16 | tf NLL | "
          "NLL - bf16 [paired 95%] | fidelity | KL(bf16||arm) | tf agree | degenerate |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    fa_all = {i: float(gr[i]["correct"]) for i in ids}
    print(f"| bf16 | {sr['meta']['batch']} | {len(ids)} | {100 * np.mean(list(fa_all.values())):.1f} | - | - | "
          f"{np.mean([tr[i]['nll'] for i in ids]):.4f} | - | {np.mean([tr[i]['fidelity'] for i in ids]):.4f} | 0 | "
          f"1.000 | {sr['gen']['greedy']['degenerate']:.3f} |")
    for lab, d in arms:
        if not os.path.exists(d + "/greedy.jsonl"):
            print(f"| {lab} | missing {d} |")
            continue
        g, t, s = load(d)
        com = [i for i in ids if i in g and i in t]
        if len(com) < n:
            print(f"| {lab} | INCOMPLETE {len(com)}/{n} rows paired |")
        if not com:
            continue
        a = np.array([g[i]["correct"] for i in com], float)
        fa = np.array([fa_all[i] for i in com])
        nl = np.array([t[i]["nll"] for i in com])
        fn = np.array([tr[i]["nll"] for i in com])
        m, lo, hi = boot(a - fa)
        mn, ln, hn = boot(nl - fn)
        wins = int(((a == 1) & (fa == 0)).sum())
        loss = int(((a == 0) & (fa == 1)).sum())
        verdict = "above bf16" if lo > 0 else ("below bf16" if hi < 0 else "interval covers zero")
        print(f"| {lab} | {s['meta']['batch']} | {len(com)} | {100 * a.mean():.1f} | {100 * m:+.1f} [{100 * lo:+.1f}, "
              f"{100 * hi:+.1f}] ({verdict}) | {wins}/{loss} | {nl.mean():.4f} | {mn:+.4f} [{ln:+.4f}, {hn:+.4f}] | "
              f"{np.mean([t[i]['fidelity'] for i in com]):.4f} | {np.mean([t[i]['kl'] for i in com]):.4f} | "
              f"{np.mean([t[i]['tf_agree'] for i in com]):.3f} | {s['gen']['greedy']['degenerate']:.3f} |")
        print(f"   summary: acc {s['gen']['greedy']['acc']} nll {s['tf']['nll']:.4f} fid {s['tf']['fidelity']:.4f} "
              f"kl {s['tf']['kl']:.4f} quant {s['meta']['quant']} n_quantized_linears {t.get('n_quantized_linears')} "
              f"gpu {s['meta']['gpu']} transformers {s['meta']['transformers']} script {s['meta']['script_sha1']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="EdgeTRM/results/looped/babel/lmwf")
    ap.add_argument("--n", type=int, default=1319)
    ap.add_argument("--models", default="huginn,olmo_rec32")
    ap.add_argument("--arms", default="wf0.05,wf0.1")
    ap.add_argument("--steps", type=int, default=32)
    a = ap.parse_args()
    for model in a.models.split(","):
        cell = lambda arm: f"{a.root}/{model}_{arm}/{model}_s{a.steps}_{arm}_core"  # noqa: E731
        table(f"{model}, {a.steps} loops, GSM8K test[:{a.n}], greedy, batch-matched", cell("bf16"),
              [(arm, cell(arm)) for arm in a.arms.split(",")], a.n)


if __name__ == "__main__":
    main()
