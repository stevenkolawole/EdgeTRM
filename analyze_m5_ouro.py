#!/usr/bin/env python3
"""Read-out of ouro_exit.py summaries: overthinking across loops and the learned exit gate's recovery.

Per (model, weight spec), from per-problem greedy GSM8K correctness at L = 1, 2, 3 loops and at 4 loops (tau 1.0):
  V_rel     share of the 4-loop errors that were solved with fewer loops (visit then lose across loops)
  oracle    accuracy of choosing the best loop count per problem (right at any L)
  gate      the trained exit gate at threshold tau (per token, 4 loops computed): accuracy, mean exit loop
  recovery  (acc_gate - acc_4) / (oracle - acc_4)

    python analyze_m5_ouro.py results/m5/ouro/ouro14_summary.json [...]
"""
import json
import sys

import numpy as np


def load(paths):
    """merge serial-driver summaries (<model>_summary.json) and per-cell files (<model>_<spec>_<cfg>.summary.json)."""
    by_model = {}
    for p in paths:
        model = p.split("/")[-1].split("_")[0]
        by_model.setdefault(model, {}).update(json.load(open(p)))
    return by_model


def main(paths):
    out = {}
    for model, d in load(paths).items():
        specs = sorted({k.split("|")[0] for k in d})
        for spec in specs:
            g = lambda k: d.get(f"{spec}|{k}")  # noqa: E731
            full = g("tau1.0")
            Ls = [g(f"L{i}") for i in (1, 2, 3)]
            if full is None or any(x is None for x in Ls):
                print(f"{model} {spec}: incomplete ({sorted(k for k in d if k.startswith(spec + '|'))})")
                continue
            C = np.array([x["correct"] for x in Ls] + [full["correct"]], bool)   # [4, n]
            fin = C[-1]
            anyc = C.any(0)
            V_abs = (anyc & ~fin).mean() * 100
            V_rel = (anyc & ~fin).sum() / max((~fin).sum(), 1) * 100
            orc = anyc.mean() * 100
            row = {"n": int(C.shape[1]), "acc_by_loops": [float(c.mean() * 100) for c in C], "V_abs": V_abs,
                   "V_rel": V_rel, "oracle": orc, "gate": {}}
            for tau in ("0.7", "0.5"):
                x = g(f"tau{tau}")
                if x is None:
                    continue
                a = x["acc"] * 100
                gap = orc - fin.mean() * 100
                row["gate"][tau] = {"acc": a, "exit_loop": x["mean_exit_loop"],
                                    "recovery": (a - fin.mean() * 100) / gap if gap >= 2 else None,
                                    "degenerate": x["degenerate"] * 100}
            out[f"{model}|{spec}"] = row
            gs = " | ".join(f"gate tau {t}: {v['acc']:.1f} at {v['exit_loop']:.2f} loops, recovery "
                            + (f"{v['recovery']:+.2f}" if v["recovery"] is not None else "-") for t, v in row["gate"].items())
            print(f"{model} {spec:6s} n={row['n']}: acc at 1/2/3/4 loops " + " / ".join(f"{a:.1f}" for a in row["acc_by_loops"])
                  + f" | V_rel {V_rel:.1f}% (V_abs {V_abs:.1f}) | oracle {orc:.1f} | " + gs)
    return out


def cli():
    gen = [p for p in sys.argv[1:] if not p.endswith("_tf.json")]
    tf = [p for p in sys.argv[1:] if p.endswith("_tf.json")]
    res = {"gen": main(gen) if gen else {}, "gen_ci": gen_table(gen) if gen else {}, "tf": tf_table(tf) if tf else {}}
    json.dump(res, open("results/breadth/m5_ouro.json", "w"), indent=1, default=float)


# ----------------------------------------------------------------------------- CIs and the teacher-forced control
RNG = np.random.default_rng(0)


def boot(fn, n, B=2000):
    idx = RNG.integers(0, n, size=(B, n))
    vals = np.array([fn(ix) for ix in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def vrel_gen(C):
    """C: [4, n] bool (L1, L2, L3, L4). share of L4 errors right at some L < 4."""
    def f(ix):
        c = C[:, ix]
        wrong = ~c[-1]
        return 100 * (c[:-1].any(0) & wrong).sum() / max(wrong.sum(), 1)
    return f


def gen_table(paths):
    """free generation: V_rel with a problem bootstrap, per spec and its paired difference to bf16."""
    out = {}
    for model, d in load(paths).items():
        C = {}
        for spec in sorted({k.split("|")[0] for k in d}):
            ks = [f"{spec}|L1", f"{spec}|L2", f"{spec}|L3", f"{spec}|tau1.0"]
            if all(k in d for k in ks):
                C[spec] = np.array([d[k]["correct"] for k in ks], bool)
        for spec, c in C.items():
            n = c.shape[1]
            f = vrel_gen(c)
            v = f(np.arange(n)); lo, hi = boot(f, n)
            row = {"V_rel": v, "ci": [lo, hi], "n": n}
            if spec != "bf16" and "bf16" in C and C["bf16"].shape[1] == n:
                fb = vrel_gen(C["bf16"])
                g = lambda ix: f(ix) - fb(ix)  # noqa: E731
                row["d_vs_bf16"] = g(np.arange(n)); row["d_ci"] = list(boot(g, n))
            out[f"{model}|{spec}"] = row
            print(f"[gen] {model} {spec:6s} n={n}: V_rel {v:5.1f} [{lo:5.1f}, {hi:5.1f}]"
                  + (f"   vs bf16 {row['d_vs_bf16']:+5.1f} [{row['d_ci'][0]:+5.1f}, {row['d_ci'][1]:+5.1f}]" if "d_vs_bf16" in row else ""))
    return out


def unbits(h):
    hx, n = h.split(":")
    return np.unpackbits(np.frombuffer(bytes.fromhex(hx), np.uint8))[:int(n)].astype(bool)


def tf_table(paths):
    """teacher-forced gold solutions (fixed input): token-level V_rel (positions wrong at 4 loops that some L < 4
    predicts), answer-level V_rel on tf_ans_acc, and the gate's recovery of the token-level oracle gap."""
    out = {}
    per = {}
    for p in paths:
        d = json.load(open(p))
        S = d["settings"]
        if not all(k in S for k in ("L1", "L2", "L3", "L4")):
            continue
        n = d["n"]
        T = {k: [unbits(h) for h in S[k]["tok_bits"]] for k in S}
        wrong4 = np.array([(~t).sum() for t in T["L4"]])
        vis = np.array([((~T["L4"][i]) & (T["L1"][i] | T["L2"][i] | T["L3"][i])).sum() for i in range(n)])
        tokf = lambda ix: 100 * vis[ix].sum() / max(wrong4[ix].sum(), 1)  # noqa: E731
        A = np.array([S[k]["tf_ans_acc"] for k in ("L1", "L2", "L3", "L4")], bool)
        ansf = vrel_gen(A)
        row = {"n": n, "tok_acc": {k: S[k]["tok_acc"] for k in S}, "V_tok": tokf(np.arange(n)), "V_tok_ci": list(boot(tokf, n)),
               "V_ans": ansf(np.arange(n)), "V_ans_ci": list(boot(ansf, n)),
               "ans_acc": {k: 100 * float(np.mean(S[k]["tf_ans_acc"])) for k in S}}
        orc_tok = np.array([(T["L1"][i] | T["L2"][i] | T["L3"][i] | T["L4"][i]).sum() for i in range(n)])
        base = np.array([T["L4"][i].sum() for i in range(n)])
        for tau in ("tau0.7", "tau0.5"):
            if tau in T:
                g = np.array([T[tau][i].sum() for i in range(n)])
                gap = orc_tok.sum() - base.sum()
                row[f"recovery_tok_{tau}"] = float((g.sum() - base.sum()) / gap) if gap > 0 else None
        out[f"{d['model']}|{d['spec']}"] = row
        per[(d["model"], d["spec"])] = (vis, wrong4)
        print(f"[tf ] {d['model']} {d['spec']:6s} n={n}: token acc L1-4 " + " / ".join(f"{100 * row['tok_acc'][k]:.1f}" for k in ("L1", "L2", "L3", "L4"))
              + f" | V_tok {row['V_tok']:5.1f} [{row['V_tok_ci'][0]:.1f}, {row['V_tok_ci'][1]:.1f}]"
              + f" | answer acc L1-4 " + " / ".join(f"{row['ans_acc'][k]:.1f}" for k in ("L1", "L2", "L3", "L4"))
              + f" V_ans {row['V_ans']:5.1f} [{row['V_ans_ci'][0]:.1f}, {row['V_ans_ci'][1]:.1f}]"
              + "".join(f" | gate {t} recovers {row[f'recovery_tok_{t}']:+.2f} of the token oracle gap" for t in ("tau0.7", "tau0.5")
                        if row.get(f"recovery_tok_{t}") is not None))
    for (m, s), (vis, w4) in per.items():
        if s != "bf16" and (m, "bf16") in per and len(per[(m, "bf16")][0]) == len(vis):
            vb, wb = per[(m, "bf16")]
            g = lambda ix: 100 * vis[ix].sum() / max(w4[ix].sum(), 1) - 100 * vb[ix].sum() / max(wb[ix].sum(), 1)  # noqa: E731
            d0 = g(np.arange(len(vis))); lo, hi = boot(g, len(vis))
            out[f"{m}|{s}"]["V_tok_d_vs_bf16"] = [d0, lo, hi]
            print(f"[tf ] {m} {s}: V_tok minus bf16 {d0:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    return out


if __name__ == "__main__":
    cli()
