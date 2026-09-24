#!/usr/bin/env python3
"""M4 read-out: the (model, configuration, backend) table with device-faithful labels and the release
check's host-side evidence, threshold fitting, and held-out-category validation (DriftBench rule:
leave one backend / model / configuration family out; never random splits).

Inputs (pulled into EdgeTRM/results/m4/):
  labels.json                 m4_labels.py (X3 host simulations of AI Hub graphs, TFLite chains, devices)
  ev/onnx__*.json             m4_onnx_sweep.py (check on the compiled QDQ graphs)
  ev/tflite__*.json           release_check.py tflite (check on TFLite graphs, cross-runtime T0)
  ev_torch/torch__*.json      m4_torch_sweep.py (model + config path, with its own label run)
  ../breadth/x2_contraction_summary.json   X2 torch-harness cells (delta_T, cell agreement, phi_H)

    python m4_table.py            -> results/m4/m4_table.json, results/m4/m4_readout.md
"""
import json
import pathlib
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE / "results" / "m4"
X2 = HERE / "results" / "breadth" / "x2_contraction_summary.json"
OK_T, BROKEN_T = 0.9, 0.5

# torch emulation config -> compiled-graph cell names (AI Hub QDQ, TFLite AEQ)
MATCH = {"w8c+a8all": {"qnn_qdq_host": "w8a8", "tflite_auto": "int8", "tflite_ref": "int8", "qnn_device": "w8a8"},
         "w4c+a8all": {"qnn_qdq_host": "w4a8", "tflite_auto": "w4a8_c", "qnn_device": "w4a8"},
         "w8c+a16all": {"qnn_qdq_host": "w8a16", "tflite_auto": "w8a16", "tflite_ref": "w8a16", "qnn_device": "w8a16"},
         "w4c+a16all": {"qnn_qdq_host": "w4a16", "tflite_auto": "w4a16_c", "qnn_device": "w4a16"},
         "w4t+a8all": {"tflite_auto": "w4a8_t"},
         "w8c+a8lin": {"tflite_auto": "w8a8_fc"}}
TASK_ALIAS = {"maze_attn": "maze"}          # breadth-harness name -> AI Hub / TFLite name


def cls(r):
    if r is None:
        return None
    return "OK" if r >= OK_T else ("DEGRADED" if r >= BROKEN_T else "BROKEN")


def family_of(backend, cell):
    if backend in ("torch_harness", "x2_harness"):
        if "@" in cell:
            return "weight_group"
        return "weight_noise" if cell.startswith("noise") else "weight_quant"
    if cell.startswith("abl_A"):
        return "act_group_8bit"
    if cell.startswith("abl_B"):
        return "act_group_16bit"
    if cell.endswith("a8all") or cell in ("w8a8", "w4a8", "int8", "w4a8_c", "w4a8_t", "w4a8_allops"):
        return "act8_all"
    if cell.endswith("a16all") or cell in ("w8a16", "w4a16", "w4a16_c", "abl_A0_w8a16"):
        return "act16_all"
    if cell.endswith("a8lin") or cell == "w8a8_fc":
        return "act8_linear_only"
    return "weight_quant" if cell.startswith("w") else ("weight_noise" if cell.startswith("noise") else "other")


def load_rows():
    rows = []
    labels = json.loads((RES / "labels.json").read_text()) if (RES / "labels.json").exists() else []
    lab = {}
    for L in labels:
        if L.get("slice"):                       # the 1000-puzzle slices: kept as a separate check
            continue
        lab[(L["backend"], L["task"], L["cell"], L.get("device"))] = L
    ev = {}
    for f in sorted((RES / "ev").glob("onnx__*.json")):
        e = json.loads(f.read_text())
        ev[("onnx", e["task"], e["cell"])] = e
    for f in sorted((RES / "ev").glob("tflite__*.json")):
        _, task, v = f.stem.split("__")
        e = json.loads(f.read_text())
        ev[("tflite", task, v)] = e
    # graph-mode rows: the check run on the very artifact whose label we have
    for (backend, task, cell, dev), L in lab.items():
        if backend in ("qnn_qdq_host", "qnn_device"):
            e = ev.get(("onnx", task, cell))
        elif backend in ("tflite_auto", "tflite_ref"):
            e = ev.get(("tflite", task, cell))
        else:
            e = None
        if e is None:
            continue
        r = dict(mode="graph", backend=backend, device=dev, task=task, cell=cell, family=family_of(backend, cell),
                 retained=L["retained"], label=cls(L["retained"]), exact=L["exact"], exact_fp32=L["exact_fp32"],
                 n_label=L["n"], evidence_src=e.get("graph"))
        for k in ("delta_T_median", "cell_agree", "cell_agree_settled", "answer_agree", "answer_agree_settled",
                  "phi_H", "qsign_agree", "layer_local_max", "layer_prop_out", "n_nonfinite_tensors",
                  "xrt_max_lsb", "tflm_vs_opt_max_lsb", "onestep_cell_agree", "weight_err_max"):
            r[k] = e.get(k)
        r["xrt_bad"] = bool(e.get("xrt_first_bad")) if backend == "tflite_ref" else False
        rows.append(r)
    # torch path: its own label run (torch_emulation / torch_harness backends) and cross-backend
    # predictions of the matched compiled graphs
    for f in sorted((RES / "ev_torch").glob("torch__*.json")) if (RES / "ev_torch").exists() else []:
        e = json.loads(f.read_text())
        cfg = e["config"]
        task = e["task"]
        base = dict(task=TASK_ALIAS.get(task, task), cell=cfg)
        feats = {k: e.get(k) for k in ("delta_T_median", "cell_agree", "cell_agree_settled", "answer_agree",
                                       "answer_agree_settled", "phi_H", "qsign_agree", "kick_growth_median")}
        ret = e["label_exact"] / e["label_exact_fp32"] if e.get("label_exact_fp32") else None
        rows.append(dict(mode="torch_self", backend=e["backend"], family=family_of(e["backend"], cfg),
                         retained=ret, label=cls(ret), exact=e.get("label_exact"), exact_fp32=e.get("label_exact_fp32"),
                         n_label=e.get("label_n"), **base, **feats))
        for backend, gcell in MATCH.get(cfg, {}).items():
            for (b2, t2, c2, dev), L in lab.items():
                if b2 == backend and t2 == base["task"] and c2 == gcell:
                    rows.append(dict(mode="torch_to_graph", backend=backend, device=dev, family=family_of(backend, gcell),
                                     retained=L["retained"], label=cls(L["retained"]), exact=L["exact"],
                                     exact_fp32=L["exact_fp32"], n_label=L["n"], task=base["task"],
                                     cell=f"{cfg}->{gcell}", **feats))
    # X2 torch-harness cells (weight quantizers, noise, weight groups; labels from the same run)
    if X2.exists():
        d = json.loads(X2.read_text())
        acc0 = {}
        for key, blob in d.items():
            for c in blob["cells"]:
                if c["spec"] == "fp32":
                    acc0[blob["task"]] = c["acc"]
        for key, blob in d.items():
            task = blob["task"]
            if key.endswith("_p2"):          # the DT period-2 map (G o G) re-measures the same cells
                continue
            for c in blob["cells"]:
                if c["spec"] == "fp32" or c.get("deltaT_q") is None:
                    continue
                ret = c["acc"] / acc0[task] if acc0.get(task) else None
                rows.append(dict(mode="x2", backend="x2_harness", task=task, cell=f"{c['spec']}#{c.get('draw', 0)}",
                                 family=family_of("x2_harness", c["spec"]), retained=ret, label=cls(ret),
                                 exact=c["acc"], exact_fp32=acc0.get(task), n_label=c["n"],
                                 delta_T_median=c["deltaT_q"][1], cell_agree=c.get("agree_mean"), phi_H=c.get("phi_H"),
                                 kick_growth_median=(c.get("kick_q_refcorrect") or [None, None])[1]))
    return rows


# ------------------------------------------------------------------------------------------ fitting
PRED = {  # name -> (feature, direction): +1 means larger is worse
    "T1 delta_T": ("delta_T_median", +1),
    "T2 cell agreement (all)": ("cell_agree", -1),
    "T2 cell agreement (fp32-settled)": ("cell_agree_settled", -1),
    "T2 answer agreement (fp32-settled)": ("answer_agree_settled", -1),
    "T2 phi_H": ("phi_H", -1),
    "T0 max local act. error": ("layer_local_max", +1),
    "T2 halting-sign agreement": ("qsign_agree", -1),
}


def score_of(r, feat, sign):
    v = r.get(feat)
    return None if v is None else sign * float(v)


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ties
    for v in np.unique(s):
        m = s == v
        ranks[m] = ranks[m].mean()
    npos, nneg = y.sum(), (~y).sum()
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def fit_threshold(s, y):
    """Threshold on score s (larger = worse) flagging y (bad): maximise balanced accuracy, ties broken
    toward the midpoint of the widest gap."""
    s, y = np.asarray(s, float), np.asarray(y, bool)
    cands = np.unique(s)
    mids = np.concatenate([[cands[0] - 1e-9], (cands[:-1] + cands[1:]) / 2, [cands[-1] + 1e-9]])
    best, bt = -1, None
    for t in mids:
        pred = s > t
        tpr = (pred & y).sum() / max(y.sum(), 1)
        tnr = (~pred & ~y).sum() / max((~y).sum(), 1)
        ba = (tpr + tnr) / 2 if y.any() and (~y).any() else (tpr if y.any() else tnr)
        if ba > best + 1e-12:
            best, bt = ba, t
    return float(bt), float(best)


def envelope_threshold(s, y):
    """Conservative gate: pass only what lies inside the range of scores seen on OK cells (threshold =
    the largest OK score in training). No training false rejects; false accepts only where a bad
    cell scores inside the OK envelope."""
    s, y = np.asarray(s, float), np.asarray(y, bool)
    return float(s[~y].max()), None


def heldout(rows, pred, key, bad=lambda r: r["label"] != "OK", rule="balanced"):
    feat, sign = PRED[pred]
    R = [r for r in rows if score_of(r, feat, sign) is not None and r["label"] is not None]
    cats = sorted({r[key] for r in R})
    out = []
    for c in cats:
        tr = [r for r in R if r[key] != c]
        te = [r for r in R if r[key] == c]
        ytr = [bad(r) for r in tr]
        if not any(ytr) or all(ytr):
            out.append(dict(category=c, n=len(te), note="training set has one class"))
            continue
        fit = fit_threshold if rule == "balanced" else envelope_threshold
        t, _ = fit([score_of(r, feat, sign) for r in tr], ytr)
        yte = np.array([bad(r) for r in te])
        pte = np.array([score_of(r, feat, sign) > t for r in te])
        out.append(dict(category=c, n=len(te), n_bad=int(yte.sum()), threshold=sign * t,
                        correct=int((pte == yte).sum()), false_accept=int((~pte & yte).sum()),
                        false_reject=int((pte & ~yte).sum()),
                        false_accept_cells=[r["cell"] + "|" + r["task"] for r, p_, y_ in zip(te, pte, yte) if y_ and not p_],
                        false_reject_cells=[r["cell"] + "|" + r["task"] for r, p_, y_ in zip(te, pte, yte) if p_ and not y_]))
    return out


def summarize(rows, subset_name, R):
    lines = [f"\n### {subset_name} ({len(R)} rows; not-OK = retained < {OK_T})\n",
             "| predictor | rows | AUROC (not-OK) | fitted threshold (all rows) | balanced acc. |",
             "|---|---|---|---|---|"]
    res = {}
    for pred, (feat, sign) in PRED.items():
        S = [(score_of(r, feat, sign), r["label"] != "OK") for r in R if score_of(r, feat, sign) is not None
             and r["label"] is not None]
        if len(S) < 4:
            continue
        s, y = zip(*S)
        a = auroc(s, y)
        t, ba = fit_threshold(s, y) if any(y) and not all(y) else (None, None)
        res[pred] = dict(n=len(S), auroc=a, threshold=None if t is None else sign * t, bal_acc=ba)
        lines.append(f"| {pred} | {len(S)} | {a if a is None else round(a, 3)} | "
                     f"{'-' if t is None else round(sign * t, 4)} | {'-' if ba is None else round(ba, 3)} |")
    return lines, res


def main():
    rows = load_rows()
    RES.mkdir(parents=True, exist_ok=True)
    out = dict(rows=rows)
    md = ["# M4 read-out (generated by m4_table.py)", ""]
    by = defaultdict(int)
    for r in rows:
        by[(r["mode"], r["backend"])] += 1
    md.append("Rows by mode/backend: " + ", ".join(f"{k[0]}/{k[1]}: {v}" for k, v in sorted(by.items())))
    for name, sel in (("graph mode: check run on the labelled artifact", lambda r: r["mode"] == "graph"),
                      ("torch path, cross-backend (model + config -> compiled graph label)", lambda r: r["mode"] == "torch_to_graph"),
                      ("torch path, own label", lambda r: r["mode"] == "torch_self"),
                      ("X2 torch harness", lambda r: r["mode"] == "x2"),
                      ("all rows", lambda r: True)):
        R = [r for r in rows if sel(r)]
        if not R:
            continue
        lines, res = summarize(rows, name, R)
        md += lines
        out.setdefault("summary", {})[name] = res
    # held-out validation on the pooled table (every row that carries the feature)
    md.append("\n## Held-out categories (threshold fitted on the other categories, applied unchanged)\n")
    out["heldout"] = {}
    for pred, key, rule in [(p_, k_, r_) for p_ in PRED for r_ in ("balanced", "envelope")
                            for k_ in ("backend", "task", "family")]:
        if True:
            h = heldout(rows, pred, key, rule=rule)
            out["heldout"][f"{pred} | {key} | {rule}"] = h
            tot = sum(x.get("n", 0) for x in h if "correct" in x)
            cor = sum(x.get("correct", 0) for x in h)
            fa = sum(x.get("false_accept", 0) for x in h)
            fr = sum(x.get("false_reject", 0) for x in h)
            md.append(f"- **{pred}** [{rule}], leave one {key} out: {cor}/{tot} correct, false accepts {fa}, "
                      f"false rejects {fr}")
            for x in h:
                if "correct" in x and (x["false_accept"] or x["false_reject"]):
                    md.append(f"    - {key}={x['category']}: thr {x['threshold']:.4g}, {x['correct']}/{x['n']} "
                              f"(FA {x['false_accept_cells']}, FR {x['false_reject_cells']})")
    # the table itself
    md.append("\n## Table\n")
    md.append("| mode | backend | task | cell | family | retained | label | delta_T | cell agr. | cell agr. settled | "
              "phi_H | T0 local max | xrt LSB |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    def f(v, d=3):
        return "-" if v is None else (f"{v:.{d}g}" if isinstance(v, float) else str(v))
    for r in sorted(rows, key=lambda r: (r["mode"], r["backend"], r["task"], r["cell"])):
        if r["mode"] == "x2":
            continue
        md.append(f"| {r['mode']} | {r['backend']}{' ' + r['device'] if r.get('device') else ''} | {r['task']} | "
                  f"{r['cell']} | {r['family']} | {f(r['retained'])} | {r['label']} | {f(r.get('delta_T_median'))} | "
                  f"{f(r.get('cell_agree'))} | {f(r.get('cell_agree_settled'))} | {f(r.get('phi_H'))} | "
                  f"{f(r.get('layer_local_max'))} | {f(r.get('xrt_max_lsb'))} |")
    (RES / "m4_table.json").write_text(json.dumps(out, indent=1, default=float))
    (RES / "m4_readout.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:80]))


if __name__ == "__main__":
    main()
