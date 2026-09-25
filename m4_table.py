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
    if "@" in cell:
        return "vendor_emulation"
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
        lvl = e.get("ort_level", "all")
        ev[("onnx" if lvl == "all" else "onnx_basic", e["task"], e["cell"])] = e
    for f in sorted((RES / "ev").glob("tflite__*.json")):
        _, task, v = f.stem.split("__")
        e = json.loads(f.read_text())
        ev[("tflite", task, v)] = e
    # graph-mode rows: the check run on the very artifact whose label we have
    # device labels, predicted by the host check under each kernel model (literal QDQ, fused RMSNorm)
    for (backend, task, cell, dev), L in lab.items():
        if backend != "qnn_device":
            continue
        for suf, tag in (("", "literal QDQ"), ("@qnnfused", "fused norm, 8-bit out"),
                         ("@qnnfused16", "fused norm, float stream")):
            e = ev.get(("onnx_basic", task, cell + suf)) or ev.get(("onnx", task, cell + suf))
            if e is None:
                continue
            r = dict(mode="device_from_host_check", backend=backend, device=dev, task=task,
                     cell=f"{cell} [{tag}]", family=family_of(backend, cell), retained=L["retained"],
                     label=cls(L["retained"]), exact=L["exact"], exact_fp32=L["exact_fp32"], n_label=L["n"])
            for k in ("delta_T_median", "cell_agree", "cell_agree_settled", "answer_agree", "answer_agree_settled",
                      "changed_agree", "changed_agree_settled", "phi_H", "qsign_agree", "layer_local_max",
                      "n_nonfinite_tensors"):
                r[k] = e.get(k)
            rows.append(r)
    for (backend, task, cell, dev), L in lab.items():
        if backend == "qnn_device":
            continue
        if backend == "qnn_qdq_host":
            e = ev.get(("onnx", task, cell))
        elif backend == "qnn_qdq_ref":
            e = ev.get(("onnx_basic", task, cell))
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
                  "changed_agree", "changed_agree_settled",
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
                                       "changed_agree", "changed_agree_settled",
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
    "T2 written-cell agreement (all)": ("changed_agree", -1),
    "T2 written-cell agreement (fp32-settled)": ("changed_agree_settled", -1),
    "T2 phi_H": ("phi_H", -1),
    "T0 max local act. error": ("layer_local_max", +1),
    "T2 halting-sign agreement": ("qsign_agree", -1),
    "reference-free: random-kick growth (16 steps)": ("kick_growth_median", +1),
    "T2 check quantity (settled if available)": ("_t2", -1),
    "CHECK: T0 hard rules + T2 written cells (settled)": ("_combined:changed_agree_settled", -1),
    "CHECK: T0 hard rules + T2 whole answer (settled)": ("_combined:answer_agree_settled", -1),
}


def t2_score(r):
    """The check's T2 quantity: cell agreement on fp32-settled inputs when available, else all inputs."""
    for k in ("changed_agree_settled", "changed_agree", "cell_agree_settled", "cell_agree"):
        if r.get(k) is not None:
            return r[k]
    return None


def hard_fail(r):
    """T0 rules that are not fitted: kernel disagreement between runtimes (> 16 LSB) and non-finite
    intermediates where fp32 is finite."""
    return bool(r.get("xrt_bad")) or bool(r.get("n_nonfinite_tensors"))


def score_of(r, feat, sign):
    if feat.startswith("_combined"):
        key = feat.split(":")[1] if ":" in feat else None
        t2 = r.get(key) if key else t2_score(r)
        if t2 is None and key:
            t2 = r.get(key.replace("_settled", ""))
        if t2 is None:
            return None
        return 1e6 if hard_fail(r) else -float(t2)
    if feat == "_t2":
        t2 = t2_score(r)
        return None if t2 is None else -float(t2)
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



# ------------------------------------------------------------------------------------------ case studies
def _ev(name):
    f = RES / "ev" / name
    return json.loads(f.read_text()) if f.exists() else None


def _evt(task, cfg):
    f = RES / "ev_torch" / f"torch__{task}__{cfg}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _verdict(e):
    if e is None:
        return None, []
    import sys
    sys.path.insert(0, str(HERE))
    import release_check as R
    return R.decide(e)


def _fmt_ev(e):
    if e is None:
        return "not run"
    v, rules = _verdict(e)
    parts = [f"**{v}**", f"delta_T {e.get('delta_T_median', float('nan')):.3f}"]
    for k, lab in (("changed_agree_settled", "written-cell agr. (settled)"), ("answer_agree_settled", "answer agr. (settled)"),
                   ("cell_agree", "cell agr.")):
        if e.get(k) is not None:
            parts.append(f"{lab} {e[k]:.3f}")
    if e.get("n_nonfinite_tensors"):
        parts.append(f"{e['n_nonfinite_tensors']} inf/nan tensors")
    if e.get("layer_local_top"):
        t = e["layer_local_top"][0]
        parts.append(f"top local Q/DQ error {t['rel_err']:.2f} at {t['tensor'].split('/')[-1]} ({t['op']})")
    if e.get("layer_first_prop_gt_0.5"):
        parts.append(f"first tensor with propagated error > 0.5: {e['layer_first_prop_gt_0.5']['tensor'].split('/inner/')[-1]}")
    if e.get("xrt_first_bad"):
        b = e["xrt_first_bad"]
        parts.append(f"cross-runtime: op {b.get('op_index')} {b.get('op')} "
                     + (f"fails on reference kernels" if b.get("error") else f"differs {b.get('maxdiff'):.0f} LSB (inputs {b.get('in_diff_lsb', 0):.1f})"))
    if e.get("onnx_xrt_rel_median") is not None:
        parts.append(f"ORT basic-vs-all one-step carry diff {e['onnx_xrt_rel_median']:.3f}")
    return "; ".join(parts)


def _lab(labels, backend, task, cell, device=None):
    for L in labels:
        if L["backend"] == backend and L["task"] == task and L["cell"] == cell and not L.get("slice") \
                and (device is None or L.get("device") == device):
            return f"{L['exact'] * 100:.0f}% vs fp32 {L['exact_fp32'] * 100:.0f}% (n={L['n']}, retained {L['retained']:.2f})"
    return "no label"


def case_studies():
    labels = json.loads((RES / "labels.json").read_text()) if (RES / "labels.json").exists() else []
    md = ["\n## Case studies\n"]
    md.append("### A. Sudoku-MLP w8a8 (AI Hub all-op 8-bit activations; RMSNorm variance path)\n")
    md.append(f"- Labels: QDQ host sim (ORT all, X3) {_lab(labels, 'qnn_qdq_host', 'sudoku_mlp', 'w8a8')}; "
              f"reference QDQ semantics {_lab(labels, 'qnn_qdq_ref', 'sudoku_mlp', 'w8a8')}; "
              f"S24 on device {_lab(labels, 'qnn_device', 'sudoku_mlp', 'w8a8')}; "
              f"TFLite int8 (XNNPACK) {_lab(labels, 'tflite_auto', 'sudoku_mlp', 'int8')}.")
    for name, e in (("check on the exported QDQ graph (ORT all)", _ev("onnx__sudoku_mlp__w8a8.json")),
                    ("check on the exported QDQ graph (reference QDQ semantics)", _ev("onnx__sudoku_mlp__w8a8#basic.json")),
                    ("QNN kernel emulation: fused RMSNorm, 8-bit norm output", _ev("onnx__sudoku_mlp__w8a8@qnnfused#basic.json") or _ev("onnx__sudoku_mlp__w8a8@qnnfused.json")),
                    ("QNN kernel emulation: fused RMSNorm, normalised stream unquantized", _ev("onnx__sudoku_mlp__w8a8@qnnfused16#basic.json") or _ev("onnx__sudoku_mlp__w8a8@qnnfused16.json")),
                    ("model + config before export: w8c + every op output 8-bit (toolchain emulation)", _evt("sudoku_mlp", "w8c+a8all")),
                    ("model + config, harness-style: w8c + 8-bit linear inputs only", _evt("sudoku_mlp", "w8c+a8lin")),
                    ("TFLite int8 graph", _ev("tflite__sudoku_mlp__int8.json"))):
        md.append(f"- {name}: {_fmt_ev(e)}")
    md.append("\n### B. TFLite Micro REDUCE_SUM (reference kernels)\n")
    for task in ("sudoku_mlp", "sudoku_attn", "maze"):
        for v in ("int8", "w8a16"):
            e = _ev(f"tflite__{task}__{v}.json")
            if e is None:
                continue
            md.append(f"- {task} {v}: labels XNNPACK {_lab(labels, 'tflite_auto', task, v)}, reference kernels "
                      f"{_lab(labels, 'tflite_ref', task, v)}; check: {_fmt_ev(e)}")
    md.append("\n### C. ARC w8a8 (partial loss)\n")
    for v in ("w8a8", "w4a8", "w8a16", "w4a16"):
        e = _ev(f"onnx__arc__{v}.json")
        md.append(f"- ARC {v}: label {_lab(labels, 'qnn_qdq_host', 'arc', v)}; check: {_fmt_ev(e)}")
    md.append("\n### D. Attention models\n")
    for task in ("sudoku_attn", "maze"):
        for v in ("w8a8", "w4a8", "w8a16", "w4a16"):
            e = _ev(f"onnx__{task}__{v}.json")
            md.append(f"- {task} {v}: label {_lab(labels, 'qnn_qdq_host', task, v)}; check: {_fmt_ev(e)}")
    return md


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
                      ("device labels from the host check", lambda r: r["mode"] == "device_from_host_check"),
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
    md += case_studies()
    (RES / "m4_table.json").write_text(json.dumps(out, indent=1, default=float))
    (RES / "m4_readout.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:80]))


if __name__ == "__main__":
    main()
