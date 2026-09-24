#!/usr/bin/env python3
"""X8 (angles C and D): label-free "the answer is wrong" detectors side by side, per quantizer.

For one task and a list of quantizer specs, runs nsup supervision steps from the model's own
initial carry and records, per puzzle, at the steps in --record (default 1,2,4,8,12,16; halting logit and exact correctness at every step):
  correct            puzzle-exact correctness (every cell equals the reference)
  n_wrong            wrong cells among the answer-bearing cells (Sudoku: blanks; maze: non-wall)
  qhalt              the ACT halting logit (trained with BCE on exact correctness)
  resid_H, resid_L   ||z(k) - z(k-1)|| / ||z(k)||   (convergence residual)
  msp_mean, msp_min  max-softmax probability over the answer-bearing cells (mean, min)
  mrg_mean, mrg_min  top-1 minus top-2 probability over the same cells (mean, min)
  chk                hand-coded, label-free checker score (0 = passes)
                       Sudoku: row/col/box conflict count E(y) (Ren & Liu 2601.10679 sec. 5.2)
                               + given-cell mismatches + non-digit cells
                       Maze:   format violations (walls/S/G changed, path on a wall) + degree
                               violations (S, G need degree 1, path cells degree 2) + extra
                               connected components; 0 <=> a simple S-to-G path
and at the final step only:
  valid, vshort      maze: valid S-to-G path; valid and as short as the BFS shortest path
  plen, conflicts, given_mis, nondigit, n_format, n_degree, n_comp   (checker parts)
  fid_H              cosine(z_H, z_H of fp32 on the same puzzle)  (reference-based)
  agree_fp32         fraction of answer-bearing cells whose prediction equals fp32's
  pi_*               path independence (Anil et al. 2211.09961): a second trajectory of the
                     same (compressed) model from a random initial carry z_H, z_L ~ N(0, 1)
                     (same draw for every spec); pi_dis = fraction of answer-bearing cells whose
                     answers differ, pi_same = identical answer, pi_cos = cos(z_H, z_H'),
                     pi_correct = exact correctness of the second trajectory.
Correctness definitions for mazes (valid / valid-shortest / exact) are computed here from the
predicted grid, so the read-out can score every detector under each.

Spec syntax: breadth_sweep specs (fp32, w8c, w4t, w3g32, w4c+a8, noise0.1, anoise0.3, ...) with
an optional "@s<seed>" suffix that seeds the weight-noise / activation-noise draw.  Resumable:
specs already in --out are skipped (fp32 is always rerun in-process as the reference).

    python detector_sweep.py --task sudoku_mlp --specs fp32,w8c,w4t,noise0.1@s0 --n 1000 \
        --out ~/breadth_results/x8/det_sudoku_mlp.json
"""
import argparse
import json
import os
import socket
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import breadth_sweep as bs  # noqa: E402
from breadth_sweep import DEV, TRM_TASKS, HRM_TASKS, _ActQuant, _ActNoise, _linears, build, load_qa, parse_quant, quantize_weights_  # noqa: E402

# token ids (breadth_sweep.load_qa: value + 1)
SUD_BLANK = 1                       # "." in the question
MZ_WALL, MZ_OPEN, MZ_S, MZ_G, MZ_PATH = 1, 2, 3, 4, 5    # "# SGo" + 1

_units = []
for i in range(9):
    _units.append([i * 9 + j for j in range(9)])                     # rows
    _units.append([j * 9 + i for j in range(9)])                     # cols
for bi in range(3):
    for bj in range(3):
        _units.append([(bi * 3 + a) * 9 + bj * 3 + b for a in range(3) for b in range(3)])
SUD_UNITS = torch.tensor(_units, dtype=torch.long)                   # [27, 9]


def split_spec(spec):
    if "@s" in spec:
        base, s = spec.split("@s")
        return base, int(s)
    return spec, None


# ----------------------------------------------------------------------------- checkers
def sudoku_check(pred, inp):
    """pred, inp: [B, 81] token ids (torch, any device). Returns dict of [B] int tensors."""
    digit = pred - 1                                                 # 1..9 for tokens 2..10
    isdig = (pred >= 2) & (pred <= 10)
    oh = F.one_hot(digit.clamp(0, 9), 10)[..., 1:] * isdig[..., None]   # [B, 81, 9]
    cnt = oh[:, SUD_UNITS.to(pred.device), :].sum(2)                 # [B, 27, 9]
    conflicts = (cnt - 1).clamp_min(0).sum((1, 2))
    given = inp != SUD_BLANK
    given_mis = (given & (pred != inp)).sum(1)
    nondigit = (~isdig).sum(1)
    return {"conflicts": conflicts, "given_mis": given_mis, "nondigit": nondigit,
            "chk": conflicts + given_mis + nondigit}


def bfs_dist(grid, side):
    """Shortest S->G distance in edges over non-wall cells of one input maze (flat array)."""
    s = int(np.where(grid == MZ_S)[0][0]); g = int(np.where(grid == MZ_G)[0][0])
    dist = {s: 0}
    q = deque([s])
    while q:
        u = q.popleft()
        if u == g:
            return dist[u]
        r, c = divmod(u, side)
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < side and 0 <= cc < side:
                v = rr * side + cc
                if v not in dist and grid[v] != MZ_WALL:
                    dist[v] = dist[u] + 1
                    q.append(v)
    return -1


def maze_check(pred, inp, bfs, side=30):
    """pred, inp: [B, side*side] numpy int arrays; bfs: [B] shortest S->G distance (edges).
    valid <=> no format violation, S and G have degree 1, every path cell degree 2, one component
    (then the node set is a simple S-to-G path).  plen = path cells + 1 (edges)."""
    B = pred.shape[0]
    fixed = (inp == MZ_WALL) | (inp == MZ_S) | (inp == MZ_G)
    n_format = ((fixed & (pred != inp)) | ((inp == MZ_OPEN) & (pred != MZ_OPEN) & (pred != MZ_PATH))).sum(1)
    P = (pred == MZ_PATH) & (inp == MZ_OPEN)
    N = P | (inp == MZ_S) | (inp == MZ_G)
    Ng = N.reshape(B, side, side)
    deg = np.zeros_like(Ng, dtype=np.int32)
    deg[:, 1:, :] += Ng[:, :-1, :]; deg[:, :-1, :] += Ng[:, 1:, :]
    deg[:, :, 1:] += Ng[:, :, :-1]; deg[:, :, :-1] += Ng[:, :, 1:]
    deg = deg.reshape(B, -1)
    req = np.where((inp == MZ_S) | (inp == MZ_G), 1, 2)
    n_degree = (N & (deg != req)).sum(1)
    n_comp = np.zeros(B, np.int64)
    for b in range(B):
        nodes = np.flatnonzero(N[b])
        if len(nodes) == 0:
            continue
        seen = set()
        nset = set(nodes.tolist())
        comps = 0
        for s0 in nodes.tolist():
            if s0 in seen:
                continue
            comps += 1
            st = [s0]; seen.add(s0)
            while st:
                u = st.pop()
                r, c = divmod(u, side)
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < side and 0 <= cc < side:
                        v = rr * side + cc
                        if v in nset and v not in seen:
                            seen.add(v); st.append(v)
        n_comp[b] = comps
    chk = n_format + n_degree + np.maximum(n_comp - 1, 0)
    valid = chk == 0
    plen = P.sum(1) + 1
    vshort = valid & (plen == bfs)
    return {"n_format": n_format, "n_degree": n_degree, "n_comp": n_comp, "chk": chk,
            "valid": valid, "vshort": vshort, "plen": plen}


# ----------------------------------------------------------------------------- run
def explicit_carry(m, bt, zH0=None, zL0=None):
    """Carry that the model will NOT reset (halted=False), starting from (zH0, zL0); default
    = the model's own H_init / L_init broadcast, which is what reset_carry would produce."""
    c = m.initial_carry(bt)
    ic = c.inner_carry
    B, S, D = ic.z_H.shape
    if zH0 is None:
        zH0 = m.inner.H_init.to(DEV).view(1, 1, D).expand(B, S, D).clone()
        zL0 = m.inner.L_init.to(DEV).view(1, 1, D).expand(B, S, D).clone()
    return type(c)(inner_carry=type(ic)(z_H=zH0, z_L=zL0),
                   steps=torch.zeros(B, dtype=torch.int32, device=DEV),
                   halted=torch.zeros(B, dtype=torch.bool, device=DEV),
                   current_data={k: v for k, v in bt.items()})


def soft_stats(logits, ansmask):
    p = torch.softmax(logits.float(), -1)
    top2 = p.topk(2, -1).values
    msp, mrg = top2[..., 0], top2[..., 0] - top2[..., 1]
    big = torch.full_like(msp, 2.0)
    cnt = ansmask.sum(1).clamp_min(1)
    return {"msp_mean": (msp * ansmask).sum(1) / cnt, "msp_min": torch.where(ansmask, msp, big).amin(1),
            "mrg_mean": (mrg * ansmask).sum(1) / cnt, "mrg_min": torch.where(ansmask, mrg, big).amin(1)}


@torch.no_grad()
def trajectory(m, x, y, nsup, record, kind, bfs=None, init=None, keep_final=False):
    """One nsup-step trajectory for a batch. Returns {k: {signal: np array [B]}} for k in record,
    plus the final z_H / predictions if keep_final."""
    B = x.shape[0]
    bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
    carry = explicit_carry(m, bt, *(init or (None, None)))
    ansmask = (x == SUD_BLANK) if kind == "sudoku" else (x != MZ_WALL)
    prevH, prevL = carry.inner_carry.z_H.float(), carry.inner_carry.z_L.float()
    out_rec, final = {}, {"q_all": [], "c_all": []}
    for k in range(1, nsup + 1):
        carry, out = m(carry, bt)
        zH, zL = carry.inner_carry.z_H.float(), carry.inner_carry.z_L.float()
        final["q_all"].append(out["q_halt_logits"].float())
        final["c_all"].append((out["logits"].argmax(-1) == y).all(1))
        if k in record or k == nsup:
            logits = out["logits"]
            pred = logits.argmax(-1)
            wrong = (pred != y)
            r = {"correct": (~wrong).all(1), "n_wrong": (wrong & ansmask).sum(1), "n_wrong_all": wrong.sum(1),
                 "qhalt": out["q_halt_logits"].float(),
                 "resid_H": (zH - prevH).flatten(1).norm(dim=1) / zH.flatten(1).norm(dim=1).clamp_min(1e-8),
                 "resid_L": (zL - prevL).flatten(1).norm(dim=1) / zL.flatten(1).norm(dim=1).clamp_min(1e-8)}
            r.update(soft_stats(logits, ansmask))
            if kind == "sudoku":
                r.update(sudoku_check(pred, x))
                r = {kk: vv.cpu().numpy() for kk, vv in r.items()}
            else:
                r = {kk: vv.cpu().numpy() for kk, vv in r.items()}
                r.update(maze_check(pred.cpu().numpy(), x.cpu().numpy(), bfs))
            out_rec[k] = r
            if k == nsup and keep_final:
                final.update({"zH": zH, "pred": pred})
        prevH, prevL = zH, zL
    final["q_all"] = torch.stack(final["q_all"], 1).cpu().numpy()        # [B, nsup]
    final["c_all"] = torch.stack(final["c_all"], 1).cpu().numpy()
    return out_rec, final


def rnd_init(B, S, D, chunk_id):
    g = torch.Generator().manual_seed(12345 + chunk_id)
    return torch.randn(B, S, D, generator=g).to(DEV), torch.randn(B, S, D, generator=g).to(DEV)


FINAL_ONLY = ("valid", "vshort", "plen", "conflicts", "given_mis", "nondigit", "n_format", "n_degree", "n_comp",
              "n_wrong_all")
STEP_KEYS = ("correct", "n_wrong", "qhalt", "resid_H", "resid_L", "msp_mean", "msp_min", "mrg_mean", "mrg_min", "chk")


def to_list(a):
    a = np.asarray(a)
    if a.dtype == bool:
        return a.astype(int).tolist()
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(int).tolist()
    return [round(float(v), 5) for v in a]


@torch.no_grad()
def run_spec(m, inp, lab, batch, nsup, record, kind, bfs, act_quant=None, act_noise=None, ref=None, pi=True):
    m = m.to(DEV)
    n = inp.shape[0]
    handles, obs = [], []
    if act_noise is not None:
        for mod in _linears(m):
            handles.append(mod.register_forward_pre_hook(_ActNoise(act_noise)))
    if act_quant is not None:
        for mod in _linears(m):
            q = _ActQuant(act_quant)
            obs.append(q)
            handles.append(mod.register_forward_pre_hook(q))
    steps = {k: {} for k in record}
    fin = {}
    keep = {"zH": [], "pred": []}
    allk = {"q_all": [], "c_all": []}
    for ci, s in enumerate(range(0, n, batch)):
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        bb = bfs[s:s + batch] if bfs is not None else None
        if act_quant is not None and ci == 0:                     # calibrate static scales, then freeze
            trajectory(m, x, y, nsup, set(), kind, bb)
            for q in obs:
                q.freeze()
        rec, final = trajectory(m, x, y, nsup, set(record), kind, bb, keep_final=True)
        allk["q_all"].append(np.round(final["q_all"], 3)); allk["c_all"].append(final["c_all"].astype(np.int8))
        for k in record:
            for kk, vv in rec[k].items():
                if k != nsup and kk in FINAL_ONLY:
                    continue
                steps[k].setdefault(kk, []).append(vv)
        B = x.shape[0]
        ansmask = (x == SUD_BLANK) if kind == "sudoku" else (x != MZ_WALL)
        if ref is None:
            keep["zH"].append(final["zH"].half().cpu())
            keep["pred"].append(final["pred"].to(torch.uint8).cpu())
        else:
            rz = ref["zH"][s:s + B].to(DEV).float()
            rp = ref["pred"][s:s + B].to(DEV).long()
            fin.setdefault("fid_H", []).append(F.cosine_similarity(final["zH"].flatten(1), rz.flatten(1), dim=-1).cpu().numpy())
            agree = ((final["pred"] == rp) & ansmask).sum(1).float() / ansmask.sum(1).clamp_min(1)
            fin.setdefault("agree_fp32", []).append(agree.cpu().numpy())
            fin.setdefault("same_fp32", []).append((final["pred"] == rp).all(1).cpu().numpy())
        if pi:
            S, D = final["zH"].shape[1:]
            rec2, final2 = trajectory(m, x, y, nsup, {nsup}, kind, bb, init=rnd_init(B, S, D, ci), keep_final=True)
            dis = ((final2["pred"] != final["pred"]) & ansmask).sum(1).float() / ansmask.sum(1).clamp_min(1)
            fin.setdefault("pi_dis", []).append(dis.cpu().numpy())
            fin.setdefault("pi_same", []).append((final2["pred"] == final["pred"]).all(1).cpu().numpy())
            fin.setdefault("pi_cos", []).append(F.cosine_similarity(final2["zH"].flatten(1), final["zH"].flatten(1), dim=-1).cpu().numpy())
            fin.setdefault("pi_correct", []).append(rec2[nsup]["correct"])
            fin.setdefault("pi_qhalt", []).append(rec2[nsup]["qhalt"])
    for h in handles:
        h.remove()
    m.cpu()
    torch.cuda.empty_cache()
    steps = {k: {kk: np.concatenate(vv) for kk, vv in v.items()} for k, v in steps.items()}
    fin = {kk: np.concatenate(vv) for kk, vv in fin.items()}
    fin_all = {kk: np.concatenate(vv) for kk, vv in allk.items()}
    refout = None
    if ref is None:
        refout = {"zH": torch.cat(keep["zH"]), "pred": torch.cat(keep["pred"])}
    return steps, (fin, fin_all), refout


def pack(task, spec, q, seed, meta, n, nsup, record, steps, fin, secs):
    fin, fin_all = fin
    last = steps[nsup]
    rec = {"task": task, "spec": spec, "quant": q, "seed": seed, "H_cycles": meta["H_cycles"], "nsup": nsup, "n": n,
           "acc": float(np.mean(last["correct"])), "valid_acc": float(np.mean(last["valid"])) if "valid" in last else None,
           "vshort_acc": float(np.mean(last["vshort"])) if "vshort" in last else None,
           "secs": secs, "host": socket.gethostname(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__,
           "final": {kk: to_list(vv) for kk, vv in last.items()},
           "steps": {str(k): {kk: to_list(vv) for kk, vv in steps[k].items() if kk in STEP_KEYS} for k in record if k != nsup}}
    rec["final"].update({kk: to_list(vv) for kk, vv in fin.items()})
    rec["per_step"] = {kk: vv.tolist() for kk, vv in fin_all.items()}     # [n][nsup]: halting logit, exact
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TRM_TASKS) + list(HRM_TASKS))
    ap.add_argument("--specs", default="fp32")
    ap.add_argument("--nsup", type=int, default=16)
    ap.add_argument("--record", default="1,2,4,8,12,16")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--no_pi", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow_cpu", action="store_true", help="local logic smoke only")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA"
    t = TRM_TASKS.get(a.task) or HRM_TASKS[a.task]
    kind = t["data"]
    batch = a.batch or t["batch"]
    record = sorted({int(k) for k in a.record.split(",")} | {a.nsup})
    inp, lab = load_qa(kind, a.n)
    bfs = None
    meta_task = {"n": int(len(inp))}
    if kind == "maze":
        bfs = np.array([bfs_dist(g, 30) for g in inp])
        lab_chk = maze_check(lab, inp, bfs)
        meta_task.update({"bfs": bfs.tolist(), "ref_plen": lab_chk["plen"].tolist(), "ref_valid": lab_chk["valid"].astype(int).tolist(),
                          "ref_vshort": lab_chk["vshort"].astype(int).tolist(), "n_open": (inp != MZ_WALL).sum(1).tolist()})
        print(f"reference paths: valid {lab_chk['valid'].mean()*100:.1f}%  shortest {lab_chk['vshort'].mean()*100:.1f}%", flush=True)
    else:
        lab_chk = sudoku_check(torch.from_numpy(lab), torch.from_numpy(inp))
        meta_task.update({"n_blank": (inp == SUD_BLANK).sum(1).tolist()})
        print(f"reference grids: conflict-free {(lab_chk['chk'] == 0).float().mean()*100:.1f}%", flush=True)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    if out_path.exists():
        try:
            records = json.load(open(out_path)).get("records", [])
        except (json.JSONDecodeError, ValueError):
            print(f"{out_path} unreadable; starting fresh", flush=True)
    done = {r["spec"] for r in records}
    specs = [s.strip() for s in a.specs.split(",") if s.strip()]
    print(f"{a.task}: n={len(inp)} batch={batch} nsup={a.nsup} record={record} specs={specs}; done={sorted(done)}", flush=True)

    def dump():
        tmp = str(out_path) + ".tmp"
        json.dump({"task": a.task, "meta": meta_task, "records": records}, open(tmp, "w"))
        os.replace(tmp, out_path)

    t0 = time.time()
    torch.manual_seed(0)
    m, meta = build(a.task, None, a.nsup, batch)
    steps, fin, ref = run_spec(m, inp, lab, batch, a.nsup, record, kind, bfs, pi=not a.no_pi)
    del m
    if "fp32" not in done:
        records.append(pack(a.task, "fp32", parse_quant("fp32"), None, meta, len(inp), a.nsup, record, steps, fin, time.time() - t0))
        dump()
    print(f"  fp32        acc {np.mean(steps[a.nsup]['correct'])*100:6.2f} ({time.time()-t0:.0f}s)", flush=True)
    for spec in specs:
        if spec == "fp32" or spec in done:
            continue
        base, seed = split_spec(spec)
        q = parse_quant(base)
        t1 = time.time()
        mq, _ = build(a.task, None, a.nsup, batch)
        torch.manual_seed(seed if seed is not None else 0)
        if q["gran"] is not None:
            quantize_weights_(mq, q["bits"], q["gran"], q.get("sigma"))
        steps, fin, _ = run_spec(mq, inp, lab, batch, a.nsup, record, kind, bfs, act_quant=q["act"],
                                 act_noise=q.get("anoise"), ref=ref, pi=not a.no_pi)
        del mq
        rec = pack(a.task, spec, q, seed, meta, len(inp), a.nsup, record, steps, fin, time.time() - t1)
        records.append(rec)
        dump()
        extra = f" valid {rec['valid_acc']*100:5.1f} vshort {rec['vshort_acc']*100:5.1f}" if rec["valid_acc"] is not None else ""
        print(f"  {spec:12s} acc {rec['acc']*100:6.2f}{extra} fid {np.mean(fin[0]['fid_H']):.3f} ({time.time()-t1:.0f}s)", flush=True)
    print("wrote", out_path, len(records), "records", flush=True)


if __name__ == "__main__":
    main()
