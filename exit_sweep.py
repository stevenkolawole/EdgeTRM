#!/usr/bin/env python3
"""M5 (halting as early exit): per-step records for exit rules, at H=1 granularity.

One trajectory per (task, quantizer spec) from the model's own initial carry, run with
H_cycles = 1 for K steps and read out after EVERY step. Because one H=h supervision step is
exactly h consecutive H=1 steps (TRM: L L-updates then one H update per H cycle; HRM: the same
order; X3 verified identical accuracies for H x n fixed), this single run holds the default-H
trajectory at every h-th step AND the finer "inner step" exits in between. The checkpoint
default H (TRM 3, HRM 2) is written into the file; K = h x (number of default-H steps).

Per puzzle and step (arrays [n, K], stored in one compressed .npz per spec):
  correct   puzzle-exact correctness of the argmax answer
  qh, qc    halting logits q_halt, q_continue (TRM decides q_halt > 0; HRM q_halt > q_continue)
  qh32, qc32  the same logits from the unquantized fp32 halting head applied to the (compressed)
            state z_H[:, 0] without activation hooks (the head is read-only, so this is the exit an
            fp32 head would take on the same trajectory)
  resH      ||z_H(k) - z_H(k-1)|| / ||z_H(k)||  (convergence residual, per H=1 step)
  msp       min over answer-bearing cells of the max softmax probability (confidence exit)
  chg       number of answer-bearing cells whose argmax changed since the previous step
  nwrong    wrong answer-bearing cells (clipped to 255)
Maze tasks with --maze_flags also store valid / vshort (valid simple S-G path; valid and
shortest; detector_sweep.maze_check) at every step.

Spec syntax = breadth_sweep / detector_sweep (fp32, w4c, w3g32, w4c+a8, noise0.1@s1, anoise0.3@s0), plus an
optional suffix "%qh": the halting head (inner.q_head, hidden x 2) is kept in fp32 and outside the
activation quantizer / noise hooks, everything else as the spec says (is an over- or under-confident
exit caused by the head's own weights or by the state it reads?).
Resumable: specs whose npz exists are skipped.

    python exit_sweep.py --task sudoku_mlp --K 1536 --specs fp32,w4c,w3g32 --n 1000 \
        --outdir ~/breadth_results/m5/sudoku_mlp
    --offset 1000: rows 1000.. of the test set (held-out); --phi 4,8,16: store phi = cos(z_H, fp32 z_H) at those
    default-H steps from an in-process fp32 reference (same trajectories as the correctness records).
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from breadth_sweep import DEV, TRM_TASKS, HRM_TASKS, _ActQuant, _ActNoise, _linears, build, load_qa, parse_quant, quantize_weights_  # noqa: E402
from detector_sweep import SUD_BLANK, MZ_WALL, bfs_dist, explicit_carry, maze_check, split_spec  # noqa: E402

DEFAULT_H = {"sudoku_mlp": 3, "sudoku_attn": 3, "maze_attn": 3, "hrm_sudoku": 2, "hrm_maze": 2}


@torch.no_grad()
def trajectory(m, x, y, K, kind, h_def, bfs=None, maze_flags=False, head32=None, zsteps=(), zref=None):
    """zsteps: H=1 step indices (0-based) at which z_H is captured (zref None: returned as "zcap", fp16 on the CPU)
    or compared with the reference (zref = {k: [B, S, D]}: cosine per instance, returned as "phi" [B, len(zsteps)])."""
    B = x.shape[0]
    zcap, phi = {}, []
    bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
    carry = explicit_carry(m, bt)
    ans = (x == SUD_BLANK) if kind == "sudoku" else (x != MZ_WALL)
    big = torch.full(ans.shape, 2.0, device=DEV)
    prevH = carry.inner_carry.z_H.float()
    prevP = None
    rec = {k: torch.zeros(B, K, device=DEV, dtype=d) for k, d in
           (("correct", torch.bool), ("qh", torch.float16), ("qc", torch.float16), ("resH", torch.float16),
            ("msp", torch.float16), ("chg", torch.int16), ("nwrong", torch.uint8), ("qh32", torch.float16), ("qc32", torch.float16))}
    mz = {"valid": np.zeros((B, K), bool), "vshort": np.zeros((B, K), bool)} if maze_flags else None
    for k in range(K):
        carry, out = m(carry, bt)
        zH = carry.inner_carry.z_H.float()
        logits = out["logits"]
        pred = logits.argmax(-1)
        wrong = pred != y
        rec["correct"][:, k] = ~wrong.any(1)
        rec["qh"][:, k] = out["q_halt_logits"].float()
        rec["qc"][:, k] = out["q_continue_logits"].float()
        if head32 is not None:
            z0 = carry.inner_carry.z_H[:, 0]
            q32 = torch.nn.functional.linear(z0, head32[0].to(z0.dtype), head32[1].to(z0.dtype)).float()
            rec["qh32"][:, k], rec["qc32"][:, k] = q32[:, 0], q32[:, 1]
        rec["resH"][:, k] = (zH - prevH).flatten(1).norm(dim=1) / zH.flatten(1).norm(dim=1).clamp_min(1e-8)
        p = torch.softmax(logits.float(), -1).amax(-1)
        rec["msp"][:, k] = torch.where(ans, p, big).amin(1)
        rec["chg"][:, k] = ((pred != prevP) & ans).sum(1).to(torch.int16) if prevP is not None else ans.sum(1).to(torch.int16)
        rec["nwrong"][:, k] = (wrong & ans).sum(1).clamp_max(255).to(torch.uint8)
        if maze_flags:
            chk = maze_check(pred.cpu().numpy(), x.cpu().numpy(), bfs)
            mz["valid"][:, k] = chk["valid"]
            mz["vshort"][:, k] = chk["vshort"]
        if k in zsteps:
            if zref is None:
                zcap[k] = carry.inner_carry.z_H.detach().half().cpu()
            else:
                phi.append(torch.nn.functional.cosine_similarity(zH.flatten(1), zref[k].to(DEV).float().flatten(1), dim=-1))
        prevH, prevP = zH, pred
        # keep the model from resetting the carry at its own halt_max_steps
        carry.halted.zero_()
    out = {k: v.cpu().numpy() for k, v in rec.items()}
    if mz is not None:
        out.update(mz)
    if zsteps and zref is None:
        out["zcap"] = zcap
    if phi:
        out["phi"] = torch.stack(phi, 1).float().cpu().numpy()
    return out


def run_spec(m, inp, lab, batch, K, kind, h_def, bfs, maze_flags, act_quant=None, act_noise=None, keep_head=False, head32=None,
             zsteps=(), zref=None):
    m = m.to(DEV)
    handles, obs = [], []
    lins = [mod for mod in _linears(m) if not (keep_head and mod is m.inner.q_head)]
    if act_noise is not None:
        for mod in lins:
            handles.append(mod.register_forward_pre_hook(_ActNoise(act_noise)))
    if act_quant is not None:
        for mod in lins:
            q = _ActQuant(act_quant)
            obs.append(q)
            handles.append(mod.register_forward_pre_hook(q))
    parts = []
    for ci, s in enumerate(range(0, inp.shape[0], batch)):
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        bb = bfs[s:s + batch] if bfs is not None else None
        if act_quant is not None and ci == 0:        # calibrate static scales on the first chunk (16 default steps)
            trajectory(m, x, y, 16 * h_def, kind, h_def)
            for q in obs:
                q.freeze()
        zr = None if zref is None else {k: v[s:s + x.shape[0]] for k, v in zref.items()}
        parts.append(trajectory(m, x, y, K, kind, h_def, bb, maze_flags, head32, zsteps, zr))
        print(f"    chunk {ci} done ({s + x.shape[0]}/{inp.shape[0]})", flush=True)
    for h in handles:
        h.remove()
    m.cpu()
    torch.cuda.empty_cache()
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0] if k != "zcap"}
    if "zcap" in parts[0]:
        out["zcap"] = {k: torch.cat([p["zcap"][k] for p in parts]) for k in parts[0]["zcap"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TRM_TASKS) + list(HRM_TASKS))
    ap.add_argument("--specs", default="fp32")
    ap.add_argument("--K", type=int, required=True, help="H=1 steps (a multiple of the default H)")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--maze_flags", action="store_true")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--allow_cpu", action="store_true")
    ap.add_argument("--ckpt", default="", help="TRM: a trained variant's step_N file (breadth_sweep --ckpt semantics)")
    ap.add_argument("--offset", type=int, default=0, help="first test row (held-out rows: e.g. 1000)")
    ap.add_argument("--phi", default="", help="default-H steps (e.g. 4,8,16) at which to store phi = cos(z_H, z_H of fp32 at the same "
                    "step, same puzzle), from an fp32 reference run in the same process (M5 overthinking: phi and V from one trajectory)")
    a = ap.parse_args()
    assert a.allow_cpu or torch.cuda.is_available(), "NO CUDA"
    if a.ckpt:
        import breadth_sweep as bs
        bs.CKPT_OVERRIDE = a.ckpt
    t = TRM_TASKS.get(a.task) or HRM_TASKS[a.task]
    kind = t["data"]
    h_def = DEFAULT_H[a.task]
    assert a.K % h_def == 0
    batch = a.batch or t["batch"]
    inp, lab = load_qa(kind, a.n + a.offset)
    inp, lab = inp[a.offset:], lab[a.offset:]
    bfs = np.array([bfs_dist(g, 30) for g in inp]) if kind == "maze" else None
    od = Path(a.outdir)
    od.mkdir(parents=True, exist_ok=True)
    specs = [s.strip() for s in a.specs.split(",") if s.strip()]
    print(f"{a.task}: n={len(inp)} batch={batch} K={a.K} (H=1 steps; default H={h_def}) specs={specs}", flush=True)
    phi_steps = [int(v) for v in a.phi.split(",") if v]
    zsteps, zref = tuple(v * h_def - 1 for v in phi_steps), None
    assert all(0 <= k < a.K for k in zsteps)
    if phi_steps and any(not (od / f"{s}.npz").exists() for s in specs):
        t1 = time.time()
        m, meta = build(a.task, 1, a.K, batch)
        torch.manual_seed(0)
        r = run_spec(m, inp, lab, batch, a.K, kind, h_def, bfs, a.maze_flags and kind == "maze", zsteps=zsteps)
        del m
        zref = r.pop("zcap")
        print(f"  fp32 reference for phi at default-H steps {phi_steps} ({time.time() - t1:.0f}s)", flush=True)
    for spec in specs:
        path = od / f"{spec}.npz"
        if path.exists():
            print(f"  {spec}: exists, skipped", flush=True)
            continue
        keep_head = spec.endswith("%qh")
        base, seed = split_spec(spec[:-3] if keep_head else spec)
        q = parse_quant("fp32" if base.startswith("gq") else base)        # gq<fmt>: GPTQ weights (m5_gptq.py)
        t1 = time.time()
        m, meta = build(a.task, 1, a.K, batch)             # H=1, halt_max_steps = K
        torch.manual_seed(seed if seed is not None else 0)
        w_head = m.inner.q_head.weight.data.clone()
        head32 = (w_head.to(DEV), m.inner.q_head.bias.data.clone().to(DEV))
        if q["gran"] is not None:
            quantize_weights_(m, q["bits"], q["gran"], q.get("sigma"))
        if base.startswith("gq"):
            import m5_gptq
            m5_gptq.load_if_gptq(m, a.task, base)
        if keep_head:
            m.inner.q_head.weight.data = w_head
        r = run_spec(m, inp, lab, batch, a.K, kind, h_def, bfs, a.maze_flags and kind == "maze",
                     act_quant=q["act"], act_noise=q.get("anoise"), keep_head=keep_head, head32=head32, zsteps=zsteps, zref=zref)
        del m
        info = {"task": a.task, "ckpt": a.ckpt, "spec": spec, "quant": q, "seed": seed, "keep_head": keep_head, "H_run": 1, "H_default": h_def, "K": a.K,
                "phi_steps": phi_steps, "offset": a.offset,
                "n": int(len(inp)), "secs": time.time() - t1, "host": socket.gethostname(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "torch": torch.__version__}
        tmp = str(path) + ".tmp.npz"
        np.savez_compressed(tmp, info=json.dumps(info), **r)
        os.replace(tmp, path)
        c = r["correct"]
        msg = " ".join(f"{s}:{100 * c[:, s * h_def - 1].mean():.1f}" for s in (1, 2, 4, 8, 16, 64, 512) if s * h_def <= a.K)
        print(f"  {spec:14s} exact at default-H step {msg} ({time.time() - t1:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
