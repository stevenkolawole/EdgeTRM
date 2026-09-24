#!/usr/bin/env python3
"""Q7 (trainable basin width): evaluate TRM checkpoints for basin width and per-tensor INT4 fidelity.

For one step_N checkpoint of a pretrain.py / pretrain_q7.py run (all_config.yaml beside it), on the first
n Sudoku-Extreme test puzzles at the checkpoint's H_cycles and n_sup=16 (fp32 forward, breadth_sweep.py's
harness and quantizers):
  * fp32, w8c, w4c, w4t (per-tensor INT4, the target), w4g32, w3g32
  * the noise cliff: Gaussian weight noise sigma x per-tensor std on every CastedLinear, sigma in
    SIGMAS, draws 0..D-1, each cell seeded by (draw, sigma) so a cell does not depend on the order run
  * jac_fro2: Hutchinson estimate of ||J||_F^2/dim of one L-level step z_L -> f(z_L, z_H + x) at the
    final carry (the quantity arm "jac" penalises), 8 probe vectors on the first 256 puzzles; jac_spec_*:
    top singular value of that J per puzzle (20 power iterations on J^T J), mean / p90 / max over puzzles
Every record carries per-puzzle correctness. Output: one JSON per checkpoint, rewritten after every cell
and resumed cell by cell.

    python q7_eval.py --ckpt ~/q7/ckpt/q7_base_s0/step_6510 --out ~/q7/ckpt/q7_base_s0/eval_step_6510.json
    python q7_eval.py --scan ~/q7/ckpt          # every step_N under every run lacking a complete eval

pretrain_q7.py calls evaluate_checkpoint() in-process after each checkpoint save (no extra GPU process).
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_sweep as bs  # noqa: E402  (adds TinyRecursiveModels to sys.path)

QUANTS = ["w8c", "w4c", "w4t", "w4g32", "w3g32"]
SIGMAS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]
DRAWS = 3


def _cells(quants, sigmas, draws):
    cells = [("fp32", 0)] + [(q, 0) for q in quants]
    cells += [(f"noise{s}", d) for s in sigmas for d in range(draws)]
    return cells


@torch.no_grad()
def _jac_stats(m, inp, lab, probes=8, n=256, iters=20, chunk=64):
    """||J||_F^2/dim (Hutchinson, `probes` vectors) and the top singular value per puzzle (power iteration on
    J^T J) of z_L -> L_level(z_L, z_H + x) at the carry reached after n_sup ACT steps; J is block-diagonal
    over puzzles, so puzzles are processed in chunks of `chunk` (memory)."""
    from torch.nn.attention import sdpa_kernel, SDPBackend
    dev = bs.DEV
    m = m.to(dev)
    inner = m.inner
    cs = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None
    fro, specs = [], []
    for c0 in range(0, min(n, len(inp)), chunk):
        x = torch.from_numpy(inp[c0:min(n, c0 + chunk)]).to(dev)
        y = torch.from_numpy(lab[c0:min(n, c0 + chunk)]).to(dev)
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(x.shape[0], dtype=torch.int32, device=dev)}
        c = m.initial_carry(bt)
        ic = c.inner_carry
        carry = type(c)(inner_carry=type(ic)(z_H=ic.z_H.to(dev), z_L=ic.z_L.to(dev)), steps=c.steps.to(dev),
                        halted=c.halted.to(dev), current_data={k: v.to(dev) for k, v in c.current_data.items()})
        for _ in range(m.config.halt_max_steps):
            carry, _ = m(carry, bt)
            if carry.halted.all():
                break
        inj = carry.inner_carry.z_H + inner._input_embeddings(x, bt["puzzle_identifiers"])

        def f(zz):
            return inner.L_level(zz, inj, cos_sin=cs)

        z0 = carry.inner_carry.z_L.detach()
        with torch.enable_grad():
            z = z0.clone().requires_grad_(True)
            out = f(z)
            for _ in range(probes):                      # Hutchinson: E||v^T J||^2 = ||J||_F^2
                v = torch.randn_like(out)
                (g,) = torch.autograd.grad(out, z, v, retain_graph=True)
                fro.append((float(g.pow(2).sum()), v.numel()))
            dims = tuple(range(1, z0.ndim))
            shp = (-1,) + (1,) * (z0.ndim - 1)
            v = torch.randn_like(z0)
            v = v / v.flatten(1).norm(dim=1).view(shp)
            s2 = None
            for _ in range(iters):
                with sdpa_kernel([SDPBackend.MATH]):      # forward-mode AD: only the math SDPA kernel has it
                    _, jv = torch.func.jvp(f, (z0,), (v,))
                (w,) = torch.autograd.grad(out, z, jv, retain_graph=True)
                s2 = (v * w).sum(dims)                    # v^T J^T J v with |v| = 1
                v = w / w.flatten(1).norm(dim=1).clamp_min(1e-12).view(shp)
            specs.append(s2.clamp_min(0).sqrt().float().cpu().numpy())
            del out, z, w, jv, v
        torch.cuda.empty_cache()
    spec = np.concatenate(specs)
    # per-probe estimate pooled over chunks
    per = np.array([a / b for a, b in fro]).reshape(-1, probes)          # chunks x probes
    wts = np.array([b for _, b in fro]).reshape(-1, probes)[:, 0]
    est = (per * wts[:, None]).sum(0) / wts.sum()
    m.cpu()
    torch.cuda.empty_cache()
    return {"jac_fro2": float(est.mean()), "jac_fro2_sd": float(est.std()),
            "jac_spec_mean": float(spec.mean()), "jac_spec_p90": float(np.percentile(spec, 90)),
            "jac_spec_max": float(spec.max())}


SENS_SIGMAS = (0.02, 0.1)


@torch.no_grad()
def _act_map(m, zH, zL, x):
    inner = m.inner
    cs = inner.rotary_emb() if hasattr(inner, "rotary_emb") else None
    for _ in range(inner.config.H_cycles):
        for _ in range(inner.config.L_cycles):
            zL = inner.L_level(zL, zH + x, cos_sin=cs)
        zH = inner.L_level(zH, zL, cos_sin=cs)
    return zH, zL


@torch.no_grad()
def _sens_stats(build, prep, inp, lab, batch, nsup, n=256, draws=DRAWS, tag=""):
    """X2's one-step sensitivity: delta_T = ||F~(s_T) - F(s_T)|| / ||s_T|| with s_T = (z_H, z_L) the end state
    (n_sup ACT steps) of the model `prep` produces, F one full ACT step of that model and F~ the same map
    with Gaussian weight noise sigma x per-tensor std on every CastedLinear (after `prep`). S_sigma = median
    over puzzles of delta_T / sigma, median over `draws` draws. Returns {f"S{tag}_{sigma}": ..., spreads}."""
    dev = bs.DEV
    m = build()
    prep(m)
    _, _, zH, zL = bs.run(m, inp[:n], lab[:n], batch, nsup)
    m = m.to(dev)
    x = m.inner._input_embeddings(torch.from_numpy(inp[:n]).to(dev),
                                  torch.zeros(n, dtype=torch.int32, device=dev)).float()
    zH, zL = zH.to(dev), zL.to(dev)
    h0, l0 = _act_map(m, zH, zL, x)
    norm = torch.sqrt(zH.pow(2).flatten(1).sum(1) + zL.pow(2).flatten(1).sum(1))
    m.cpu()
    out = {}
    for sg in SENS_SIGMAS:
        per = []
        for d in range(draws):
            mq = build()
            prep(mq)
            torch.manual_seed(100003 * d + int(round(sg * 10000)) + 13)
            bs.quantize_weights_(mq, None, "noise", sg)
            mq = mq.to(dev)
            h1, l1 = _act_map(mq, zH, zL, x)
            dl = torch.sqrt((h1 - h0).pow(2).flatten(1).sum(1) + (l1 - l0).pow(2).flatten(1).sum(1)) / norm
            per.append(float(dl.median()) / sg)
            del mq
        out[f"S{tag}_{sg}"] = float(np.median(per))
        out[f"S{tag}_{sg}_draws"] = per
    torch.cuda.empty_cache()
    return out


def _lsq_split(ckpt):
    """pretrain_q7 checkpoints may carry a learned weight step `...q7_lsq_s` (weight-LSQ) and a frozen
    activation scale `...q7_act_s` (w+a runs) per CastedLinear. Returns (path of a copy without them,
    {module: weight step}, {module: activation scale}); the plain path and empty dicts otherwise."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    extra = {k: v for k, v in sd.items() if k.endswith("q7_lsq_s") or k.endswith("q7_act_s")}
    if not extra:
        return str(ckpt), {}, {}
    clean = str(ckpt) + ".nolsq"
    if not os.path.exists(clean):
        torch.save({k: v for k, v in sd.items() if k not in extra}, clean + ".tmp")
        os.replace(clean + ".tmp", clean)

    def names(suffix):
        return {bs._clean({k[: -len(suffix)]: 0}).popitem()[0]: float(v) for k, v in extra.items()
                if k.endswith(suffix)}
    return clean, names(".q7_lsq_s"), names(".q7_act_s")


class _StaticActQ:
    """Activation fake-quant at a stored (trained) per-tensor scale."""

    def __init__(self, s, qp):
        self.s, self.qp = s, qp

    def __call__(self, mod, args):
        x = args[0]
        return (torch.clamp(torch.round(x / self.s), -self.qp, self.qp) * self.s,) + tuple(args[1:])


def _attach_actq(m, scales, bits=4):
    for name, mod in m.named_modules():
        if mod.__class__.__name__ == "CastedLinear":
            mod.register_forward_pre_hook(_StaticActQ(scales[name], 2 ** (bits - 1) - 1))


@torch.no_grad()
def _quantize_lsq_(m, steps, bits=4):
    """Round every CastedLinear to its learned per-tensor step: clamp(round(W/s), -Qp, Qp) * s."""
    qp = 2 ** (bits - 1) - 1
    n = 0
    for name, mod in m.named_modules():
        if mod.__class__.__name__ == "CastedLinear":
            s = steps[name]
            mod.weight.data = torch.clamp(torch.round(mod.weight.data / s), -qp, qp) * s
            n += 1
    assert n == len(steps), (n, len(steps))


QNOISE_SIGMAS = [0.05, 0.1, 0.15, 0.2, 0.3]


def _run_cfg(ckpt):
    import yaml
    f = Path(ckpt).parent / "all_config.yaml"
    return yaml.safe_load(f.read_text()) if f.exists() else {}


def _qd(spec):
    """Record descriptor of a cell spec."""
    base, _, nz = spec.partition("~")
    if base in ("w4lsq", "fp32+a4s", "w4t+a4s", "w4lsq+a4s"):
        d = {"spec": spec, "bits": None if base.startswith("fp32") else 4,
             "gran": "lsq" if base.startswith("w4lsq") else ("t" if base.startswith("w4t") else None),
             "act": "a4s" if base.endswith("+a4s") else None}
    else:
        d = dict(bs.parse_quant(base), spec=spec)
    if nz:
        d["noise_after_quant"] = float(nz[5:])
    return d


def evaluate_checkpoint(ckpt, out, n=1000, nsup=16, quants=None, sigmas=None, draws=DRAWS, batch=256,
                        label="", log=print, ref_ckpt=None, quant_noise=None, light=None):
    """Evaluate one checkpoint into `out` (resumable). Leaves the caller's RNG states untouched.
    From the run's all_config.yaml: a QAT-style run (q7_reg qat/lsq or activation fake-quant) also gets the
    noise cliff AROUND ITS QUANTIZED POINT (cells 'w4t~noiseS', 'w4lsq~noiseS': quantize, then Gaussian
    noise S x per-tensor std of the quantized weight), because its latent fp32 weights need not work; and a
    run started from another checkpoint (load_checkpoint) gets `fidelity_H_ref` in every record, the
    fidelity against that starting model's fp32 trajectory (the reference a release check would use)."""
    quants = QUANTS if quants is None else quants
    sigmas = SIGMAS if sigmas is None else sigmas
    rng_cpu, rng_np = torch.get_rng_state(), np.random.get_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        cfg = _run_cfg(ckpt)
        if ref_ckpt is None and cfg.get("load_checkpoint"):
            ref_ckpt = cfg["load_checkpoint"]
        if ref_ckpt is None and cfg.get("q7_fork_from") and (cfg.get("q7_start_step") or -1) >= 0:
            fork_ref = Path(cfg["q7_fork_from"]).parent / f"step_{cfg['q7_start_step']}"
            if fork_ref.exists():          # late-phase arms: the base run's checkpoint at the fork point
                ref_ckpt = str(fork_ref)
        if quant_noise is None:
            quant_noise = cfg.get("q7_reg") in ("qat", "lsq") or (cfg.get("q7_act_bits") or 0) > 0
        ck_path, lsq_steps, act_scales = _lsq_split(ckpt)
        arch = bs._variant_arch(ckpt)
        H = arch.get("H_cycles", 3)
        inp, lab = bs.load_qa("sudoku", n)
        out = Path(out)
        rec = {"ckpt": str(ckpt), "label": label, "n": int(len(inp)), "nsup": nsup, "H_cycles": H,
               "arch": arch, "ref_ckpt": ref_ckpt, "records": []}
        if out.exists():
            try:
                old = json.load(open(out))
                if old.get("n") == rec["n"] and old.get("nsup") == nsup:
                    rec["records"] = old.get("records", [])
                    rec.update({k: v for k, v in old.items() if k.startswith("jac") or k.startswith("ref_fp32")
                                or k.startswith("S")})
            except (json.JSONDecodeError, ValueError):
                log(f"[q7_eval] {out} unreadable; starting fresh")
        if ref_ckpt:   # records written before the reference was known are redone
            rec["records"] = [r for r in rec["records"] if "fidelity_H_ref" in r]
        done = {(r["quant"]["spec"], r.get("draw", 0)) for r in rec["records"]}
        cells = _cells(quants, sigmas, draws)
        qbases = []
        if lsq_steps:
            cells.insert(1, ("w4lsq", 0))
            rec["lsq_steps"] = lsq_steps
            qbases.append("w4lsq")
        if act_scales:
            extra = [("fp32+a4s", 0), ("w4t+a4s", 0)] + ([("w4lsq+a4s", 0)] if lsq_steps else [])
            cells[1:1] = extra
            rec["act_scales"] = act_scales
            qbases = [b + "+a4s" for b in (["w4t"] + qbases)]
        elif quant_noise:
            qbases = ["w4t"] + qbases
        if quant_noise:
            cells += [(f"{b}~noise{sg}", d) for b in qbases for sg in QNOISE_SIGMAS for d in range(draws)]
        step = int(re.search(r"step_(\d+)", str(ckpt)).group(1)) if re.search(r"step_(\d+)", str(ckpt)) else 0
        final = int(cfg.get("epochs", 0) * 1000 / max(cfg.get("global_batch_size", 768), 1))
        if light is None:
            light = bool(cfg.get("load_checkpoint")) and step < final
        if light:   # intermediate checkpoint of a 12k-step run from a trained model: the trajectory cells only
            keep = {"fp32", "w4t", "w4c", "w4lsq", "fp32+a4s", "w4t+a4s", "w4lsq+a4s", "noise0.15"}
            cells = [(sp, d) for sp, d in cells if d == 0 and (sp in keep or sp.endswith("~noise0.1")
                                                                  or sp.endswith("~noise0.2"))]
            rec["light"] = True
        todo = [c for c in cells if c not in done]
        if not todo and "jac_spec_mean" in rec and "S_0.02" in rec and (not quant_noise or "Sq_0.02" in rec):
            return rec
        t0 = time.time()
        ref_zH2 = None
        if ref_ckpt:
            bs.CKPT_OVERRIDE = str(_lsq_split(ref_ckpt)[0])
            mr, _ = bs.build("sudoku_mlp", H, nsup, batch)
            rc, _, ref_zH2, _ = bs.run(mr, inp, lab, batch, nsup)
            rec["ref_fp32_pexact"] = float(rc.mean())
            del mr
        bs.CKPT_OVERRIDE = ck_path
        m, _ = bs.build("sudoku_mlp", H, nsup, batch)
        ref_c, ref_cell, ref_zH, ref_zL = bs.run(m, inp, lab, batch, nsup)
        if "jac_spec_mean" not in rec:
            rec.update(_jac_stats(m, inp, lab))
        del m
        mk = lambda: bs.build("sudoku_mlp", H, nsup, batch)[0]  # noqa: E731
        if "S_0.02" not in rec:
            rec.update(_sens_stats(mk, lambda mm: None, inp, lab, batch, nsup))
        if quant_noise and "Sq_0.02" not in rec:     # sensitivity around the deployed (quantized) point
            if lsq_steps:
                rec.update(_sens_stats(mk, lambda mm: _quantize_lsq_(mm, lsq_steps, 4), inp, lab, batch, nsup, tag="q"))
            else:
                rec.update(_sens_stats(mk, lambda mm: bs.quantize_weights_(mm, 4, "t"), inp, lab, batch, nsup, tag="q"))
        for spec, draw in todo:
            t1 = time.time()
            if spec == "fp32":
                c, cell, zH, fH, fL = ref_c, ref_cell, ref_zH, 1.0, 1.0
            else:
                base, _, nz = spec.partition("~")
                mq, _ = bs.build("sudoku_mlp", H, nsup, batch)
                if base.startswith("w4lsq"):
                    _quantize_lsq_(mq, lsq_steps, 4)
                elif base.startswith("w4t+"):
                    bs.quantize_weights_(mq, 4, "t")
                elif not base.startswith("fp32"):
                    q = bs.parse_quant(base)
                    if q["gran"] == "noise":
                        torch.manual_seed(100003 * draw + int(round(q["sigma"] * 10000)))
                    bs.quantize_weights_(mq, q["bits"], q["gran"], q.get("sigma"))
                if nz:
                    sg = float(nz[5:])
                    torch.manual_seed(100003 * draw + int(round(sg * 10000)) + 7)
                    bs.quantize_weights_(mq, None, "noise", sg)
                if base.endswith("+a4s"):
                    _attach_actq(mq, act_scales, 4)
                c, cell, zH, zL = bs.run(mq, inp, lab, batch, nsup)
                del mq
                fH, fL = bs.fidelity(zH, ref_zH), bs.fidelity(zL, ref_zL)
            r = {"quant": _qd(spec), "draw": draw, "pexact": float(c.mean()), "cell": float(cell),
                 "fidelity_H": fH, "fidelity_L": fL, "correct": c.astype(int).tolist(), "secs": time.time() - t1}
            if ref_zH2 is not None:
                r["fidelity_H_ref"] = bs.fidelity(zH, ref_zH2)
            rec["records"].append(r)
            tmp = str(out) + ".tmp"
            json.dump(rec, open(tmp, "w"))
            os.replace(tmp, out)
        fp = next(r for r in rec["records"] if r["quant"]["spec"] == "fp32")
        w4t = next((r for r in rec["records"] if r["quant"]["spec"] == "w4t"), None)
        lsq = next((r for r in rec["records"] if r["quant"]["spec"] == "w4lsq"), None)
        fr = (lambda r: f"/ref {r['fidelity_H_ref']:.3f}" if "fidelity_H_ref" in r else "")
        log(f"[q7_eval] {label or ckpt}: fp32 {fp['pexact']*100:.1f}"
            + (f" w4t {w4t['pexact']*100:.1f} (fidH {w4t['fidelity_H']:.3f}{fr(w4t)})" if w4t else "")
            + (f" w4lsq {lsq['pexact']*100:.1f} (fidH {lsq['fidelity_H']:.3f}{fr(lsq)})" if lsq else "")
            + f" S0.02 {rec['S_0.02']:.2f} S0.1 {rec['S_0.1']:.2f}"
            + (f" Sq0.1 {rec['Sq_0.1']:.2f}" if "Sq_0.1" in rec else "")
            + f" jac_fro2 {rec['jac_fro2']:.3f}; {len(todo)} cells in {time.time()-t0:.0f}s")
        return rec
    finally:
        torch.set_rng_state(rng_cpu)
        np.random.set_state(rng_np)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--scan", default="", help="root holding <run>/step_N checkpoints; evaluates every one")
    ap.add_argument("--rundir", action="append", default=[], help="one run directory; evaluates its step_N")
    ap.add_argument("--full", action="store_true", help="never use the light cell set")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--nsup", type=int, default=16)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--sigmas", default=",".join(map(str, SIGMAS)))
    ap.add_argument("--quants", default=",".join(QUANTS))
    a = ap.parse_args()
    sig = [float(s) for s in a.sigmas.split(",") if s]
    qs = [q for q in a.quants.split(",") if q]
    jobs = []
    if a.ckpt:
        jobs.append((a.ckpt, a.out or str(Path(a.ckpt).parent / f"eval_{Path(a.ckpt).name}.json")))
    if a.scan:
        for p in sorted(glob.glob(os.path.join(a.scan, "*", "step_*"))):
            if re.fullmatch(r"step_\d+", Path(p).name):
                jobs.append((p, str(Path(p).parent / f"eval_{Path(p).name}.json")))
    for rd in a.rundir:
        for p in sorted(glob.glob(os.path.join(rd, "step_*")), key=lambda q: int(re.sub(r"\D", "", Path(q).name) or 0)):
            if re.fullmatch(r"step_\d+", Path(p).name):
                jobs.append((p, str(Path(p).parent / f"eval_{Path(p).name}.json")))
    for ck, out in jobs:
        evaluate_checkpoint(ck, out, n=a.n, nsup=a.nsup, quants=qs, sigmas=sig, draws=a.draws,
                            label=f"{Path(ck).parent.name}/{Path(ck).name}", light=False if a.full else None)


if __name__ == "__main__":
    main()
