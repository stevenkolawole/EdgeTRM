#!/usr/bin/env python3
"""X1 (angle B beyond the convnets): accuracy vs test-time iterations per Sudoku-Extreme
difficulty bin, fp32 vs quantizers vs weight/activation noise, for TRM Sudoku-MLP, TRM
Sudoku-attn and HRM Sudoku.

Difficulty = the dataset's `rating` column (number of backtracks the tdoku solver needs; the
dataset card says higher is harder). The puzzles are a stratified sample of `test.csv` rows
fixed in `results/breadth/sudoku_extreme_test_meta.json` ("strat": 6 rating bins x 1000
puzzles, numpy default_rng(0)); `--set first` evaluates the first n rows instead, as every
earlier breadth sweep did.

One trajectory per quantizer: halt_max_steps = K and the answer is read out after every ACT
step k = 1..K. In eval mode TRM/HRM halt only at halt_max_steps and never reset a carry
before it, so the step-k read-out is exactly what a run with nsup = k returns (checked: 0
mismatching puzzles against the nsup = 1,2,4,8,16 breadth records, same platform).

Per quantizer and puzzle it stores
  bits[i]        hex bitmask, bit k-1 set = puzzle-exact correct after step k
  wrong@k[i]     wrong cells (of the blanks) after step k, k in WRONG_AT
  qhalt[i]       halting-head logit after step K (learned "answer is correct" signal)
  fid_H[i]       cosine(z_H, z_H^fp32) after step K, same puzzle (fp32 run in the same process)
Specs: breadth_sweep's (fp32 | w<bits><gran>[+a8] | noise<s> | anoise<s> | w..+anoise<s>) plus
an optional '@s<seed>' suffix: the RNG is reseeded with <seed> immediately before the weight
draw and before the run, so every noise draw is reproducible on its own (default seed 0).

    cd ~/edgetrm_src && python difficulty_sweep.py --task sudoku_mlp --K 64 \
        --quants fp32,w8c,w4c,w4t,noise0.15@s0 --out ~/breadth_results/x1/diff_sudoku_mlp_q.json

Resumable: specs already in --out are skipped (fp32 is always rerun first, as the fidelity
reference, and checked bit-for-bit against the stored fp32 record).
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import breadth_sweep as bs  # noqa: E402
from breadth_sweep import (DEV, HRM_TASKS, TRM_TASKS, _ActNoise, _ActQuant, _linears, build,  # noqa: E402
                           parse_quant, quantize_weights_)

WRONG_AT = (1, 4, 16, 32, 64, 128, 256, 512)
META_CANDIDATES = [HERE / "results" / "breadth" / "sudoku_extreme_test_meta.json",
                   HERE / "sudoku_extreme_test_meta.json"]


def load_meta():
    for p in META_CANDIDATES:
        if p.exists():
            return json.load(open(p))
    raise FileNotFoundError(f"sudoku_extreme_test_meta.json not found in {META_CANDIDATES}")


def load_rows(rows):
    """test.csv rows (0-based data-row indices) -> inputs, labels, ratings (same encoding as
    breadth_sweep.load_qa: blanks 1, digits 2..10)."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("sapientinc/sudoku-extreme", "test.csv", repo_type="dataset")
    want = {r: i for i, r in enumerate(rows)}
    q, a, rat = [None] * len(rows), [None] * len(rows), [None] * len(rows)
    last = max(rows)
    with open(p, newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for j, row in enumerate(rd):
            i = want.get(j)
            if i is not None:
                qq, aa = row[1], row[2]
                q[i] = (np.frombuffer(qq.replace(".", "0").encode(), np.uint8) - ord("0")).astype(np.int64) + 1
                a[i] = (np.frombuffer(aa.encode(), np.uint8) - ord("0")).astype(np.int64) + 1
                rat[i] = int(row[3])
            if j >= last:
                break
    assert all(x is not None for x in q), "some rows not found"
    return np.stack(q), np.stack(a), rat


def split_spec(spec):
    if "@s" in spec:
        base, s = spec.split("@s")
        return base, int(s)
    return spec, 0


def _fresh(m, bt):
    c = m.initial_carry(bt)
    ic = c.inner_carry
    return type(c)(inner_carry=type(ic)(z_H=ic.z_H.to(DEV), z_L=ic.z_L.to(DEV)),
                   steps=c.steps.to(DEV), halted=c.halted.to(DEV),
                   current_data={k: v.to(DEV) for k, v in c.current_data.items()})


class _ActNoiseFixed:
    """Activation noise drawn once per layer and reused at every call (X7's 'af' arm, the fixed error placed in the
    activations): x + sigma * std(x) * G, with G over the last two dims at rms 1, from a generator seeded by
    (seed, layer index), so every loop and every chunk sees the same direction."""

    def __init__(self, sigma, seed, li):
        self.sigma, self.seed, self.li, self.G = sigma, seed, li, None

    def __call__(self, mod, args):
        x = args[0]
        shape = tuple(x.shape[-2:]) if x.dim() >= 3 else (x.shape[-1],)
        if self.G is None or tuple(self.G.shape) != shape:
            gen = torch.Generator(device="cpu").manual_seed(self.seed * 1000 + self.li)
            G = torch.randn(shape, generator=gen)
            self.G = (G / G.pow(2).mean().sqrt()).to(x.device, x.dtype)
        return (x + self.sigma * x.std() * self.G,) + tuple(args[1:])


class _WFresh:
    """Weight noise redrawn at every call (the fresh error placed in the weights): W0 + sigma * std(W0) * randn,
    a new draw before every forward of the linear; restore() puts W0 back."""

    def __init__(self, mod, sigma):
        self.W0, self.sigma = mod.weight.data.clone(), sigma

    def __call__(self, mod, args):
        mod.weight.data = self.W0 + torch.randn_like(self.W0) * (self.sigma * self.W0.std())
        return None

    def restore(self, mod):
        mod.weight.data = self.W0


@torch.no_grad()
def run_traj(m, inp, lab, batch, K, act_quant=None, act_noise=None, ref_zH=None, keep_zH=False, partial=None, tag=None,
             act_noise_fixed=None, w_fresh=None, seed=0):
    """partial: path of a chunk-level checkpoint (p4d reboots daily); resumes a spec at the next chunk.
    Not used with static activation quantization (its scales are calibrated on chunk 0)."""
    m = m.to(DEV)
    n = inp.shape[0]
    if act_quant is not None:
        partial = None
    handles, obs = [], []
    if act_noise is not None:
        for mod in _linears(m):
            handles.append(mod.register_forward_pre_hook(_ActNoise(act_noise)))
    if act_noise_fixed is not None:
        for li, mod in enumerate(_linears(m)):
            handles.append(mod.register_forward_pre_hook(_ActNoiseFixed(act_noise_fixed, seed, li)))
    wf = []
    if w_fresh is not None:
        for mod in _linears(m):
            h = _WFresh(mod, w_fresh)
            wf.append((mod, h))
            handles.append(mod.register_forward_pre_hook(h))
    if act_quant is not None:
        for mod in _linears(m):
            q = _ActQuant(act_quant)
            obs.append(q)
            handles.append(mod.register_forward_pre_hook(q))
    bits = np.zeros((n, K), dtype=bool)
    wrong = {k: np.zeros(n, dtype=np.int32) for k in WRONG_AT if k <= K}
    qhalt = np.zeros(n, dtype=np.float32)
    fid = np.full(n, np.nan, dtype=np.float32)
    zs = []
    start = 0
    if partial is not None and Path(partial).exists():
        ck = torch.load(partial, weights_only=False)
        if ck.get("tag") == tag:
            start = ck["next"]
            bits, wrong, qhalt, fid, zs = ck["bits"], ck["wrong"], ck["qhalt"], ck["fid"], ck["zs"]
            print(f"    resuming {tag} at chunk {start}", flush=True)
    for ci, s in enumerate(range(0, n, batch)):
        if ci < start:
            continue
        x = torch.from_numpy(inp[s:s + batch]).to(DEV)
        y = torch.from_numpy(lab[s:s + batch]).to(DEV)
        B = x.shape[0]
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
        carry = _fresh(m, bt)
        if act_quant is not None and ci == 0:        # static scales observed on the first chunk, full depth
            cc = carry
            for _ in range(K):
                cc, _ = m(cc, bt)
            for q in obs:
                q.freeze()
            carry = _fresh(m, bt)
        blank = x == 1                               # the cells the model has to fill
        mask = y != 0
        out = None
        for k in range(1, K + 1):
            carry, out = m(carry, bt)
            preds = out["logits"].argmax(-1)
            ok = (preds == y) | ~mask
            bits[s:s + B, k - 1] = ok.all(-1).cpu().numpy()
            if k in wrong:
                wrong[k][s:s + B] = ((~ok) & blank).sum(-1).cpu().numpy()
        qh = out.get("q_halt_logits")
        if qh is not None:
            qhalt[s:s + B] = qh.float().cpu().numpy()
        zH = carry.inner_carry.z_H.float()
        if ref_zH is not None:
            ref = ref_zH[s:s + B].to(DEV).float()
            fid[s:s + B] = F.cosine_similarity(zH.flatten(1), ref.flatten(1), dim=-1).cpu().numpy()
        if keep_zH:
            zs.append(zH.half().cpu())
        if partial is not None and s + batch < n:
            tmp = str(partial) + ".tmp"
            torch.save({"tag": tag, "next": ci + 1, "bits": bits, "wrong": wrong, "qhalt": qhalt, "fid": fid,
                        "zs": zs}, tmp)
            os.replace(tmp, partial)
    for h in handles:
        h.remove()
    for mod, h in wf:
        h.restore(mod)
    m.cpu()
    torch.cuda.empty_cache()
    return bits, wrong, qhalt, fid, (torch.cat(zs) if keep_zH else None)


def pack(bits):
    w = (bits.shape[1] + 3) // 4
    weights = 1 << np.arange(bits.shape[1], dtype=object)
    return [format(int(sum(int(b) * wt for b, wt in zip(row, weights))), f"0{w}x") for row in bits]


def atomic_dump(obj, path):
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["sudoku_mlp", "sudoku_attn", "hrm_sudoku"])
    ap.add_argument("--quants", default="fp32,w8c,w4c,w4t")
    ap.add_argument("--K", type=int, default=64, help="ACT steps (halt_max_steps); read out after each")
    ap.add_argument("--H", default="", help="H_cycles override; empty = checkpoint default")
    ap.add_argument("--set", default="strat", choices=["strat", "first"])
    ap.add_argument("--n", type=int, default=0, help="first-n puzzles of the chosen set (0 = all); smoke tests")
    ap.add_argument("--per_bin", type=int, default=0, help="strat set: keep the first N puzzles of each rating bin "
                                                          "(the deep K=512 sweep uses 300)")
    ap.add_argument("--batch", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ref_cache", default="", help="file for the fp32 reference (final z_H): written after the fp32 "
                                                    "run, reused on restart instead of rerunning fp32")
    ap.add_argument("--ref_wait", action="store_true", help="do not run fp32 here: wait for --ref_cache (written by "
                                                          "another process on the same puzzles) and pair against it")
    ap.add_argument("--checkpoint", action="store_true", help="chunk-level resume (<out>.partial.pt)")
    ap.add_argument("--ckpt", default="", help="TRM tasks: a trained variant's step_N file (arch from its all_config.yaml)")
    a = ap.parse_args()
    if a.ckpt:
        assert a.task in TRM_TASKS, "--ckpt applies to TRM tasks only"
        bs.CKPT_OVERRIDE = a.ckpt

    meta = load_meta()
    if a.set == "strat":
        rows = meta["strat"]["rows"]
        bins = meta["strat"]["bin"]
    else:
        rows = list(range(len(meta["first1000"]["rating"])))
        bins = None
    if a.per_bin and a.set == "strat":
        seen = {}
        keep = []
        for r, b in zip(rows, bins):
            if seen.get(b, 0) < a.per_bin:
                keep.append(r)
                seen[b] = seen.get(b, 0) + 1
        rows = keep
    if a.n:
        rows = rows[:a.n] if a.set == "first" else [r for i, r in enumerate(rows) if i % (len(rows) // a.n or 1) == 0][:a.n]
    inp, lab, rat = load_rows(rows)
    want_rat = {r: v for r, v in zip(meta["strat"]["rows"], meta["strat"]["rating"])} if a.set == "strat" else \
        {i: v for i, v in enumerate(meta["first1000"]["rating"])}
    assert all(want_rat[r] == v for r, v in zip(rows, rat)), "rating mismatch between test.csv and meta file"
    t = TRM_TASKS.get(a.task) or HRM_TASKS[a.task]
    batch = a.batch or t["batch"]
    H = int(a.H) if a.H else None
    quants = [q.strip() for q in a.quants.split(",") if q.strip()]
    if "fp32" not in quants and not a.ref_wait:
        quants = ["fp32"] + quants
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"task": a.task, "set": a.set, "rows": rows, "rating": rat, "K": a.K, "ckpt": a.ckpt, "records": []}
    if out_path.exists():
        try:
            old = json.load(open(out_path))
            assert old["rows"] == rows and old["K"] == a.K, "existing file has different rows/K"
            doc["records"] = old["records"]
        except (json.JSONDecodeError, ValueError):
            print(f"{out_path} unreadable; starting fresh", flush=True)
    done = {r["spec"]: r for r in doc["records"]}
    print(f"{a.task}: {len(rows)} puzzles ({a.set}), K={a.K}, batch {batch}, specs {quants}; "
          f"{len(done)} done already; device {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'cpu'}", flush=True)

    t0 = time.time()
    partial = str(out_path) + ".partial.pt" if a.checkpoint else None
    ref_cache = Path(a.ref_cache) if a.ref_cache else None
    torch.manual_seed(0)
    m, meta_m = build(a.task, H, a.K, batch)
    del m
    if a.ref_wait:
        while not (ref_cache and ref_cache.exists()):
            print(f"  waiting for {ref_cache}", flush=True)
            time.sleep(300)
    if ref_cache and ref_cache.exists() and ("fp32" in done or a.ref_wait):
        rc = torch.load(ref_cache, weights_only=False)
        assert rc["rows"] == rows and rc["K"] == a.K, "reference cache is for different puzzles or K"
        ref_zH = rc["zH"]
        print(f"  fp32 reference loaded from {ref_cache}", flush=True)
        run_fp32 = False
    else:
        run_fp32 = True
    if run_fp32:
        m, meta_m = build(a.task, H, a.K, batch)
        bits, wrong, qh, fid, ref_zH = run_traj(m, inp, lab, batch, a.K, keep_zH=True, partial=partial,
                                                tag=f"fp32|K{a.K}|n{len(rows)}|b{batch}")
        del m
        if ref_cache:
            tmp = str(ref_cache) + ".tmp"
            torch.save({"rows": rows, "K": a.K, "zH": ref_zH}, tmp)
            os.replace(tmp, ref_cache)
        if partial and Path(partial).exists():
            os.remove(partial)
    for spec in ([] if not run_fp32 else ["fp32"]):
        packed = pack(bits)
        acc = bits.mean(0)
        print(f"  fp32 ({time.time()-t0:.0f}s): " + " ".join(f"k{k}={100*acc[k-1]:.1f}" for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= a.K), flush=True)
        if "fp32" in done:
            same = sum(x == y for x, y in zip(done["fp32"]["bits"], packed))
            print(f"  fp32 rerun vs stored: {same}/{len(packed)} identical trajectories", flush=True)
            continue
        rec = {"spec": "fp32", "quant": parse_quant("fp32"), "seed": 0, "H_cycles": meta_m["H_cycles"],
               "L_cycles": meta_m["L_cycles"], "K": a.K, "n": len(rows), "acc": acc.tolist(), "bits": packed,
               "wrong": {str(k): v.tolist() for k, v in wrong.items()}, "qhalt": np.round(qh.astype(float), 3).tolist(),
               "fid_H": [1.0] * len(rows), "fid_H_mean": 1.0, "secs": time.time() - t0,
               "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"}
        doc["records"].append(rec)
        atomic_dump(doc, out_path)
    for spec in quants:
        if spec == "fp32" or spec in done:
            continue
        base, seed = split_spec(spec)
        if base.startswith("afnoise"):        # fixed activation noise (same direction on every loop)
            q = {"spec": base, "bits": None, "gran": None, "act": None, "afnoise": float(base[7:])}
        elif base.startswith("wfresh"):       # fresh weight noise (redrawn at every call)
            q = {"spec": base, "bits": None, "gran": None, "act": None, "wfresh": float(base[6:])}
        else:
            q = parse_quant(base)
        t1 = time.time()
        mq, _ = build(a.task, H, a.K, batch)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if q["gran"] is not None:
            quantize_weights_(mq, q["bits"], q["gran"], q.get("sigma"))
        torch.manual_seed(seed)
        b2, w2, qh2, fid2, _ = run_traj(mq, inp, lab, batch, a.K, act_quant=q["act"], act_noise=q.get("anoise"),
                                        ref_zH=ref_zH, partial=partial, tag=f"{spec}|K{a.K}|n{len(rows)}|b{batch}",
                                        act_noise_fixed=q.get("afnoise"), w_fresh=q.get("wfresh"), seed=seed)
        del mq
        if partial and Path(partial).exists():
            os.remove(partial)
        acc2 = b2.mean(0)
        rec = {"spec": spec, "quant": q, "seed": seed, "H_cycles": meta_m["H_cycles"], "L_cycles": meta_m["L_cycles"],
               "K": a.K, "n": len(rows), "acc": acc2.tolist(), "bits": pack(b2),
               "wrong": {str(k): v.tolist() for k, v in w2.items()}, "qhalt": np.round(qh2.astype(float), 3).tolist(),
               "fid_H": np.round(fid2.astype(float), 4).tolist(), "fid_H_mean": float(np.nanmean(fid2)), "secs": time.time() - t1,
               "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu"}
        doc["records"].append(rec)
        atomic_dump(doc, out_path)
        print(f"  {spec:14s} ({time.time()-t1:.0f}s) fidH {rec['fid_H_mean']:.3f}: "
              + " ".join(f"k{k}={100*acc2[k-1]:.1f}" for k in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512) if k <= a.K), flush=True)
    print("wrote", out_path, len(doc["records"]), "records", flush=True)


if __name__ == "__main__":
    main()
