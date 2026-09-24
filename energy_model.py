#!/usr/bin/env python3
"""Analytical energy ESTIMATE per recursion step, per answer and per solved instance (not a
measurement; every number this script writes is labelled `estimate`).

Model (Horowitz, ISSCC 2014, 45 nm, as tabulated in Han et al. arXiv 1506.02626): energy =
MACs x e_mac(format) + bytes moved x e_byte(memory level). Per MAC: fp32 4.6 pJ (3.7 mult + 0.9 add),
fp16 1.5 pJ (1.1 + 0.4), int8 0.23 pJ (0.2 + 0.03); INT4 weights are charged as int8 MACs (Hexagon
and Ethos-U execute W4 as 8-bit MACs after unpacking). Memory: on-chip SRAM 5 pJ / 32 bit (1.25 pJ/B),
DRAM 640 pJ / 32 bit (160 pJ/B). Two bounds per step: weights resident on chip (read once per
application of f from SRAM) and weights streamed from DRAM once per application. Activations are
charged one SRAM write + read per layer output. 45 nm constants overstate a 4 nm phone NPU by
roughly an order of magnitude in absolute terms; use the model for ratios between formats and
between depths, not as an absolute device figure.

MAC counts are exact for the exported graphs (same dims as the ONNX step): hidden 512, 8 heads,
SwiGLU inner 1536; Sudoku-MLP token mixer SwiGLU over 97 positions with inner 512.

    python energy_model.py --out results/devices/energy_estimate.json
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PJ_MAC = {"fp32": 4.6, "fp16": 1.5, "int8": 0.23}
PJ_B = {"sram": 1.25, "dram": 160.0}
WBYTES = {"fp32": 4, "fp16": 2, "w8": 1, "w4": 0.5}
TASKS = {  # seq incl. 16 puzzle-embedding rows, L_cycles, mixer
    "sudoku_mlp": dict(S=97, L=6, mixer="mlp", vocab=11),
    "sudoku_attn": dict(S=97, L=6, mixer="attn", vocab=11),
    "maze": dict(S=916, L=4, mixer="attn", vocab=6),
    "arc": dict(S=916, L=4, mixer="attn", vocab=12),
}
D, INNER, TOK_INNER = 512, 1536, 512


def f_counts(t):
    """MACs, weight parameters and activation bytes-per-element-count of one application of f."""
    S = t["S"]
    mlp = S * (D * 2 * INNER + INNER * D)
    if t["mixer"] == "attn":
        mix = S * (D * 3 * D + D * D) + 2 * S * S * D
        wmix = D * 3 * D + D * D
    else:
        mix = D * (S * 2 * TOK_INNER + TOK_INNER * S)
        wmix = S * 2 * TOK_INNER + TOK_INNER * S
    macs = 2 * (mlp + mix)
    params = 2 * (wmix + D * 2 * INNER + INNER * D)
    act_elems = 2 * S * (D * 4 + 2 * INNER)          # layer outputs written and re-read (coarse)
    return macs, params, act_elems


def step_estimate(task, H, fmt):
    t = TASKS[task]
    macs, params, act = f_counts(t)
    n_f = H * (t["L"] + 1)
    mac_fmt = {"fp32": "fp32", "fp16": "fp16", "w8a8": "int8", "w4a8": "int8", "w4a16": "int8", "w8a16": "int8"}[fmt]
    wb = {"fp32": 4, "fp16": 2, "w8a8": 1, "w8a16": 1, "w4a8": 0.5, "w4a16": 0.5}[fmt]
    ab = {"fp32": 4, "fp16": 2, "w8a8": 1, "w4a8": 1, "w8a16": 2, "w4a16": 2}[fmt]
    e_mac = n_f * macs * PJ_MAC[mac_fmt]
    e_act = n_f * act * ab * 2 * PJ_B["sram"]
    e_w_sram = n_f * params * wb * PJ_B["sram"]
    e_w_dram = n_f * params * wb * PJ_B["dram"]
    head = t["S"] * D * t["vocab"] * PJ_MAC[mac_fmt]
    lo = (e_mac + e_act + e_w_sram + head) * 1e-9      # mJ
    hi = (e_mac + e_act + e_w_dram + head) * 1e-9
    return {"task": task, "H": H, "format": fmt, "f_per_step": n_f, "gmac_per_step": n_f * macs / 1e9,
            "weight_mb": params * wb / 1e6, "mj_per_step_weights_on_chip": lo, "mj_per_step_weights_dram": hi,
            "kind": "estimate (Horowitz 45 nm)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "devices" / "energy_estimate.json"))
    a = ap.parse_args()
    rows = [step_estimate(t, H, f) for t in TASKS for H in (1, 3) for f in ("fp32", "fp16", "w8a8", "w4a8",
                                                                              "w8a16", "w4a16")]
    Path(a.out).write_text(json.dumps(rows, indent=1))
    for r in rows:
        if r["H"] == 1:
            print(f"{r['task']:12s} H1 {r['format']:6s} {r['gmac_per_step']:6.2f} GMAC/step  weights {r['weight_mb']:5.2f} MB  "
                  f"{r['mj_per_step_weights_on_chip']:8.3f} .. {r['mj_per_step_weights_dram']:8.3f} mJ/step (estimate)")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
