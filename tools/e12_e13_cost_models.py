#!/usr/bin/env python3
"""Cost models for the board week, from file sizes and existing timings only (no run): E12 tiered finishing (carry
bytes vs loop bytes, link break-even), E13 embedding-table traffic (table vs backbone by loop count, SD-card time),
and the launch-tax prediction for the Jetson Nano from E1's eager / CUDA-graph / TensorRT rows. Prints Markdown for
reports/edgetrm_exp/E3_GATE_PHI_AND_COST.md and writes results/devices/e12_e13_cost.json.

Every constant carries its source in a comment; the link and SD rates are stated assumptions to be replaced by an
iperf / dd measurement on the boards.
"""
import argparse
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent.parent

# ---- graphs (stepc_H1.onnx initializers, counted on p4d Sep 29: sum of initializer elements; inputs z_H / z_L (1, seq, 512),
#      puzzle_embedding (1, 512)); passes per supervision step = L_cycles + 1 (the one net is applied to z_L L times and to z_H once)
GRAPHS = {"sudoku_mlp": dict(params=5_028_866, seq=97, passes=7, attn=False),
          "sudoku_attn": dict(params=6_828_034, seq=97, passes=7, attn=True),
          "maze": dict(params=6_822_914, seq=916, passes=5, attn=True),
          "arc": dict(params=6_829_058, seq=916, passes=5, attn=True)}
D = 512
# bytes per weight: fp32, fp16, INT8 per-tensor, INT8 g32 (b_eff = 8 + 16/32), INT4 g32 (4 + 16/32) -- ICLR App. H b_eff = b + 16/g
WBYTES = {"fp32": 4.0, "fp16": 2.0, "int8": 1.0, "int8_g32": 8.5 / 8, "int4_g32": 4.5 / 8}
# carry element sizes: fp32 (host / fp32 graph), fp16 (what the 2.45 fused DLCs return, class E), uint16 (w8a16 graph), uint8 (all-integer DLC, guard match)
CBYTES = {"fp32": 4, "fp16": 2, "uint16": 2, "uint8": 1}
# link rates in MB/s (assumptions; nominal in the comment): USB 2.0 480 Mbit/s bulk -> ~35 MB/s practical; Wi-Fi 5 802.11ac
# 2x2 80 MHz 866 Mbit/s -> ~40 MB/s TCP practical, and the Pi 4 radio (1x1, 433 Mbit/s) -> ~12 MB/s; BLE LE 1M PHY with DLE -> ~0.1 MB/s
LINKS = {"USB 2.0 (35 MB/s)": 35.0, "Wi-Fi 5 2x2 (40 MB/s)": 40.0, "Wi-Fi 5 Pi 4 radio (12 MB/s)": 12.0, "BLE (0.1 MB/s)": 0.1}
# SD-card sequential read MB/s: class 10 floor (10 MB/s spec minimum), Pi 4 / Nano SDHCI at DDR50 (~45 MB/s practical), UHS-I SDR104 card on a full-speed reader (~90)
SD = {"class 10 floor (10 MB/s)": 10.0, "Pi 4 / Nano slot, DDR50 (45 MB/s)": 45.0, "UHS-I SDR104 reader (90 MB/s)": 90.0}
SD_ROW_MS = 1.0   # one random 4 kB read on an SD card, ~1 ms (1k IOPS class); assumption
# measured per-step device times (E5_PHONE_CTL.md table 4, AI Hub profile jobs; ms per supervision step, H=1)
PHONE = {("S24", "sudoku_mlp"): dict(cheap=3.13, precise=2.92, cheap_graph="w8a8", precise_graph="w8a16"),
         ("S25", "sudoku_mlp"): dict(cheap=2.78, precise=2.76, cheap_graph="w8a8", precise_graph="w8a16"),
         ("S24", "sudoku_attn"): dict(cheap=3.86, precise=4.09, cheap_graph="w8a8", precise_graph="w8a16"),
         ("S25", "sudoku_attn"): dict(cheap=2.97, precise=3.25, cheap_graph="w8a8", precise_graph="w8a16"),
         ("S24", "maze"): dict(cheap=22.35, precise=43.77, cheap_graph="w8a8rope16", precise_graph="fp32"),
         ("S25", "maze"): dict(cheap=17.75, precise=44.75, cheap_graph="w8a8rope16", precise_graph="fp32")}
# ICLR finishing counts and cost ratio r = b_f / b_eff (App. H, tab:app-ratio): j in {2, 4}; 8-bit finishing after w4g32 1.78, w3g32 2.29, w2g32 3.2 ("2-4"); fp32 7.1-12.8 ("8-16")
J = (2, 4)
R = {"8-bit after w4g32": 8 / 4.5, "8-bit after w3g32": 8 / 3.5, "8-bit after w2g32": 8 / 2.5, "fp32 after w4g32": 32 / 4.5}
# E13: puzzle-embedding tables. Notebook-era ARC checkpoint (EdgeTRM/notebook_comparison_report.md 14.1; X3.md): traced step 126.1 MB,
# INT8 table 25.5 MB, i.e. ~99 MB fp32; arc_v1_public/step_518071 (X3.md): 876,406 x 512 fp32 = 1,794.9 MB, non-embedding 27.3 MB
TABLES = {"ARC notebook checkpoint (99 MB)": dict(mb=99.0, rows=int(99e6 / (D * 4))), "arc_v1_public step_518071 (1,795 MB)": dict(mb=1794.9, rows=876_406)}
# E1 (L40S, H=3, batch 1, fixed 16 steps, E1.md R2 table: ms per solved / 16 = ms per step): eager fp32, CUDA graph fp16, TRT fp16
E1 = {"sudoku_mlp": dict(eager=274 / 16, graph=51.6 / 16, trt=36.2 / 16, passes_h3=21),
      "sudoku_attn": dict(eager=306 / 16, graph=71.2 / 16, trt=35.0 / 16, passes_h3=21),
      "maze": dict(eager=253 / 16, graph=77.0 / 16, trt=44.6 / 16, passes_h3=15)}
A100 = dict(eager_fp32=16.1, eager_fp16=19.4, graph_fp16=3.76)   # E1.md p4d partial, Sudoku-MLP H=3, ms per step
# executable ONNX nodes per H=1 step (p4d Sep 29: total nodes minus Constant / Identity / Shape)
NODES = {"sudoku_mlp": 958 - 239 - 48 - 28, "sudoku_attn": 2260 - 827 - 48 - 84, "maze": 1622 - 595 - 32 - 60}
# Nano assumptions: Cortex-A57 1.43 GHz Python dispatch 25-50 us per eager op (5-8x the x86 host's ~5-7 us); fp16 peak 472 GFLOP/s at
# 20-40 % on these shapes (M = 97 / 916 rows); LPDDR4 25.6 GB/s
NANO = dict(us_per_op=(25, 50), gflops=(472 * 0.2, 472 * 0.4), bw_gbs=25.6, trt_us_per_layer=(10, 20), call_ms=(0.5, 1.0))


def carry_bytes(task, dtype):
    return 2 * GRAPHS[task]["seq"] * D * CBYTES[dtype]


def loop_bytes(task, wfmt, physical=False):
    b = GRAPHS[task]["params"] * WBYTES[wfmt]
    return b * GRAPHS[task]["passes"] if physical else b


def gflop_step(task):
    g = GRAPHS[task]
    mm = 2 * g["params"] * g["seq"] * g["passes"]
    at = (2 * 2 * g["seq"] ** 2 * D * 2) * g["passes"] if g["attn"] else 0   # 2 layers x (QK^T + PV) x 2 flop/MAC
    return (mm + at) / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results" / "devices" / "e12_e13_cost.json"))
    a = ap.parse_args()
    out = {}

    print("## E12.1 Carry bytes per puzzle and weight bytes per loop\n")
    print("| graph | seq | carry elems (z_H + z_L) | carry fp32 | fp16 | uint8 | params | loop fp32 | fp16 | INT8 | INT8 g32 | INT4 g32 | physical x passes (fp32) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for t, g in GRAPHS.items():
        cb = {d: carry_bytes(t, d) for d in CBYTES}
        lb = {w: loop_bytes(t, w) for w in WBYTES}
        out[f"bytes/{t}"] = dict(carry=cb, loop=lb, loop_physical_fp32=loop_bytes(t, "fp32", True), passes=g["passes"])
        print(f"| {t} | {g['seq']} | {2 * g['seq'] * D:,} | {cb['fp32'] / 1e6:.3f} MB | {cb['fp16'] / 1e6:.3f} | {cb['uint8'] / 1e6:.3f} | {g['params']:,} | "
              f"{lb['fp32'] / 1e6:.2f} MB | {lb['fp16'] / 1e6:.2f} | {lb['int8'] / 1e6:.2f} | {lb['int8_g32'] / 1e6:.2f} | {lb['int4_g32'] / 1e6:.2f} | "
              f"{g['passes']} x = {loop_bytes(t, 'fp32', True) / 1e6:.0f} MB |")

    print("\n## E12.2 Bytes: carry vs j finishing loops run locally (ICLR convention, one weight read per loop)\n")
    print("| graph | carry (fp32 / uint8) | 8-bit finishing j=2 / j=4 | fp32 finishing j=2 / j=4 | carry fp32 as a share of one 8-bit loop |")
    print("|---|---|---|---|---|")
    for t in ("sudoku_mlp", "sudoku_attn", "maze"):
        c4, c1 = carry_bytes(t, "fp32"), carry_bytes(t, "uint8")
        l8, l32 = loop_bytes(t, "int8"), loop_bytes(t, "fp32")
        print(f"| {t} | {c4 / 1e6:.3f} / {c1 / 1e6:.3f} MB | {2 * l8 / 1e6:.1f} / {4 * l8 / 1e6:.1f} MB | {2 * l32 / 1e6:.1f} / {4 * l32 / 1e6:.1f} MB | {c4 / l8:.3f} |")

    print("\n## E12.3 Link time of one carry and its cost in cheap loops (n* = t_link / t_cheap); send wins on time when n* < j x r\n")
    print("Return path (the finished answer, 81 or 900 tokens, or the finished carry if the loop continues) is not charged; charge 2 n* for a round trip of the carry.\n")
    print("| device / graph (cheap ms) | link | t_link fp32 carry | n* fp32 | t_link uint8 carry | n* uint8 | j x r (8-bit after w4g32, j=2 / 4) | j x r (fp32 after w4g32, j=2 / 4) |")
    print("|---|---|---|---|---|---|---|---|")
    be = {}
    r8, r32 = R["8-bit after w4g32"], R["fp32 after w4g32"]
    for (dev, t), p in PHONE.items():
        for ln, rate in LINKS.items():
            t4 = carry_bytes(t, "fp32") / (rate * 1e6) * 1e3
            t1 = carry_bytes(t, "uint8") / (rate * 1e6) * 1e3
            n4, n1 = t4 / p["cheap"], t1 / p["cheap"]
            be[f"{dev}/{t}/{ln}"] = dict(t_link_fp32_ms=t4, n_fp32=n4, t_link_uint8_ms=t1, n_uint8=n1, jr8=[2 * r8, 4 * r8], jr32=[2 * r32, 4 * r32])
            print(f"| {dev} {t} ({p['cheap']} ms, {p['cheap_graph']}) | {ln} | {t4:.1f} ms | {n4:.1f} | {t1:.1f} ms | {n1:.1f} | {2 * r8:.1f} / {4 * r8:.1f} | {2 * r32:.1f} / {4 * r32:.1f} |")
    out["break_even"] = be
    print("\nParametric: n* = carry bytes / (rate x t_cheap). For a board with t_cheap = T ms, n* (fp32 carry) = "
          + "; ".join(f"{t}: {carry_bytes(t, 'fp32') / 1e6:.3f} MB / (rate MB/s x T)" for t in ("sudoku_mlp", "maze")))

    print("\n## E13.1 Puzzle-embedding table vs backbone\n")
    print(f"Per puzzle the step graph reads one table row: puzzle_embedding (1, 512) = {D * 4:,} B fp32 / {D:,} B INT8 "
          f"(stepc_H1.onnx input; the Sudoku and Maze checkpoints have a single puzzle identifier, so their table is one row and E13 is an ARC question).\n")
    print("| table | rows | fp32 MB | INT8 MB | backbone fp32 / INT8 / INT4 per loop (MB) | loops until backbone traffic = one full table load: fp32 / INT8 / INT4 |")
    print("|---|---|---|---|---|---|")
    bb = {w: loop_bytes("arc", w) / 1e6 for w in ("fp32", "int8", "int4_g32")}
    for name, tb in TABLES.items():
        k = {w: tb["mb"] / bb[w] for w in bb}
        out[f"table/{name}"] = dict(rows=tb["rows"], fp32_mb=tb["mb"], int8_mb=tb["mb"] / 4, backbone_mb=bb, loops_to_match=k)
        print(f"| {name} | {tb['rows']:,} | {tb['mb']:.1f} | {tb['mb'] / 4:.1f} | {bb['fp32']:.2f} / {bb['int8']:.2f} / {bb['int4_g32']:.2f} | {k['fp32']:.1f} / {k['int8']:.1f} / {k['int4_g32']:.0f} |")

    print("\n## E13.2 Ratio table traffic / backbone traffic as a function of loop count K (one puzzle per model load; divide by P for P puzzles per load)\n")
    Ks = (1, 2, 4, 8, 16, 32, 64)
    print("| table, backbone format | " + " | ".join(f"K={k}" for k in Ks) + " | row fetch only (2 kB), K=8 |")
    print("|---|" + "---|" * (len(Ks) + 1))
    for name, tb in TABLES.items():
        for w in ("fp32", "int8", "int4_g32"):
            print(f"| {name}, {w} | " + " | ".join(f"{tb['mb'] / (k * bb[w]):.2f}" for k in Ks) + f" | {D * 4 / (8 * bb[w] * 1e6):.1e} |")

    print("\n## E13.3 SD-card time for the table (sequential) and per-puzzle row (random)\n")
    print("| table | " + " | ".join(SD) + " | one row, random read |")
    print("|---|" + "---|" * (len(SD) + 1))
    for name, tb in TABLES.items():
        print(f"| {name} fp32 | " + " | ".join(f"{tb['mb'] / r:.1f} s" for r in SD.values()) + f" | ~{SD_ROW_MS:.0f} ms |")
        print(f"| {name} INT8 | " + " | ".join(f"{tb['mb'] / 4 / r:.1f} s" for r in SD.values()) + f" | ~{SD_ROW_MS:.0f} ms |")
    print(f"| backbone 27.3 MB fp32 (one load) | " + " | ".join(f"{bb['fp32'] / r:.1f} s" for r in SD.values()) + " | - |")

    print("\n## Launch tax: what E1 measured (L40S, H=3, batch 1) and the Nano prediction (H=1)\n")
    print("| graph | eager fp32 ms/step | CUDA graph fp16 | TRT fp16 | dispatch = eager - graph | share of eager | passes (H=3) | dispatch per pass | ONNX ops per H=1 step | us per op (x86 host) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    lt = {}
    for t, e in E1.items():
        disp = e["eager"] - e["graph"]
        per_pass = disp / e["passes_h3"]
        ops_h1 = NODES[t]
        us_op = per_pass * 1e3 / (ops_h1 / GRAPHS[t]["passes"])
        lt[t] = dict(dispatch_ms_h3=disp, share=disp / e["eager"], per_pass_ms=per_pass, ops_h1=ops_h1, us_per_op=us_op)
        print(f"| {t} | {e['eager']:.1f} | {e['graph']:.1f} | {e['trt']:.2f} | {disp:.1f} | {disp / e['eager']:.0%} | {e['passes_h3']} | {per_pass:.2f} ms | {ops_h1} | {us_op:.1f} |")
    print(f"\nA100 cross-check (E1 partial, Sudoku-MLP H=3): eager fp32 {A100['eager_fp32']} ms, CUDA graph fp16 {A100['graph_fp16']} ms: dispatch {A100['eager_fp32'] - A100['graph_fp16']:.1f} ms = {(A100['eager_fp32'] - A100['graph_fp16']) / A100['eager_fp32']:.0%}.\n")
    print("| graph (H=1) | GFLOP / step | Nano compute floor fp16 (20-40 % of 472 GFLOP/s) | weight traffic / step fp16 (x passes) at 25.6 GB/s | eager launch floor (25-50 us x ops) | predicted eager step | launch share | TRT per-layer launches (10-20 us) | unrolled-k saving per call |")
    print("|---|---|---|---|---|---|---|---|---|")
    pred = {}
    for t in ("sudoku_mlp", "sudoku_attn", "maze"):
        gf = gflop_step(t)
        comp = (gf / NANO["gflops"][1] * 1e3, gf / NANO["gflops"][0] * 1e3)
        wt = loop_bytes(t, "fp16", True) / (NANO["bw_gbs"] * 1e9) * 1e3
        launch = (NODES[t] * NANO["us_per_op"][0] / 1e3, NODES[t] * NANO["us_per_op"][1] / 1e3)
        step = (comp[0] + launch[0], comp[1] + launch[1])
        share = (launch[0] / (launch[0] + comp[1]), launch[1] / (launch[1] + comp[0]))   # low launch / high compute .. high launch / low compute
        trt = (NODES[t] / 4 * NANO["trt_us_per_layer"][0] / 1e3, NODES[t] / 4 * NANO["trt_us_per_layer"][1] / 1e3)   # TRT fuses ~4 ONNX ops per layer (E1: ~430 layers for the H=3 step)
        pred[t] = dict(gflop=gf, compute_ms=comp, weight_ms=wt, launch_ms=launch, step_ms=step, launch_share=share, trt_launch_ms=trt, call_ms=NANO["call_ms"])
        print(f"| {t} | {gf:.1f} | {comp[0]:.0f}-{comp[1]:.0f} ms | {wt:.1f} ms | {launch[0]:.0f}-{launch[1]:.0f} ms | {step[0]:.0f}-{step[1]:.0f} ms | {share[0]:.0%}-{share[1]:.0%} | "
              f"{trt[0]:.1f}-{trt[1]:.1f} ms | {NANO['call_ms'][0]}-{NANO['call_ms'][1]} ms |")
    out["launch_tax_measured"] = lt
    out["nano_prediction"] = pred
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
