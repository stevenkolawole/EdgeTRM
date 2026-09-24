#!/usr/bin/env python3
"""Assemble the H=1 MCU-tier results pulled from p4d (mcu_h1.py --stage collect,
the Corstone-300 FVP logs, the INT4 route checks) into
results/devices/{vela_h1.json, tflm_h1.json, fvp_h1.json, mcu_h1.md}.

    python3 mcu_h1_report.py --raw results/devices/mcu_h1_raw --out results/devices
"""
import argparse
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
H3_SEP10 = HERE / "results" / "breadth" / "vela_breadth.json"
FVP_H3_SEP10 = dict(task="sudoku_mlp", H=3, variant="int8 (Sep 10, unit-normal calibration)", npu="u55_256",
                    npu_active_cycles=166255297, npu_total_cycles=166256106,
                    axi0_rd_beats=69115449, axi0_wr_beats=29763636, axi1_rd_beats=13048360,
                    source="~/corstone/fvp_sudoku_mlp_npu2.log")
# Clock for converting FVP NPU cycles: the Vela system config the graph was compiled for
# (Ethos_U55_High_End_Embedded core_clock=500e6; Ethos_U65_High_End core_clock=1e9).
CLK = {"u55_256": 500e6, "u65_256": 1e9}
# AXI data width (bytes per beat): Ethos-U55 64-bit, Ethos-U65 128-bit interfaces.
BEAT = {"u55_256": 8, "u65_256": 16}
STEPS = (8, 16)
NOTES = [
    "Maze on U55/U65: Vela's per-layer CSV for maze H=1 INT8 (U55-256, Shared_Sram) puts 82.0% of NPU cycles in "
    "the 20 attention BatchMatMuls (916 tokens). The 263 MiB off-chip flash is the compiled graph's 'read_only' "
    "buffer (275,560,752 B); the Softmax-lowering Add ops are the ones reading it (2.68e8 flash accesses), "
    "consistent with one [8,916,916] 32-bit constant (26.8 MB) per softmax x 10 softmax ops at H=1 (30 at H=3: "
    "776 MiB). On U85-256 (SYS_Flash_High) the same H=1 graph needs 7,620 KiB (7.4 MiB) of flash.",
    "W8A16 cycles exclude CPU fallbacks: on U55/U65 the attention BATCH_MATMULs with INT16 inputs stay on the "
    "CPU (Vela: 'IF Matmul or (FC with dynamic weights): IFM precision must be Int8'; 28 ops for sudoku_attn "
    "H=1, 20 for maze H=1), so W8A16 attention rows are not comparable with INT8 rows.",
    "Vela version sensitivity: the eval kit's Vela 5.1.0 estimates 136.8 M / 332.5 M cycles for the same H=1 "
    "Sudoku-MLP / Sudoku-Attn INT8 graphs on U55-256 (5.2.0: 52.0 M / 162.5 M; H=3: 410.3 M / 997.1 M vs 155.8 M / "
    "487.5 M). The FVP runs the 5.2.0-compiled stream (mcu_h1_raw/vela510_crosscheck.json).",
    "FVP attention caveat: Vela lowers each U55 BATCH_MATMUL into ELEMENTWISE MUL (int8 x int8 -> int32) plus POOL "
    "REDUCE_SUM with an int32 IFM and int8 OFM. The FVP's Ethos-U55/U65 model (rev 136b7d75, Apr 2023) prints "
    "'[critical] Op=4: Invalid data type, ifm=5 ofm=1' once per such command (2,254 per H=1 Sudoku-Attn step on "
    "U55, the exact count in the register stream). The run completes, but its attention cycle count is not a "
    "validated measurement. The MLP-mixing graphs contain no matmul and run clean.",
    "TFLite Micro numerics: every graph TFLM invokes returns a single repeated output value (for sudoku_mlp H=1 "
    "INT8, every logit at the output zero point). Traced on sudoku_mlp H=1 INT8 (mcu_ref_kernel_diff.py): the builtin reference kernels, which TFLM "
    "shares, get the RMS-norm INT8 REDUCE SUM wrong (exact integer result -127/-128, optimized kernel -127/-128, "
    "reference kernel and TFLM -1); earlier ops differ by at most 14 LSB. TFLM arenas are valid; a CPU-only "
    "TFLM deployment of these graphs is not numerically valid without a fix to that kernel. Maze invokes abort "
    "(SIGABRT, no message).",
    "INT4 on Ethos-U: no TFLite route (Vela docs: 'Weight tensors must be 8-bit precision' for FULLY_CONNECTED; "
    "Vela 5.2.0 crashes with SIGFPE in regor.compile on every graph with an INT4 FULLY_CONNECTED, down to a single "
    "Linear; 5.1.0 crashes the same way on the Sep 10 W4A8 graph). The documented route is TOSA (EXT-INT4; CONV2D (i8,i4,i32)), e.g. ExecuTorch's Arm backend; the "
    "TOSA tables above show Vela 5.2 accepting INT4 weights. Evidence: mcu_h1_raw/int4/int4_notes.json.",
    "Ethos_U55_High_End_Embedded + Dedicated_Sram is not a valid Vela configuration (arena_mem_area=OffChipFlash); "
    "it fails at every H exactly as in the Sep 10 sweep and is kept for completeness.",
]
CFG_ORDER = [
    ("ethos-u55-32", "Ethos_U55_Deep_Embedded", "Sram_Only"),
    ("ethos-u55-64", "Ethos_U55_Deep_Embedded", "Shared_Sram"),
    ("ethos-u55-128", "Ethos_U55_High_End_Embedded", "Shared_Sram"),
    ("ethos-u55-256", "Ethos_U55_High_End_Embedded", "Shared_Sram"),
    ("ethos-u55-256", "Ethos_U55_High_End_Embedded", "Dedicated_Sram"),
    ("ethos-u65-256", "Ethos_U65_Embedded", "Shared_Sram"),
    ("ethos-u65-256", "Ethos_U65_Mid_End", "Dedicated_Sram"),
    ("ethos-u65-512", "Ethos_U65_High_End", "Dedicated_Sram"),
    ("ethos-u85-128", "Ethos_U85_SYS_Flash_Low", "Shared_Sram"),
    ("ethos-u85-256", "Ethos_U85_SYS_Flash_High", "Shared_Sram"),
    ("ethos-u85-512", "Ethos_U85_SYS_DRAM_Low", "Dedicated_Sram"),
    ("ethos-u85-1024", "Ethos_U85_SYS_DRAM_Mid", "Dedicated_Sram"),
    ("ethos-u85-2048", "Ethos_U85_SYS_DRAM_High", "Dedicated_Sram"),
]
TASKS = ["sudoku_mlp", "sudoku_attn", "maze"]
SWEEP_KEEP = ["task", "H", "variant", "accelerator", "system_config", "memory_mode", "rc", "ok", "error", "signal",
              "cpu_operators", "npu_operators", "sram_memory_used", "off_chip_flash_memory_used",
              "on_chip_flash_memory_used", "dram_memory_used", "arena_cache_size", "core_clock", "cycles_total",
              "cycles_npu", "cycles_sram_access", "cycles_dram_access", "cycles_on_chip_flash_access",
              "cycles_off_chip_flash_access", "inference_time", "inference_ms", "nn_macs",
              "total_original_weights", "total_npu_encoded_weights",
              "sram_feature_map_read_bytes", "sram_feature_map_write_bytes", "sram_weight_read_bytes",
              "sram_total_bytes", "dram_feature_map_read_bytes", "dram_feature_map_write_bytes",
              "dram_weight_read_bytes", "dram_total_bytes", "on_chip_flash_feature_map_read_bytes",
              "on_chip_flash_weight_read_bytes", "on_chip_flash_total_bytes",
              "off_chip_flash_feature_map_read_bytes", "off_chip_flash_weight_read_bytes",
              "off_chip_flash_total_bytes", "seconds"]


def rj(p):
    p = pathlib.Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def slim_accept(r):
    q, i, v, t = r.get("quantizer") or {}, r.get("interpreter") or {}, r.get("vela_default") or {}, r.get("tflm") or {}
    out = dict(task=r["task"], H=r["H"], variant=r["variant"])
    if r["variant"] != "fp32":
        au = q.get("audit") or {}
        out["quantizer"] = dict(ok=q.get("ok"), error=q.get("error"), n_calibration=q.get("n_calibration"),
                                file_bytes=au.get("file_bytes"), const_bytes_by_type=au.get("const_bytes_by_type"),
                                weight_quant_layout=au.get("weight_quant_layout"),
                                float_compute_ops=au.get("float_compute_ops"), n_ops=au.get("n_ops"), ops=au.get("ops"))
    out["interpreter"] = None if not i else dict(
        ok=i.get("ok"), resolver=i.get("resolver"), error=i.get("error"),
        default_resolver_error=i.get("default_resolver_error"),
        heldout_max_abs_logit_diff=i.get("heldout_max_abs"), heldout_mean_abs_logit_diff=i.get("heldout_mean_abs"),
        heldout_argmax_agree=i.get("heldout_argmax_agree"), calib_sample_max_abs=i.get("calib_sample_max_abs"),
        calib_sample_argmax_agree=i.get("calib_sample_argmax_agree"), per_sample=i.get("per_sample"), io=i.get("io"))
    if r["variant"] != "fp32":
        out["vela_default"] = None if not v else dict(
            config="ethos-u55-256 / Ethos_U55_High_End_Embedded / Shared_Sram", ok=v.get("ok"), rc=v.get("rc"),
            signal=v.get("signal"), error=v.get("error"), sram_kib=v.get("sram_memory_used"),
            off_chip_flash_kib=v.get("off_chip_flash_memory_used"), cycles=v.get("cycles_total"),
            inference_ms=v.get("inference_ms"), nn_macs=v.get("nn_macs"), cpu_operators=v.get("cpu_operators"),
            npu_operators=v.get("npu_operators"))
        out["tflm"] = None if not t else {k: t.get(k) for k in (
            "ok", "error", "tflm_messages", "arena_total_bytes", "arena_head_bytes", "arena_tail_bytes", "invoke_ok",
            "invoke_error",
            "invoke_s", "inputs_from", "sample", "vs_tflite_frac_equal", "vs_tflite_max_abs_raw", "vs_torch_max_abs",
            "vs_torch_argmax_agree", "output_distinct_values", "rc")}
    return out


def parse_fvp(log):
    txt = pathlib.Path(log).read_text(errors="replace")
    rec = {}
    for m in re.finditer(r"NPU (\w+): (\d+) (cycles|beats)", txt):
        rec[m.group(1)] = int(m.group(2))
    m = re.search(r"Activation buffer \(a.k.a tensor arena\) size used: (\d+)", txt)
    rec["tensor_arena_bytes"] = int(m.group(1)) if m else None
    ops = re.findall(r"Operator \d+: (\S+)", txt)
    rec["operators"] = ops
    for k, pat in (("simulated_time_s", r"Simulated time\s*:\s*([\d.]+)s"), ("wall_time_s", r"Wall time\s*:\s*([\d.]+)s")):
        m = re.search(pat, txt)
        rec[k] = float(m.group(1)) if m else None
    m = re.search(r"\[fvp\] rc=(\d+)", txt)
    rec["fvp_rc"] = int(m.group(1)) if m else None
    rec["inference_completed"] = "Inference completed" in txt
    m = re.search(r"MACs/cc:\s*(\d+)", txt)
    rec["npu_macs_per_cc_reported"] = int(m.group(1)) if m else None
    m = re.search(r"Using (\S+) memory mode", txt)
    rec["memory_mode_reported"] = m.group(1) if m else None
    rec["errors"] = [l.strip() for l in txt.splitlines() if re.search(r"ERROR|Error|error|failed", l)][:10]
    crit = [l.split("] ", 2)[-1].strip() for l in txt.splitlines() if l.startswith("[critical]")]
    rec["npu_model_critical_count"] = len(crit)
    rec["npu_model_critical_messages"] = sorted(set(crit))
    m = re.search(r"\[condensed\] (\d+) '\[critical\]' lines removed here; distinct: (.*)", txt)
    if m:  # logs pulled with the repeated FVP [critical] lines folded into one count line
        rec["npu_model_critical_count"] = int(m.group(1))
        rec["npu_model_critical_messages"] = [m.group(2).strip()]
    return rec


def fmt(x, nd=1):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:,.{nd}f}"
    return f"{x:,}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    raw, out = pathlib.Path(a.raw), pathlib.Path(a.out)
    matrix = rj(raw / "mcu_h1_matrix.json") or []
    sweep = rj(raw / "vela_h1_sweep.json") or []
    tflm = rj(raw / "tflm_h1.json") or []
    tosa = rj(raw / "int4" / "tosa_int4.json")
    tosa_real = rj(raw / "int4" / "tosa_int4_real.json")
    minrep = rj(raw / "int4" / "min.json")
    v510 = rj(raw / "vela510_crosscheck.json")
    int4_notes = rj(raw / "int4" / "int4_notes.json")

    sweep_rows = [{k: r.get(k) for k in SWEEP_KEEP} for r in sweep]
    # Same-toolchain check of the H=3 re-run against the Sep 10 sweep.
    sep10 = rj(H3_SEP10) or []
    h3chk = []
    for r in sweep_rows:
        if r["H"] != 3 or r["variant"] != "int8" or not r["ok"]:
            continue
        old = [o for o in sep10 if o["task"] == r["task"] and o["graph"] == "inner_step_int8.tflite"
               and o["accelerator"] == r["accelerator"] and o["system_config"] == r["system_config"]
               and o["memory_mode"] == r["memory_mode"] and o["rc"] == 0]
        if old:
            o = old[0]
            h3chk.append(dict(task=r["task"], accelerator=r["accelerator"], system_config=r["system_config"],
                              memory_mode=r["memory_mode"], cycles_rerun=r["cycles_total"],
                              cycles_sep10=o.get("cycles_total"), sram_kib_rerun=r["sram_memory_used"],
                              sram_kib_sep10=o.get("sram_memory_used"),
                              flash_kib_rerun=r["off_chip_flash_memory_used"],
                              flash_kib_sep10=o.get("off_chip_flash_memory_used")))

    vela_out = dict(
        meta=dict(
            produced_by="EdgeTRM/mcu_h1.py (p4d, ~/venvs/mcu: torch 2.13, litert_torch 0.9.4, "
                        "ai_edge_quantizer 0.9.0, ai_edge_litert 2.2.0, ethos-u-vela 5.2.0)",
            units="sram/flash/dram *_memory_used in KiB (Vela summary CSV); inference_time in seconds as in the "
                  "CSV, inference_ms = 1e3 * inference_time; *_bytes are bytes per inference; cycles are NPU "
                  "cycles and exclude operators Vela leaves on the CPU (see cpu_operators)",
            calibration="real carries from ~/aihub_artifacts/h1/<task>/calib_H<H>.npz (aihub_h1.py --stage calib); "
                        "last 4 samples held out for the numerical check",
            acceptance_vela_config="ethos-u55-256 / Ethos_U55_High_End_Embedded / Shared_Sram (the Corstone-300 "
                                   "FVP configuration)",
            configs=[list(c) for c in CFG_ORDER], notes=NOTES),
        acceptance=[slim_accept(r) for r in matrix],
        sweep=sweep_rows,
        h3_rerun_vs_sep10=h3chk,
        vela510_crosscheck=v510,
        int4_route=dict(tosa_ffn_random_weights=tosa, tosa_swiglu_real_weights=tosa_real,
                        tflite_single_linear=minrep, notes=int4_notes),
    )
    (out / "vela_h1.json").write_text(json.dumps(vela_out, indent=1))
    (out / "tflm_h1.json").write_text(json.dumps(tflm, indent=1))

    # ---------------------------------------------------------------- FVP
    fvp = []
    for log in sorted((raw / "fvp").glob("fvp_*_h*_*.run.log")):
        m = re.match(r"fvp_(sudoku_mlp|sudoku_attn|maze)_h(\d)_(\w+?)_(u\d\d_\d+)\.run\.log", log.name)
        if not m:
            continue
        task, H, var, npu = m.group(1), int(m.group(2)), m.group(3), m.group(4)
        rec = dict(task=task, H=H, variant=var, npu=npu, log=log.name, **parse_fvp(log))
        vj = rj(raw / "fvp" / f"vela_{task}_h{H}_{var}_{npu}.json")
        if vj:
            rec["vela_estimate"] = dict(cycles_total=vj.get("cycles_total"), inference_ms=vj.get("inference_ms"),
                                        sram_kib=vj.get("sram_memory_used"),
                                        flash_kib=vj.get("off_chip_flash_memory_used"),
                                        dram_kib=vj.get("dram_memory_used"), core_clock=vj.get("core_clock"),
                                        system_config=vj.get("system_config"), memory_mode=vj.get("memory_mode"),
                                        arena_cache_size_kib=vj.get("arena_cache_size"))
        act = rec.get("ACTIVE")
        if act:
            clk = CLK[npu]
            rec["clock_hz_for_ms"] = clk
            rec["npu_active_ms"] = act / clk * 1e3
            rec["npu_total_ms"] = rec.get("TOTAL", act) / clk * 1e3
            for n in STEPS:
                rec[f"ms_per_answer_n{n}"] = rec["npu_total_ms"] * n
            if vj and vj.get("cycles_total"):
                rec["fvp_over_vela_cycles"] = rec.get("TOTAL", act) / vj["cycles_total"]
            bb = BEAT[npu]
            rec["axi_bytes_assuming_%dB_beats" % bb] = {k: v * bb for k, v in rec.items()
                                                      if isinstance(v, int) and "BEAT" in k}
            if task == "sudoku_mlp" and npu == "u55_256":
                rec["h3_fvp_sep10_total_cycles"] = FVP_H3_SEP10["npu_total_cycles"]
                rec["h3_over_h1_fvp_cycles"] = FVP_H3_SEP10["npu_total_cycles"] / rec.get("TOTAL", act)
        fvp.append(rec)
    (out / "fvp_h1.json").write_text(json.dumps(dict(
        meta=dict(platform="Arm Corstone-300 FVP (FVP_Corstone_SSE-300_Ethos-U55 / -U65), ml-embedded-evaluation-kit "
                           "26.06 inference_runner, one inference with random inputs (token ids zero)",
                  cycles_note="NPU cycles from the Ethos-U PMU as printed by the kit. Arm's kit FAQ: the Cortex-M "
                              "part of the Fast Model is not cycle accurate or approximate, the Ethos-U part is "
                              "cycle approximate. ms use the clock of the Vela system config the graph was "
                              "compiled for (U55 High-End Embedded 500 MHz; U65 High-End 1 GHz).",
                  h3_reference=FVP_H3_SEP10),
        runs=fvp), indent=1))

    # ---------------------------------------------------------------- markdown
    L = ["# EdgeTRM inner step on Arm Ethos-U at H=1 (and H=3, same toolchain)", "",
         "Produced by `mcu_h1.py` / `mcu_h1_fvp.sh` / `mcu_h1_report.py` on p4d (CPU only). Vela 5.2.0; "
         "graphs quantized with the AI Edge Quantizer 0.9.0 on real carries. Vela numbers are compiler "
         "estimates of NPU cycles; operators left on the CPU are not in them. One recursion step = one "
         "supervision step (H x (L+1) applications of the shared block); ms per answer = ms per step x n.", ""]

    def get(task, H, var, cfg):
        for r in sweep_rows:
            if (r["task"], r["H"], r["variant"], r["accelerator"], r["system_config"], r["memory_mode"]) == \
                    (task, H, var, *cfg):
                return r
        return None

    fvp_by = {(r["task"], r["H"], r["npu"]): r for r in fvp}
    L += ["## Headline: INT8 step on Ethos-U55-256 and Ethos-U85-256, H=1 vs H=3", "",
          "Vela estimates (U55-256: Ethos_U55_High_End_Embedded, Shared_Sram, 500 MHz; U85-256: "
          "Ethos_U85_SYS_Flash_High, Shared_Sram, 500 MHz) and, where run, the Corstone-300 FVP NPU cycle "
          "count for the same compiled U55 graph.", "",
          "| task | H | NPU | SRAM KiB | flash KiB | Vela cycles | ms/step | ms/answer n=8 | ms/answer n=16 | "
          "FVP NPU cycles | FVP ms/step |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for task in TASKS:
        for cfg, npu in ((CFG_ORDER[3], "U55-256"), (CFG_ORDER[9], "U85-256")):
            for H in (1, 3):
                r = get(task, H, "int8", cfg)
                if not r or not r["ok"]:
                    continue
                ms = r["inference_ms"]
                fv = fvp_by.get((task, H, "u55_256")) if npu == "U55-256" else None
                fvc = (fv or {}).get("TOTAL")
                flag = " (unvalidated: NPU-model errors)" if (fv or {}).get("npu_model_critical_count") else ""
                L.append(f"| {task} | {H} | {npu} | {fmt(r['sram_memory_used'])} | "
                         f"{fmt(r['off_chip_flash_memory_used'])} | {fmt(int(r['cycles_total']))} | {fmt(ms, 2)} | "
                         f"{fmt(ms * 8, 1)} | {fmt(ms * 16, 1)} | {fmt(fvc)}{flag} | "
                         f"{fmt(fvc / 500e6 * 1e3 if fvc else None, 2)} |")
    L.append("")
    for var in ("int8", "w8a16"):
        L += [f"## {var.upper()} step, H=1 vs H=3, every Vela configuration", "",
              "| task | accelerator | system config | memory | H | SRAM KiB | flash KiB | DRAM KiB | NPU cycles | "
              "ms/step | ms/answer n=8 | ms/answer n=16 | CPU ops |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for task in TASKS:
            for cfg in CFG_ORDER:
                for H in (1, 3):
                    r = get(task, H, var, cfg)
                    if r is None:
                        continue
                    if not r["ok"]:
                        err = (r.get("error") or "").strip()
                        m = re.search(r"Invalid configuration of [^']*", err)
                        why = m.group(0) if m else (r.get("signal") or str(r["rc"]))
                        L.append(f"| {task} | {cfg[0]} | {cfg[1]} | {cfg[2]} | {H} | fails: {why[:90]} |"
                                 + " |" * 7)
                        continue
                    ms = r.get("inference_ms")
                    L.append(f"| {task} | {cfg[0]} | {cfg[1]} | {cfg[2]} | {H} | {fmt(r['sram_memory_used'])} | "
                             f"{fmt(r['off_chip_flash_memory_used'])} | {fmt(r.get('dram_memory_used'))} | "
                             f"{fmt(int(r['cycles_total']))} | {fmt(ms, 2)} | {fmt(ms * 8 if ms else None, 1)} | "
                             f"{fmt(ms * 16 if ms else None, 1)} | {fmt(r.get('cpu_operators'))} |")
        L.append("")

    # Table 2: FVP
    L += ["## Corstone-300 FVP (INT8)", "",
          "Same Vela-compiled graph the kit embeds; NPU cycles from the Ethos-U PMU; ms at the Vela system-config "
          "clock (U55 High-End Embedded 500 MHz, U65 High-End 1 GHz). U65 runs use the kit's U65 default: "
          "Dedicated_Sram with a 384 KiB NPU cache, weights and arena in DRAM.", "",
          "| task | H | NPU | NPU ACTIVE cycles | NPU TOTAL cycles | ms/step @ clock | Vela est. cycles | FVP/Vela | "
          "AXI0 rd beats | AXI0 wr beats | AXI1 rd beats | arena B | wall s | NPU-model critical msgs |",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(fvp, key=lambda r: (TASKS.index(r["task"]), r["npu"], r["H"])):
        ve = (r.get("vela_estimate") or {}).get("cycles_total")
        L.append(f"| {r['task']} | {r['H']} | {r['npu']} | {fmt(r.get('ACTIVE'))} | {fmt(r.get('TOTAL'))} | "
                 f"{fmt(r.get('npu_total_ms'), 2)} @ {fmt((r.get('clock_hz_for_ms') or 0) / 1e6, 0)} MHz | "
                 f"{fmt(int(ve) if ve else None)} | {fmt(r.get('fvp_over_vela_cycles'), 3)} | "
                 f"{fmt(r.get('AXI0_RD_DATA_BEAT_RECEIVED'))} | {fmt(r.get('AXI0_WR_DATA_BEAT_WRITTEN'))} | "
                 f"{fmt(r.get('AXI1_RD_DATA_BEAT_RECEIVED'))} | {fmt(r.get('tensor_arena_bytes'))} | "
                 f"{fmt(r.get('wall_time_s'), 0)} | {fmt(r.get('npu_model_critical_count'))} |")
    L += ["", f"H=3 reference (Sep 10, `{FVP_H3_SEP10['source']}`, unit-normal calibration): Sudoku-MLP INT8 on "
              f"U55-256, NPU TOTAL {FVP_H3_SEP10['npu_total_cycles']:,} cycles."]
    a1, a3 = fvp_by.get(("sudoku_mlp", 1, "u55_256")), fvp_by.get(("sudoku_mlp", 3, "u55_256"))
    if a1 and a3 and a1.get("TOTAL") and a3.get("TOTAL"):
        L += [f"Sudoku-MLP on U55-256, same toolchain and real-carry calibration: H=3 / H=1 FVP NPU cycles = "
              f"{a3['TOTAL']:,} / {a1['TOTAL']:,} = {a3['TOTAL'] / a1['TOTAL']:.4f}; the H=3 re-run differs from "
              f"the Sep 10 H=3 run by {a3['TOTAL'] - FVP_H3_SEP10['npu_total_cycles']:+,} cycles "
              f"({(a3['TOTAL'] / FVP_H3_SEP10['npu_total_cycles'] - 1) * 100:+.3f}%)."]
    L.append("")

    # Table 3: acceptance matrix
    L += ["## Acceptance matrix (quantizer / TFLite interpreter / Vela / TFLite Micro)", "",
          "Interpreter: held-out real carries (4 samples), max |logit diff| vs torch fp32 and argmax agreement. "
          "Vela: U55-256 High-End Embedded, Shared_Sram. TFLM: arena from the recording allocator; one invoke "
          "compared with torch.", "",
          "| task | H | variant | quantizer | interpreter (max abs / argmax) | Vela | TFLM arena B | TFLM invoke vs torch |",
          "|---|---|---|---|---|---|---:|---|"]
    for r in vela_out["acceptance"]:
        if r["variant"] == "fp32":
            i = r["interpreter"] or {}
            L.append(f"| {r['task']} | {r['H']} | fp32 | - | {fmt(i.get('heldout_max_abs_logit_diff'), 4)} / "
                     f"{fmt(i.get('heldout_argmax_agree'), 3)} | - | - | - |")
            continue
        q, i, v, t = r["quantizer"], r.get("interpreter") or {}, r.get("vela_default") or {}, r.get("tflm") or {}
        qs = "ok" if q["ok"] else "REFUSED: " + (q["error"] or "")[:110].replace("|", "/")
        if not q["ok"]:
            L.append(f"| {r['task']} | {r['H']} | {r['variant']} | {qs} | - | - | - | - |")
            continue
        is_ = (f"{fmt(i.get('heldout_max_abs_logit_diff'), 1)} / {fmt(i.get('heldout_argmax_agree'), 3)}"
               + (f" ({i.get('resolver')})" if i.get("resolver") and "XNNPACK" not in i.get("resolver") else "")
               if i.get("ok") else "FAILS: " + (i.get("error") or "")[:100].replace("|", "/"))
        vs = (f"ok, {fmt(int(v['cycles']))} cyc, {fmt(v.get('cpu_operators'))} CPU ops" if v.get("ok")
              else f"FAILS ({v.get('signal') or v.get('rc')})" + ("" if v.get("signal") else ": " +
                                                                    (v.get("error") or "")[-80:].replace("|", "/")))
        if t.get("ok"):
            if t.get("invoke_ok"):
                ts = (f"argmax {fmt(t.get('vs_torch_argmax_agree'), 3)}, {t.get('output_distinct_values')} distinct "
                      f"output values")
            elif t.get("invoke_ok") is False:
                ts = "invoke FAILS: " + (t.get("invoke_error") or "")[:60].replace("|", "/").replace("\n", " ")
            else:
                ts = "not invoked"
            ta = fmt(t.get("arena_total_bytes"))
        else:
            ta = "-"
            ts = "FAILS: " + ((t.get("tflm_messages") or [""])[0] or t.get("error") or "")[:100].replace("|", "/")
            if t.get("arena_total_bytes"):
                ta = fmt(t.get("arena_total_bytes"))
        L.append(f"| {r['task']} | {r['H']} | {r['variant']} | {qs} | {is_} | {vs} | {ta} | {ts} |")
    L.append("")
    for title, tt in (("real weights: one sudoku_attn block's SwiGLU, gate_up 3072x512 and down 512x1536, "
                       "97 tokens, from the H=1 int8 / w4a8_c graphs", tosa_real),
                      ("random weights: 97 tokens, 512->1536->512", tosa)):
        if not tt:
            continue
        L += [f"## INT4 weights on Ethos-U through TOSA 1.0 ({title})",
              "", "| weights | accelerator | original weight B | NPU-encoded weight B | NPU cycles |",
              "|---|---|---:|---:|---:|"]
        for g in tt:
            for v in g.get("vela", []):
                L.append(f"| {g['weights']} | {v['accelerator']} | {fmt(v.get('total_original_weights'), 0)} | "
                         f"{fmt(v.get('total_npu_encoded_weights'), 0)} | {fmt(v.get('cycles_total'), 0)} |"
                         if v["rc"] == 0 else f"| {g['weights']} | {v['accelerator']} | fails rc={v['rc']} |||")
        L.append("")
    L += ["## Notes", ""] + [f"- {n}" for n in NOTES] + [""]
    (out / "mcu_h1.md").write_text("\n".join(L) + "\n")
    print("wrote", out / "vela_h1.json", out / "tflm_h1.json", out / "fvp_h1.json", out / "mcu_h1.md")


if __name__ == "__main__":
    main()
