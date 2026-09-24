#!/usr/bin/env python3
"""M4: the release check's "model + quantization config" path (release_check.torch_check) over the
configurations whose compiled graphs have device-faithful labels (AI Hub QDQ host simulation, TFLite
interpreter), plus a label run of the torch emulation itself on the evaluation puzzles.

Matched configurations (torch emulation <-> compiled graph):
  w8c+a8all  <-> AI Hub w8a8, TFLite int8         w4c+a8all  <-> AI Hub w4a8, TFLite w4a8_c
  w8c+a16all <-> AI Hub w8a16, TFLite w8a16       w4c+a16all <-> AI Hub w4a16, TFLite w4a16_c
  w4t+a8all  <-> TFLite w4a8_t                    w8c+a8lin  <-> TFLite w8a8_fc (FC-only), harness +a8
plus weight-only cells (w8c, w4c, w4t, w3g32, w3c, noise) for the torch-harness backend.

    python m4_torch_sweep.py --device cuda --out_dir ~/m4/ev_torch [--tasks sudoku_mlp,...]
"""
import argparse
import json
import pathlib
import sys
import time
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

ACT = ["w8c+a8all", "w4c+a8all", "w8c+a16all", "w4c+a16all", "w4t+a8all", "w8c+a8lin", "fp32+a8all"]
WEIGHT = ["w8c", "w4c", "w4t", "w3g32", "w3c", "noise0.1", "noise0.2"]
CFG = {
    "sudoku_mlp": dict(H=1, T=16, n=32, label_n=100, configs=ACT + WEIGHT),
    "sudoku_attn": dict(H=1, T=16, n=32, label_n=100, configs=ACT + WEIGHT + ["w3t", "noise0.5", "noise0.8"]),
    "maze_attn": dict(H=1, T=8, n=16, label_n=50, configs=ACT + ["w4c", "w4t", "w3c", "noise1.0", "noise2.0"]),
    "hrm_sudoku": dict(H=None, T=16, n=32, label_n=100, configs=["w8c+a8all", "w8c+a16all", "w4c+a8all", "w8c",
                                                                "w4c", "w4t", "w3c", "w2a", "noise0.5", "noise1.0"]),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_dir", default=str(pathlib.Path.home() / "m4" / "ev_torch"))
    p.add_argument("--tasks", default=",".join(CFG))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    import torch
    if a.device == "cuda":
        assert torch.cuda.is_available(), "NO CUDA"
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    import release_check as R
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task in a.tasks.split(","):
        c = CFG[task]
        for cfg in c["configs"]:
            w, _, am = cfg.partition("+")
            am = am if am else "none"
            out = out_dir / f"torch__{task}__{cfg}.json"
            if out.exists():
                continue
            t0 = time.time()
            try:
                ev = R.torch_check(task, w, am, c["H"], c["T"], None, c["n"], threads=a.threads,
                                   device=a.device, label_n=c["label_n"], batch=32 if "maze" not in task else 8)
                ev.update(config=cfg, backend="torch_emulation" if am != "none" else "torch_harness")
                v, rules = R.decide(ev)
                ev.update(verdict=v, rules=rules)
                out.write_text(json.dumps(R._strip(ev), indent=1, default=float))
                print(f"[{time.strftime('%H:%M:%S')}] {task} {cfg}: {v} delta={ev['delta_T_median']:.3f} "
                      f"cell={ev['cell_agree']:.3f} label {ev['label_exact']:.3f}/{ev['label_exact_fp32']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as e:  # noqa: BLE001
                (out_dir / f"torch__{task}__{cfg}.err").write_text(traceback.format_exc())
                print(f"{task} {cfg}: ERROR {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
