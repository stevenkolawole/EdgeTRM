"""Per-step cost of the re-draw (dither) on a GPU: time one supervision step of a TRM/HRM at batch 1 and batch 300,
against the time of re-rounding every linear from its master (fp32 or w8c) before the step. Reports the ratio the
MLSys draft needs (Q7, GPU half). Uses m2_bestofk's sr_quantize_ (same rounding as the exhibits).
    python scratch/dither_cost_gpu.py --task sudoku_mlp --spec w4c --master fp32 --out results/breadth/dither_cost_gpu.json
"""
import argparse, json, sys, time, pathlib
import torch
HERE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import m2_bestofk as M                      # noqa: E402
from breadth_sweep import build, parse_quant, _linears, DEV  # noqa: E402


def timed(fn, iters, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sudoku_mlp"); ap.add_argument("--spec", default="w4c")
    ap.add_argument("--master", default="fp32"); ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    q = parse_quant(a.spec)
    out = {"task": a.task, "spec": a.spec, "master": a.master, "gpu": torch.cuda.get_device_name(0), "rows": []}
    for B in (1, 32, 300):
        m, meta = build(a.task, 3, 16, B)
        m = m.to(DEV).eval()
        master = [mod.weight.data.clone() for mod in _linears(m)]
        gen = torch.Generator(device=DEV).manual_seed(0)
        inp, lab = M.load_rows(a.task, 0, B) if hasattr(M, "load_rows") else (None, None)
        from difficulty_sweep import _fresh, load_meta  # noqa: E402
        import numpy as np  # noqa: E402
        x = torch.randint(1, 10, (B, 81), device=DEV) if "sudoku" in a.task else torch.randint(0, 5, (B, 900), device=DEV)
        y = torch.zeros_like(x)
        bt = {"inputs": x, "labels": y, "puzzle_identifiers": torch.zeros(B, dtype=torch.int32, device=DEV)}
        carry = _fresh(m, bt)
        with torch.no_grad():
            t_step = timed(lambda: m(carry, bt), a.iters)
            t_round = timed(lambda: M.sr_quantize_(m, master, q["bits"], q["gran"], gen), a.iters)
        nparam = sum(w.numel() for w in master)
        out["rows"].append({"batch": B, "step_ms": 1e3 * t_step, "redraw_ms": 1e3 * t_round,
                            "redraw_over_step": t_round / t_step, "params_rerounded": nparam})
        print(f"{a.task} {a.spec} B={B}: step {1e3*t_step:.3f} ms, re-draw {1e3*t_round:.3f} ms, ratio {t_round/t_step:.3f}", flush=True)
        del m; torch.cuda.empty_cache()
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
