# edge/ — board runner for the MLSys edge measurements

One Python 3.6-compatible loop (`edge_loop.py`, stdlib + numpy) runs the X3 step graph on every board: Jetson
Nano 4 GB x2 and 2 GB (JetPack 4.6, TensorRT 8.2, `--backend trt`) and Raspberry Pi 4 8 GB (Python 3.11,
onnxruntime CPU, `--backend ort`). Checklist and measurement protocol: `reports/edgetrm_exp/BRINGUP.md`,
`reports/edgetrm_exp/ENERGY_PROTOCOL.md`. The exit rules and the record layout are E1's (`e1_energy.py`).

## Files

| file | what |
|---|---|
| `edge_loop.py` | the runner: check pass (fixed 16 steps vs labels and host reference), record pass (exit rule / fixed depth, optional finishing graph), idle window, energy windows; one JSON per run |
| `build_engines.sh` | `trtexec` fp32 + fp16 engines per `stepc_H*.onnx` (and per `stepc_k*_H*.onnx` if an unrolled export exists) |
| `convert_opset.py` | host-side restamp of the opset-17 exports to opset 13 (`*_op13.onnx`): onnxruntime 1.10 (last Python 3.6 wheel) refuses opset > 15 and TensorRT 8.2's parser is of that generation; checked bit-identical on the host |
| `make_puzzles.py` | puzzle npz (inputs int32, labels int64) from the Sapient test CSVs, same row order as everywhere else |
| `selftest_power.py` | the INA3221 sampler + integration on a fake sysfs directory (no hardware) |
| `assets/<task>/` | `stepc_H1.onnx`, `stepc_H3.onnx` (X3 exports, copied from p4d `~/aihub_artifacts/h1/<task>/`) and their `_op13` copies (use these on the boards), `init.npz` (H_init, L_init, puzzle-embedding row 0), `host_fp32.npz` (X3's ORT host chain: 100 Sudoku / 50 Maze puzzles, 16 / 8 steps), `puzzles_*.npz`. Only `init.npz` is in git; the rest is gitignored (regenerate: p4d `aihub_h1.py --stage export`, `aihub_infer.py --stage host`; here `convert_opset.py --opset 13 assets/*/stepc_H*.onnx`, `make_puzzles.py`) |

## 1. Copy to a board

```sh
# from the repo (EdgeTRM/), one task at a time; the Nano needs only H1 + H3 onnx, init, puzzles, host ref
rsync -av --exclude '*.plan' --exclude 'stepc_H?.onnx' edge/ edge@nano-a:~/edge/   # ships the _op13 graphs only
```

## 2. Build (Jetson only)

```sh
ssh edge@nano-a
sudo nvpmodel -m 0 && sudo jetson_clocks          # 10 W + pinned clocks; or -m 1 for the 5 W rows; record which
sh ~/edge/build_engines.sh sudoku_mlp             # -> ~/edge/engines/sudoku_mlp/stepc_H{1,3}_op13_{fp32,fp16}.plan
python3 -c "import tensorrt; print(tensorrt.__version__)"   # JetPack's bindings; pycuda optional (ctypes fallback)
```

On the Pi: `python3 -m venv ~/venv && . ~/venv/bin/activate && pip install onnxruntime numpy`; fix the governor
(`echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`).

## 3. Sanity gate, then runs

```sh
cd ~/edge && mkdir -p results
# gate (BRINGUP A.7): fp32 engine, 100 puzzles, check pass only; exact counts per step must equal the host's
python3 edge_loop.py --backend trt --task sudoku_mlp --model engines/sudoku_mlp/stepc_H1_op13_fp32.plan \
    --init assets/sudoku_mlp/init.npz --puzzles assets/sudoku_mlp/puzzles_0_100.npz \
    --ref assets/sudoku_mlp/host_fp32.npz --board nano-a --precision fp32 --nwin 0 --no_power \
    --out results/nano-a_sudoku_mlp_trt_fp32_gate.json
# energy row: fp16 engine, exit rule, 1 idle window + 5 x 60 s windows on the INA3221 rails (defaults)
python3 edge_loop.py --backend trt --task sudoku_mlp --model engines/sudoku_mlp/stepc_H1_op13_fp16.plan \
    --init assets/sudoku_mlp/init.npz --puzzles assets/sudoku_mlp/puzzles_0_100.npz \
    --ref assets/sudoku_mlp/host_fp32.npz --board nano-a --precision fp16 --mode exit \
    --note "nvpmodel 0, jetson_clocks on, barrel jack" --out results/nano-a_sudoku_mlp_trt_fp16_exit.json
# fixed depth, finishing (j fp32 steps after the fp16 exit), unrolled-k engine:
#   --mode fixed --depth 16 | --finish engines/.../stepc_H1_op13_fp32.plan --finish_steps 2 --finish_precision fp32
#   | --model engines/.../stepc_k4_H1_fp16.plan --steps_per_call 4
# Pi (no power monitor): --backend ort --model assets/sudoku_mlp/stepc_H1_op13.onnx --threads 4 --no_power
# Nano 2 GB (no INA3221): --no_power, or --power_dir/--power_files for whatever meter exposes sysfs values
```

Every record has `meta` (board, backend, runtime version, precision, mode, rule, nvpmodel / jetson_clocks /
cpufreq state, thermal zones at start and end, power file paths, argv), `check` (exact per step, cell agreement and
final z_H cosine vs the host reference, offline exit statistics, per-step ms), `record` (per puzzle: steps,
correct, per-step ms, q_halt), `idle[]` and `windows[cfg]` in E1's `energy.json` layout (per window: `E_J`, `P_W`,
`E_dyn_J`, `mJ_per_step`, `J_per_solved`, `ms_per_step`, `acc_window`, per-rail energy, temperatures, `clean`).
A window that throttled (Pi `get_throttled`) is marked `clean: false` and left out of `median_clean`.

## 4. Results back

```sh
rsync -av edge@nano-a:~/edge/results/ edge/results/nano-a/      # the card is the only copy: after every session
```

## Local test (host, no board)

```sh
python3 edge/selftest_power.py                                      # fake sysfs rails
python3 edge/edge_loop.py --backend ort --task sudoku_mlp --model edge/assets/sudoku_mlp/stepc_H1.onnx \
    --init edge/assets/sudoku_mlp/init.npz --puzzles edge/assets/sudoku_mlp/puzzles_0_100.npz --n 20 \
    --ref edge/assets/sudoku_mlp/host_fp32.npz --no_power --nwin 0 --out /tmp/localtest.json
```
