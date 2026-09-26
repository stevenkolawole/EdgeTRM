#!/usr/bin/env python3
"""Device-breadth chains on AI Hub (Sep 26 evening): the same host-loop chains as aihub_acct.py, on every hosted
device class we can reach, gate-first. Reuses aihub_acct.Acct / Chain; tokens are never read or printed here
(qai_hub.Client(profile=...) loads them). Runs on p4d from ~/x3.

Plan: for each device, Sudoku-MLP {fp32, w8a8, w8a16}, Sudoku-attn {w8a8}, Maze {fp32, w8a8, w8a8rope16}, 8 steps,
100 puzzles (Maze 50), qnn_dlc where the device lists framework:qnn, tflite otherwise (Pixel, Exynos). Chains advance
one step per loop across all devices, so every device reaches the two-step probe before any reaches step 8.
Records land beside the S24 / 8 Elite ones: ~/aihub_artifacts/h1/<task>/infer/dev_<prec>_<device>_<acct>.{json,npz};
device_acc.py picks them up.

    python aihub_breadth.py --plan tier1 [--smoke]      (loops until every chain is done or failed; resumable)
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import aihub_acct as X  # noqa: E402

QNN = {"acct2": ["Samsung Galaxy S25", "Samsung Galaxy S23", "Samsung Galaxy S22 5G", "Samsung Galaxy S21",
                 "Samsung Galaxy Tab S8", "Snapdragon 8 Elite Gen 5 QRD", "Snapdragon 7 Gen 4 QRD"],
       "acct3": ["Dragonwing RB3 Gen 2 Vision Kit", "Dragonwing IQ-9075 EVK", "QCS8550 (Proxy)", "SA8775P ADP",
                 "Arduino VENTUNO Q", "Snapdragon X Elite CRD", "Dragonwing Q-6690 MTP"]}
TFLITE = {"acct2": ["Google Pixel 9", "Google Pixel 10"], "acct3": ["Samsung Galaxy A53 5G"]}
CELLS = [("sudoku_mlp", "fp32"), ("sudoku_mlp", "w8a8"), ("sudoku_mlp", "w8a16"), ("sudoku_attn", "w8a8"),
         ("maze", "fp32"), ("maze", "w8a8"), ("maze", "w8a8rope16")]
TFL_CELLS = [("sudoku_mlp", "fp32"), ("sudoku_mlp", "w8a8")]


def device_os(client):
    os_of = {}
    for d in client.get_devices():
        os_of.setdefault(d.name, d.os)
    return os_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="tier1")
    ap.add_argument("--smoke", action="store_true", help="one chain (S25, Sudoku-MLP fp32, 1 step) on acct2")
    ap.add_argument("--accts", default="acct2,acct3")
    a = ap.parse_args()
    accts = {n: X.Acct(n) for n in a.accts.split(",")}
    for acct in accts.values():
        X.OS.update(device_os(acct.C))
    chains = []
    if a.smoke:
        chains.append(X.Chain(accts["acct2"], "sudoku_mlp", "fp32", "Samsung Galaxy S25", "qnn_dlc", 1))
    else:
        for n, devs in QNN.items():
            if n not in accts:
                continue
            for d in devs:
                for t, p in CELLS:
                    chains.append(X.Chain(accts[n], t, p, d, "qnn_dlc", 8))
        for n, devs in TFLITE.items():
            if n not in accts:
                continue
            for d in devs:
                for t, p in TFL_CELLS:
                    chains.append(X.Chain(accts[n], t, p, d, "tflite", 8))
    missing = [c.device for c in chains if c.device not in X.OS]
    assert not missing, f"devices without an OS entry: {sorted(set(missing))}"
    print(f"[breadth] {len(chains)} chains on {list(accts)}", flush=True)
    while True:
        for acct in accts.values():
            bs = acct.status()
            for c in chains:
                if c.a is acct:
                    try:
                        c.step(bs)
                    except Exception as exc:  # noqa: BLE001
                        print(f"[chain {c.task} {c.tag}] ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        active = [c for c in chains if not c.done]
        done = [c for c in chains if c.meta.get("done")]
        failed = [c for c in chains if c.meta.get("failed")]
        print(f"[breadth] {time.strftime('%H:%M:%S', time.gmtime())} active {len(active)} done {len(done)} failed {len(failed)}",
              flush=True)
        if not active:
            break
        time.sleep(90)


if __name__ == "__main__":
    main()
