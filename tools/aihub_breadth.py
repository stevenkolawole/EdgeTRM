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
    python aihub_breadth.py --plan qairt245 --slots 2   (Sep 29 rerun of the chains whose compile ran on QAIRT 2.50,
                                                         on the pinned 2.45 compiles; see aihub_acct.QAIRT and
                                                         DIAG_ACCT_SPLIT.md; the old records were moved to *.qairt250)
--slots N caps the device jobs in flight per account (0 = every chain advances each loop, the Sep 26-28 behaviour);
chains are served fewest-steps-first so the gate (two-step probe) is reached everywhere before any chain finishes.
Devices an account no longer hosts are dropped at startup with a message instead of failing the OS assert.
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
# Sep 29: the cells whose Sep 26 compiles ran on QAIRT 2.50 (carry clipped), rerun on the 2.45 pin; the S21 / 7 Gen 4
# (acct2) and RB3 Gen 2 / Q-6690 (acct3) retries ask whether their MODEL_GRAPH_ERROR was version-dependent.
RERUN = {"acct2": (["Samsung Galaxy S25", "Samsung Galaxy S23", "Samsung Galaxy S22 5G", "Samsung Galaxy Tab S8",
                    "Snapdragon 8 Elite Gen 5 QRD", "Samsung Galaxy S21", "Snapdragon 7 Gen 4 QRD"],
                   [("sudoku_mlp", "w8a8"), ("sudoku_mlp", "w8a16"), ("sudoku_attn", "w8a8"), ("maze", "w8a8")]),
         "acct3": (["Arduino VENTUNO Q", "Dragonwing IQ-9075 EVK", "QCS8550 (Proxy)", "SA8775P ADP",
                    "Snapdragon X Elite CRD", "Dragonwing RB3 Gen 2 Vision Kit", "Dragonwing Q-6690 MTP"],
                   [("maze", "w8a8rope16")])}


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
    ap.add_argument("--slots", type=int, default=0, help="device jobs in flight per account (0 = unlimited)")
    ap.add_argument("--sleep", type=int, default=90)
    a = ap.parse_args()
    accts = {n: X.Acct(n) for n in a.accts.split(",")}
    for acct in accts.values():
        X.OS.update(device_os(acct.C))
    chains = []
    if a.smoke:
        chains.append(X.Chain(accts["acct2"], "sudoku_mlp", "fp32", "Samsung Galaxy S25", "qnn_dlc", 1))
    elif a.plan == "qairt245":
        for n, (devs, cells) in RERUN.items():
            if n not in accts:
                continue
            for d in devs:
                for t, p in cells:
                    chains.append(X.Chain(accts[n], t, p, d, "qnn_dlc", 8))
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
    hosted = {n: set(device_os(acct.C)) for n, acct in accts.items()}
    for c in [c for c in chains if c.device not in hosted[c.a.name]]:
        print(f"[breadth] DROPPED {c.task}/{c.tag}: {c.device} is not hosted on {c.a.name}", flush=True)
        chains.remove(c)
    missing = [c.device for c in chains if c.device not in X.OS]
    assert not missing, f"devices without an OS entry: {sorted(set(missing))}"
    print(f"[breadth] {len(chains)} chains on {list(accts)} (plan {a.plan}, qairt {X.QAIRT}, slots {a.slots or 'unlimited'})",
          flush=True)
    while True:
        for acct in accts.values():
            bs = acct.status()
            mine = sorted([c for c in chains if c.a is acct], key=lambda c: c.steps_done)   # gate-first
            free = (a.slots - sum(c.inflight(bs) for c in mine)) if a.slots else None
            for c in mine:
                try:
                    used = c.step(bs, free)
                    if free is not None:
                        free -= used
                except Exception as exc:  # noqa: BLE001
                    print(f"[chain {c.task} {c.tag}] ERROR {type(exc).__name__}: {str(exc)[:200]}", flush=True)
        active = [c for c in chains if not c.done]
        done = [c for c in chains if c.meta.get("done")]
        failed = [c for c in chains if c.meta.get("failed")]
        print(f"[breadth] {time.strftime('%H:%M:%S', time.gmtime())} active {len(active)} done {len(done)} failed {len(failed)}",
              flush=True)
        if not active:
            break
        time.sleep(a.sleep)


if __name__ == "__main__":
    main()
