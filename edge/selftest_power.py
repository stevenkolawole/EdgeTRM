#!/usr/bin/env python3
"""Self-test of edge_loop's power sampler and window integration on a fake sysfs directory: three "rails"
whose files a writer thread rewrites at 200 Hz with a known power profile (idle 1500 mW, then a 4000 mW plateau
during the "run" window), so the integrated energy has a closed form to check against. No hardware needed.

    python3 selftest_power.py [--dir /tmp/fake_ina3221]
"""
import argparse
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import edge_loop as E  # noqa: E402


class FakeRails(threading.Thread):
    def __init__(self, d, hz=200.0):
        threading.Thread.__init__(self)
        self.daemon = True
        self.d, self.hz = d, hz
        self.level = [1500.0, 300.0, 900.0]                  # mW: total, gpu, cpu (idle)
        self.stop_ = threading.Event()

    def write(self):
        for k, v in enumerate(self.level):                    # in place, like a sysfs node (same inode)
            with open(os.path.join(self.d, "in_power%d_input" % k), "w") as f:
                f.write("%d\n" % int(v))

    def run(self):
        while not self.stop_.is_set():
            self.write()
            time.sleep(1.0 / self.hz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/fake_ina3221_%d" % os.getpid())
    a = ap.parse_args()
    if not os.path.isdir(a.dir):
        os.makedirs(a.dir)
    rails = FakeRails(a.dir)
    rails.write()
    rails.start()
    files = [os.path.join(a.dir, "in_power%d_input" % k) for k in range(3)]
    names = ["POM_5V_IN", "GPU", "CPU"]
    ok = True
    # (1) the sampler's kept fds see value changes through pread(0), as on sysfs
    s = E.PowerSampler(files, names, hz=100.0)
    s.start()
    assert s.read_once() == [1500.0, 300.0, 900.0], s.read_once()
    rails.level = [4000.0, 2000.0, 1200.0]
    time.sleep(0.05)
    assert s.read_once()[0] == 4000.0, s.read_once()
    print("fd re-read after value change: ok")
    rails.level = [1500.0, 300.0, 900.0]
    time.sleep(0.05)
    # (2) idle window: 1.0 s at 1.5 W -> 1.5 J
    w = E.idle_window(1.0, s, names, 0, "start")
    print("idle: %.3f W over %.3f s, %d samples (%.0f Hz), E %.4f J" % (w["P_W"], w["T_s"], w["n_samples"], w["sample_hz"], w["E_J"]))
    ok &= abs(w["P_W"] - 1.5) < 0.05 and w["n_samples"] >= 80
    # (3) run window: 0.5 s idle level then 1.0 s at 4 W then back: integrate over [t0, t1] = 1.5 s window
    s.take()
    s.on.set()
    t0 = time.perf_counter()
    time.sleep(0.5)
    rails.level = [4000.0, 2000.0, 1200.0]
    time.sleep(1.0)
    rails.level = [1500.0, 300.0, 900.0]
    t1 = time.perf_counter()
    s.on.clear()
    e = E.integrate(s.take(), names, t0, t1, 0)
    expect = 1.5 * 0.5 + 4.0 * 1.0
    print("run: E %.4f J expected ~%.3f J (%.1f%%), rails %s" % (
        e["E_J"], expect, 100 * (e["E_J"] / expect - 1), dict((k, round(v["E_J"], 3)) for k, v in e["rails"].items())))
    ok &= abs(e["E_J"] - expect) / expect < 0.06
    ok &= abs(e["rails"]["GPU"]["E_J"] - (0.3 * 0.5 + 2.0 * 1.0)) / 2.15 < 0.06
    # (4) idle subtraction as in edge_loop.window: dyn = E - idle_P * T
    Ed = e["E_J"] - w["P_W"] * (t1 - t0)
    print("dyn: %.4f J expected ~%.3f J" % (Ed, expect - 1.5 * 1.5))
    ok &= abs(Ed - (expect - 1.5 * 1.5)) < 0.2
    # (5) missing files -> the loop runs without energy (edge_loop main checks existence); integrate() on no rows
    e0 = E.integrate([], names, 0.0, 1.0, 0)
    ok &= e0["E_J"] is None and e0["n_samples"] == 0
    s.close()
    rails.stop_.set()
    print("selftest_power:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
