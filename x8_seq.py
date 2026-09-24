#!/usr/bin/env python3
"""Run the lines of a file sequentially, each as `python <line>` (a JOBFILE line can only start one
process; this lets one GPU slot run several resumable sweeps back to back). Keeps going if one
line fails and reports each exit code."""
import shlex
import subprocess
import sys

for line in open(sys.argv[1]):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    print(f"[seq] python {line}", flush=True)
    rc = subprocess.call([sys.executable] + shlex.split(line))
    print(f"[seq] rc={rc} for {line.split()[0]} {' '.join(line.split()[1:5])}", flush=True)
