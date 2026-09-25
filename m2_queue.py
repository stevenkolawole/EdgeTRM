#!/usr/bin/env python3
"""Dynamic work queue for M2 (priorities can change while jobs run). Queue dir layout: pending/, running/, done/,
failed/. Each task is a file whose single line is "<script> <args>" (run as `python <line>`); tasks are claimed in
file-name order by atomic rename, so several workers (JOBFILE lines, jobs) can share one queue. On (re)start a
worker returns its own stale claims (suffix .<SLURM_JOB_ID>.<worker>) to pending. Exits when pending/ is empty.

    python m2_queue.py ~/m2_results/q2 <worker-name>
"""
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

q = Path(sys.argv[1]); name = sys.argv[2]
tag = f"{os.environ.get('SLURM_JOB_ID', 'local')}.{name}"
for d in ("pending", "running", "done", "failed"):
    (q / d).mkdir(parents=True, exist_ok=True)
for f in (q / "running").glob(f"*.{tag}"):
    os.replace(f, q / "pending" / f.name[: -len(tag) - 1])
while True:
    tasks = sorted((q / "pending").iterdir())
    if not tasks:
        break
    claimed = None
    for t in tasks:
        dst = q / "running" / f"{t.name}.{tag}"
        try:
            os.rename(t, dst)
            claimed = dst
            break
        except FileNotFoundError:
            continue
    if claimed is None:
        continue
    line = claimed.read_text().strip()
    print(f"[q {name}] {time.strftime('%H:%M:%S', time.gmtime())} start {claimed.name}: {line[:160]}", flush=True)
    rc = subprocess.call([sys.executable] + shlex.split(line))
    base = claimed.name[: -len(tag) - 1]
    os.replace(claimed, q / ("done" if rc == 0 else "failed") / base)
    print(f"[q {name}] rc={rc} {base}", flush=True)
print(f"[q {name}] queue empty", flush=True)
