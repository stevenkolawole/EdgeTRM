#!/bin/bash
# Pull the X3 device records from p4d (AI Hub job records, status, on-device inference records)
# and Babel (E0 / carry-precision accuracy) into results/devices/. Idempotent.
set -u
cd "$(dirname "$0")"
D=results/devices
mkdir -p $D/aihub_h1 $D/chipq $D/infer
for t in sudoku_mlp sudoku_attn maze arc; do
  scp -q sk-p4d:~/aihub_artifacts/h1/$t/jobs.json $D/aihub_h1/${t}_jobs.json 2>/dev/null
  mkdir -p $D/infer/$t
  scp -q "sk-p4d:~/aihub_artifacts/h1/$t/infer/*.json" $D/infer/$t/ 2>/dev/null
done
scp -q sk-p4d:~/aihub_artifacts/h1/status.json $D/aihub_h1/status.json 2>/dev/null
scp -q sk-p4d:~/aihub_artifacts/h1/infer_report.json $D/infer/infer_report.json 2>/dev/null
scp -q sk-p4d:~/aihub_artifacts/h1/uploaded.json $D/aihub_h1/uploaded.json 2>/dev/null
# Babel: E0 and carry sweeps (results live in $HOME, visible from the login node)
scp -q "babel:~/breadth_results/x3/*.json" $D/chipq/ 2>/dev/null
ls -la $D/aihub_h1 $D/chipq $D/infer | head -60
