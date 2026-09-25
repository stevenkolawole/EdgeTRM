#!/bin/bash
# Q7 (X5): (re-)evaluate every step_N of the given runs with the current q7_eval.py (resumable per cell),
# then mark this job DONE. Launched by q7_p4d_runner.sh for manifest lines named eval_*.
#   usage: q7_evalrun.sh JOBNAME RUN [RUN ...]
set -u
JOB=$1; shift
source "$HOME/q7/env_p4d.sh"
cd "$HOME/edgetrm_src" || exit 4
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
$Q7_PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "[preflight] no CUDA"; exit 3; }
args=(); for r in "$@"; do case "$r" in --*) args+=("$r");; *) args+=(--rundir "$Q7_ROOT/ckpt/$r");; esac; done
echo "$(date -u) [q7_evalrun] $JOB: ${args[*]}"
$Q7_PY q7_eval.py "${args[@]}" 2>&1 | grep -v Warn
rc=${PIPESTATUS[0]}
if [ "$rc" = 0 ]; then mkdir -p "$Q7_ROOT/ckpt/$JOB" && date -u > "$Q7_ROOT/ckpt/$JOB/DONE"; fi
echo "$(date -u) [q7_evalrun] $JOB exited rc=$rc"
