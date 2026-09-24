#!/bin/bash
# Q7 (X5) supervisor on sk-p4d. Singleton (flock). Started by cron @reboot and by hand; safe to rerun.
# Reads ~/q7/p4d_manifest.tsv, one run per line (tab-separated):
#     card  run  seed  requires  overrides...
# `requires` is a file that must exist before the run may start (a fork point), or "-".
# A run is launched on its card when it is not DONE, not already running, its requirement exists, and
# the card holds fewer than 2 GPU processes (ours + anybody else's). Cards other than 3-5 are refused.
# Every launch and exit is logged to ~/q7/logs/runner.log; per-run logs are ~/q7/logs/<run>.log.
# Stops when every manifest run is DONE.
set -f   # no globbing: overrides carry [..] lists
exec 9>"$HOME/q7/runner2.lock"   # children close fd 9 (9>&-) so they never hold the lock
flock -n 9 || { echo "$(date -u) runner already active"; exit 0; }
source "$HOME/q7/env_p4d.sh"
MANIFEST=$HOME/q7/p4d_manifest.tsv
PIDS=$HOME/q7/pids
mkdir -p "$PIDS" "$HOME/q7/logs"
log() { echo "$(date -u '+%F %T') $*"; }

alive() {  # run name -> 0 if its wrapper is running
  local f="$PIDS/$1.pid"
  [ -f "$f" ] || return 1
  local p; p=$(cat "$f")
  [ -d "/proc/$p" ] && tr '\0' ' ' < "/proc/$p/cmdline" | grep -q -E "q7_(train|evalrun).sh $1 "
}

card_procs() {  # GPU processes on card $1 (anyone's) + our runs on that card not yet visible there
  local uuid n ours=0 visible=0
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v c="$1" '$1==c{print $2}')
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid")
  while IFS=$'\t' read -r card run _rest; do
    [ "$card" = "$1" ] || continue
    if alive "$run"; then
      ours=$((ours + 1))
      # is this run's python already on the GPU?
      if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}' \
          | while read -r gp; do { tr '\0' '\n' < "/proc/$gp/environ" 2>/dev/null | grep -qx "Q7_RUN=$run" \
              || tr '\0' ' ' < "/proc/$gp/cmdline" 2>/dev/null | grep -q "run_name=$run "; } && echo y; done | grep -q y; then
        visible=$((visible + 1))
      fi
    fi
  done < <(grep -v '^#' "$MANIFEST")
  echo $((n + ours - visible))
}

log "runner start (pid $$)"
while true; do
  pending=0
  while IFS=$'\t' read -r card run seed req over; do
    [ -z "${run:-}" ] && continue
    if [ -f "$Q7_ROOT/ckpt/$run/DONE" ]; then
      # disk: a finished run no longer needs resume.pt (fork points resume_iter*.pt are kept)
      if [ -f "$Q7_ROOT/ckpt/$run/resume.pt" ] && ! alive "$run"; then rm -f "$Q7_ROOT/ckpt/$run/resume.pt"; log "cleaned resume.pt of $run"; fi
      continue
    fi
    pending=$((pending + 1))
    alive "$run" && continue
    case "$card" in 3|4|5) ;; *) log "refusing $run on card $card (X5 owns 3-5)"; continue ;; esac
    if [ "$req" != "-" ] && [ ! -f "$req" ]; then continue; fi
    np=$(card_procs "$card")
    if [ "$np" -ge 2 ]; then continue; fi
    script=q7_train.sh; [ "${run#eval_}" != "$run" ] && script=q7_evalrun.sh
    # shellcheck disable=SC2086
    if [ "$script" = q7_evalrun.sh ]; then set -- $over; else set -- "$seed" $over; fi
    Q7_RUN=$run CUDA_VISIBLE_DEVICES=$card setsid nohup bash "$HOME/q7/$script" "$run" "$@" \
      >> "$HOME/q7/logs/$run.log" 2>&1 < /dev/null 9>&- &
    echo $! > "$PIDS/$run.pid"
    log "launched $run seed=$seed card=$card pid=$! (card had $np procs) overrides: $over"
    sleep 90 9>&-   # let it reach the GPU before counting the card again
  done < <(grep -v '^#' "$MANIFEST")
  if [ "$pending" -eq 0 ]; then log "all manifest runs DONE; runner exits"; exit 0; fi
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "${free:-99}" -lt 3 ] && log "WARNING: only ${free} GB free on /"
  sleep 300 9>&-
done
