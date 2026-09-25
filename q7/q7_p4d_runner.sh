#!/bin/bash
# Q7 / M6 (X5) supervisor on sk-p4d. Singleton (flock). Started by cron @reboot and by hand; safe to rerun.
# Reads ~/q7/p4d_manifest.tsv, one run per line (tab-separated):
#     card  run  seed  requires  overrides...
# card: a card number or "A" (any card in CARDS). requires: "-" or comma-separated files that must exist.
# Every pass launches, in manifest order, every run that is not DONE, not running, whose requirements
# exist and for which a card in CARDS holds fewer than 2 GPU processes (ours + anybody else's).
# Runs named eval_* go through q7_evalrun.sh (overrides = run names to evaluate).
# Since Sep 25 00:20 UTC X5 owns cards 3 and 4 only (card 5 went to M1).
# Logs: ~/q7/logs/runner.log (launches), ~/q7/logs/<run>.log; card of each run in ~/q7/pids/<run>.card.
set -f   # no globbing: overrides carry [..] lists
exec 9>"$HOME/q7/runner2.lock"   # children close fd 9 (9>&-) so they never hold the lock
flock -n 9 || { echo "$(date -u) runner already active"; exit 0; }
source "$HOME/q7/env_p4d.sh"
MANIFEST=$HOME/q7/p4d_manifest.tsv
PIDS=$HOME/q7/pids
CARDS="3 4"
mkdir -p "$PIDS" "$HOME/q7/logs"
log() { echo "$(date -u '+%F %T') $*"; }

alive() {  # run name -> 0 if its wrapper is running
  local f="$PIDS/$1.pid"
  [ -f "$f" ] || return 1
  local p; p=$(cat "$f")
  [ -d "/proc/$p" ] && tr '\0' ' ' < "/proc/$p/cmdline" | grep -q -E "q7_(train|evalrun).sh $1 "
}

card_procs() {  # GPU processes on card $1 (anyone's) + our live runs on that card not yet visible there
  local uuid n ours=0 visible=0 f run gp
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v c="$1" '$1==c{print $2}')
  n=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep -c "$uuid")
  for f in $(ls "$PIDS" 2>/dev/null | grep '[.]card$'); do   # (set -f: no globbing)
    f="$PIDS/$f"
    [ "$(cat "$f")" = "$1" ] || continue
    run=$(basename "$f" .card)
    alive "$run" || continue
    ours=$((ours + 1))
    for gp in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | awk -F', ' '{print $2}'); do
      if tr '\0' '\n' < "/proc/$gp/environ" 2>/dev/null | grep -qx "Q7_RUN=$run" \
         || tr '\0' ' ' < "/proc/$gp/cmdline" 2>/dev/null | grep -q "run_name=$run "; then
        visible=$((visible + 1)); break
      fi
    done
  done
  echo $((n + ours - visible))
}

log "runner start (pid $$), cards $CARDS"
while true; do
  pending=0
  while IFS=$'\t' read -r card run seed req over; do
    [ -z "${run:-}" ] && continue
    if [ -f "$Q7_ROOT/ckpt/$run/DONE" ]; then
      if [ -f "$Q7_ROOT/ckpt/$run/resume.pt" ] && ! alive "$run"; then rm -f "$Q7_ROOT/ckpt/$run/resume.pt"; log "cleaned resume.pt of $run"; fi
      continue
    fi
    pending=$((pending + 1))
    alive "$run" && continue
    if [ "$req" != "-" ]; then
      miss=0; IFS=',' read -ra reqs <<< "$req"
      for rf in "${reqs[@]}"; do [ -f "$rf" ] || miss=1; done
      [ "$miss" = 1 ] && continue
    fi
    nfail=$(grep -c "DONE=no" "$HOME/q7/logs/$run.log" 2>/dev/null)
    if [ "${nfail:-0}" -ge 3 ]; then
      [ -f "$PIDS/$run.failed" ] || { log "SKIP $run: exited without DONE $nfail times (see logs/$run.log)"; touch "$PIDS/$run.failed"; }
      continue
    fi
    # pick a card: the manifest card if it is one of ours, else (or for "A") the least loaded of ours
    target=""; best=9
    cands=$CARDS; case " $CARDS " in *" $card "*) cands=$card ;; esac
    for c in $cands; do
      np=$(card_procs "$c")
      if [ "$np" -lt 2 ] && [ "$np" -lt "$best" ]; then best=$np; target=$c; fi
    done
    [ -z "$target" ] && continue
    script=q7_train.sh; [ "${run#eval_}" != "$run" ] && script=q7_evalrun.sh
    # shellcheck disable=SC2086
    if [ "$script" = q7_evalrun.sh ]; then set -- $over; else set -- "$seed" $over; fi
    Q7_RUN=$run CUDA_VISIBLE_DEVICES=$target setsid nohup bash "$HOME/q7/$script" "$run" "$@" \
      >> "$HOME/q7/logs/$run.log" 2>&1 < /dev/null 9>&- &
    echo $! > "$PIDS/$run.pid"
    echo "$target" > "$PIDS/$run.card"
    log "launched $run seed=$seed card=$target pid=$! (card had $best procs) overrides: $over"
    sleep 2 9>&-
  done < <(grep -v '^#' "$MANIFEST")
  if [ "$pending" -eq 0 ]; then log "all manifest runs DONE; runner exits"; exit 0; fi
  free=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "${free:-99}" -lt 3 ] && log "WARNING: only ${free} GB free on /"
  sleep 60 9>&-
done
