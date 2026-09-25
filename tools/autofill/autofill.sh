#!/bin/bash
# autofill: keep every GPU on this node busy from a shared, prioritised backlog.
#
# Decisions are gated on card occupancy (power, memory, process count), never on process names.
# Backlog line format (one job per line; '#' comments allowed):
#   <prio> <mem_GB> <owner> <tag> <command ...>
#   prio: lower runs first. mem_GB: peak GPU memory the job needs. tag: unique id (also the log name).
#   command: a self-contained, resumable shell command (cd, env, python ...). It must NOT set CUDA_VISIBLE_DEVICES.
# Agents append lines with:  flock $ROOT/backlog.lock -c 'echo "..." >> $ROOT/backlog.txt'
#
# A card is fillable when it has fewer than MAXPROC compute processes, enough free memory for the job plus a
# margin, and its power has been below POWER_LOW for two consecutive checks (so jobs still loading are not
# double-booked). After a launch the card cools down for COOLDOWN seconds.
set -u
ROOT=${AUTOFILL_ROOT:-$HOME/autofill}
INTERVAL=${INTERVAL:-90}
MAXPROC=${MAXPROC:-3}
POWER_LOW=${POWER_LOW:-250}
MARGIN_GB=${MARGIN_GB:-3}
COOLDOWN=${COOLDOWN:-300}
mkdir -p "$ROOT/logs" "$ROOT/state"
touch "$ROOT/backlog.txt" "$ROOT/launched.txt"
STATUS="$ROOT/status.txt"

RETRY_WINDOW=${RETRY_WINDOW:-900}

reap() {  # finished jobs: record exit code; requeue once if it failed within RETRY_WINDOW (e.g. an env error)
  local f tag pid start line rc
  for f in "$ROOT"/state/pid_*; do
    [ -e "$f" ] || continue
    tag=${f##*/pid_}; pid=$(cat "$f")
    kill -0 "$pid" 2>/dev/null && continue
    rc=$(grep -oE 'AUTOFILL_EXIT=[0-9]+' "$ROOT/logs/$tag.log" 2>/dev/null | tail -1 | cut -d= -f2)
    start=$(cat "$ROOT/state/start_$tag" 2>/dev/null || echo 0)
    echo "$(date -u +%FT%TZ) tag=$tag END rc=${rc:-unknown} runtime=$(( $(date +%s) - start ))s" >> "$ROOT/launched.txt"
    if [ "${rc:-1}" != "0" ] && [ $(( $(date +%s) - start )) -lt "$RETRY_WINDOW" ] && [ ! -e "$ROOT/state/retried_$tag" ] \
       && [ -s "$ROOT/state/line_$tag" ]; then
      line=$(cat "$ROOT/state/line_$tag")
      mv "$ROOT/logs/$tag.log" "$ROOT/logs/$tag.fail1.log"
      flock "$ROOT/backlog.lock" bash -c "cat '$ROOT/state/line_$tag' >> '$ROOT/backlog.txt'"
      touch "$ROOT/state/retried_$tag"
      echo "$(date -u +%FT%TZ) tag=$tag REQUEUED once (fast failure; see logs/$tag.fail1.log)" >> "$ROOT/launched.txt"
    fi
    rm -f "$f"
  done
}

while true; do
  reap
  now=$(date +%s)
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,uuid,power.draw,memory.used,memory.total --format=csv,noheader,nounits)
  procs=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)
  {
    echo "# $(date -u +%FT%TZ) autofill status (idx power_W used_GB free_GB nproc state)"
    for g in "${GPUS[@]}"; do
      IFS=', ' read -r idx uuid pw mu mt <<<"$(echo "$g" | tr -s ' ')"
      pw=${pw%.*}; used=$((mu/1024)); free=$(( (mt-mu)/1024 ))
      np=$(grep -c "$uuid" <<<"$procs" || true)
      st="busy"
      low="$ROOT/state/low_$idx"; cool="$ROOT/state/cool_$idx"
      if [ "$pw" -lt "$POWER_LOW" ]; then
        if [ -f "$low" ]; then lowcount=2; else lowcount=1; touch "$low"; fi
      else
        rm -f "$low"; lowcount=0
      fi
      cooling=0
      if [ -f "$cool" ] && [ $(( now - $(cat "$cool") )) -lt "$COOLDOWN" ]; then cooling=1; fi
      if [ "$np" -lt "$MAXPROC" ] && [ "$lowcount" -ge 2 ] && [ "$cooling" -eq 0 ]; then
        st="fillable"
        line=$(flock "$ROOT/backlog.lock" awk -v free="$free" -v m="$MARGIN_GB" '
          /^[[:space:]]*#/ || NF < 5 {next}
          ($2 + m) <= free { if (best == "" || $1 < bp) {best=$0; bp=$1} }
          END {print best}' "$ROOT/backlog.txt")
        if [ -n "$line" ]; then
          read -r prio mem owner tag cmd <<<"$line"
          flock "$ROOT/backlog.lock" bash -c "grep -vxF -- \"\$1\" '$ROOT/backlog.txt' > '$ROOT/backlog.tmp'; mv '$ROOT/backlog.tmp' '$ROOT/backlog.txt'" _ "$line"
          echo "$now" > "$cool"; rm -f "$low"
          printf '%s\n' "$line" > "$ROOT/state/line_$tag"; echo "$now" > "$ROOT/state/start_$tag"
          ( export CUDA_VISIBLE_DEVICES=$idx; nohup bash -c "$cmd; echo AUTOFILL_EXIT=\$?" > "$ROOT/logs/$tag.log" 2>&1 & echo $! > "$ROOT/state/pid_$tag" )
          echo "$(date -u +%FT%TZ) card=$idx owner=$owner tag=$tag prio=$prio mem=$mem pid=$(cat "$ROOT/state/pid_$tag")" >> "$ROOT/launched.txt"
          st="LAUNCHED:$tag"
        else
          st="IDLE_NO_FITTING_JOB"
        fi
      fi
      echo "$idx $pw $used $free $np $st"
    done
    echo "# backlog_lines $(grep -cv '^[[:space:]]*#' "$ROOT/backlog.txt" || true)"
  } > "$STATUS.tmp" && mv "$STATUS.tmp" "$STATUS"
  sleep "$INTERVAL"
done
