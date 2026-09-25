#!/bin/bash
# Emits one line whenever the fill state gets worse: an idle card with no fitting backlog job, a short backlog,
# an empty Babel backlog, or a dead/unreachable fill daemon. Used by the coordinator's Monitor.
prev=""
while true; do
  out=""
  p4=$(timeout 40 ssh sk-p4d 'pgrep -f 'autofill.sh$' >/dev/null || echo DAEMON_DEAD; cat ~/autofill/status.txt' 2>/dev/null || echo UNREACHABLE)
  if [ -n "${SKIP_P5:-}" ]; then p5="p5 skipped (released to Steven)"; else
  p5=$(timeout 40 ssh uk-p5 'pgrep -f 'autofill.sh$' >/dev/null || echo DAEMON_DEAD; cat /opt/dlami/nvme/edgetrm/autofill/status.txt' 2>/dev/null || echo UNREACHABLE); fi
  bb=$(cat /home/stevenkolawole/workspace/emnlp-26/reports/edgetrm_exp/autofill_babel_status.txt 2>/dev/null)
  pgrep -f babel_fill.sh >/dev/null || out="$out babel_fill:DAEMON_DEAD;"
  for pair in "p4d:$p4" "p5:$p5"; do
    name=${pair%%:*}; s=${pair#*:}
    bad=$(grep -oE 'DAEMON_DEAD|UNREACHABLE' <<<"$s" | head -1)
    [ -n "$bad" ] && out="$out $name:$bad;"
    idle=$(awk '$6=="IDLE_NO_FITTING_JOB" && ($5==0 || $3<10){printf "%s ", $1}' <<<"$s")
    [ -n "$idle" ] && out="$out $name:idle_cards[$idle];"
    bl=$(awk '/backlog_lines/{print $3}' <<<"$s")
    [ -n "$bl" ] && [ "$bl" -lt 4 ] && out="$out $name:backlog=$bl;"
  done
  grep -q HOME_QUOTA_LOW <<<"$bb" && out="$out babel:HOME_QUOTA_LOW;"
  # an empty Babel backlog matters only if that partition is below its cap (held/pending + running)
  for part in general preempt; do
    if grep -q "^$part EMPTY_BACKLOG" <<<"$bb"; then
      hp=$(grep -oE "^$part held_or_pending=[0-9]+" <<<"$bb" | cut -d= -f2); cap=$(grep -oE "^$part .*cap=[0-9]+" <<<"$bb" | grep -oE "cap=[0-9]+" | cut -d= -f2)
      [ -n "$hp" ] && [ -n "$cap" ] && [ "$hp" -lt "$cap" ] && out="$out babel:$part:empty_backlog_below_cap($hp/$cap);"
    fi
  done
  if [ -n "$out" ] && [ "$out" != "$prev" ]; then echo "$(date -u +%H:%M) ALERT$out"; fi
  prev="$out"
  sleep 180
done
