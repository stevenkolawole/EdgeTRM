#!/bin/bash
# Emits one line whenever the fill state gets worse: an idle card with no fitting backlog job, a short backlog,
# an empty Babel backlog, or a dead/unreachable fill daemon. Used by the coordinator's Monitor.
prev=""
while true; do
  out=""
  p4=$(timeout 40 ssh sk-p4d 'pgrep -f 'autofill.sh$' >/dev/null || echo DAEMON_DEAD; cat ~/autofill/status.txt' 2>/dev/null || echo UNREACHABLE)
  p5=$(timeout 40 ssh uk-p5 'pgrep -f 'autofill.sh$' >/dev/null || echo DAEMON_DEAD; cat /opt/dlami/nvme/edgetrm/autofill/status.txt' 2>/dev/null || echo UNREACHABLE)
  bb=$(cat /home/stevenkolawole/workspace/emnlp-26/reports/edgetrm_exp/autofill_babel_status.txt 2>/dev/null)
  pgrep -f babel_fill.sh >/dev/null || out="$out babel_fill:DAEMON_DEAD;"
  for pair in "p4d:$p4" "p5:$p5"; do
    name=${pair%%:*}; s=${pair#*:}
    bad=$(grep -oE 'DAEMON_DEAD|UNREACHABLE' <<<"$s" | head -1)
    [ -n "$bad" ] && out="$out $name:$bad;"
    idle=$(awk '$6=="IDLE_NO_FITTING_JOB"{printf "%s ", $1}' <<<"$s")
    [ -n "$idle" ] && out="$out $name:idle_cards[$idle];"
    bl=$(awk '/backlog_lines/{print $3}' <<<"$s")
    [ -n "$bl" ] && [ "$bl" -lt 4 ] && out="$out $name:backlog=$bl;"
  done
  grep -q EMPTY_BACKLOG <<<"$bb" && out="$out babel:$(grep EMPTY_BACKLOG <<<"$bb" | awk '{print $1}' | tr '\n' ',')empty;"
  if [ -n "$out" ] && [ "$out" != "$prev" ]; then echo "$(date -u +%H:%M) ALERT$out"; fi
  prev="$out"
  sleep 180
done
