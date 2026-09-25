#!/bin/bash
# X2: move X5 mixer-seed measurement jobs into the Babel autofill backlog once their final checkpoint exists.
# Runs on the local machine (like babel_fill.sh), one ssh every 10 min through the ControlMaster; exits when done.
LOG=/home/stevenkolawole/workspace/emnlp-26/reports/edgetrm_exp/x2_seed_watch.log
while true; do
  left=$(timeout 90 ssh babel 'cd ~/x2_jobs/pending 2>/dev/null || exit 0
    for f in 20_X2_x2fit_*.sbatch; do
      [ -f "$f" ] || continue
      ck=$(grep -o "/home/skolawol/q7/ckpt/var_[a-z0-9_]*/step_65100" "$f" | head -1)
      if [ -f "$ck" ] && [ -f "$(dirname $ck)/DONE" -o $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 600 ]; then
        mv "$f" ~/autofill/babel_backlog/preempt/ && echo "QUEUED $f"
      fi
    done
    ls 20_X2_x2fit_*.sbatch 2>/dev/null | wc -l' 2>&1)
  echo "$(date -u +%FT%TZ) $left" | tr '\n' ' ' >> $LOG; echo >> $LOG
  case "$left" in *$'\n'0|0) break;; esac
  [ "$(echo "$left" | tail -1)" = "0" ] && break
  sleep 600
done
echo "$(date -u +%FT%TZ) all seed jobs queued; exiting" >> $LOG
