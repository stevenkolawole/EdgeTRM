#!/bin/bash
# babel_fill: keep our Babel general (8 GPUs, QOS normal) and preempt (24) allocations full from a backlog of
# sbatch scripts. Runs on the LOCAL machine and reuses the ssh ControlMaster (one short ssh per cycle; Babel admins
# flagged bursts of logins, so this polls every 5 minutes).
#
# Backlog: ~/autofill/babel_backlog/{general,preempt}/<prio>_<owner>_<tag>.sbatch on Babel ($HOME is shared).
# Each script must carry its own #SBATCH lines: --constraint=L40S, the node exclusions, --gres=gpu:1, a CUDA
# preflight, output to $HOME, and --requeue for preempt. Submitted scripts move to ~/autofill/babel_submitted/.
set -u
INTERVAL=${INTERVAL:-300}
LOCAL_STATUS=${LOCAL_STATUS:-/home/stevenkolawole/workspace/emnlp-26/reports/edgetrm_exp/autofill_babel_status.txt}
while true; do
  timeout 120 ssh babel 'bash -s' > "$LOCAL_STATUS.tmp" 2>&1 <<'EOF'
set -u
B=$HOME/autofill; mkdir -p $B/babel_backlog/general $B/babel_backlog/preempt $B/babel_submitted
declare -A CAP=([general]=8 [preempt]=24)
echo "# $(date -u +%FT%TZ) babel_fill"
for part in general preempt; do
  # count GPUs held or requested (running + pending) in this partition
  # (user-held jobs, e.g. the paused EpiKV array, do not count)
  n=$(squeue -u $USER -h -p $part -t R,PD,CF -o "%b|%r" | grep -v 'JobHeldUser' | cut -d'|' -f1 | grep -o 'gpu[^,]*' | awk -F: '{s+=($NF~/^[0-9]+$/?$NF:1)} END{print s+0}')
  run=$(squeue -u $USER -h -p $part -t R -o %i | wc -l)
  echo "$part held_or_pending=$n running=$run cap=${CAP[$part]} backlog=$(ls $B/babel_backlog/$part | wc -l)"
  while [ "$n" -lt "${CAP[$part]}" ]; do
    f=$(ls $B/babel_backlog/$part/*.sbatch 2>/dev/null | sort | head -1)
    [ -z "$f" ] && { echo "$part EMPTY_BACKLOG"; break; }
    if out=$(sbatch --partition=$part "$f" 2>&1); then
      echo "$part SUBMITTED $(basename $f): $out"; mv "$f" $B/babel_submitted/
      n=$((n+1))
    else
      echo "$part FAILED $(basename $f): $out"; mv "$f" "$f.failed"; break
    fi
  done
done
EOF
  mv "$LOCAL_STATUS.tmp" "$LOCAL_STATUS"
  sleep "$INTERVAL"
done
