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
declare -A QOS=([general]=normal [preempt]=preempt_qos)
echo "# $(date -u +%FT%TZ) babel_fill"
# $HOME quota guard: jobs write their output under $HOME, so a full quota makes every job fail at once
avail_kb=$(df -Pk $HOME | awk 'NR==2{print $4}')
if [ "${avail_kb:-0}" -lt 5000000 ]; then echo "HOME_QUOTA_LOW avail_kb=$avail_kb: not submitting"; exit 0; fi
for part in general preempt; do
  # count GPUs held or requested (running + pending) in this partition
  # (user-held jobs, e.g. the paused EpiKV array, do not count)
  n=$(squeue -u $USER -h -p $part -t R,PD,CF -o "%b|%r" | grep -v 'JobHeldUser' | cut -d'|' -f1 | grep -o 'gpu[^,]*' | awk -F: '{s+=($NF~/^[0-9]+$/?$NF:1)} END{print s+0}')
  run=$(squeue -u $USER -h -p $part -t R -o %i | wc -l)
  echo "$part held_or_pending=$n running=$run cap=${CAP[$part]} backlog=$(ls $B/babel_backlog/$part | wc -l)"
  while [ "$n" -lt "${CAP[$part]}" ]; do
    f=$(ls $B/babel_backlog/$part/*.sbatch 2>/dev/null | sort | head -1)
    # general has room but no general work: borrow preempt work (preempt is congested; general is not)
    if [ -z "$f" ] && [ "$part" = general ]; then
      j=$(squeue -u $USER -h -p preempt -t PD -o "%i %r" | grep -v JobHeldUser | awk '$2!="Dependency"{print $1}' | head -1)
      if [ -n "$j" ] && scontrol update jobid=$j partition=general qos=normal >/dev/null 2>&1; then
        echo "general MOVED pending preempt job $j"; n=$((n+1)); continue
      fi
      f=$(ls $B/babel_backlog/preempt/*.sbatch 2>/dev/null | sort | head -1)
    fi
    # preempt has room but no preempt work: borrow general-backlog work (submitted with --requeue so preemption resumes it)
    if [ -z "$f" ] && [ "$part" = preempt ]; then
      f=$(ls $B/babel_backlog/general/*.sbatch 2>/dev/null | sort | head -1)
      [ -n "$f" ] && ! grep -q -- "--requeue" "$f" && sed -i "2i #SBATCH --requeue" "$f"
    fi
    [ -z "$f" ] && { echo "$part EMPTY_BACKLOG"; break; }
    # PACKING: small puzzle-model jobs (M1, M2 puzzle runs, M3, X2) use a few GB of a 48 GB L40S, so run up to 3
    # of them concurrently inside one GPU allocation. LM decodes (X4, M2 m2o) and training (X5) stay one per GPU.
    bf=$(basename $f); d=$(dirname $f)
    if [[ $bf =~ ^[0-9]+_(M1|M2|M3|X2)_ ]] && [[ $bf != *m2o* ]]; then
      mem=$(ls $d/*.sbatch 2>/dev/null | sort | while read x; do b=$(basename $x); [[ $b =~ ^[0-9]+_(M1|M2|M3|X2)_ ]] && [[ $b != *m2o* ]] && echo $x; done | head -3)
      mkdir -p $B/packs $B/packlogs; pk=$B/packs/pack_$(date +%s)_$RANDOM.sbatch; ok=1; names=""
      for x in $mem; do mv "$x" $B/babel_submitted/ 2>/dev/null || { ok=0; break; }; names="$names $(basename $x)"; done
      [ "$ok" = 1 ] || { echo "$part MOVE_FAILED (pack): not submitting"; break; }
      { echo '#!/bin/bash'; echo '#SBATCH --job-name=pack'; echo '#SBATCH --requeue'; echo '#SBATCH --gres=gpu:1'
        echo '#SBATCH --constraint=L40S'; echo '#SBATCH --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20'
        echo '#SBATCH --cpus-per-task=12'; echo '#SBATCH --mem=96G'; echo '#SBATCH --time=12:00:00'
        echo "#SBATCH --output=$B/packlogs/%x_%j.out"; echo "echo members:$names"
        for n1 in $names; do echo "bash $B/babel_submitted/$n1 > $B/packlogs/${n1%.sbatch}.\$SLURM_JOB_ID.log 2>&1 &"; done
        echo 'wait'; } > $pk
      if out=$(sbatch --partition=$part --qos=${QOS[$part]} "$pk" 2>&1); then echo "$part PACKED$names: $out"; n=$((n+1)); continue
      else echo "$part PACK_FAILED$names: $out"; for n1 in $names; do mv $B/babel_submitted/$n1 $d/$n1.failed; done; break; fi
    fi
    # move first, so a failed move can never cause a duplicate submission
    g=$B/babel_submitted/$(basename $f)
    mv "$f" "$g" 2>/dev/null || { echo "$part MOVE_FAILED $(basename $f): not submitting"; break; }
    if out=$(sbatch --partition=$part --qos=${QOS[$part]} "$g" 2>&1); then
      echo "$part SUBMITTED $(basename $f): $out"
      n=$((n+1))
    else
      echo "$part FAILED $(basename $f): $out"; mv "$g" "$f.failed"; break
    fi
  done
done
EOF
  mv "$LOCAL_STATUS.tmp" "$LOCAL_STATUS"
  sleep "$INTERVAL"
done
