#!/bin/bash
# X2 v2: when an X5 mixer seed's final checkpoint exists, (1) submit an S-only job at once (sens_fit + x2_predict:
# the prediction is written from X2's own S before any width is read), (2) move the width job into the autofill backlog.
LOG=/home/stevenkolawole/workspace/emnlp-26/reports/edgetrm_exp/x2_seed_watch.log
while true; do
  left=$(timeout 90 ssh babel 'cd ~/x2_jobs/pending 2>/dev/null || exit 0
    for f in 20_X2_x2fit_*.sbatch; do
      [ -f "$f" ] || continue
      ck=$(grep -o "/home/skolawol/q7/ckpt/var_[a-z0-9_]*/step_65100" "$f" | head -1)
      nm=$(grep -o "x2_model.sh [a-z0-9_]*" "$f" | head -1 | cut -d" " -f2)
      if [ -f "$ck" ] && [ -f "$(dirname $ck)/DONE" -o $(( $(date +%s) - $(stat -c %Y "$ck") )) -gt 600 ]; then
        id=$(sbatch --parsable --partition=preempt --qos=preempt_qos --requeue --gres=gpu:1 --constraint=L40S \
          --exclude=babel-o9-24,babel-o5-16,babel-p9-28,babel-t5-28,babel-p9-20 --cpus-per-task=4 --mem=32G --time=00:30:00 \
          --job-name=x2S_$nm --output=$HOME/breadth_results/logs/%x_%j.out \
          --wrap="source ~/miniconda3/etc/profile.d/conda.sh && conda activate kvcache && python -c \"import torch; assert torch.cuda.is_available()\" && ~/x2_jobs/x2_sonly.sh $nm $ck")
        mv "$f" ~/autofill/babel_backlog/preempt/ && echo "QUEUED $f Sjob=$id"
      fi
    done
    ls 20_X2_x2fit_*.sbatch 2>/dev/null | wc -l' 2>&1)
  echo "$(date -u +%FT%TZ) $left" | tr '\n' ' ' >> $LOG; echo >> $LOG
  [ "$(echo "$left" | tail -1)" = "0" ] && break
  sleep 600
done
echo "$(date -u +%FT%TZ) all seed jobs queued; exiting" >> $LOG
