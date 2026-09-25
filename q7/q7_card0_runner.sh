#!/bin/bash
# Q7/X5: supervisor for the maze-MLP diagnostic on p4d card 0 (coordinator Sep 25 ~05:00 UTC: X7 holds the
# other process on card 0; at most 2 processes on the card in total, so ONE run of ours). Singleton, cron
# @reboot, resumable (pretrain_q7 resume.pt); gives up after 3 exits without DONE.
set -f
exec 8>"$HOME/q7/card0.lock"
flock -n 8 || { echo "$(date -u) card0 runner already active"; exit 0; }
source "$HOME/q7/env_p4d.sh"
RUN=mazemlp_aug_w512
D=$HOME/q7/data_maze/maze-30x30-hard-1k-aug8
ARGS="data_paths=[$D] arch.L_cycles=4 global_batch_size=128 epochs=3200 eval_interval=800 +q7_eval_n=200 arch.hidden_size=512 arch.num_heads=8"
log() { echo "$(date -u '+%F %T') $*"; }
log "card0 runner start (pid $$)"
while [ ! -f "$Q7_ROOT/ckpt/$RUN/DONE" ]; do
  nfail=$(grep -c "DONE=no" "$HOME/q7/logs/$RUN.log" 2>/dev/null)
  [ "${nfail:-0}" -ge 3 ] && { log "giving up on $RUN after $nfail failures"; exit 1; }
  n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1==0{print $2}')")
  if [ "$n" -ge 2 ]; then log "card 0 has $n processes; waiting"; sleep 300 8>&-; continue; fi
  log "launching $RUN on card 0 (card had $n procs)"
  # shellcheck disable=SC2086
  Q7_RUN=$RUN CUDA_VISIBLE_DEVICES=0 bash "$HOME/q7/q7_train2.sh" $RUN 0 $ARGS >> "$HOME/q7/logs/$RUN.log" 2>&1 < /dev/null 8>&-
  log "$RUN exited rc=$?"
  sleep 30 8>&-
done
log "$RUN DONE; card0 runner exits"
