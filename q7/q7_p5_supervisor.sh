#!/bin/bash
# X5 on p5 (H200 card 7): run the listed Q7/M6 fine-tunes in parallel, each resumable, <= 3 failures each.
#   usage: q7_p5_supervisor.sh   (reads q7_p5_manifest.tsv: run<TAB>seed<TAB>overrides)
set -f
exec 7>"/opt/dlami/nvme/edgetrm/results/x5/p5sup.lock"
flock -n 7 || { echo "$(date -u) supervisor already active"; exit 0; }
R=/opt/dlami/nvme/edgetrm
export Q7_ROOT=$R/results/x5 Q7_DATA=$R/results/x5/data/sudoku-extreme-1k-aug-1000 Q7_PY=$R/venv/bin/python \
       Q7_SRC=$R/EdgeTRM/TinyRecursiveModels HF_HOME=$R/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=7
mkdir -p $Q7_ROOT/logs $Q7_ROOT/ckpt
log() { echo "$(date -u '+%F %T') $*"; }
runone() {  # run seed overrides...
  local run=$1 seed=$2; shift 2
  while [ ! -f "$Q7_ROOT/ckpt/$run/DONE" ]; do
    n=$(grep -c "DONE=no" "$Q7_ROOT/logs/$run.log" 2>/dev/null)
    [ "${n:-0}" -ge 3 ] && { log "SKIP $run after $n failures"; return; }
    log "start $run"
    Q7_RUN=$run bash $R/EdgeTRM/q7/q7_train2.sh "$run" "$seed" "$@" >> "$Q7_ROOT/logs/$run.log" 2>&1 < /dev/null 7>&-
    log "$run exited rc=$?"
    sleep 20 7>&-
  done
  log "$run DONE"
}
while IFS=$'\t' read -r run seed over; do
  [ -z "${run:-}" ] && continue; case "$run" in \#*) continue;; esac
  # shellcheck disable=SC2086
  runone "$run" "$seed" $over &
  sleep 30 7>&-
done < $R/EdgeTRM/q7/q7_p5_manifest.tsv
wait
log "all p5 runs finished"
