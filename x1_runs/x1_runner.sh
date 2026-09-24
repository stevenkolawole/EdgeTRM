#!/bin/bash
# EdgeTRM X1 slot runner (p4d; pattern of X4's lm_runs/slot_runner.sh). usage: x1_runner.sh SLOT CARD NEED_GB
# Runs the lines of ~/x1/queue/SLOT.txt ("<script> <args>", run in ~/edgetrm_src) one after the other on
# card CARD, skipping lines in SLOT.done / SLOT.failed. Before each job it waits until the card has fewer
# than 2 compute processes and NEED_GB free. The sweeps resume (per spec, and per chunk with
# --checkpoint), so a restart after the 00:05 UTC reboot loses at most one chunk per job.
SLOT=$1; CARD=$2; NEED=$3
X=$HOME/x1
Q=$X/queue/$SLOT.txt; D=$X/queue/$SLOT.done; FL=$X/queue/$SLOT.failed; LOG=$X/logs/$SLOT.log
touch $D $FL
echo "$(date -u) runner $SLOT card $CARD pid $$" >> $LOG
while true; do
  line=$(cat $D $FL | grep -vxF -f - $Q | grep -v '^#' | grep -v '^ *$' | head -1)
  if [ -z "$line" ]; then echo "$(date -u) queue empty" >> $LOG; exit 0; fi
  while true; do
    napps=$(nvidia-smi -i $CARD --query-compute-apps=pid --format=csv,noheader | grep -c .)
    free=$(nvidia-smi -i $CARD --query-gpu=memory.free --format=csv,noheader,nounits)
    # another X1 slot that started on this card in the last 3 min may not show in nvidia-smi yet
    recent=$(find $X/run -name "starting.card$CARD.*" ! -name "starting.card$CARD.$SLOT" -mmin -3 2>/dev/null | grep -c .)
    if [ "$napps" -lt 2 ] && [ "$recent" -eq 0 ] && [ "$free" -gt $((NEED * 1024)) ]; then break; fi
    sleep $((50 + RANDOM % 20))
  done
  mkdir -p $X/run && touch $X/run/starting.card$CARD.$SLOT
  echo "$(date -u) START $line" >> $LOG
  cd $HOME/edgetrm_src && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 \
    $HOME/venvs/kvcache/bin/python -c "import torch; assert torch.cuda.is_available()" && \
  cd $HOME/edgetrm_src && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 \
    $HOME/venvs/kvcache/bin/python $line >> $LOG 2>&1
  rc=$?
  echo "$(date -u) END rc=$rc $line" >> $LOG
  if [ $rc -eq 0 ]; then
    echo "$line" >> $D
  else
    nfail=$(grep -cxF "$line" $X/queue/$SLOT.failcount 2>/dev/null)
    echo "$line" >> $X/queue/$SLOT.failcount
    if [ "$nfail" -ge 1 ]; then echo "$line" >> $FL; fi
    sleep 120
  fi
done
