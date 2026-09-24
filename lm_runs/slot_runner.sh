#!/bin/bash
# EdgeTRM X4 slot runner (p4d).  usage: slot_runner.sh SLOT CARD NEED_GB
# Runs the lines of queue/SLOT.txt (each = looped_decode.py arguments) one after the
# other on card CARD, skipping lines listed in queue/SLOT.done or queue/SLOT.failed.
# The queue is re-read before every job, so lines can be appended while it runs.
# Before each job it waits until the card has fewer than 2 compute processes and at
# least NEED_GB free.  looped_decode.py resumes at batch granularity, so a restart
# (e.g. after the 00:05 UTC reboot) loses at most one batch per job.
SLOT=$1; CARD=$2; NEED=$3
B=$HOME/breadth/lm
Q=$B/queue/$SLOT.txt; D=$B/queue/$SLOT.done; FL=$B/queue/$SLOT.failed; LOG=$B/logs/$SLOT.log
touch $D $FL
echo "$(date -u) runner $SLOT card $CARD pid $$" >> $LOG
while true; do
  line=$(cat $D $FL | grep -vxF -f - $Q | grep -v '^#' | grep -v '^ *$' | head -1)
  if [ -z "$line" ]; then echo "$(date -u) queue empty" >> $LOG; exit 0; fi
  while true; do
    napps=$(nvidia-smi -i $CARD --query-compute-apps=pid --format=csv,noheader | grep -c .)
    free=$(nvidia-smi -i $CARD --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$napps" -lt 2 ] && [ "$free" -gt $((NEED * 1024)) ]; then break; fi
    sleep 60
  done
  echo "$(date -u) START $line" >> $LOG
  cd $B && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=$HOME/pylibs/tf4554 HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 \
    $HOME/venvs/kvcache/bin/python looped_decode.py $line >> $LOG 2>&1
  rc=$?
  echo "$(date -u) END rc=$rc $line" >> $LOG
  if [ $rc -eq 0 ]; then
    echo "$line" >> $D
  else
    nfail=$(grep -cxF "$line" $B/queue/$SLOT.failcount 2>/dev/null)
    echo "$line" >> $B/queue/$SLOT.failcount
    if [ "$nfail" -ge 1 ]; then echo "$line" >> $FL; fi
    sleep 120
  fi
done
