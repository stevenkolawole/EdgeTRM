#!/bin/bash
# EdgeTRM X7 slot runner (p4d).  usage: x7_runner.sh SLOT CARD NEED_GB
# All slots pull from ONE shared queue ($X/queue/all.txt, one x7_ingolfsson.py argument line per job,
# in priority order), under flock: a slot claims the first line that is not done, failed, or claimed by
# another slot. The queue is re-read before every job, so lines can be appended while it runs.
# Before each job it waits until the card has fewer than 2 compute processes and NEED_GB free.
# x7_ingolfsson.py resumes from its JSON (cells already written are skipped), so a restart after the
# 00:05 UTC reboot loses at most one cell group per job. A line that fails twice goes to failed.txt.
SLOT=$1; CARD=$2; NEED=$3
X=$HOME/x7
Q=$X/queue/all.txt; D=$X/queue/done.txt; FL=$X/queue/failed.txt; LOG=$X/logs/$SLOT.log
touch $D $FL
rm -f $X/queue/claim_$SLOT          # this slot's previous claim (if any) is resumable by anyone
echo "$(date -u) runner $SLOT card $CARD pid $$" >> $LOG
while true; do
  line=$(flock $X/queue/.lock bash -c "cat $D $FL $X/queue/claim_* 2>/dev/null | grep -vxF -f - $Q | grep -v '^#' | grep -v '^ *\$' | head -1 | tee $X/queue/claim_$SLOT")
  if [ -z "$line" ]; then rm -f $X/queue/claim_$SLOT; echo "$(date -u) queue empty" >> $LOG; exit 0; fi
  while true; do
    napps=$(nvidia-smi -i $CARD --query-compute-apps=pid --format=csv,noheader | grep -c .)
    free=$(nvidia-smi -i $CARD --query-gpu=memory.free --format=csv,noheader,nounits)
    if [ "$napps" -lt 2 ] && [ "$free" -gt $((NEED * 1024)) ]; then break; fi
    sleep 60
  done
  echo "$(date -u) START card $CARD $line" >> $LOG
  cd $HOME/edgetrm_src && CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$CARD \
    HF_HOME=$HOME/hf_cache HF_HUB_OFFLINE=1 \
    $HOME/venvs/kvcache/bin/python x7_ingolfsson.py $line >> $LOG 2>&1
  rc=$?
  echo "$(date -u) END rc=$rc $line" >> $LOG
  if [ $rc -eq 0 ]; then
    echo "$line" >> $D
  else
    nfail=$(grep -cxF "$line" $X/queue/failcount 2>/dev/null)
    echo "$line" >> $X/queue/failcount
    if [ "$nfail" -ge 1 ]; then echo "$line" >> $FL; fi
    sleep 120
  fi
  rm -f $X/queue/claim_$SLOT
done
