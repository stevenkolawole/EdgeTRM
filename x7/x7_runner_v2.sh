#!/bin/bash
# EdgeTRM X7 slot runner v2 (p4d).  usage: x7_runner_v2.sh SLOT CARD NEED_GB
# Same shared queue as x7_runner.sh, but a slot waits for its card BEFORE claiming a line, so a slot
# parked on a busy card never holds work another slot could run. After the card frees up it waits a
# random 0-45 s and re-checks (fewer simultaneous starts with other agents' runners on the same card).
SLOT=$1; CARD=$2; NEED=$3
X=$HOME/x7
Q=$X/queue/all.txt; D=$X/queue/done.txt; FL=$X/queue/failed.txt; LOG=$X/logs/$SLOT.log
touch $D $FL
rm -f $X/queue/claim_$SLOT
echo "$(date -u) runner(v2) $SLOT card $CARD pid $$" >> $LOG
card_ok() {
  napps=$(nvidia-smi -i $CARD --query-compute-apps=pid --format=csv,noheader | grep -c .)
  free=$(nvidia-smi -i $CARD --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$napps" -lt 2 ] && [ "$free" -gt $((NEED * 1024)) ]
}
while true; do
  until card_ok; do sleep 60; done
  sleep $((RANDOM % 45))
  card_ok || continue
  line=$(flock $X/queue/.lock bash -c "cat $D $FL $X/queue/claim_* 2>/dev/null | grep -vxF -f - $Q | grep -v '^#' | grep -v '^ *\$' | head -1 | tee $X/queue/claim_$SLOT")
  if [ -z "$line" ]; then rm -f $X/queue/claim_$SLOT; echo "$(date -u) queue empty" >> $LOG; exit 0; fi
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
