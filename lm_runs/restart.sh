#!/bin/bash
# EdgeTRM X4: (re)start every p4d slot runner that is not already running.
# Called from cron @reboot; safe to call by hand.  Slots are listed in queue/slots.txt
# as "SLOT CARD NEED_GB".
B=$HOME/breadth/lm
while read -r slot card need; do
  [ -z "$slot" ] && continue
  case "$slot" in \#*) continue;; esac
  if pgrep -f "slot_runner.sh $slot " > /dev/null; then echo "$(date -u) $slot already running"; continue; fi
  setsid nohup bash $B/slot_runner.sh $slot $card $need > /dev/null 2>&1 < /dev/null &
  echo "$(date -u) started $slot on card $card"
done < $B/queue/slots.txt
