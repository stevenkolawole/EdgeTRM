#!/bin/bash
# EdgeTRM X1: (re)start every p4d slot runner that is not already running. Called from cron @reboot;
# safe to call by hand. Slots: ~/x1/queue/slots.txt, lines "SLOT CARD NEED_GB".
X=$HOME/x1
while read -r slot card need; do
  [ -z "$slot" ] && continue
  case "$slot" in \#*) continue;; esac
  if pgrep -f "x1_runner.sh $slot " > /dev/null; then echo "$(date -u) $slot already running"; continue; fi
  setsid nohup bash $X/x1_runner.sh $slot $card $need > /dev/null 2>&1 < /dev/null &
  echo "$(date -u) started $slot on card $card"
  sleep 20
done < $X/queue/slots.txt
