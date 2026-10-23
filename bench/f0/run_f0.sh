#!/bin/zsh
# Runs one F0 measurement under /usr/bin/time -l with a 1 s sampler that records
# system swap, memory pressure and the Python process's phys_footprint.
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
cd ~/acestep-f0
TAG=${1:-run}
/usr/bin/time -l ACE-Step-1.5/.venv/bin/python f0_measure.py > f0_$TAG.log 2>&1 &
TIMEPID=$!
sleep 0.5
PYPID=$(pgrep -P $TIMEPID | head -1)
( while kill -0 $TIMEPID 2>/dev/null; do
    fp=$(footprint -p $PYPID 2>/dev/null | awk '/phys_footprint:/{print $2$3; exit}')
    echo "$(date +%T) $(sysctl -n vm.swapusage | awk '{print $6}') $(memory_pressure | tail -1 | awk '{print $NF}') ${fp:-na}"
    sleep 1
  done ) > sampler_$TAG.log 2>&1 &
wait $TIMEPID
echo "EXIT $?" >> f0_$TAG.log
