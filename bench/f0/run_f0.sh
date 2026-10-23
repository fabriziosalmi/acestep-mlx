#!/bin/zsh
# Runs one F0 measurement with a 2 s system sampler and /usr/bin/time -l.
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
cd ~/acestep-f0
TAG=${1:-run}
( while true; do echo "$(date +%T) $(sysctl -n vm.swapusage | awk '{print $6}') $(memory_pressure | tail -1 | awk '{print $NF}')"; sleep 2; done ) > sampler_$TAG.log 2>&1 &
SAMPLER=$!
/usr/bin/time -l ACE-Step-1.5/.venv/bin/python f0_measure.py > f0_$TAG.log 2>&1
echo "EXIT $?" >> f0_$TAG.log
kill $SAMPLER
