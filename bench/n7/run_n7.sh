#!/bin/zsh
# N7 on the Mac mini: AceStepE2E with a 20 s idle period and 3 sleep/wake cycles, plus a 1 s
# sampler of every engine process's phys_footprint, the E2E process's RSS and free+inactive memory.
# Usage: ~/acestep-mlx/bench/n7/run_n7.sh <out dir>
out=$1
mkdir -p $out
cd ~/acestep-mlx/app
swift build -c release --product AceStepE2E > $out/build.log 2>&1 || { echo build failed; exit 1; }
E2E=$(swift build -c release --show-bin-path)/AceStepE2E
ACESTEP_IDLE_SECONDS=20 ACESTEP_E2E_CYCLES=3 $E2E > $out/e2e.log 2>&1 &
epid=$!
( while kill -0 $epid 2>/dev/null; do
    engines=""
    for pid in $(pgrep -f acestep_mlx.engine_service); do
      fp=$(footprint -p $pid 2>/dev/null | awk '/phys_footprint:/{print $2$3; exit}')
      engines="$engines $pid:${fp:-na}"
    done
    rss=$(ps -o rss= -p $epid 2>/dev/null | tr -d ' ')
    pages=$(vm_stat | awk '/Pages free|Pages inactive/{gsub(/\./,"",$3); s+=$3} END {print s}')
    echo "$(date +%s) e2e_rss_kb=$rss avail_mb=$((pages * 16 / 1024)) engines=[${engines# }]"
    sleep 1
  done ) > $out/sampler.log 2>&1 &
wait $epid
echo "e2e exit $?" >> $out/e2e.log
