#!/bin/zsh
# Runs the N4 duration/LM series on the Mac mini: one process per song under
# /usr/bin/time -l, with a 1 s sampler of swap and the process's phys_footprint.
cd ~/acestep-f0
PY=ACE-Step-1.5/.venv/bin/python
SCRIPT=~/acestep-mlx/bench/n4/run_song.py
LOG=n4_series.log
run() {  # tag lm seconds [prompt]
  local tag=$1
  /usr/bin/time -l $PY $SCRIPT $2 $3 $tag "${@:4}" > n4_$tag.log 2>&1 &
  local tpid=$!
  sleep 0.5
  local pid=$(pgrep -P $tpid | head -1)
  ( while kill -0 $tpid 2>/dev/null; do
      fp=$(footprint -p $pid 2>/dev/null | awk '/phys_footprint:/{print $2$3; exit}')
      echo "$(date +%T) $(sysctl -n vm.swapusage | awk '{print $6}') ${fp:-na}"
      sleep 1
    done ) > n4_sampler_$tag.log 2>&1 &
  wait $tpid
  echo "$tag exit $? peak $(grep 'peak memory footprint' n4_$tag.log | awk '{print $1}') swap $(head -1 n4_sampler_$tag.log | awk '{print $2}')->$(tail -1 n4_sampler_$tag.log | awk '{print $2}') $(grep -o '^N4 .*' n4_$tag.log)" >> $LOG
  sleep 20
}
rm -f $LOG
run s06_030 acestep-5Hz-lm-0.6B 30
run s06_060 acestep-5Hz-lm-0.6B 60
run s06_120 acestep-5Hz-lm-0.6B 120
run s06_240 acestep-5Hz-lm-0.6B 240
run v06_240 acestep-5Hz-lm-0.6B 240 "dark techno, 140 bpm in A minor, with a female voice singing"
run s17_030 acestep-5Hz-lm-1.7B 30
run s17_240 acestep-5Hz-lm-1.7B 240
echo SERIES_DONE >> $LOG
