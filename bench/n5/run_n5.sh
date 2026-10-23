#!/bin/zsh
# N5 clips on the Mac mini: the 0.6B series, then the 1.7B series, one process each.
# Usage: ~/acestep-mlx/bench/n5/run_n5.sh <out dir>   (logs next to it)
set -u
out=$1
mkdir -p "$out"
cd ~/acestep-f0
for lm in 0.6B 1.7B; do
  /usr/bin/time -l ACE-Step-1.5/.venv/bin/python ~/acestep-mlx/bench/n5/make_clips.py $lm "$out/$lm" \
    > "$out/n5_$lm.log" 2>&1
  echo "$lm exit $?" >> "$out/status.txt"
done
echo done >> "$out/status.txt"
