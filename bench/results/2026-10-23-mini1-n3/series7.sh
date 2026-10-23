#!/bin/zsh
cd ~/acestep-f0
export F0_ENGINE=lean F0_REPLAY=$HOME/acestep-f0/fixed.json ACESTEP_LEAN_MEMLOG=1
run() { rm -rf out/$1; F0_TAG=$1 zsh ./run_f0.sh $1; echo "$1 $(grep ^EXIT f0_$1.log) peak $(grep "peak memory footprint" f0_$1.log | awk "{print \$1}") md5 $(md5 -q out/$1/*.flac) maxfp $(awk "{print \$4}" sampler_$1.log | sort -h | tail -1) $(grep -o "audio in [0-9.]*s" f0_$1.log | tail -1)" >> series7.log; sleep 15; }
ACESTEP_LEAN_CONDITION=compare run n3_compare
run n3_mlx
run n3_mlx2
ACE-Step-1.5/.venv/bin/python compare.py ref1 te_mlx n3_compare n3_mlx n3_mlx2 >> series7.log 2>&1
ACE-Step-1.5/.venv/bin/python compare.py te_mlx n3_mlx >> series7.log 2>&1
echo SERIES_DONE >> series7.log
