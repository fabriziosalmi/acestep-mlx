#!/bin/zsh
cd ~/acestep-f0
export F0_ENGINE=lean F0_REPLAY=$HOME/acestep-f0/fixed.json
run() { rm -rf out/$1; F0_TAG=$1 zsh ./run_f0.sh $1; echo "$1 $(grep ^EXIT f0_$1.log) peak $(grep "peak memory footprint" f0_$1.log | awk "{print \$1}") md5 $(md5 -q out/$1/*.flac) maxfp $(awk "{print \$4}" sampler_$1.log | sort -h | tail -1)" >> series4.log; grep "\[lean\]" f0_$1.log | cut -c 90- >> series4.log; sleep 15; }
run inst
ACESTEP_MLX_CLEAR_BEFORE_VAE=1 run clr
echo SERIES_DONE >> series4.log
