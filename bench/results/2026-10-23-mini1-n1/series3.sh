#!/bin/zsh
cd ~/acestep-f0
export F0_ENGINE=lean F0_REPLAY=$HOME/acestep-f0/fixed.json
run() { rm -rf out/$1; echo "[$(date +%T)] start $1" >> series3.log; F0_TAG=$1 zsh ./run_f0.sh $1; echo "[$(date +%T)] end $1 $(grep ^EXIT f0_$1.log) peak $(grep "peak memory footprint" f0_$1.log | awk "{print \$1}")" >> series3.log; sleep 15; }
ACESTEP_MLX_VAE_CHUNK=256 run vc256
ACESTEP_MLX_VAE_CHUNK=192 run vc192
ACESTEP_MLX_VAE_FP16=1 run vf16
ACE-Step-1.5/.venv/bin/python compare.py ref1 n1b vc256 vc192 vf16 >> series3.log 2>&1
for t in n1b vc256 vc192 vf16; do echo "$t md5 $(md5 -q out/$t/*.flac) maxfp $(awk "{print \$4}" sampler_$t.log | sort -h | tail -1) maxswap $(awk "{print \$2}" sampler_$t.log | sort -n | tail -1)" >> series3.log; done
echo SERIES_DONE >> series3.log
