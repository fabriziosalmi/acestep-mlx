#!/bin/zsh
cd ~/acestep-f0
export F0_ENGINE=lean F0_REPLAY=$HOME/acestep-f0/fixed.json ACESTEP_LEAN_MEMLOG=1
run() { rm -rf out/$1; F0_TAG=$1 zsh ./run_f0.sh $1; echo "$1 $(grep ^EXIT f0_$1.log) peak $(grep "peak memory footprint" f0_$1.log | awk "{print \$1}") md5 $(md5 -q out/$1/*.flac) maxfp $(awk "{print \$4}" sampler_$1.log | sort -h | tail -1) tiles $(grep -c "VAE tile" f0_$1.log) maxtile $(grep -o "MLX peak [0-9]*" f0_$1.log | awk "{print \$3}" | sort -n | tail -1) $(grep -o "audio in [0-9.]*s" f0_$1.log | tail -1)" >> series6.log; sleep 15; }
ACESTEP_LEAN_VAE_CHUNK=256 ACESTEP_LEAN_VAE_OVERLAP=64 run t256_64
ACESTEP_LEAN_VAE_CHUNK=128 ACESTEP_LEAN_VAE_OVERLAP=32 run t128_32
ACESTEP_LEAN_VAE_CHUNK=64 ACESTEP_LEAN_VAE_OVERLAP=16 run t64_16
ACESTEP_LEAN_VAE_CHUNK=48 ACESTEP_LEAN_VAE_OVERLAP=12 run t48_12
ACE-Step-1.5/.venv/bin/python compare.py ref1 t256_64 t128_32 t64_16 t48_12 >> series6.log 2>&1
echo SERIES_DONE >> series6.log
