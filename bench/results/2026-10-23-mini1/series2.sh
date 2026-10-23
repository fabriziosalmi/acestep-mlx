#!/bin/zsh
cd ~/acestep-f0
export F0_SLIM=1 F0_MLX_BF16=1 F0_REPLAY=$HOME/acestep-f0/fixed.json
run() { echo "[$(date +%T)] start $1" >> series2.log; F0_TAG=$1 zsh ./run_f0.sh $1; echo "[$(date +%T)] end $1 $(grep ^EXIT f0_$1.log)" >> series2.log; sleep 20; }
F0_TORCH_BF16=1 run bt
ACESTEP_MLX_VAE_FP16=1 run bv
ACE-Step-1.5/.venv/bin/python compare.py ref1 bt bv b >> series2.log 2>&1
echo SERIES_DONE >> series2.log
