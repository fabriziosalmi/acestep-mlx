#!/bin/zsh
# F1b series: gen + replays with frozen inputs, one at a time.
cd ~/acestep-f0
export F0_SLIM=1
run() { echo "[$(date +%T)] start $1" >> series.log; F0_TAG=$1 zsh ./run_f0.sh $1; echo "[$(date +%T)] end $1 $(grep ^EXIT f0_$1.log)" >> series.log; sleep 20; }
F0_MLX_BF16=1 F0_SAVE_FIXED=$HOME/acestep-f0/fixed.json run gen
test -s fixed.json || { echo "no fixed.json, abort" >> series.log; exit 1; }
F0_REPLAY=$HOME/acestep-f0/fixed.json run ref1
F0_REPLAY=$HOME/acestep-f0/fixed.json run ref2
F0_REPLAY=$HOME/acestep-f0/fixed.json F0_MLX_BF16=1 run a
F0_REPLAY=$HOME/acestep-f0/fixed.json F0_MLX_BF16=1 F0_TORCH_BF16=1 ACESTEP_MLX_VAE_FP16=1 run b
F0_REPLAY=$HOME/acestep-f0/fixed.json F0_MLX_BF16=1 F0_COMPACT=1 run c
ACE-Step-1.5/.venv/bin/python compare.py ref1 ref2 a b c >> series.log 2>&1
echo SERIES_DONE >> series.log
