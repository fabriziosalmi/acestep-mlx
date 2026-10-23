#!/bin/zsh
set -euo pipefail
export PATH=$HOME/.local/bin:/opt/homebrew/bin:$PATH
cd ~/acestep-f0/ACE-Step-1.5
echo "[$(date +%T)] uv sync start"
uv sync 2>&1 | tail -5
echo "[$(date +%T)] uv sync done"
echo "[$(date +%T)] download start"
uv run hf download ACE-Step/Ace-Step1.5 --include "acestep-v15-turbo/*" "vae/*" "Qwen3-Embedding-0.6B/*" --local-dir checkpoints 2>&1 | tail -2
uv run hf download ACE-Step/acestep-5Hz-lm-0.6B --local-dir checkpoints/acestep-5Hz-lm-0.6B 2>&1 | tail -2
echo "[$(date +%T)] download done"
du -sh checkpoints/*
echo SETUP_OK
