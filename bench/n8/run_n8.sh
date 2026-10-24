#!/bin/zsh
# N8 on the Mac mini: the app's own engine, with the development install out of reach.
# 1. ~/acestep-f0 is renamed (and renamed back on any exit), so nothing can come from it.
# 2. AceStepE2E drives the app's runtime twice: the first run downloads the models into
#    an empty ~/Library/Application Support/ACE-Step, the second must not download.
# 3. Every second, lsof lists the files the engine process has open.
# Logs go to ~/n8 (bench/results/<date>-mini1-n8 once copied).
set -euo pipefail
APP=$HOME/acestep-mlx/app/build/ACE-Step.app
RUNTIME=$APP/Contents/Resources/runtime
SUPPORT="$HOME/Library/Application Support/ACE-Step"
OUT=$HOME/n8
mkdir -p "$OUT"
if [[ -e $SUPPORT ]]; then echo "$SUPPORT exists; N8 needs it empty"; exit 1; fi
if pgrep -f acestep_mlx.engine_service >/dev/null; then echo "an engine is running"; exit 1; fi

(cd "$HOME/acestep-mlx/app" && swift build --product AceStepE2E 2>&1 | tail -1)
E2E=$HOME/acestep-mlx/app/.build/debug/AceStepE2E
{ df -k "$HOME" | tail -1; du -sk "$HOME/.cache/huggingface" 2>/dev/null || echo "no hf cache"; } > "$OUT/before.txt"

mv "$HOME/acestep-f0" "$HOME/acestep-f0.n8off"
sampler=""
cleanup() {
  [[ -n $sampler ]] && kill "$sampler" 2>/dev/null || true
  mv "$HOME/acestep-f0.n8off" "$HOME/acestep-f0"
}
trap cleanup EXIT
trap "exit 130" INT TERM

(
  while true; do
    for pid in $(pgrep -f acestep_mlx.engine_service); do
      print "T $(date +%s) $pid"
      lsof -p "$pid" -Fn 2>/dev/null | sed -n 's/^n//p' | sort -u
    done
    sleep 1
  done
) > "$OUT/open_files.log" 2>&1 &
sampler=$!

for run in 1 2; do
  echo "=== run $run $(date +%s)" | tee -a "$OUT/e2e.log"
  ACESTEP_RUNTIME=$RUNTIME "$E2E" >> "$OUT/e2e.log" 2>&1 || echo "E2E run $run exit $?" | tee -a "$OUT/e2e.log"
done
{ df -k "$HOME" | tail -1; du -sk "$HOME/.cache/huggingface" 2>/dev/null || echo "no hf cache"; } > "$OUT/after.txt"
cp "$HOME/Library/Logs/ACE-Step/engine.log" "$OUT/engine.log"
echo N8_DONE
