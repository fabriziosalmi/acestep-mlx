#!/bin/zsh
# Builds build/ACE-Step.app with the engine inside, signed ad hoc.
#
# Contents/Resources/runtime holds:
#   python/    a standalone CPython (uv-managed python-build-standalone) with upstream's
#              dependencies installed from its uv.lock, hashes required, minus scripts/prune.txt;
#   upstream/  upstream's acestep package and LICENSE, from git archive of UPSTREAM_COMMIT;
#   engine/    this repo's engine.
# Model weights are not bundled: the engine downloads them on the first start into
# ~/Library/Application Support/ACE-Step/checkpoints (engine/acestep_mlx/models.py).
# Needs uv and a git checkout of upstream (UPSTREAM, default ~/acestep-f0/ACE-Step-1.5).
set -euo pipefail
cd "${0:A:h}/.."
UPSTREAM=${UPSTREAM:-$HOME/acestep-f0/ACE-Step-1.5}
UPSTREAM_COMMIT=ca1e85f
PYTHON_VERSION=3.12.13

swift build -c release --product AceStep

# The runtime is built once per set of inputs and cached under build/; the engine is
# copied fresh every time.
key=$( { echo "$PYTHON_VERSION"; git -C "$UPSTREAM" rev-parse "$UPSTREAM_COMMIT^{commit}";
         git -C "$UPSTREAM" show "${UPSTREAM_COMMIT}:uv.lock"; cat "$0" scripts/prune.txt; } | shasum -a 256 | cut -c1-12)
RUNTIME=build/runtime-$key
if [[ ! -f $RUNTIME/.complete ]]; then
  rm -rf build/runtime-*(N)
  mkdir -p "$RUNTIME/upstream"
  src=$(mktemp -d)
  git -C "$UPSTREAM" archive "$UPSTREAM_COMMIT" | tar -x -C "$src"
  base=$(uv python find --managed-python "$PYTHON_VERSION")
  cp -R "${base:A:h:h}" "$RUNTIME/python"
  # The copy is this app's own interpreter, not a shared one.
  rm -f "$RUNTIME/python/lib/python${PYTHON_VERSION%.*}/EXTERNALLY-MANAGED"
  # Local packages cannot be hash-checked; at this commit the only one is nano-vllm, which
  # upstream's lock excludes on macOS arm64.
  (cd "$src" && uv export --frozen --no-emit-project --no-emit-local --format requirements.txt \
     -o requirements.txt >/dev/null)
  uv pip install --python "$RUNTIME/python/bin/python3" --no-deps --require-hashes -r "$src/requirements.txt"
  # What no engine path imports (scripts/prune.txt, measured by bench/n11).
  uv pip uninstall --python "$RUNTIME/python/bin/python3" $(grep -v '^#' scripts/prune.txt)
  cp -R "$src/acestep" "$src/LICENSE" "$RUNTIME/upstream/"
  rm -rf "$src"
  touch "$RUNTIME/.complete"
fi

APP=build/ACE-Step.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/runtime/engine"
cp "$(swift build -c release --show-bin-path)/AceStep" "$APP/Contents/MacOS/AceStep"
# Drawn by scripts/make_icon.swift.
cp Resources/AppIcon.icns "$APP/Contents/Resources/AppIcon.icns"
# APFS clones: instant, and no extra space until a file changes.
cp -cR "$RUNTIME/python" "$RUNTIME/upstream" "$APP/Contents/Resources/runtime/"
rsync -a --exclude __pycache__ ../engine/acestep_mlx "$APP/Contents/Resources/runtime/engine/"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>ACE-Step</string>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleLocalizations</key><array><string>en</string></array>
  <key>CFBundleDisplayName</key><string>ACE-Step</string>
  <key>CFBundleIdentifier</key><string>com.fabriziosalmi.acestep</string>
  <key>CFBundleExecutable</key><string>AceStep</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.3.0</string>
  <key>CFBundleVersion</key><string>3</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
codesign --force --sign - "$APP"
codesign --verify --strict "$APP"
du -sh "$APP"
echo "$APP"
