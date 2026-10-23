#!/bin/zsh
# Builds a release binary and wraps it into build/ACE-Step.app, signed ad hoc.
# The engine is not bundled yet: the app finds the development install (EngineConfig).
set -euo pipefail
cd "${0:A:h}/.."
swift build -c release --product AceStep
APP=build/ACE-Step.app
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$(swift build -c release --show-bin-path)/AceStep" "$APP/Contents/MacOS/AceStep"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>ACE-Step</string>
  <key>CFBundleDisplayName</key><string>ACE-Step</string>
  <key>CFBundleIdentifier</key><string>com.fabriziosalmi.acestep</string>
  <key>CFBundleExecutable</key><string>AceStep</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST
codesign --force --sign - "$APP"
echo "$APP"
