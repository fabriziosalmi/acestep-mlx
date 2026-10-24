#!/bin/zsh
# Signs "ACE-Step MLX.app" with a Developer ID, notarizes it with Apple, staples the ticket and
# zips the result for distribution as build/ACE-Step-MLX-<version>-mac.zip.
# Usage: scripts/notarize.sh [path/to/ACE-Step MLX.app]
# Needs the Developer ID Application certificate in the login keychain (SIGN_ID overrides the one
# found) and a notarytool keychain profile (NOTARY_PROFILE, default fab-notary, made once with
# `xcrun notarytool store-credentials`). Runs where the certificate is; builds with bundle.sh.
set -euo pipefail
cd "${0:A:h}/.."
APP=${1:-"build/ACE-Step MLX.app"}
PROFILE=${NOTARY_PROFILE:-fab-notary}
ID=${SIGN_ID:-$(security find-identity -v -p codesigning | awk -F'"' '/Developer ID Application/ {print $2; exit}')}
if [[ -z $ID ]]; then echo "no Developer ID Application identity in the keychain"; exit 1; fi
VERSION=$(/usr/libexec/PlistBuddy -c "Print :CFBundleShortVersionString" "$APP/Contents/Info.plist")
echo "signing $APP $VERSION as $ID"

# Every Mach-O file inside the bundle (by its magic bytes, not its name), deepest first; the
# app's own executable is signed with the bundle at the end.
list=$(mktemp)
python3 - "$APP" > "$list" <<'PY'
import os, struct, sys
app = sys.argv[1]
main = os.path.join(app, "Contents", "MacOS")
found = []
for root, _dirs, names in os.walk(os.path.join(app, "Contents")):
    if root == main:
        continue
    for name in names:
        path = os.path.join(root, name)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        with open(path, "rb") as fh:
            head = fh.read(8)
        if len(head) < 8:
            continue
        magic, count = struct.unpack(">II", head)
        thin = magic in (0xFEEDFACF, 0xCFFAEDFE, 0xFEEDFACE, 0xCEFAEDFE)
        fat = magic in (0xCAFEBABE, 0xCAFEBABF) and count < 20  # a Java class has a version here
        if thin or fat:
            found.append(path)
found.sort(key=lambda p: (-p.count(os.sep), p))
sys.stdout.write("\0".join(found))
PY
count=$(tr -cd '\0' < "$list" | wc -c | tr -d ' ')
echo "Mach-O files inside: $((count + 1))"
sign() { codesign --force --timestamp --options runtime --sign "$ID" "$@"; }
# In parallel: each call fetches its own timestamp from Apple.
xargs -0 -n 1 -P 8 codesign --force --timestamp --options runtime --sign "$ID" < "$list"
sign "$APP"
codesign --verify --strict --deep "$APP"
unsigned=$(xargs -0 -n 1 codesign -dv < "$list" 2>&1 | grep -c "^TeamIdentifier=not set" || true)
if [[ $unsigned != 0 ]]; then echo "$unsigned files are not signed with a team"; exit 1; fi
rm -f "$list"

zip=build/notarize-upload.zip
rm -f "$zip"
ditto -c -k --keepParent "$APP" "$zip"
echo "submitting $(du -h "$zip" | cut -f1) to Apple"
result=$(xcrun notarytool submit "$zip" --keychain-profile "$PROFILE" --wait --output-format json)
echo "$result"
verdict=$(print -r -- "$result" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
submission=$(print -r -- "$result" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
xcrun notarytool log "$submission" --keychain-profile "$PROFILE" build/notarize-log.json >/dev/null
if [[ $verdict != Accepted ]]; then echo "notarization $verdict; see build/notarize-log.json"; exit 1; fi
rm -f "$zip"

xcrun stapler staple "$APP"
xcrun stapler validate "$APP"
spctl --assess --type execute -vv "$APP"
out=build/ACE-Step-MLX-$VERSION-mac.zip
rm -f "$out"
ditto -c -k --keepParent "$APP" "$out"
echo "$out $(du -h "$out" | cut -f1)"
