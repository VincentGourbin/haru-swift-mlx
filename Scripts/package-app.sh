#!/usr/bin/env bash
# Packages haru-studio as a runnable "Haru Studio.app" (adapted from convertvoxtral Scripts/package-app.sh).
#
# Built with xcodebuild in Release (`swift build` does not find MLX's default.metallib). The SwiftPM
# resource bundles (MLX metallib, HaruCore cards, tokenizer data) go to Contents/Resources, where
# Bundle.module and MLX look them up. Ad hoc signature: runs on this Mac.
#
# Usage: Scripts/package-app.sh [output .app path]   (default: .build/Haru Studio.app)
set -euo pipefail
cd "$(dirname "$0")/.."
DERIVED_DATA=.xcodebuild
PRODUCTS="$DERIVED_DATA/Build/Products/Release"
APP="${1:-.build/Haru Studio.app}"

xcodebuild -scheme haru-studio -configuration Release -derivedDataPath "$DERIVED_DATA" \
    -destination 'platform=macOS' -skipMacroValidation build > .local-runs/package-app.log 2>&1 \
    || { grep -n "error:" .local-runs/package-app.log | head -20; echo "PACKAGE FAILED"; exit 1; }

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$PRODUCTS/haru-studio" "$APP/Contents/MacOS/HaruStudio"
cp Sources/HaruStudio/Info.plist "$APP/Contents/Info.plist"
for bundle in "$PRODUCTS"/*.bundle; do
    cp -R "$bundle" "$APP/Contents/Resources/"
done
printf 'APPL????' > "$APP/Contents/PkgInfo"
codesign --force --deep --sign - "$APP" 2>/dev/null
echo "PACKAGE OK $APP ($(du -sh "$APP" | cut -f1)); run: open \"$APP\""
