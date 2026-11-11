#!/bin/bash
# Assemble the macOS .app bundle for Project-EAT (teacher zero-barrier install).
# Usage: scripts/make-app.sh [output-dir]   (default: dist/)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-dist}"
APP="$OUT/Project-EAT.app"

cargo build --release -p eat-gui -p eat-cli

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp target/release/eat-gui              "$APP/Contents/MacOS/Project-EAT"
cp target/release/eat                  "$APP/Contents/MacOS/"
cp vendor/pdfium-macos-arm64/lib/libpdfium.dylib "$APP/Contents/MacOS/"
mkdir -p "$APP/Contents/MacOS/models"
cp vendor/ppocr/PP-OCRv6_det_small.onnx vendor/ppocr/PP-OCRv6_rec_small.onnx vendor/ppocr/rec_keys.txt "$APP/Contents/MacOS/models/"
cp vendor/fonts/NotoSansCJKtc-Regular.otf "$APP/Contents/MacOS/fonts/NotoSansCJKtc-Regular.otf" 2>/dev/null || {
  mkdir -p "$APP/Contents/MacOS/fonts"
  cp vendor/fonts/NotoSansCJKtc-Regular.otf "$APP/Contents/MacOS/fonts/"
}

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>Project-EAT</string>
  <key>CFBundleDisplayName</key>       <string>AI 作文批改</string>
  <key>CFBundleIdentifier</key>        <string>tw.edu.project-eat</string>
  <key>CFBundleVersion</key>           <string>0.1.0</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleExecutable</key>        <string>Project-EAT</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>LSMinimumSystemVersion</key>    <string>13.0</string>
  <key>NSHighResolutionCapable</key>   <true/>
  <key>NSCameraUsageDescription</key>  <string>即時批改模式需要使用攝影機拍攝學生作文</string>
</dict>
</plist>
PLIST

# ad-hoc codesign (removes Gatekeeper "unidentified developer" friction locally;
# distribution builds should sign with a Developer ID)
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true

echo "Bundle: $APP"
du -sh "$APP"
