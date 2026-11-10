#!/bin/bash
# Assemble the Windows x64 distribution (CLI + GUI + runtime DLLs + models).
# Cross-builds from macOS via mingw-w64. Expects /tmp/pdfium.dll and
# /tmp/ort-win/.../onnxruntime.dll fetched beforehand (see comments below).
#   curl -sL -o /tmp/pdfium-win.tgz https://github.com/bblanchon/pdfium-binaries/releases/latest/download/pdfium-win-x64.tgz
#   curl -sL -o /tmp/ort-win.zip https://github.com/microsoft/onnxruntime/releases/download/v1.31.0/onnxruntime-win-x64-1.31.0.zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${1:-dist/win}"
OUT="${OUT%/}"
APP="$OUT/Project-EAT-win"

cargo build --release --target x86_64-pc-windows-gnu -p eat-cli -p eat-gui

rm -rf "$APP"
mkdir -p "$APP/models" "$APP/fonts"
cp target/x86_64-pc-windows-gnu/release/eat.exe     "$APP/"
cp target/x86_64-pc-windows-gnu/release/eat-gui.exe "$APP/Project-EAT.exe"
cp /tmp/pdfium.dll                                  "$APP/"
find /tmp/ort-win -name onnxruntime.dll -exec cp {} "$APP/" \;
cp vendor/ppocr/*.onnx vendor/ppocr/rec_keys.txt    "$APP/models/"
cp vendor/fonts/NotoSansCJKtc-Regular.otf           "$APP/fonts/"

( cd "$(dirname "$APP")" && zip -qr Project-EAT-win.zip "$(basename "$APP")" )
echo "Bundle: $APP"
du -sh "$APP" "$(dirname "$APP")/Project-EAT-win.zip"
