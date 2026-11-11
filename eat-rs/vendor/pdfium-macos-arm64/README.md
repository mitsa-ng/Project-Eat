# Vendored pdfium (macOS arm64)

Prebuilt binary from https://github.com/bblanchon/pdfium-binaries (latest release,
asset `pdfium-mac-arm64.tgz`). Loaded at runtime via `Pdfium::bind_to_library`;
path resolution order: `$EAT_PDFIUM_LIB` → beside the executable (installed-app
layout) → this vendored path (dev). Other platforms: download the matching
asset and adjust `EatPdf::bind()` candidates.
