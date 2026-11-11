# Vendored PP-OCR models (same files the Python `rapidocr` package ships)

- `PP-OCRv6_det_small.onnx` — text detection (DBNet)
- `PP-OCRv6_rec_small.onnx` — recognition (CRNN, dictionary embedded in ONNX
  metadata; also exported to `rec_keys.txt` for the Rust loader)
- `ch_ppocr_mobile_v2.0_cls_mobile.onnx` — orientation classifier. NOT used by
  the Rust path: its behaviour regressed under ONNX Runtime ≥1.31 (flips
  horizontal handwriting upside-down; verified against constant tensors).
  eat-ocr instead retries a 180° rotation per crop when rec confidence is
  low. Kept here for reference/testing parity with Python.

Source: site-packages/rapidocr/models (rapidocr PyPI distribution).
