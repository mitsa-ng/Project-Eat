## Why

The current Project-EAT workflow only supports processing static PDF files, requiring users to scan or convert handwritten essays to PDF before analysis. A live mode with camera integration would allow users to directly capture and analyze essay manuscripts in real-time, eliminating the PDF conversion step and streamlining the correction workflow.

## What Changes

- Add a new `live-mode` feature accessible via CLI (`--live` flag) and GUI (new dedicated button)
- Implement camera device enumeration and selection UI for users to choose their input camera
- Integrate OpenCV for real-time camera frame capture and document boundary detection
- Extend the existing OCR engine to process individual camera frames and extract text from detected document regions using EasyOCR
- Reuse the existing NLP engine (Phi-3 via Ollama) to analyze extracted text in real-time
- Display live analysis results (spelling/grammar/semantic errors) in the GUI or CLI output stream

## Non-Goals

- No generation of annotated PDF files in live mode (focus is real-time text analysis, not document annotation)
- No support for video file input (only live camera streams)
- No offline mode for live detection (requires active Ollama server for NLP analysis)

## Capabilities

### New Capabilities

- `live-camera-mode`: Core pipeline for live camera operation, document detection, text extraction, and real-time analysis
- `camera-device-selection`: Logic and UI for listing available cameras, letting users select input devices, and handling camera initialization

### Modified Capabilities

(none)

## Impact

- Affected specs: `live-camera-mode`, `camera-device-selection`
- Affected code:
  - New: `live_camera/live_mode.py`, `camera/camera_selector.py`
  - Modified: `gui.py`, `ocr_engine.py`, `main.py`, `requirements.txt`
  - Removed: (none)
