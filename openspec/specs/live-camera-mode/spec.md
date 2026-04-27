# Purpose
Provides real-time camera-based essay document detection and analysis capabilities, extending Project-EAT to process live handwritten essays without PDF conversion.

## ADDED

### FR1: Live mode entry points
- CLI flag `--live` added to main.py to enable live camera-based analysis
- GUI button "Live Mode" added to main interface, opens camera selector on click

### FR2: Real-time document detection
- DocumentDetector component added using OpenCV contour/edge detection
- Detects document boundaries in camera frames and returns warped ROI
- Latency target: <500ms per frame under normal lighting

### FR3: Frame-based OCR extraction
- ocr_engine.py extended with `extract_from_frame(frame)` method
- Accepts OpenCV frame, extracts text from detected document region
- Reuses EasyOCR pipeline for consistency

### FR4: Streaming NLP analysis
- Extracted text sent to Phi-3 via Ollama in real-time
- Results displayed as streaming output (GUI panel or CLI stdout)

### FR5: Camera lifecycle management
- CameraSelector enumerates devices using cv2.VideoCapture API
- LiveModeController orchestrates start/stop and cleanup
- Proper camera release and resource management

## MODIFIED

### N/A

## REMOVED

### N/A
