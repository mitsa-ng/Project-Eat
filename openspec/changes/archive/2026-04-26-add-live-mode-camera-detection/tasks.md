## 1. Setup

- [x] 1.1 Create a new directory `live_mode` and add `live_mode.py`
- [x] 1.2 Create a new directory `camera` and add `camera_selector.py`
- [x] 1.3 Add `opencv-python-headless` dependency to `requirements.txt`

## 2. Design Integration

- [x] 2.1 Update `main.py` to add `--live` flag and route to `LiveModeController`
- [x] 2.2 Update `gui.py` to add a "Live Mode" button and camera selector dropdown
- [x] 2.3 Reference design.md while implementing

## 3. Core Implementation

- [x] 3.1 Implement `CameraSelector.enumerate_devices()` to list devices
- [x] 3.2 Implement `LiveModeController.start_camera()` to open selected device
- [x] 3.3 Implement `LiveModeController.capture_loop()` to read frames and pass to `DocumentDetector`
- [x] 3.4 Implement `DocumentDetector.detect_document(frame)` returning ROI
- [x] 3.5 Extend `ocr_engine.extract_from_frame(frame)` to accept ROI and use EasyOCR
- [x] 3.6 Connect OCR output to the existing NLP pipeline
- [x] 3.7 Stream analysis results to GUI/CLI

## 4. Testing & Validation

- [x] 4.1 Write unit tests for `CameraSelector`
- [x] 4.2 Write integration test for `DocumentDetector` with sample frames
- [x] 4.3 Write integration test to ensure full live mode passes a sample document

## 5. Documentation

- [x] 5.1 Update README to include live mode usage
- [x] 5.2 Document required hardware permissions
