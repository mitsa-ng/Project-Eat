## Context

Project-EAT is a command-line and GUI application for correcting English essays. It currently processes static PDF files through a pipeline that extracts text, runs NLP analysis with an Ollama-hosted Phi-3 model, and produces annotated PDFs. The goal of this change is to allow users to capture essays directly from a camera, stream the document to the same analysis pipeline, and provide real‑time feedback.

## Architecture Overview

The live mode feature extends the existing Project‑EAT pipeline to support real‑time camera input while reusing the established OCR and NLP components. This document describes the high‑level architecture, component responsibilities, and data flow for the live mode workflow.

## Component Architecture

### New Components

#### `LiveModeController` (live_mode.py)
- Central orchestrator for live mode workflow
- Manages camera lifecycle (start/stop/capture)
- Coordinates document detection, text extraction, and NLP analysis
- Provides streaming interface for real‑time results

#### `CameraSelector` (camera_selector.py)
- Enumerates available video capture devices using OpenCV (VideoCapture API)
- Presents structured device metadata (device_id, name, resolution, fps)
- Handles camera initialization and configuration (resolution, exposure, focus)
- Implements safe camera release on shutdown

#### `DocumentDetector` (integrated into LiveModeController)
- Uses OpenCV computer vision to detect document boundaries in real‑time video frames
- Applies edge detection, perspective transform, and contour analysis
- Returns warped document region for OCR processing

#### `LiveTextProcessor` (integrated into LiveModeController)
- Processes detected document region frame‑by‑frame
- Extracts text using existing EasyOCR pipeline
- Buffers multiple frames for improved OCR accuracy
- Passes extracted text to NLP engine

### Modified Components

#### `ocr_engine.py`
- Extended with `extract_from_frame(frame)` method that accepts OpenCV frame
- Supports region‑of‑interest (ROI) extraction for detected document area
- Maintains existing PDF/image processing for backward compatibility

#### `gui.py`
- Added live mode toggle button in main interface
- Camera device selection dropdown populated from `CameraSelector`
- Real‑time video preview panel with document detection overlay
- Results panel displaying streaming analysis output
- Mode switcher preserving existing PDF analysis functionality

#### `main.py`
- Added `--live` CLI flag to enable live mode from command line
- Mode detection logic routes to LiveModeController or PDFProcessor
- Shared configuration and state management

#### `requirements.txt`
- Added `opencv-python-headless` as dependency for camera/document detection
- Existing `pymupdf`, `easyocr`, and Ollama dependencies retained

## Data Flow

1. **User Input**: User selects live mode (CLI flag or GUI button) 
2. **Device Enumeration**: CameraSelector lists available devices 
3. **Camera Selection**: User picks input device 
4. **Frame Capture**: Live video stream initiated 
5. **Document Detection**: OpenCV detects document boundaries in each frame 
6. **Text Extraction**: OCR processes detected document region 
7. **Analysis**: NLP engine analyzes extracted text using Phi‑3 
8. **Results Output**: Streaming results displayed in GUI or CLI

## Constraints and Assumptions

- Requires active Ollama server running Phi‑3 model (as with existing NLP pipeline) 
- Camera access requires appropriate system permissions 
- Real‑time performance depends on document complexity and hardware 
- Live mode does not generate PDF output files (analysis‑only mode) 
- All existing functionality (PDF mode) remains unchanged and backward compatible
