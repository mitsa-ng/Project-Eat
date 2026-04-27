# Project-EAT — Agent Guide

## What this project does
AI English Essay Corrector: extracts text from PDFs, uses Phi-3 LLM to detect spelling/grammar/semantic errors, and produces an annotated PDF with a sidebar showing corrections.

## Entry points
```bash
python main.py essay.pdf                    # CLI mode
python gui.py                                # GUI batch mode
```

## Critical setup

### 1. CUDA PyTorch MUST be installed BEFORE other deps
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
Without cu118 wheel, EasyOCR runs on CPU and is very slow.

### 2. Pull Phi-3 model
```bash
ollama pull phi3
ollama serve
```
Ollama must be running (`http://localhost:11434`) for NLP analysis to work.

### 3. Install other deps
```bash
pip install -r requirements.txt
```

## No tests exist
There are no test files in this repo. Run manually with sample PDFs in `pdfs/` directory.

## Architecture
- `ocr_engine.py` — text extraction (PyMuPDF for digital PDFs, EasyOCR for scanned/handwritten)
- `nlp_engine.py` — grammar analysis via Ollama Phi-3
- `annotator.py` — sidebar annotation with error highlighting
- `gui.py` — Tkinter batch processor (PDFs → ZIP)

## Common failures
- **"OCR found 0 words"** → PDF unreadable or corrupted
- **No errors found** → Ollama not running or phi3 model not pulled
- **CUDA error** → Wrong PyTorch installed; reinstall with cu118 index URL