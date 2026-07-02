<h1 align="center">
  AI English Essay Annotation Tool
</h1>

<p align="center">
<em>Project-EAT: Automated Essay Proofreading via Hybrid OCR and Large Language Models</em>
</p>

<p align="center">
<a href="#" target="_blank">
<img src="https://badgen.net/badge/python/3.10%2B/blue" alt="python" />
</a>
<a href="#" target="_blank">
<img src="https://badgen.net/badge/license/MIT/green" alt="license" />
</a>
<a href="#" target="_blank">
<img src="https://badgen.net/badge/platform/macOS%20%7C%20Windows%20%7C%20Linux/cyan" alt="platform" />
</a>
<a href="#" target="_blank">
<img src="https://badgen.net/badge/LLM/Ollama%20(phi3)/orange" alt="LLM" />
</a>
</p>

A fully automated pipeline for detecting spelling, grammar, and semantic errors in English essays. The system accepts scanned handwritten documents, digital-born PDFs, and live camera feeds, then produces annotated PDFs with a sidebar showing every correction.

---

### Highlights

🌟 **Hybrid OCR Engine** — Automatically detects page type and selects the optimal extraction strategy: PyMuPDF for digital PDFs (instant, pixel-perfect) and EasyOCR with optional CUDA acceleration for scanned or handwritten pages.

🌟 **LLM-Powered Error Detection** — Leverages a local LLM via Ollama (default: phi3, configurable with `--model` or the `OLLAMA_MODEL` env var) with a structured few-shot prompt to identify spelling, grammar, and semantic errors. Long essays are chunked to avoid context-window overflow; results are deduplicated and validated.

🌟 **Sidebar Annotation Model** — Errors are highlighted in the body text with color-coded underlines (red for spelling, orange for grammar, blue for semantic). Correction tags are placed in a reserved right-side column with leader lines, ensuring no overlap with the original essay text.

🌟 **Live Camera Mode** — Point your webcam at a printed essay for real-time OCR and error analysis. Supports OpenCV-based camera selection and live streaming feedback.

🌟 **Batch GUI Processor** — A dark-themed Tkinter desktop application that processes entire folders of PDF essays and packs all annotated outputs into a single ZIP archive, complete with per-file progress tracking and error summaries.

### System Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  OCREngine   │ ──→ │  NLPEngine   │ ──→ │  Annotator   │
│  (PyMuPDF /  │     │  (Ollama LLM │     │  (fitz shape │
│   EasyOCR)   │     │   e.g. phi3) │     │   drawing)   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
  OCR words            Error JSON            Annotated PDF
  (page → word         (spelling,            (original text +
   dicts with           grammar,              sidebar tags +
   bounding boxes)      semantic)             color legend)
```

### Installation

#### 1. Install PyTorch (GPU recommended)

**Windows/Linux with NVIDIA GPU** — install the CUDA-enabled wheel:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify CUDA is detected:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Expected: True
```

**macOS (Apple Silicon)** — the standard wheel already includes MPS support, which the OCR engine auto-detects:

```bash
pip install torch torchvision
python -c "import torch; print(torch.backends.mps.is_available())"
# Expected: True
```

#### 2. Install Ollama and Pull a Model

Download from [ollama.com](https://ollama.com), then:

```bash
ollama pull phi3
ollama serve
```

Any Ollama model works — select it with `--model` (CLI), the Model field (GUI), or the `OLLAMA_MODEL` environment variable.

#### 3. Install Dependencies

```bash
git clone <repository-url>
cd Project-EAT
pip install -r requirements.txt
```

### Usage

**CLI Mode** — process a single essay PDF:

```bash
python main.py essay.pdf
python main.py essay.pdf --output corrected.pdf --save-intermediate
python main.py essay.pdf --model phi3 --ollama-url http://localhost:11434/api/generate
python main.py essay.pdf --no-spell-correct   # keep raw OCR words so the LLM sees real misspellings
```

**Live Camera Mode** — real-time analysis via webcam:

```bash
python main.py --live --camera 0
```

**GUI Batch Mode** — process an entire folder:

```bash
python gui.py
```

Select an input folder containing PDFs, configure settings, and click **Process Folder**. All annotated PDFs are exported as a single ZIP file.

### Output

The annotated PDF preserves the original essay layout, with:

- **Highlighted error regions** with color-coded underlines
- **Sidebar correction tags** — one per error, showing the type and suggested fix
- **Color legend** at the top of the sidebar (Spelling / Grammar / Semantic)
- **Intermediate JSON files** (optional, `--save-intermediate`): `_ocr.json` (per-page word data) and `_errors.json` (full error list)

### Important Notes

1. **Ollama must be running** (`http://localhost:11434`) for NLP analysis. If the server is unreachable, the pipeline will exit with a connection error. Defaults can be overridden with the `OLLAMA_MODEL` and `OLLAMA_URL` environment variables.

2. **GPU vs CPU**: EasyOCR auto-detects CUDA (NVIDIA) or MPS (Apple Silicon) and falls back to CPU, which is substantially slower for scanned/handwritten pages.

3. **OCR quality**: For best results, provide clean scans at ≥200 DPI. Low-resolution or heavily distorted images may yield garbled text.

4. **Spell-correction trade-off**: By default, OCR output from scanned/handwritten pages is auto-corrected to remove OCR noise — which can also erase the student's genuine misspellings before analysis. Use `--no-spell-correct` (CLI) or untick the checkbox (GUI) to keep raw OCR words.

### Error Types

| Type      | Color  | Description                              |
|-----------|--------|------------------------------------------|
| Spelling  | Red    | Misspelled or non-standard words         |
| Grammar   | Orange | Subject-verb agreement, tense, plurality  |
| Semantic  | Blue   | Collocation errors, unnatural phrasing   |

### Acknowledgments

- [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) — PDF rendering, text extraction, and annotation drawing
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — GPU-accelerated OCR for scanned/handwritten text
- [Ollama](https://ollama.com/) — Local LLM inference server
- [Phi-3](https://ollama.com/library/phi3) — Default small language model for grammar analysis
- [OpenCV](https://opencv.org/) — Camera access for live mode

### Cite

If you use this software in academic work, please cite it as:

```bibtex
@software{project_eat,
  author = {Project-EAT Contributors},
  title = {AI English Essay Corrector},
  year = {2026},
  url = {https://github.com/mitsa-ng/Project-EAT}
}
```
