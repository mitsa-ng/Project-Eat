"""
ocr_engine.py — Base Module (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Auto-detects page type and picks the best extraction strategy:

  ┌──────────────────┬─────────────────────────────────────────────────────┐
  │ Page type        │ Strategy                                            │
  ├──────────────────┼─────────────────────────────────────────────────────┤
  │ Digital PDF      │ PyMuPDF get_text("words") — instant, pixel-perfect  │
  │ Scanned / image  │ EasyOCR  (GPU if CUDA available, else CPU)          │
  │ Handwritten      │ EasyOCR  (same path — handles printed handwriting)  │
  └──────────────────┴─────────────────────────────────────────────────────┘

Decision rule: if PyMuPDF finds ≥ DIGITAL_THRESHOLD words on a page the page
is treated as digital.  Otherwise the page is rendered to a NumPy image and
fed to EasyOCR.

Output format (identical regardless of source):
    {
      0: [{"text": "word", "box": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]}, ...],
      1: [...],
    }
All coordinates are in PDF point-space (origin = top-left, 1 pt = 1/72 inch).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import fitz  # PyMuPDF ≥ 1.23
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── GPU device detection (shared with live_camera) ───────────────────────────

def detect_easyocr_device(use_gpu: bool = True):
    """
    Pick the best device flag for easyocr.Reader(gpu=...).

    Returns True (CUDA), 'mps' (Apple Silicon), or False (CPU).
    easyocr accepts a device string, so 'mps' is passed through to torch.
    """
    if not use_gpu:
        return False
    try:
        import torch
    except ImportError:
        logger.warning("  PyTorch not found — EasyOCR will run on CPU.")
        return False
    if torch.cuda.is_available():
        logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
        return True
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        logger.info("  Apple Silicon MPS device available")
        return "mps"
    logger.warning("  No CUDA/MPS device — EasyOCR will run on CPU.")
    return False


def load_easyocr_reader(use_gpu: bool = True):
    """
    Create an easyocr.Reader on the best available device.
    Falls back to CPU if GPU/MPS initialisation fails.
    """
    import easyocr

    device = detect_easyocr_device(use_gpu)
    if device:
        try:
            reader = easyocr.Reader(['en'], gpu=device)
            logger.info(f"EasyOCR loaded  → {'GPU' if device is True else device.upper()}")
            return reader
        except Exception as exc:
            logger.warning(f"  EasyOCR failed on {device!r} ({exc}) — falling back to CPU")
    reader = easyocr.Reader(['en'], gpu=False)
    logger.info("EasyOCR loaded  → CPU")
    return reader


# ── lazy spell-check loader ─────────────────────────────────────────────────────
_SPELL_CHECKER = None

def _get_spell_checker():
    """Lazy singleton for SpellChecker (pyspellchecker)."""
    global _SPELL_CHECKER
    if _SPELL_CHECKER is None:
        try:
            from spellchecker import SpellChecker
            _SPELL_CHECKER = SpellChecker()
            logger.info("SpellChecker loaded")
        except ImportError:
            logger.warning("pyspellchecker not installed — spell-check disabled")
            _SPELL_CHECKER = False
    return _SPELL_CHECKER


def _spell_correct_word(word: str) -> str:
    """
    If *word* is a known English word → return as-is.
    Otherwise return the closest correction if edit-distance ≤ 2,
    or an empty string (meaning the token is garbage / unreadable OCR).
    When spell-checker is unavailable, returns the word unchanged.
    """
    spell = _get_spell_checker()
    if spell is False:
        return word
    if word in spell:
        return word
    candidates = spell.candidates(word)
    if not candidates:
        return ""
    corrected = spell.correction(word)
    if corrected and corrected != word and len(corrected) >= 2:
        return corrected
    return ""

# ── tunables ──────────────────────────────────────────────────────────────────
DIGITAL_THRESHOLD = 15      # native word count ≥ this → treat page as digital
DIGITAL_DPI       = 150     # render resolution for digital pages
SCAN_DPI          = 300     # render resolution for scanned / handwritten pages
USE_GPU           = True    # attempt CUDA; falls back gracefully to CPU
EASYOCR_TEXT_THRESHOLD = 0.5
EASYOCR_CANVAS_SIZE    = 3200
CONFIDENCE_THRESHOLD   = 0.4


class OCREngine:
    """
    Hybrid text extractor.

    EasyOCR is imported lazily — digital-only PDFs never pay the
    torch / CUDA startup cost.

    Usage:
        engine    = OCREngine()
        ocr_words = engine.process_pdf("essay.pdf")
        # ocr_words[0] → [{text, box}, ...]  for page 0
    """

    def __init__(self, use_gpu: bool = USE_GPU, spell_correct: bool = True):
        self.use_gpu       = use_gpu
        self.spell_correct = spell_correct
        self._reader = None   # EasyOCR reader, loaded on first scanned/HW page
        logger.info("OCREngine ready  (auto-detect: digital=PyMuPDF | scan/HW=EasyOCR)")

    # ── lazy EasyOCR loader ───────────────────────────────────────────────────

    def _get_reader(self):
        """Load EasyOCR once, on the best available device (CUDA/MPS/CPU)."""
        if self._reader is None:
            self._reader = load_easyocr_reader(self.use_gpu)
        return self._reader

    # ── digital extraction (PyMuPDF) ──────────────────────────────────────────

    @staticmethod
    def _extract_digital(page: fitz.Page) -> list:
        """
        PyMuPDF word-level extraction.

        page.get_text("words") yields tuples:
            (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        We convert the axis-aligned rect to a 4-corner polygon so the
        output schema matches EasyOCR's [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].
        """
        words = []
        for entry in page.get_text("words"):
            x0, y0, x1, y1 = entry[0], entry[1], entry[2], entry[3]
            text = entry[4].strip()
            if not text:
                continue
            box = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            words.append({"text": text, "box": box})
        return words

    # ── scanned / handwritten extraction (EasyOCR) ───────────────────────────

    def _extract_easyocr(self, page: fitz.Page, dpi: int = SCAN_DPI) -> list:
        """
        Render page → preprocess → EasyOCR → scale boxes to PDF point-space.
        """
        zoom   = dpi / 72
        mat    = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=mat, alpha=False)
        img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr    = np.array(img)

        import cv2

        gray    = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        arr     = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 31, 2
        )

        h_img, w_img = arr.shape[:2] if arr.ndim == 3 else (arr.shape[0], arr.shape[1])
        scale_x      = page.rect.width  / w_img
        scale_y      = page.rect.height / h_img

        reader = self._get_reader()
        try:
            results = reader.readtext(
                arr, detail=1, paragraph=False,
                text_threshold=EASYOCR_TEXT_THRESHOLD,
                canvas_size=EASYOCR_CANVAS_SIZE,
            )
        except RuntimeError as exc:
            # MPS ops occasionally fail at inference time — retry once on CPU
            logger.warning(f"  EasyOCR inference failed ({exc}) — retrying on CPU")
            self._reader = load_easyocr_reader(use_gpu=False)
            results = self._reader.readtext(
                arr, detail=1, paragraph=False,
                text_threshold=EASYOCR_TEXT_THRESHOLD,
                canvas_size=EASYOCR_CANVAS_SIZE,
            )

        words = []
        for (raw_box, text, conf) in results:
            text = text.strip()
            if not text or conf < CONFIDENCE_THRESHOLD:
                continue
            box = [[pt[0] * scale_x, pt[1] * scale_y] for pt in raw_box]
            words.append({"text": text, "box": box})

        # spell-check post-processing — filter / correct OCR garbage.
        # Optional: correcting here also erases genuine spelling errors the
        # LLM is supposed to flag, so callers can disable it (spell_correct=False).
        if not self.spell_correct:
            logger.debug(
                f"  EasyOCR: {len(words)} words (spell-correction disabled, "
                f"conf ≥ {CONFIDENCE_THRESHOLD})"
            )
            return words

        cleaned = []
        for w in words:
            corrected = _spell_correct_word(w["text"])
            if corrected:
                w["text"] = corrected
                cleaned.append(w)

        logger.debug(
            f"  EasyOCR: {len(cleaned)} words after spell-check "
            f"(was {len(words)}, conf ≥ {CONFIDENCE_THRESHOLD})"
        )
        return cleaned

    # ── per-page dispatch ─────────────────────────────────────────────────────

    def process_page(self, page: fitz.Page) -> list:
        """
        Auto-detect page type and extract words.

        Steps:
          1. Run PyMuPDF native extraction.
          2. If ≥ DIGITAL_THRESHOLD words found → digital mode (return now).
          3. Otherwise → EasyOCR (scanned or handwritten).

        Heuristic: pages with DIGITAL_THRESHOLD ≤ count < 2× threshold whose
        native words are mostly short / numeric (page numbers, headers) are
        re-routed to EasyOCR even if the raw count passes the threshold.
        """
        native = self._extract_digital(page)

        if len(native) >= DIGITAL_THRESHOLD:
            # guard against handwritten pages with embedded page furniture
            if len(native) < DIGITAL_THRESHOLD * 2:
                short = sum(1 for w in native if len(w["text"]) <= 3 or w["text"].isdigit())
                ratio = short / max(len(native), 1)
                if ratio > 0.6:
                    logger.info(
                        f"  Page {page.number + 1}: LOW-QUALITY DIGITAL "
                        f"({len(native)} words, {ratio:.0%} short/numeric → "
                        f"falling back to EasyOCR @ {SCAN_DPI} DPI)"
                    )
                    return self._extract_easyocr(page, dpi=SCAN_DPI)

            logger.info(
                f"  Page {page.number + 1}: DIGITAL  "
                f"({len(native)} words via PyMuPDF)"
            )
            return native

        logger.info(
            f"  Page {page.number + 1}: SCAN/HANDWRITTEN  "
            f"(only {len(native)} native words → EasyOCR @ {SCAN_DPI} DPI)"
        )
        return self._extract_easyocr(page, dpi=SCAN_DPI)

    # ── public API ────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str) -> dict:
        """
        Process all pages of a PDF.

        Returns:
            { page_index: [{"text": str, "box": [[x1,y1],...]}, ...] }
        """
        doc    = fitz.open(pdf_path)
        output = {}

        for page_num in range(len(doc)):
            logger.info(f"Processing page {page_num + 1}/{len(doc)} …")
            output[page_num] = self.process_page(doc[page_num])

        doc.close()
        total = sum(len(v) for v in output.values())
        logger.info(f"OCR complete: {total} words across {len(output)} page(s)")
        return output


# ── CLI smoke-test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s  %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <essay.pdf> [--json]")
        sys.exit(1)

    engine  = OCREngine()
    results = engine.process_pdf(sys.argv[1])

    for pg, words in results.items():
        print(f"\n=== Page {pg}  ({len(words)} words) ===")
        for w in words[:15]:
            print(f"  {w['text']!r:30s}  box_tl={w['box'][0]}")

    if "--json" in sys.argv:
        out = sys.argv[1].replace(".pdf", "_ocr.json")
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nSaved → {out}")
