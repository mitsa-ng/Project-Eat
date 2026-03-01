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

# ── tunables ──────────────────────────────────────────────────────────────────
DIGITAL_THRESHOLD = 15      # native word count ≥ this → treat page as digital
RENDER_DPI        = 200     # render resolution for EasyOCR path
USE_GPU           = True    # attempt CUDA; falls back gracefully to CPU


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

    def __init__(self, use_gpu: bool = USE_GPU):
        self.use_gpu = use_gpu
        self._reader = None   # EasyOCR reader, loaded on first scanned/HW page
        logger.info("OCREngine ready  (auto-detect: digital=PyMuPDF | scan/HW=EasyOCR)")

    # ── lazy EasyOCR loader ───────────────────────────────────────────────────

    def _get_reader(self):
        """
        Load EasyOCR once.

        CUDA detection order:
          1. Check torch.cuda.is_available() at runtime (respects the actual
             installed torch+CUDA wheel, not just the presence of a GPU).
          2. If unavailable → warn and fall back to CPU.

        FIX for "Neither CUDA nor MPS are available":
          Install the CUDA-enabled wheel first:
            pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
          then re-run.  No code change needed — this function auto-detects.
        """
        if self._reader is not None:
            return self._reader

        import easyocr

        gpu_ok = False
        if self.use_gpu:
            try:
                import torch
                gpu_ok = torch.cuda.is_available()
                if gpu_ok:
                    logger.info(f"  CUDA device: {torch.cuda.get_device_name(0)}")
                else:
                    logger.warning(
                        "  torch.cuda.is_available() = False.  "
                        "Install CUDA-enabled PyTorch:\n"
                        "  pip install torch torchvision "
                        "--index-url https://download.pytorch.org/whl/cu118"
                    )
            except ImportError:
                logger.warning("  PyTorch not found — EasyOCR will run on CPU.")

        self._reader = easyocr.Reader(['en'], gpu=gpu_ok)
        logger.info(f"EasyOCR loaded  → {'GPU' if gpu_ok else 'CPU'}")
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

    def _extract_easyocr(self, page: fitz.Page) -> list:
        """
        Render page → RGB image → EasyOCR → scale boxes to PDF point-space.
        """
        zoom   = RENDER_DPI / 72
        mat    = fitz.Matrix(zoom, zoom)
        pix    = page.get_pixmap(matrix=mat, alpha=False)
        img    = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        arr    = np.array(img)

        h_img, w_img = arr.shape[:2]
        scale_x      = page.rect.width  / w_img
        scale_y      = page.rect.height / h_img

        reader  = self._get_reader()
        results = reader.readtext(arr, detail=1, paragraph=False)

        words = []
        for (raw_box, text, conf) in results:
            text = text.strip()
            if not text or conf < 0.25:
                continue
            box = [[pt[0] * scale_x, pt[1] * scale_y] for pt in raw_box]
            words.append({"text": text, "box": box})

        logger.debug(f"  EasyOCR: {len(words)} words (conf ≥ 0.25)")
        return words

    # ── per-page dispatch ─────────────────────────────────────────────────────

    def process_page(self, page: fitz.Page) -> list:
        """
        Auto-detect page type and extract words.

        Steps:
          1. Run PyMuPDF native extraction.
          2. If ≥ DIGITAL_THRESHOLD words found → digital mode (return now).
          3. Otherwise → EasyOCR (scanned or handwritten).
        """
        native = self._extract_digital(page)

        if len(native) >= DIGITAL_THRESHOLD:
            logger.info(
                f"  Page {page.number + 1}: DIGITAL  "
                f"({len(native)} words via PyMuPDF)"
            )
            return native

        logger.info(
            f"  Page {page.number + 1}: SCAN/HANDWRITTEN  "
            f"(only {len(native)} native words → EasyOCR)"
        )
        return self._extract_easyocr(page)

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
