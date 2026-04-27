"""
main.py — Pipeline Orchestrator (v2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Runs: OCREngine → NLPEngine → Annotator

Usage examples:
    python main.py essay.pdf
    python main.py essay.pdf --output corrected.pdf
    python main.py essay.pdf --no-gpu --save-intermediate
    python main.py essay.pdf --model phi3 --ollama-url http://localhost:11434/api/generate
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ocr_engine import OCREngine
from nlp_engine  import NLPEngine
from annotator   import Annotator
from live_camera import LiveModeController
from camera    import camera_selector

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AI English Essay Corrector — hybrid OCR + Phi-3 grammar analysis"
    )
    p.add_argument("input_pdf",
                   help="Path to the essay PDF (digital or scanned/handwritten)")
    p.add_argument("--output",  "-o",
                   help="Output annotated PDF (default: <input>_annotated.pdf)")
    p.add_argument("--no-gpu",  action="store_true",
                   help="Disable GPU; force CPU for EasyOCR")
    p.add_argument("--save-intermediate", action="store_true",
                   help="Also save _ocr.json and _errors.json alongside output")
    p.add_argument("--model",   default="phi3",
                   help="Ollama model name (default: phi3)")
    p.add_argument("--ollama-url", default="http://localhost:11434/api/generate",
                   help="Ollama endpoint URL")
    p.add_argument("--live", action="store_true",
                   help="Enable live camera mode for real-time essay analysis")
    p.add_argument("--camera", type=int, default=0,
                   help="Camera device index for live mode (default: 0)")
    return p.parse_args()


def run_pipeline(args: argparse.Namespace) -> str:
    t0 = time.time()
    inp = Path(args.input_pdf)
    if not inp.exists():
        logger.error(f"File not found: {inp}")
        sys.exit(1)

    out_pdf = args.output or str(inp.with_name(inp.stem + "_annotated.pdf"))

    # ── Step 1: OCR ───────────────────────────────────────────────────────────
    logger.info("━━━ Step 1/3 — OCR ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  Digital PDF → PyMuPDF  |  Scanned/Handwritten → EasyOCR")
    ocr = OCREngine(use_gpu=not args.no_gpu)
    ocr_words = ocr.process_pdf(str(inp))
    total_words = sum(len(v) for v in ocr_words.values())
    logger.info(
        f"OCR done: {total_words} words  |  {len(ocr_words)} page(s)  "
        f"[{time.time()-t0:.1f}s]"
    )

    if total_words == 0:
        logger.error(
            "OCR found 0 words. Check that the PDF is readable and not corrupted."
        )
        sys.exit(1)

    # ── Step 2: NLP ───────────────────────────────────────────────────────────
    logger.info("━━━ Step 2/3 — Grammar Analysis (Phi-3) ━━━━━━━━━━━━━━━━━━━━")
    nlp    = NLPEngine(model=args.model, ollama_url=args.ollama_url)
    errors = nlp.analyse(ocr_words)
    logger.info(
        f"Analysis done: {len(errors)} error(s)  [{time.time()-t0:.1f}s]"
    )

    if not errors:
        logger.warning(
            "No errors returned by Phi-3.  Possible causes:\n"
            "  • Ollama not running  →  run:  ollama serve\n"
            "  • Model not pulled   →  run:  ollama pull phi3\n"
            "  • OCR text garbled   →  check _ocr.json with --save-intermediate\n"
            "Annotated PDF will still be saved with legend only."
        )
    else:
        for i, e in enumerate(errors, 1):
            logger.info(
                f"  {i:3d}. [{e['type']:8s}] "
                f"{e['original']!r:35s} → {e['suggestion']!r}"
            )

    # ── Step 3: Annotate ──────────────────────────────────────────────────────
    logger.info("━━━ Step 3/3 — Annotating PDF ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    ann = Annotator()
    result = ann.annotate_pdf(str(inp), ocr_words, errors, out_pdf)
    logger.info(
        f"Annotated PDF saved → {result}  [{time.time()-t0:.1f}s]"
    )

    # ── Optional: save intermediate files ─────────────────────────────────────
    if args.save_intermediate:
        base = inp.with_suffix("")
        ocr_path = str(base) + "_ocr.json"
        err_path = str(base) + "_errors.json"
        with open(ocr_path, "w", encoding="utf-8") as f:
            json.dump(ocr_words, f, ensure_ascii=False, indent=2)
        with open(err_path, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        logger.info(f"OCR data   → {ocr_path}")
        logger.info(f"Error list → {err_path}")

    elapsed = time.time() - t0
    logger.info(f"━━━ Pipeline complete in {elapsed:.1f}s ━━━━━━━━━━━━━━━━━━━")
    return result


if __name__ == "__main__":
    args = parse_args()
    if args.live:
        run_live_mode(args)
    else:
        run_pipeline(args)


def run_live_mode(args: argparse.Namespace) -> None:
    """Run live camera mode for real-time essay analysis."""
    t0 = time.time()
    logger.info("━━━ Live Mode — Camera Detection ━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    selectors = camera_selector.CameraSelector.enumerate_devices()
    if not selectors:
        logger.error("No cameras found. Connect a camera and try again.")
        sys.exit(1)

    logger.info(f"Found {len(selectors)} camera(s):")
    for d in selectors:
        logger.info(f"  [{d['id']}] {d['name']} - {d['resolution']}")

    cam_id = args.camera
    if cam_id >= len(selectors):
        logger.warning(f"Camera {cam_id} not available, using camera 0")
        cam_id = 0

    logger.info(f"Starting live analysis on camera {cam_id}...")
    nlp = NLPEngine(model=args.model, ollama_url=args.ollama_url)

    def on_results(ocr_words, errors):
        n = sum(len(v) for v in ocr_words.values())
        logger.info(f"  → {n} words, {len(errors)} errors")
        for i, e in enumerate(errors[:3], 1):
            logger.info(f"    {i}. [{e['type']:8s}] {e['original']!r} → {e['suggestion']!r}")

    controller = LiveModeController(use_gpu=not args.no_gpu)
    try:
        controller.start(camera_id=cam_id, callback=on_results, nlp_engine=nlp)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    finally:
        elapsed = time.time() - t0
        logger.info(f"━━━ Live mode ended in {elapsed:.1f}s ━━━━━━━━━━━━━━━━━━━")
