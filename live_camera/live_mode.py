"""
live_camera/live_mode.py — Live Camera Mode Controller
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Provides real-time camera-based essay document detection and analysis.

Usage:
    from live_camera import LiveModeController
    controller = LiveModeController(use_gpu=False)
    controller.start(camera_id=0, callback=process_results)

Output format (identical to OCREngine):
    { page_index: [{"text": "word", "box": [[x1,y1],...]}, ...] }

Processing flow:
  1. Capture frame from camera
  2. Detect document boundaries (edge detection + contour)
  3. Extract text via EasyOCR with ROI
  4. Return extracted text for NLP analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
import time
from typing import Callable, Optional

import numpy as np

from camera import CameraSelector

logger = logging.getLogger(__name__)


class DocumentDetector:
    """
    Detects document boundaries in camera frames using OpenCV.

    Uses edge detection and contour analysis to find the document
    region, then applies perspective transform to get a flat view.
    """

    def __init__(self):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError(
                "opencv-python-headless is required for live mode. "
                "Install: pip install opencv-python-headless"
            )
        self._initialized = False

    def _ensure_initialized(self):
        if not self._initialized:
            self.cv2.setNumThreads(4)
            self._initialized = True

    def detect_document(self, frame: np.ndarray) -> tuple:
        """
        Detect document boundaries in a frame.

        Args:
            frame: OpenCV BGR frame (H, W, 3)

        Returns:
            (warped_region, debug_overlay) where:
              - warped_region: Document ROI as RGB image, or None if not detected
              - debug_overlay: Frame with detection overlay drawn
        """
        self._ensure_initialized()
        cv2 = self.cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            logger.debug("No contours detected in frame")
            return None, frame

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        if area < 5000:
            logger.debug(f"Contour too small: {area}")
            return None, frame

        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) != 4:
            logger.debug(f"Document outline not quadrilateral: {len(approx)} points")
            return None, frame

        pts = approx.reshape(4, 2)
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = int(max(width_a, width_b))

        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = int(max(height_a, height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(rect.astype(np.float32), dst)
        warped = cv2.warpPerspective(frame, M, (max_width, max_height))

        overlay = frame.copy()
        cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 2)
        for i, pt in enumerate(approx):
            cv2.circle(overlay, tuple(pt[0]), 5, (0, 0, 255), -1)

        rgb_warped = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)
        return rgb_warped, overlay

    def _order_points(self, pts):
        """Order points: top-left, top-right, bottom-right, bottom-left."""
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        bl = pts[np.argmin(diff)]
        tr = pts[np.argmax(diff)]

        return np.array([tl, tr, br, bl], dtype=np.int32)


class LiveModeController:
    """
    Central orchestrator for live camera mode.

    Manages camera lifecycle, document detection, OCR extraction,
    and provides streaming results for NLP analysis.

    Usage:
        from live_camera import LiveModeController

        def on_results(ocr_words, analysis_errors):
            print(f"Extracted: {sum(len(v) for v in ocr_words.values())} words")
            print(f"Errors: {len(analysis_errors)}")

        controller = LiveModeController(use_gpu=False)
        controller.start(camera_id=0, callback=on_results)
    """

    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu
        self._detector = None
        self._ocr_reader = None
        self._capture = None
        self._running = False
        self._last_text = ""
        self._frame_count = 0
        logger.info("LiveModeController ready")

    def enumerate_cameras(self):
        """List available camera devices."""
        return CameraSelector.enumerate_devices()

    def _get_ocr_reader(self):
        """Lazy load EasyOCR reader."""
        if self._ocr_reader is not None:
            return self._ocr_reader

        import easyocr
        import torch

        gpu_ok = False
        if self.use_gpu:
            try:
                gpu_ok = torch.cuda.is_available()
            except Exception:
                pass

        self._ocr_reader = easyocr.Reader(['en'], gpu=gpu_ok)
        logger.info(f"EasyOCR loaded for live mode → {'GPU' if gpu_ok else 'CPU'}")
        return self._ocr_reader

    def _extract_text_from_frame(self, frame) -> dict:
        """Extract text from a document ROI frame."""
        if self._detector is None:
            self._detector = DocumentDetector()

        roi, _ = self._detector.detect_document(frame)
        if roi is None:
            return {0: []}

        reader = self._get_ocr_reader()
        results = reader.readtext(roi, detail=1, paragraph=False)

        words = []
        for (box, text, conf) in results:
            text = text.strip()
            if not text or conf < 0.35:
                continue
            box = [[pt[0], pt[1]] for pt in box]
            words.append({"text": text, "box": box})

        logger.debug(f"Extracted {len(words)} words from frame")
        return {0: words}

    def start(
        self,
        camera_id: int = 0,
        callback: Optional[Callable] = None,
        nlp_engine=None
    ):
        """
        Start live camera mode.

        Args:
            camera_id: Video device index (0 = default camera)
            callback: Optional callback(ocr_words, analysis_errors) for each frame
            nlp_engine: Optional NLPEngine instance for live analysis
        """
        import cv2

        self._capture = cv2.VideoCapture(camera_id)
        if not self._capture.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self._capture.set(cv2.CAP_PROP_FPS, 30)

        self._running = True
        logger.info(f"Camera {camera_id} opened - starting capture loop")

        while self._running:
            ret, frame = self._capture.read()
            if not ret:
                logger.error("Failed to read frame")
                break

            self._frame_count += 1

            if self._frame_count % 3 != 0:
                continue

            ocr_words = self._extract_text_from_frame(frame)

            if sum(len(v) for v in ocr_words.values()) < 3:
                continue

            analysis_errors = []
            if nlp_engine is not None:
                analysis_errors = nlp_engine.analyse(ocr_words)

            if callback:
                callback(ocr_words, analysis_errors)

        self.stop()

    def stop(self):
        """Stop camera capture and release resources."""
        self._running = False
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        logger.info("Camera stopped, resources released")

    def process_single_frame(self, frame) -> dict:
        """
        Process a single frame (for testing/integration).

        Args:
            frame: OpenCV BGR frame

        Returns:
            OCR output format: {0: [{"text": str, "box": [[x,y]...]}]}
        """
        return self._extract_text_from_frame(frame)


def start_live_mode(
    camera_id: int = 0,
    use_gpu: bool = True,
    callback: Optional[Callable] = None
):
    """
    Convenience function to start live mode.

    Args:
        camera_id: Camera device index
        use_gpu: Use GPU for EasyOCR if available
        callback: Results callback
    """
    controller = LiveModeController(use_gpu=use_gpu)
    controller.start(camera_id=camera_id, callback=callback)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python live_mode.py [--list-cameras | --capture]")
        sys.exit(1)

    if "--list-cameras" in sys.argv:
        devices = LiveModeController().enumerate_cameras()
        print(f"Available cameras: {devices}")

    elif "--capture" in sys.argv:
        print("Starting live capture (Ctrl+C to stop)...")

        def on_frame(words, errors):
            n = sum(len(v) for v in words.values())
            print(f"  Frame: {n} words, {len(errors)} errors")

        try:
            start_live_mode(camera_id=0, callback=on_frame)
        except KeyboardInterrupt:
            print("\nStopped")