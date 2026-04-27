"""
camera_selector.py — Camera Device Selection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enumerates available camera devices, handles selection.

Usage:
    devices = CameraSelector.enumerate_devices()
    for d in devices:
        print(f"{d['id']}: {d['name']} ({d['resolution']})")

    selector = CameraSelector()
    cam = selector.select(d['id'])
    # cam is an OpenCV VideoCapture object
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class CameraSelector:
    """
    Device enumeration and selection for camera input.

    Uses OpenCV's VideoCapture API to list available cameras.
    """

    MAX_CAMERAS = 10

    @classmethod
    def enumerate_devices(cls) -> List[Dict]:
        """
        Enumerate available video capture devices.

        Returns:
            List of device dicts: [{"id": int, "name": str, "resolution": str}]
        """
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "opencv-python-headless is required for live mode. "
                "Install: pip install opencv-python-headless"
            )

        devices = []
        for idx in range(cls.MAX_CAMERAS):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                cap.release()
                break

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            devices.append({
                "id": idx,
                "name": f"Camera {idx}",
                "resolution": f"{width}x{height}",
                "fps": fps
            })

            cap.release()
            logger.debug(f"Found camera {idx}: {width}x{height}")

        logger.info(f"Enumerated {len(devices)} camera device(s)")
        return devices

    def select(self, device_id: int = 0):
        """
        Open and return a camera device.

        Args:
            device_id: Camera index to open

        Returns:
            OpenCV VideoCapture object

        Raises:
            RuntimeError: If camera cannot be opened
        """
        import cv2

        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        logger.info(f"Opened camera {device_id}")
        return cap

    def release(self, cap):
        """Safely release a camera capture object."""
        if cap is not None and cap.isOpened():
            cap.release()
            logger.info("Camera released")


def select_camera_interactive() -> int:
    """
    Interactive camera selection - prompts user.

    Returns:
        Selected camera device ID
    """
    devices = CameraSelector.enumerate_devices()
    if not devices:
        raise RuntimeError("No cameras found")

    print("\nAvailable cameras:")
    for d in devices:
        print(f"  [{d['id']}] {d['name']} - {d['resolution']} @ {d['fps']}fps")

    while True:
        try:
            choice = input("\nSelect camera [0]: ").strip()
            if not choice:
                return 0
            idx = int(choice)
            if any(d['id'] == idx for d in devices):
                return idx
            print(f"Invalid choice. Select 0-{devices[-1]['id']}")
        except ValueError:
            print("Enter a number")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    devices = CameraSelector.enumerate_devices()
    print(f"\nFound {len(devices)} camera(s):")
    for d in devices:
        print(f"  {d['id']}: {d['name']} - {d['resolution']}")