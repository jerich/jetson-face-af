"""Threaded frame grabber using OpenCV V4L2 backend."""

import threading
import cv2
import numpy as np

import config


class FrameGrabber:
    """Continuously captures frames from a V4L2 camera in a background thread.

    The main thread can call get_frame() to retrieve the most recent frame
    without blocking on camera I/O.
    """

    def __init__(self, device: str = None, width: int = None, height: int = None):
        self.device = device or config.CAM_DEV
        self.width = width or config.CAPTURE_WIDTH
        self.height = height or config.CAPTURE_HEIGHT

        self._frame = None
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._cap = None

    def start(self):
        """Open the camera and start the capture thread."""
        dev = int(self.device) if self.device.isdigit() else self.device
        self._cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, config.CAPTURE_FPS)

        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self.device}")

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera opened: {self.device} @ {actual_w}x{actual_h}")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Continuously read frames and store the latest one."""
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                continue
            with self._lock:
                self._frame = frame

    def get_frame(self) -> np.ndarray | None:
        """Return the most recent frame, or None if no frame is available yet."""
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        """Stop the capture thread and release the camera."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
