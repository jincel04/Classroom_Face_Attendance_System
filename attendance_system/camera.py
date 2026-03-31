from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class WebcamStream:
    """Continuously read frames in a background thread to reduce capture latency."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ):
        self.camera_index = camera_index
        self.capture = cv2.VideoCapture(camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self._frame_lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._stopped = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "WebcamStream":
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.camera_index}")
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def _reader_loop(self) -> None:
        while not self._stopped:
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._frame_lock:
                self._frame = frame

    def read(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stop(self) -> None:
        self._stopped = True
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self.capture.release()

