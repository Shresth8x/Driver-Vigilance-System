from __future__ import annotations

from typing import Optional
import logging

import cv2

from .config import CameraConfig


LOGGER = logging.getLogger(__name__)


class CameraStream:
    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._capture: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self._capture = cv2.VideoCapture(self.config.index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open camera index {self.config.index}. "
                "Try running with --camera 1 or check camera permissions."
            )

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        LOGGER.info("Camera opened: index=%s", self.config.index)

    def read(self):
        if self._capture is None:
            raise RuntimeError("CameraStream.open() must be called before read().")

        ok, frame = self._capture.read()
        if not ok:
            return None

        if self.config.flip_horizontal:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            LOGGER.info("Camera released.")
