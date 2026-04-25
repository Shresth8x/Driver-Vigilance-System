from __future__ import annotations

from pathlib import Path
from typing import Optional
import logging
import time

import cv2

from .alerts import AlertController
from .camera import CameraStream
from .config import AppConfig
from .fatigue import FatigueEngine
from .temperature import TemperatureProvider
from .ui import DashboardRenderer
from .vision import FaceMeshEyeDetector


LOGGER = logging.getLogger(__name__)


class DriverMonitoringApp:
    def __init__(
        self,
        config: AppConfig,
        *,
        temperature_celsius: Optional[float] = None,
        record_path: Optional[str] = None,
    ) -> None:
        self.config = config
        initial_temp = (
            config.temperature.default_celsius
            if temperature_celsius is None
            else temperature_celsius
        )
        self.temperature_provider = TemperatureProvider(initial_temp)
        self.camera = CameraStream(config.camera)
        self.detector = FaceMeshEyeDetector(config.tracking)
        self.engine = FatigueEngine(
            config.eye,
            config.perclos,
            config.temperature,
            config.alerts,
            config.tracking,
        )
        self.alerts = AlertController(config.alerts)
        self.renderer = DashboardRenderer(config.ui)
        self.record_path = record_path
        self._writer = None

    def run(self) -> int:
        try:
            self.camera.open()
            cv2.namedWindow(self.config.ui.window_name, cv2.WINDOW_NORMAL)

            while True:
                frame = self.camera.read()
                if frame is None:
                    LOGGER.warning("Camera returned an empty frame.")
                    break

                timestamp = time.time()
                observation = self.detector.process(frame)
                state = self.engine.update(
                    observation,
                    timestamp,
                    self.temperature_provider.current_celsius,
                )
                self.alerts.update(state.alert_level, timestamp)

                output = self.renderer.render(frame, state, observation)
                self._write_frame_if_needed(output)
                cv2.imshow(self.config.ui.window_name, output)

                key = cv2.waitKey(1) & 0xFF
                if self._handle_key(key):
                    break

            return 0
        finally:
            self.close()

    def close(self) -> None:
        self.alerts.close()
        self.detector.close()
        self.camera.release()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> bool:
        if key in (27, ord("q")):
            return True
        if key in (ord("+"), ord("=")):
            new_temp = self.temperature_provider.increase()
            LOGGER.info("Simulated temperature increased to %.1f C", new_temp)
        elif key in (ord("-"), ord("_")):
            new_temp = self.temperature_provider.decrease()
            LOGGER.info("Simulated temperature decreased to %.1f C", new_temp)
        elif key == ord("l"):
            self.renderer.toggle_landmarks()
        return False

    def _write_frame_if_needed(self, frame) -> None:
        if not self.record_path:
            return

        if self._writer is None:
            output_path = Path(self.record_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                max(1, self.config.camera.fps),
                (width, height),
            )
            LOGGER.info("Recording demo video to %s", output_path)

        self._writer.write(frame)
