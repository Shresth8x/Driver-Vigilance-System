from __future__ import annotations

from typing import Callable, Optional
import logging
import platform
import threading
import time

from .config import AlertConfig
from .models import AlertLevel


LOGGER = logging.getLogger(__name__)


class AlertController:
    def __init__(self, config: AlertConfig) -> None:
        self.config = config
        self._last_alert_time = 0.0
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None
        self._beeper = self._select_beeper()

    def update(self, level: AlertLevel, timestamp: float) -> None:
        if not self.config.enabled or level == AlertLevel.NONE:
            return

        with self._lock:
            if timestamp - self._last_alert_time < self.config.cooldown_seconds:
                return
            if self._worker is not None and self._worker.is_alive():
                return

            self._last_alert_time = timestamp
            self._worker = threading.Thread(
                target=self._play_pattern,
                args=(level,),
                daemon=True,
            )
            self._worker.start()

    def close(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            self._worker.join(timeout=0.2)

    def _select_beeper(self) -> Callable[[int, int], None]:
        if platform.system().lower() == "windows":
            try:
                import winsound

                return winsound.Beep
            except ImportError:
                LOGGER.warning("winsound unavailable; falling back to terminal bell.")

        def terminal_bell(_: int, duration_ms: int) -> None:
            print("\a", end="", flush=True)
            time.sleep(duration_ms / 1000.0)

        return terminal_bell

    def _play_pattern(self, level: AlertLevel) -> None:
        if level == AlertLevel.SOFT:
            self._beeper(900, 160)
        elif level == AlertLevel.STRONG:
            self._beeper(1200, 280)
            time.sleep(0.08)
            self._beeper(1200, 280)
        elif level == AlertLevel.CRITICAL:
            for _ in range(3):
                self._beeper(1500, 220)
                time.sleep(0.06)
        elif level == AlertLevel.PASSENGER:
            for _ in range(4):
                self._beeper(1800, 180)
                time.sleep(0.05)
