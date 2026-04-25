from __future__ import annotations

from typing import Tuple

import cv2

from .config import UiConfig
from .models import AlertLevel, DriverState, EyeState, FaceObservation, FatigueLevel


Color = Tuple[int, int, int]


class DashboardRenderer:
    COLORS = {
        FatigueLevel.NO_FACE: (150, 150, 150),
        FatigueLevel.NORMAL: (60, 210, 80),
        FatigueLevel.MILD: (0, 220, 255),
        FatigueLevel.DROWSY: (40, 80, 255),
        FatigueLevel.CRITICAL: (30, 30, 255),
        FatigueLevel.OBSTRUCTED: (255, 180, 70),
        FatigueLevel.DISTRACTED: (255, 120, 60),
    }

    def __init__(self, config: UiConfig) -> None:
        self.config = config
        self.show_landmarks = config.show_landmarks

    def toggle_landmarks(self) -> None:
        self.show_landmarks = not self.show_landmarks

    def render(self, frame, state: DriverState, observation: FaceObservation | None):
        output = frame.copy()
        if self.show_landmarks and observation is not None:
            self._draw_eye_landmarks(output, observation)

        self._draw_status_panel(output, state)
        self._draw_footer(output)
        return output

    def _draw_status_panel(self, frame, state: DriverState) -> None:
        color = self.COLORS.get(state.fatigue_level, (255, 255, 255))
        panel_x, panel_y = 28, 22
        panel_w, panel_h = 520, 270

        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            (18, 18, 18),
            -1,
        )
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + 8, panel_y + panel_h),
            color,
            -1,
        )
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            color,
            2,
        )

        x = panel_x + 24
        self._put_text(frame, "DRIVER VIGILANCE MONITOR", x, panel_y + 32, 0.58, (225, 225, 225), 1)
        self._put_text(frame, state.fatigue_level.value.upper(), x, panel_y + 82, 1.05, color, 2)
        self._put_text(frame, state.message, x, panel_y + 114, 0.60, (225, 225, 225), 1)

        y = panel_y + 150
        self._put_text(frame, f"Eye: {state.eye_state.value}", x, y, 0.58, (230, 230, 230), 1)
        self._put_text(frame, f"EAR {state.smoothed_ear:.3f}", x + 285, y, 0.58, (210, 210, 210), 1)

        closed_label = f"Closed time: {state.closed_seconds:.1f}s"
        self._put_text(frame, closed_label, x, y + 34, 0.58, (230, 230, 230), 1)
        self._draw_progress_bar(
            frame,
            x + 170,
            y + 20,
            300,
            12,
            min(1.0, state.closed_seconds / 3.0),
            color if state.closed_seconds > 0 else (90, 90, 90),
        )

        self._put_text(frame, f"PERCLOS: {state.perclos_percent:.1f}%", x, y + 72, 0.58, (230, 230, 230), 1)
        self._draw_perclos_bar(frame, x + 170, y + 58, 300, state.perclos_percent)

        self._put_text(frame, f"Alert: {state.alert_level.value}", x, y + 110, 0.58, color, 1)
        self._put_text(frame, f"Temp: {state.temperature_celsius:.1f} C", x + 285, y + 110, 0.58, (205, 205, 205), 1)

        if state.eye_state == EyeState.CLOSED:
            self._draw_alert_banner(frame, color)
        elif state.eye_state in {EyeState.LOOKING_AWAY, EyeState.OBSTRUCTED}:
            self._put_text(
                frame,
                f"Yaw: {state.head_yaw_degrees:.0f}  Pitch: {state.head_pitch_degrees:.0f}  Eye Q: {state.eye_texture_score:.2f}",
                x,
                y + 132,
                0.54,
                (230, 230, 230),
                1,
            )
        if state.passenger_alert:
            self._draw_passenger_banner(frame)

    def _draw_perclos_bar(self, frame, x: int, y: int, width: int, percent: float) -> None:
        height = 12
        cv2.rectangle(frame, (x, y), (x + width, y + height), (75, 75, 75), -1)
        fill = int(width * min(100.0, max(0.0, percent)) / 100.0)
        bar_color = (60, 210, 80)
        if percent >= 50:
            bar_color = (30, 30, 255)
        elif percent >= 30:
            bar_color = (0, 140, 255)
        elif percent >= 15:
            bar_color = (0, 220, 255)
        cv2.rectangle(frame, (x, y), (x + fill, y + height), bar_color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (150, 150, 150), 1)

    @staticmethod
    def _draw_progress_bar(frame, x: int, y: int, width: int, height: int, ratio: float, color: Color) -> None:
        ratio = min(1.0, max(0.0, ratio))
        cv2.rectangle(frame, (x, y), (x + width, y + height), (75, 75, 75), -1)
        cv2.rectangle(frame, (x, y), (x + int(width * ratio), y + height), color, -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (150, 150, 150), 1)

    def _draw_alert_banner(self, frame, color: Color) -> None:
        h, w = frame.shape[:2]
        if color == self.COLORS[FatigueLevel.NORMAL]:
            return
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 12), color, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    def _draw_passenger_banner(self, frame) -> None:
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 92), (w, h), (0, 0, 120), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        self._put_text(
            frame,
            "PASSENGER ALERT: Driver assistance required",
            36,
            h - 36,
            0.9,
            (255, 255, 255),
            2,
        )

    def _draw_eye_landmarks(self, frame, observation: FaceObservation) -> None:
        for eye_points in observation.eye_points.values():
            for point in eye_points:
                cv2.circle(frame, point, 3, (255, 220, 80), -1)
            for start, end in zip(eye_points, eye_points[1:] + eye_points[:1]):
                cv2.line(frame, start, end, (90, 200, 255), 1)

    def _draw_footer(self, frame) -> None:
        h, _ = frame.shape[:2]
        text = "q/Esc: Exit    +/-: Temperature    l: Landmarks"
        self._put_text(frame, text, 28, h - 22, 0.55, (230, 230, 230), 1)

    @staticmethod
    def _put_text(frame, text: str, x: int, y: int, scale: float, color: Color, thickness: int) -> None:
        cv2.putText(
            frame,
            text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
