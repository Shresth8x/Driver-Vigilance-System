from __future__ import annotations

from dataclasses import replace
from typing import Optional
import logging

from .config import AlertConfig, EyeConfig, PerclosConfig, TemperatureConfig, TrackingConfig
from .metrics import PerclosTracker, RollingAverage
from .models import AlertLevel, DriverState, EyeState, FaceObservation, FatigueLevel


LOGGER = logging.getLogger(__name__)


class FatigueEngine:
    def __init__(
        self,
        eye_config: EyeConfig,
        perclos_config: PerclosConfig,
        temperature_config: TemperatureConfig,
        alert_config: AlertConfig,
        tracking_config: TrackingConfig | None = None,
    ) -> None:
        self.eye_config = eye_config
        self.perclos_config = perclos_config
        self.temperature_config = temperature_config
        self.alert_config = alert_config
        self.tracking_config = tracking_config or TrackingConfig()
        self._ear_average = RollingAverage(eye_config.smoothing_window)
        self._perclos = PerclosTracker(
            perclos_config.window_seconds,
            perclos_config.min_samples,
            perclos_config.warmup_seconds,
        )
        self._closed_since: Optional[float] = None
        self._alert_since: Optional[float] = None
        self._open_since: Optional[float] = None
        self._recovery_alert_until: Optional[float] = None
        self._last_active_alert_level = AlertLevel.NONE
        self._attention_issue_since: Optional[float] = None
        self._attention_issue_kind: Optional[str] = None

    def update(
        self,
        observation: Optional[FaceObservation],
        timestamp: float,
        temperature_celsius: float,
    ) -> DriverState:
        open_threshold, closed_threshold = self._temperature_adjusted_thresholds(
            temperature_celsius
        )

        if observation is None:
            self._closed_since = None
            self._alert_since = None
            self._open_since = None
            self._recovery_alert_until = None
            self._last_active_alert_level = AlertLevel.NONE
            self._attention_issue_since = None
            self._attention_issue_kind = None
            return DriverState(
                timestamp=timestamp,
                face_detected=False,
                eye_state=EyeState.NO_FACE,
                fatigue_level=FatigueLevel.NO_FACE,
                alert_level=AlertLevel.NONE,
                temperature_celsius=temperature_celsius,
                effective_open_threshold=open_threshold,
                effective_closed_threshold=closed_threshold,
                message="Face not detected",
            )

        smoothed_ear = self._ear_average.update(observation.raw_ear)
        eye_state = self._classify_eye_state(
            smoothed_ear,
            open_threshold,
            closed_threshold,
        )

        tracking_issue = self._qualified_tracking_issue(
            observation,
            eye_state,
            timestamp,
        )
        if tracking_issue is not None:
            return self._build_tracking_issue_state(
                observation=observation,
                issue=tracking_issue,
                timestamp=timestamp,
                smoothed_ear=smoothed_ear,
                open_threshold=open_threshold,
                closed_threshold=closed_threshold,
                temperature_celsius=temperature_celsius,
            )

        eyes_closed = eye_state == EyeState.CLOSED
        closed_seconds = self._update_closed_timer(timestamp, eyes_closed)
        perclos = self._perclos.update(timestamp, eyes_closed)
        perclos = self._apply_open_eye_recovery(timestamp, eye_state, perclos)

        fatigue_level = self._classify_fatigue(eye_state, closed_seconds, perclos)
        active_alert_level = self._classify_alert(
            fatigue_level,
            eye_state,
            closed_seconds,
            self.eye_config.continuous_closed_seconds,
            self.alert_config.minimum_sound_closed_seconds,
        )
        alert_level, is_recovery_alarm = self._apply_recovery_alarm(
            timestamp,
            eye_state,
            active_alert_level,
        )
        passenger_alert = False
        if not is_recovery_alarm:
            passenger_alert = self._update_passenger_escalation(timestamp, alert_level)
        if passenger_alert:
            alert_level = AlertLevel.PASSENGER
            self._last_active_alert_level = AlertLevel.PASSENGER

        return DriverState(
            timestamp=timestamp,
            face_detected=True,
            eye_state=eye_state,
            fatigue_level=fatigue_level,
            alert_level=alert_level,
            raw_ear=observation.raw_ear,
            smoothed_ear=smoothed_ear,
            effective_open_threshold=open_threshold,
            effective_closed_threshold=closed_threshold,
            closed_seconds=closed_seconds,
            perclos_percent=perclos,
            temperature_celsius=temperature_celsius,
            head_pitch_degrees=observation.head_pitch_degrees,
            head_yaw_degrees=observation.head_yaw_degrees,
            eye_texture_score=min(
                observation.left_eye_texture_score,
                observation.right_eye_texture_score,
            ),
            passenger_alert=passenger_alert,
            message=self._message_for(fatigue_level, passenger_alert),
        )

    def reset(self) -> None:
        self._ear_average.reset()
        self._perclos.reset()
        self._closed_since = None
        self._alert_since = None
        self._open_since = None
        self._recovery_alert_until = None
        self._last_active_alert_level = AlertLevel.NONE
        self._attention_issue_since = None
        self._attention_issue_kind = None

    def _temperature_adjusted_thresholds(self, temperature_celsius: float) -> tuple[float, float]:
        open_threshold = self.eye_config.open_ear_threshold
        closed_threshold = self.eye_config.closed_ear_threshold

        if not self.temperature_config.enabled:
            return open_threshold, closed_threshold

        baseline = self.temperature_config.baseline_celsius
        high = self.temperature_config.high_heat_celsius
        if temperature_celsius <= baseline or high <= baseline:
            return open_threshold, closed_threshold

        heat_ratio = min(1.0, (temperature_celsius - baseline) / (high - baseline))
        offset = heat_ratio * self.temperature_config.max_ear_threshold_offset

        return open_threshold + offset, closed_threshold + (offset * 0.6)

    @staticmethod
    def _classify_eye_state(
        ear: float,
        open_threshold: float,
        closed_threshold: float,
    ) -> EyeState:
        if ear > open_threshold:
            return EyeState.OPEN
        if ear > closed_threshold:
            return EyeState.WARNING
        return EyeState.CLOSED

    def _update_closed_timer(self, timestamp: float, eyes_closed: bool) -> float:
        if not eyes_closed:
            self._closed_since = None
            return 0.0

        if self._closed_since is None:
            self._closed_since = timestamp
            return 0.0

        return timestamp - self._closed_since

    def _apply_open_eye_recovery(
        self,
        timestamp: float,
        eye_state: EyeState,
        perclos_percent: float,
    ) -> float:
        if eye_state != EyeState.OPEN:
            self._open_since = None
            return perclos_percent

        if self._open_since is None:
            self._open_since = timestamp
            return perclos_percent

        if timestamp - self._open_since >= self.perclos_config.recovery_open_seconds:
            self._perclos.reset()
            self._alert_since = None
            return 0.0

        return perclos_percent

    def _qualified_tracking_issue(
        self,
        observation: FaceObservation,
        eye_state: EyeState,
        timestamp: float,
    ) -> Optional[str]:
        if eye_state == EyeState.CLOSED:
            self._attention_issue_since = None
            self._attention_issue_kind = None
            return None

        issue: Optional[str] = None
        if (
            self.tracking_config.enable_head_pose_check
            and observation.tracking_issue == "looking_away"
        ):
            issue = "looking_away"
        elif (
            self.tracking_config.enable_eye_obstruction_check
            and
            observation.tracking_issue == "eyes_obstructed"
            and eye_state != EyeState.CLOSED
        ):
            issue = "eyes_obstructed"

        if issue is None:
            self._attention_issue_since = None
            self._attention_issue_kind = None
            return None

        if self._attention_issue_kind != issue:
            self._attention_issue_kind = issue
            self._attention_issue_since = timestamp
            return None

        if self._attention_issue_since is None:
            self._attention_issue_since = timestamp
            return None

        if timestamp - self._attention_issue_since < self.tracking_config.attention_warning_seconds:
            return None

        return issue

    def _build_tracking_issue_state(
        self,
        *,
        observation: FaceObservation,
        issue: str,
        timestamp: float,
        smoothed_ear: float,
        open_threshold: float,
        closed_threshold: float,
        temperature_celsius: float,
    ) -> DriverState:
        self._closed_since = None
        self._open_since = None
        self._recovery_alert_until = None
        self._last_active_alert_level = AlertLevel.NONE

        alert_level = AlertLevel.NONE
        if self.tracking_config.alert_on_attention_issue:
            alert_level = AlertLevel.SOFT

        if issue == "looking_away":
            eye_state = EyeState.LOOKING_AWAY
            fatigue_level = FatigueLevel.DISTRACTED
            message = "Driver looking away"
        else:
            eye_state = EyeState.OBSTRUCTED
            fatigue_level = FatigueLevel.OBSTRUCTED
            message = "Eyes obstructed"

        return DriverState(
            timestamp=timestamp,
            face_detected=True,
            eye_state=eye_state,
            fatigue_level=fatigue_level,
            alert_level=alert_level,
            raw_ear=observation.raw_ear,
            smoothed_ear=smoothed_ear,
            effective_open_threshold=open_threshold,
            effective_closed_threshold=closed_threshold,
            closed_seconds=0.0,
            perclos_percent=self._perclos.percent,
            temperature_celsius=temperature_celsius,
            head_pitch_degrees=observation.head_pitch_degrees,
            head_yaw_degrees=observation.head_yaw_degrees,
            eye_texture_score=min(
                observation.left_eye_texture_score,
                observation.right_eye_texture_score,
            ),
            passenger_alert=False,
            message=message,
        )

    def _apply_recovery_alarm(
        self,
        timestamp: float,
        eye_state: EyeState,
        active_alert_level: AlertLevel,
    ) -> tuple[AlertLevel, bool]:
        if active_alert_level != AlertLevel.NONE:
            self._last_active_alert_level = active_alert_level
            self._recovery_alert_until = None
            return active_alert_level, False

        if (
            eye_state in {EyeState.OPEN, EyeState.WARNING}
            and self._last_active_alert_level
            in {AlertLevel.STRONG, AlertLevel.CRITICAL, AlertLevel.PASSENGER}
        ):
            if self._recovery_alert_until is None:
                self._recovery_alert_until = (
                    timestamp + self.alert_config.recovery_alarm_seconds
                )

            if timestamp < self._recovery_alert_until:
                return self._last_active_alert_level, True

        self._recovery_alert_until = None
        self._last_active_alert_level = AlertLevel.NONE
        return AlertLevel.NONE, False

    def _classify_fatigue(
        self,
        eye_state: EyeState,
        closed_seconds: float,
        perclos_percent: float,
    ) -> FatigueLevel:
        if (
            perclos_percent >= self.perclos_config.critical_percent
            or closed_seconds >= self.eye_config.continuous_closed_seconds * 1.7
        ):
            return FatigueLevel.CRITICAL

        if (
            perclos_percent >= self.perclos_config.drowsy_percent
            or closed_seconds >= self.eye_config.continuous_closed_seconds
        ):
            return FatigueLevel.DROWSY

        if (
            perclos_percent >= self.perclos_config.mild_percent
            or eye_state == EyeState.WARNING
        ):
            return FatigueLevel.MILD

        return FatigueLevel.NORMAL

    def _classify_alert(
        self,
        fatigue_level: FatigueLevel,
        eye_state: EyeState,
        closed_seconds: float,
        drowsy_closed_seconds: float,
        minimum_sound_closed_seconds: float,
    ) -> AlertLevel:
        if eye_state != EyeState.CLOSED:
            return AlertLevel.NONE

        if closed_seconds >= drowsy_closed_seconds * 1.7:
            return AlertLevel.CRITICAL
        if closed_seconds >= drowsy_closed_seconds:
            return AlertLevel.STRONG

        if (
            fatigue_level in {FatigueLevel.DROWSY, FatigueLevel.CRITICAL}
            and closed_seconds >= minimum_sound_closed_seconds
        ):
            return AlertLevel.SOFT

        return AlertLevel.NONE

    def _update_passenger_escalation(
        self,
        timestamp: float,
        alert_level: AlertLevel,
    ) -> bool:
        if alert_level not in {AlertLevel.STRONG, AlertLevel.CRITICAL}:
            self._alert_since = None
            return False

        if self._alert_since is None:
            self._alert_since = timestamp
            return False

        return (timestamp - self._alert_since) >= self.alert_config.passenger_escalation_seconds

    @staticmethod
    def _message_for(level: FatigueLevel, passenger_alert: bool) -> str:
        if passenger_alert:
            return "Passenger assistance required"
        if level == FatigueLevel.CRITICAL:
            return "Critical fatigue detected"
        if level == FatigueLevel.DROWSY:
            return "Driver drowsy"
        if level == FatigueLevel.MILD:
            return "Early fatigue signs"
        if level == FatigueLevel.NORMAL:
            return "Driver attentive"
        return "Face not detected"


def with_temperature(state: DriverState, temperature_celsius: float) -> DriverState:
    return replace(state, temperature_celsius=temperature_celsius)
