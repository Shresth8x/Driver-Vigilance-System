from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional
import json


@dataclass(frozen=True)
class CameraConfig:
    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    flip_horizontal: bool = True


@dataclass(frozen=True)
class EyeConfig:
    open_ear_threshold: float = 0.21
    closed_ear_threshold: float = 0.15
    smoothing_window: int = 3
    continuous_closed_seconds: float = 3.0


@dataclass(frozen=True)
class PerclosConfig:
    window_seconds: float = 60.0
    mild_percent: float = 15.0
    drowsy_percent: float = 30.0
    critical_percent: float = 50.0
    min_samples: int = 12
    warmup_seconds: float = 0.0
    recovery_open_seconds: float = 3.0


@dataclass(frozen=True)
class TemperatureConfig:
    enabled: bool = False
    baseline_celsius: float = 30.0
    high_heat_celsius: float = 40.0
    max_ear_threshold_offset: float = 0.025
    default_celsius: float = 38.0


@dataclass(frozen=True)
class TrackingConfig:
    enable_head_pose_check: bool = False
    enable_eye_obstruction_check: bool = False
    max_yaw_degrees: float = 45.0
    max_pitch_degrees: float = 40.0
    min_eye_texture_score: float = 0.18
    attention_warning_seconds: float = 3.0
    alert_on_attention_issue: bool = False


@dataclass(frozen=True)
class AlertConfig:
    enabled: bool = True
    cooldown_seconds: float = 2.0
    passenger_escalation_seconds: float = 8.0
    recovery_alarm_seconds: float = 4.0
    minimum_sound_closed_seconds: float = 1.0


@dataclass(frozen=True)
class UiConfig:
    window_name: str = "Driver Vigilance Monitoring System"
    show_landmarks: bool = True
    show_debug_values: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: str = "logs"
    level: str = "INFO"


@dataclass(frozen=True)
class AppConfig:
    camera: CameraConfig = CameraConfig()
    eye: EyeConfig = EyeConfig()
    perclos: PerclosConfig = PerclosConfig()
    temperature: TemperatureConfig = TemperatureConfig()
    tracking: TrackingConfig = TrackingConfig()
    alerts: AlertConfig = AlertConfig()
    ui: UiConfig = UiConfig()
    logging: LoggingConfig = LoggingConfig()


def _merge_section(default_obj: Any, values: Dict[str, Any]) -> Any:
    allowed = default_obj.__dataclass_fields__.keys()
    clean_values = {key: value for key, value in values.items() if key in allowed}
    return replace(default_obj, **clean_values)


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    config = AppConfig()
    if path is None:
        return config

    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    return AppConfig(
        camera=_merge_section(config.camera, data.get("camera", {})),
        eye=_merge_section(config.eye, data.get("eye", {})),
        perclos=_merge_section(config.perclos, data.get("perclos", {})),
        temperature=_merge_section(config.temperature, data.get("temperature", {})),
        tracking=_merge_section(config.tracking, data.get("tracking", {})),
        alerts=_merge_section(config.alerts, data.get("alerts", {})),
        ui=_merge_section(config.ui, data.get("ui", {})),
        logging=_merge_section(config.logging, data.get("logging", {})),
    )


def apply_cli_overrides(
    config: AppConfig,
    *,
    camera_index: Optional[int] = None,
    no_sound: bool = False,
    show_landmarks: Optional[bool] = None,
) -> AppConfig:
    camera = config.camera
    alerts = config.alerts
    ui = config.ui

    if camera_index is not None:
        camera = replace(camera, index=camera_index)
    if no_sound:
        alerts = replace(alerts, enabled=False)
    if show_landmarks is not None:
        ui = replace(ui, show_landmarks=show_landmarks)

    return replace(config, camera=camera, alerts=alerts, ui=ui)
