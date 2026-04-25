from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


Point = Tuple[int, int]
EyePoints = Dict[str, List[Point]]


class EyeState(str, Enum):
    NO_FACE = "No Face"
    OPEN = "Eyes Open"
    WARNING = "Getting Drowsy"
    CLOSED = "Eyes Closed"
    OBSTRUCTED = "Eyes Obstructed"
    LOOKING_AWAY = "Looking Away"


class FatigueLevel(str, Enum):
    NO_FACE = "No Face"
    NORMAL = "Normal"
    MILD = "Mild Fatigue"
    DROWSY = "Drowsy"
    CRITICAL = "Critical"
    OBSTRUCTED = "Eyes Obstructed"
    DISTRACTED = "Distracted"


class AlertLevel(str, Enum):
    NONE = "None"
    SOFT = "Soft Alert"
    STRONG = "Strong Alert"
    CRITICAL = "Critical Alarm"
    PASSENGER = "Passenger Alert"


@dataclass(frozen=True)
class FaceObservation:
    left_ear: float
    right_ear: float
    raw_ear: float
    eye_points: EyePoints = field(default_factory=dict)
    left_eye_texture_score: float = 1.0
    right_eye_texture_score: float = 1.0
    head_pitch_degrees: float = 0.0
    head_yaw_degrees: float = 0.0
    head_roll_degrees: float = 0.0
    tracking_issue: Optional[str] = None


@dataclass(frozen=True)
class DriverState:
    timestamp: float
    face_detected: bool
    eye_state: EyeState
    fatigue_level: FatigueLevel
    alert_level: AlertLevel
    raw_ear: float = 0.0
    smoothed_ear: float = 0.0
    effective_open_threshold: float = 0.0
    effective_closed_threshold: float = 0.0
    closed_seconds: float = 0.0
    perclos_percent: float = 0.0
    temperature_celsius: float = 0.0
    head_pitch_degrees: float = 0.0
    head_yaw_degrees: float = 0.0
    eye_texture_score: float = 1.0
    passenger_alert: bool = False
    message: str = ""
