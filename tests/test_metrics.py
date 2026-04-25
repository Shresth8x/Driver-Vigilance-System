from dvs.config import AlertConfig, EyeConfig, PerclosConfig, TemperatureConfig, TrackingConfig
from dvs.fatigue import FatigueEngine
from dvs.metrics import PerclosTracker, eye_aspect_ratio
from dvs.models import AlertLevel, EyeState, FaceObservation, FatigueLevel


def test_eye_aspect_ratio_for_symmetric_eye():
    points = [(0, 0), (1, -1), (3, -1), (4, 0), (3, 1), (1, 1)]

    assert eye_aspect_ratio(points) == 0.5


def test_perclos_tracker_reports_closed_percentage_after_min_samples():
    tracker = PerclosTracker(window_seconds=10, min_samples=4)

    tracker.update(1.0, True)
    tracker.update(2.0, False)
    tracker.update(3.0, True)
    percent = tracker.update(4.0, False)

    assert percent == 50.0


def test_fatigue_engine_escalates_after_continuous_eye_closure():
    engine = FatigueEngine(
        EyeConfig(open_ear_threshold=0.21, closed_ear_threshold=0.15, smoothing_window=1, continuous_closed_seconds=3),
        PerclosConfig(min_samples=1, warmup_seconds=0.0),
        TemperatureConfig(enabled=False),
        AlertConfig(),
    )
    observation = FaceObservation(left_ear=0.10, right_ear=0.10, raw_ear=0.10)

    engine.update(observation, 1.0, 30.0)
    state = engine.update(observation, 4.2, 30.0)

    assert state.fatigue_level in {FatigueLevel.DROWSY, FatigueLevel.CRITICAL}
    assert state.alert_level in {AlertLevel.STRONG, AlertLevel.CRITICAL}


def test_alarm_continues_briefly_when_eyes_open_after_critical_alert():
    engine = FatigueEngine(
        EyeConfig(open_ear_threshold=0.21, closed_ear_threshold=0.15, smoothing_window=1, continuous_closed_seconds=3),
        PerclosConfig(min_samples=1, warmup_seconds=0.0),
        TemperatureConfig(enabled=False),
        AlertConfig(recovery_alarm_seconds=4.0),
    )
    closed = FaceObservation(left_ear=0.10, right_ear=0.10, raw_ear=0.10)
    open_eye = FaceObservation(left_ear=0.30, right_ear=0.30, raw_ear=0.30)

    engine.update(closed, 1.0, 30.0)
    engine.update(closed, 5.0, 30.0)
    recovered = engine.update(open_eye, 5.1, 30.0)

    assert recovered.perclos_percent > 0
    assert recovered.alert_level in {AlertLevel.STRONG, AlertLevel.CRITICAL}


def test_alarm_stops_after_recovery_alarm_period():
    engine = FatigueEngine(
        EyeConfig(open_ear_threshold=0.21, closed_ear_threshold=0.15, smoothing_window=1, continuous_closed_seconds=3),
        PerclosConfig(min_samples=1, warmup_seconds=0.0, recovery_open_seconds=3.0),
        TemperatureConfig(enabled=False),
        AlertConfig(recovery_alarm_seconds=4.0),
    )
    closed = FaceObservation(left_ear=0.10, right_ear=0.10, raw_ear=0.10)
    open_eye = FaceObservation(left_ear=0.30, right_ear=0.30, raw_ear=0.30)

    engine.update(closed, 1.0, 30.0)
    engine.update(closed, 5.0, 30.0)
    engine.update(open_eye, 5.1, 30.0)
    recovered = engine.update(open_eye, 9.2, 30.0)

    assert recovered.perclos_percent == 0.0
    assert recovered.alert_level == AlertLevel.NONE


def test_normal_blink_does_not_trigger_sound_alert():
    engine = FatigueEngine(
        EyeConfig(open_ear_threshold=0.21, closed_ear_threshold=0.15, smoothing_window=1, continuous_closed_seconds=3),
        PerclosConfig(min_samples=1, warmup_seconds=0.0),
        TemperatureConfig(enabled=False),
        AlertConfig(minimum_sound_closed_seconds=1.0),
    )
    closed = FaceObservation(left_ear=0.10, right_ear=0.10, raw_ear=0.10)
    open_eye = FaceObservation(left_ear=0.30, right_ear=0.30, raw_ear=0.30)

    engine.update(open_eye, 1.0, 30.0)
    blink = engine.update(closed, 1.1, 30.0)
    recovered = engine.update(open_eye, 1.2, 30.0)

    assert blink.alert_level == AlertLevel.NONE
    assert recovered.alert_level == AlertLevel.NONE


def test_obstructed_eyes_are_not_marked_as_drowsy():
    engine = FatigueEngine(
        EyeConfig(open_ear_threshold=0.21, closed_ear_threshold=0.15, smoothing_window=1),
        PerclosConfig(min_samples=1, warmup_seconds=0.0),
        TemperatureConfig(enabled=False),
        AlertConfig(),
        TrackingConfig(enable_eye_obstruction_check=True, attention_warning_seconds=2.0),
    )
    obstructed = FaceObservation(
        left_ear=0.18,
        right_ear=0.18,
        raw_ear=0.18,
        left_eye_texture_score=0.05,
        right_eye_texture_score=0.05,
        tracking_issue="eyes_obstructed",
    )

    first = engine.update(obstructed, 1.0, 30.0)
    state = engine.update(obstructed, 3.2, 30.0)

    assert first.fatigue_level != FatigueLevel.OBSTRUCTED
    assert state.eye_state == EyeState.OBSTRUCTED
    assert state.fatigue_level == FatigueLevel.OBSTRUCTED
    assert state.alert_level == AlertLevel.NONE


def test_looking_away_is_classified_as_distraction_not_drowsiness():
    engine = FatigueEngine(
        EyeConfig(open_ear_threshold=0.21, closed_ear_threshold=0.15, smoothing_window=1),
        PerclosConfig(min_samples=1, warmup_seconds=0.0),
        TemperatureConfig(enabled=False),
        AlertConfig(),
        TrackingConfig(enable_head_pose_check=True, attention_warning_seconds=2.0),
    )
    looking_away = FaceObservation(
        left_ear=0.18,
        right_ear=0.18,
        raw_ear=0.18,
        head_yaw_degrees=35.0,
        tracking_issue="looking_away",
    )

    first = engine.update(looking_away, 1.0, 30.0)
    state = engine.update(looking_away, 3.2, 30.0)

    assert first.fatigue_level != FatigueLevel.DISTRACTED
    assert state.eye_state == EyeState.LOOKING_AWAY
    assert state.fatigue_level == FatigueLevel.DISTRACTED
    assert state.alert_level == AlertLevel.NONE


def test_closed_eyes_are_not_overridden_by_distraction_tracking():
    engine = FatigueEngine(
        EyeConfig(open_ear_threshold=0.21, closed_ear_threshold=0.15, smoothing_window=1),
        PerclosConfig(min_samples=1, warmup_seconds=0.0),
        TemperatureConfig(enabled=False),
        AlertConfig(),
        TrackingConfig(enable_head_pose_check=True, attention_warning_seconds=0.0),
    )
    closed_with_bad_pose = FaceObservation(
        left_ear=0.10,
        right_ear=0.10,
        raw_ear=0.10,
        head_yaw_degrees=50.0,
        tracking_issue="looking_away",
    )

    state = engine.update(closed_with_bad_pose, 1.0, 30.0)

    assert state.eye_state == EyeState.CLOSED
    assert state.fatigue_level != FatigueLevel.DISTRACTED


def test_head_pose_distraction_is_disabled_by_default():
    engine = FatigueEngine(
        EyeConfig(open_ear_threshold=0.21, closed_ear_threshold=0.15, smoothing_window=1),
        PerclosConfig(min_samples=1, warmup_seconds=0.0),
        TemperatureConfig(enabled=False),
        AlertConfig(),
        TrackingConfig(attention_warning_seconds=0.0),
    )
    looking_away = FaceObservation(
        left_ear=0.18,
        right_ear=0.18,
        raw_ear=0.18,
        head_yaw_degrees=50.0,
        tracking_issue="looking_away",
    )

    state = engine.update(looking_away, 1.0, 30.0)

    assert state.eye_state != EyeState.LOOKING_AWAY
    assert state.fatigue_level != FatigueLevel.DISTRACTED
