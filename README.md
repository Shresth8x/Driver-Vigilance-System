# Driver Vigilance Monitoring System

This is a professional multi-file version of the driver fatigue prototype. It uses a camera feed, MediaPipe Face Mesh, Eye Aspect Ratio (EAR), PERCLOS, audible alerts, visual warnings, passenger escalation logic, and optional smart diagnostics.

The project is built as an industry demo/prototype. It is suitable for academic and evaluation submission, but it is not a certified automotive safety product.

## Project Structure

```text
driver_vigilance_system/
  config/default_config.json
  docs/PROJECT_EXPLANATION.md
  src/dvs/
    alerts.py
    app.py
    camera.py
    config.py
    fatigue.py
    main.py
    metrics.py
    models.py
    temperature.py
    ui.py
    vision.py
  tests/test_metrics.py
  tools/calibrate_ear.py
  requirements.txt
  run.py
```

## What It Does

- Opens a webcam and captures real-time driver video.
- Detects the driver's face using MediaPipe Face Mesh.
- Tracks left and right eye landmarks.
- Calculates EAR to classify eye openness.
- Keeps poor-tracking diagnostics optional so the main drowsiness demo stays stable.
- Smooths EAR readings to reduce noise.
- Calculates PERCLOS over a rolling time window.
- Classifies fatigue as normal, mild, drowsy, or critical.
- Includes optional high-temperature sensitivity adjustment.
- Plays escalating alert sounds.
- Keeps the alarm active briefly after eye recovery so the driver notices it.
- Ignores normal blinks for sound alerts.
- Clears the PERCLOS demo display after stable open-eye recovery.
- Can optionally show looking-away and obstructed-eye cases separately from drowsiness.
- Displays live status, EAR, PERCLOS, temperature, and alert level.
- Escalates to a passenger alert if the driver does not recover.

## Setup on Windows

Open a terminal inside this folder:

```powershell
cd C:\Users\Shresth\Documents\Codex\2026-04-25\files-mentioned-by-the-user-complete\driver_vigilance_system
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the system:

```powershell
python run.py
```

If the wrong camera opens:

```powershell
python run.py --camera 1
```

Run with simulated display temperature:

```powershell
python run.py --temperature 40
```

Run without sound:

```powershell
python run.py --no-sound
```

## Keyboard Controls

- `q` or `Esc`: exit
- `+`: increase simulated temperature
- `-`: decrease simulated temperature
- `l`: show or hide eye landmarks

## Calibration

Every face, camera, and lighting condition is slightly different. Use the calibration tool to estimate better EAR thresholds:

```powershell
python tools/calibrate_ear.py
```

During calibration:

- Keep eyes open and press `o` several times.
- Close eyes and press `c` several times.
- Press `s` to show suggested thresholds.
- Press `q` or `Esc` to exit.

Then update `config/default_config.json`.

## How PERCLOS Works Here

PERCLOS means Percentage of Eye Closure. In this project, every frame is stored as either:

- `1`: eyes closed
- `0`: eyes open

The system keeps a rolling time window, configured as `perclos.window_seconds`. The default is 60 seconds.

Example:

- If 30 seconds out of the last 60 seconds were closed, PERCLOS is about 50 percent.
- If you close your eyes for a long time, PERCLOS becomes high.
- When you open your eyes, PERCLOS normally falls slowly because it is a recent-history value, not an instant value.

For demo usability, this project also has two recovery controls:

- `alerts.recovery_alarm_seconds`: keeps the alarm active briefly after the eyes open.
- `alerts.minimum_sound_closed_seconds`: prevents short natural blinks from making sound.
- `perclos.warmup_seconds`: optional startup delay before PERCLOS is trusted; default is `0.0` for the demo.
- `perclos.recovery_open_seconds`: resets the PERCLOS display after stable open-eye recovery.

## Tracking Reliability

The default project runs in a stable core drowsiness mode. Extra tracking diagnostics are available, but they are disabled by default because normal webcams can produce noisy head-pose and eye-texture estimates.

When enabled, the project separates three cases:

- `Drowsy`: eyes are actually closed for long enough.
- `Distracted`: face/head is turned sideways or downward, such as looking at a phone.
- `Eyes Obstructed`: face is visible, but both eye regions are not reliable, such as fingers covering the eyes.

These controls live in `config/default_config.json`:

```json
"tracking": {
  "enable_head_pose_check": false,
  "enable_eye_obstruction_check": false,
  "max_yaw_degrees": 45.0,
  "max_pitch_degrees": 40.0,
  "min_eye_texture_score": 0.18,
  "attention_warning_seconds": 3.0,
  "alert_on_attention_issue": false
}
```

Keep these checks off for the main drowsiness demo. If your evaluator asks for distraction or obstruction detection, enable one feature at a time and tune it slowly.

If eye-cover detection is enabled but does not trigger, increase `min_eye_texture_score` slightly.

## Run Tests

```powershell
pytest
```

## Submission Notes

For your report or presentation, explain the project as four modules:

1. Vision module: camera, face mesh, eye landmarks.
2. Measurement module: EAR, smoothing, closed-eye timer.
3. Fatigue intelligence module: PERCLOS, optional temperature adaptation, level classification.
4. Alert module: sound warning, visual warning, passenger escalation.

The detailed architecture and phase mapping are in `docs/PROJECT_EXPLANATION.md`.
