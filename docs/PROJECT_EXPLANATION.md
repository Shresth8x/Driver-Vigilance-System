# Driver Vigilance Monitoring System - Technical Explanation

## Objective

The goal is to monitor the driver's attention level through a camera-based vision system. The system detects face and eye landmarks, measures eye closure, estimates fatigue over time, and escalates alerts when drowsiness is detected.

## Main Pipeline

```text
Camera Frame
  -> Face Mesh Detection
  -> Eye Landmark Extraction
  -> EAR Calculation
  -> EAR Smoothing
  -> Eye State Classification
  -> PERCLOS Tracking
  -> Fatigue Level Decision
  -> Visual + Audio + Passenger Alerts
```

## Phase Mapping From Project Guide

### Phase 1: Project Understanding

Implemented as a real-time Driver Monitoring System (DMS). The system observes the driver through a camera and detects signs of fatigue based on eye closure and behavior over time.

### Phase 2: System Design

Inputs:

- Camera feed.
- Simulated ambient temperature.

Outputs:

- Visual warning on screen.
- Audio warning through system sound.
- Passenger escalation message.

Logic:

- Short eye closures are treated as natural blinking.
- Continuous eye closure beyond the configured duration triggers drowsiness.
- High PERCLOS indicates fatigue trend over time.
- High temperature increases sensitivity because heat can accelerate fatigue.

### Phase 3: Core Software

Implemented in these modules:

- `camera.py`: opens and reads webcam frames.
- `vision.py`: detects face mesh and extracts eye points.
- `metrics.py`: calculates EAR and PERCLOS.
- `fatigue.py`: classifies driver condition.
- `alerts.py`: handles warning sounds.
- `ui.py`: renders professional dashboard overlay.
- `app.py`: coordinates the complete runtime loop.

## PERCLOS Logic

PERCLOS stands for Percentage of Eye Closure. The system stores recent eye states in a rolling window:

- Closed eye frame = 1.
- Open eye frame = 0.

Then:

```text
PERCLOS = closed samples / total samples * 100
```

This is intentionally a history-based metric. It does not drop immediately the moment the eyes open because old closed-eye samples remain inside the rolling window for a short time.

In the prototype, active alarm sound is separated from the PERCLOS history. PERCLOS is ignored during an initial warm-up period so a normal blink near startup does not look like a high closure percentage. Sound alerts also require the eyes to remain closed for a minimum duration, so natural blinks do not beep.

If the driver's eyes become clearly open after a drowsy or critical event, the alarm continues for a short recovery period so the warning is not missed. After a stable open-eye recovery period, the PERCLOS display is reset for a cleaner live demonstration.

### Phase 4: Smart Features

Temperature adaptation:

- The system accepts a simulated temperature input.
- Above the baseline temperature, the system slightly increases the EAR closure sensitivity.
- This means borderline eye closure is treated more carefully in hot conditions.

Passenger alert:

- If the driver remains drowsy after repeated warnings, the system escalates.
- In the prototype, this is shown on screen and logged.
- In hardware, this can be connected to a vibration motor, buzzer, or cabin alert.

Tracking reliability:

- If the full face is missing, the system reports `No Face`.
- Optional diagnostics can report `Eyes Obstructed` when both eye regions are unreliable.
- Optional diagnostics can report `Distracted` when the driver looks sideways or down for too long.
- These states are intentionally separated from `Drowsy` to reduce false positives.
- In the default demo configuration, these diagnostics are disabled so they do not interfere with the core eye-closure and PERCLOS system.

### Phase 5: Hardware Extension

The same software design can be moved to Raspberry Pi:

- Use a USB camera or Pi camera.
- Replace the desktop sound alert with GPIO buzzer output.
- Connect a passenger vibration motor through a transistor driver circuit.
- Run `python run.py` at startup using a service.

## Important Engineering Detail

EAR values are usually higher when eyes are open and lower when eyes are closed. Therefore, if we want earlier detection in hot conditions, the closed-eye threshold should become slightly more sensitive. In this implementation, hot temperature increases the effective threshold used for warning decisions.

## Fatigue Levels

| Level | Meaning | Typical Trigger |
| --- | --- | --- |
| Normal | Driver appears attentive | Low PERCLOS, eyes open |
| Mild | Early warning signs | Borderline EAR or PERCLOS above 15 percent |
| Drowsy | Strong fatigue signal | Closed eyes for several seconds or PERCLOS above 30 percent |
| Critical | Unsafe condition | PERCLOS above 50 percent or prolonged eye closure |

## Limitations

- Lighting, sunglasses, camera angle, and face occlusion can affect accuracy.
- EAR thresholds should be calibrated for the user and camera.
- This prototype does not replace certified automotive driver monitoring systems.
- Production deployment would require vehicle-grade testing, redundancy, and safety validation.

## Future Scope

- Add head pose estimation for distraction detection.
- Add yawn detection using mouth landmarks.
- Add steering pattern or CAN bus signals.
- Add infrared camera support for night operation.
- Add embedded deployment with Raspberry Pi or automotive compute hardware.
- Add event recording for validation logs.
