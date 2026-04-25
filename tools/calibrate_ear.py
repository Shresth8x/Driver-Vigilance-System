from __future__ import annotations

from pathlib import Path
import statistics
import sys

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dvs.config import load_config
from dvs.camera import CameraStream
from dvs.vision import FaceMeshEyeDetector


def mean_or_zero(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> int:
    config = load_config(PROJECT_ROOT / "config" / "default_config.json")
    camera = CameraStream(config.camera)
    detector = FaceMeshEyeDetector()
    open_samples: list[float] = []
    closed_samples: list[float] = []

    try:
        camera.open()
        cv2.namedWindow("EAR Calibration", cv2.WINDOW_NORMAL)

        while True:
            frame = camera.read()
            if frame is None:
                break

            observation = detector.process(frame)
            ear = observation.raw_ear if observation else 0.0

            cv2.rectangle(frame, (25, 25), (620, 165), (0, 0, 0), -1)
            cv2.putText(frame, f"Current EAR: {ear:.3f}", (45, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(frame, "o: save open-eye sample | c: save closed-eye sample", (45, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.putText(frame, "s: show suggestion | q/Esc: exit", (45, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1)
            cv2.imshow("EAR Calibration", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if observation is None:
                continue
            if key == ord("o"):
                open_samples.append(ear)
                print(f"Saved open-eye sample: {ear:.3f}")
            elif key == ord("c"):
                closed_samples.append(ear)
                print(f"Saved closed-eye sample: {ear:.3f}")
            elif key == ord("s"):
                open_mean = mean_or_zero(open_samples)
                closed_mean = mean_or_zero(closed_samples)
                if open_samples and closed_samples:
                    open_threshold = (open_mean + closed_mean) / 2.0
                    closed_threshold = closed_mean + ((open_threshold - closed_mean) * 0.35)
                    print("")
                    print("Calibration suggestion")
                    print(f"  Open-eye average:   {open_mean:.3f}")
                    print(f"  Closed-eye average: {closed_mean:.3f}")
                    print(f"  open_ear_threshold:   {open_threshold:.3f}")
                    print(f"  closed_ear_threshold: {closed_threshold:.3f}")
                    print("")
                else:
                    print("Collect both open and closed samples before asking for a suggestion.")
        return 0
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
