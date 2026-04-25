from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, List
import math

from .models import Point


LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)


def euclidean_distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(points: Iterable[Point]) -> float:
    eye = list(points)
    if len(eye) != 6:
        raise ValueError("EAR calculation requires exactly six eye points.")

    vertical_1 = euclidean_distance(eye[1], eye[5])
    vertical_2 = euclidean_distance(eye[2], eye[4])
    horizontal = euclidean_distance(eye[0], eye[3])

    if horizontal == 0:
        return 0.0

    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class RollingAverage:
    def __init__(self, window_size: int) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero.")
        self._values: Deque[float] = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self._values.append(value)
        return self.value

    @property
    def value(self) -> float:
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)

    def reset(self) -> None:
        self._values.clear()


class PerclosTracker:
    def __init__(
        self,
        window_seconds: float,
        min_samples: int = 10,
        warmup_seconds: float = 0.0,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero.")
        self.window_seconds = window_seconds
        self.min_samples = min_samples
        self.warmup_seconds = warmup_seconds
        self._samples: Deque[tuple[float, bool]] = deque()

    def update(self, timestamp: float, eyes_closed: bool) -> float:
        self._samples.append((timestamp, eyes_closed))
        self._remove_old_samples(timestamp)
        return self.percent

    @property
    def percent(self) -> float:
        if len(self._samples) < self.min_samples:
            return 0.0
        if self.elapsed_seconds < self.warmup_seconds:
            return 0.0
        closed_count = sum(1 for _, closed in self._samples if closed)
        return (closed_count / len(self._samples)) * 100.0

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def elapsed_seconds(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        return self._samples[-1][0] - self._samples[0][0]

    def reset(self) -> None:
        self._samples.clear()

    def _remove_old_samples(self, timestamp: float) -> None:
        cutoff = timestamp - self.window_seconds
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()


def points_from_landmarks(
    landmarks: List[object],
    indices: Iterable[int],
    frame_width: int,
    frame_height: int,
) -> List[Point]:
    points: List[Point] = []
    for index in indices:
        landmark = landmarks[index]
        points.append((int(landmark.x * frame_width), int(landmark.y * frame_height)))
    return points
