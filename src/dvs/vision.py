from __future__ import annotations

from typing import Optional
import logging

import cv2
import mediapipe as mp
import numpy as np

from .config import TrackingConfig
from .metrics import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio, points_from_landmarks
from .models import FaceObservation


LOGGER = logging.getLogger(__name__)

LEFT_EYE_CONTOUR = (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7)
RIGHT_EYE_CONTOUR = (362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382)


class FaceMeshEyeDetector:
    def __init__(self, tracking_config: TrackingConfig | None = None) -> None:
        self.tracking_config = tracking_config or TrackingConfig()
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame) -> Optional[FaceObservation]:
        frame_height, frame_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        left_points = points_from_landmarks(landmarks, LEFT_EYE, frame_width, frame_height)
        right_points = points_from_landmarks(landmarks, RIGHT_EYE, frame_width, frame_height)
        left_contour = points_from_landmarks(landmarks, LEFT_EYE_CONTOUR, frame_width, frame_height)
        right_contour = points_from_landmarks(landmarks, RIGHT_EYE_CONTOUR, frame_width, frame_height)

        left_ear = eye_aspect_ratio(left_points)
        right_ear = eye_aspect_ratio(right_points)
        raw_ear = (left_ear + right_ear) / 2.0
        left_score = self._eye_texture_score(frame, left_contour)
        right_score = self._eye_texture_score(frame, right_contour)
        pitch, yaw, roll = self._estimate_head_pose(landmarks, frame_width, frame_height)
        tracking_issue = self._tracking_issue(pitch, yaw, left_score, right_score)

        return FaceObservation(
            left_ear=left_ear,
            right_ear=right_ear,
            raw_ear=raw_ear,
            eye_points={"left": left_points, "right": right_points},
            left_eye_texture_score=left_score,
            right_eye_texture_score=right_score,
            head_pitch_degrees=pitch,
            head_yaw_degrees=yaw,
            head_roll_degrees=roll,
            tracking_issue=tracking_issue,
        )

    def close(self) -> None:
        self._face_mesh.close()

    def _tracking_issue(
        self,
        pitch: float,
        yaw: float,
        left_score: float,
        right_score: float,
    ) -> Optional[str]:
        if self.tracking_config.enable_head_pose_check and (
            abs(yaw) > self.tracking_config.max_yaw_degrees
            or abs(pitch) > self.tracking_config.max_pitch_degrees
        ):
            return "looking_away"

        if self.tracking_config.enable_eye_obstruction_check and (
            left_score < self.tracking_config.min_eye_texture_score
            and right_score < self.tracking_config.min_eye_texture_score
        ):
            return "eyes_obstructed"

        return None

    @staticmethod
    def _eye_texture_score(frame, eye_contour) -> float:
        if len(eye_contour) < 3:
            return 0.0

        polygon = np.array(eye_contour, dtype=np.int32)
        x, y, width, height = cv2.boundingRect(polygon)
        if width <= 2 or height <= 2:
            return 0.0

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi_gray = gray[y : y + height, x : x + width]
        roi_mask = mask[y : y + height, x : x + width]
        pixels = roi_gray[roi_mask > 0]
        if pixels.size < 20:
            return 0.0

        std_score = min(float(np.std(pixels)) / 32.0, 1.0)
        median = float(np.median(pixels))
        dark_ratio = float(np.mean(pixels < max(20.0, median - 25.0)))
        dark_score = min(dark_ratio / 0.18, 1.0)

        edges = cv2.Canny(roi_gray, 40, 100)
        edge_pixels = edges[roi_mask > 0]
        edge_ratio = float(np.mean(edge_pixels > 0)) if edge_pixels.size else 0.0
        edge_score = min(edge_ratio / 0.06, 1.0)

        return (0.50 * std_score) + (0.35 * edge_score) + (0.15 * dark_score)

    @staticmethod
    def _estimate_head_pose(landmarks, frame_width: int, frame_height: int) -> tuple[float, float, float]:
        image_points = np.array(
            [
                _landmark_xy(landmarks[1], frame_width, frame_height),
                _landmark_xy(landmarks[152], frame_width, frame_height),
                _landmark_xy(landmarks[33], frame_width, frame_height),
                _landmark_xy(landmarks[263], frame_width, frame_height),
                _landmark_xy(landmarks[61], frame_width, frame_height),
                _landmark_xy(landmarks[291], frame_width, frame_height),
            ],
            dtype=np.float64,
        )
        model_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (0.0, -63.6, -12.5),
                (-43.3, 32.7, -26.0),
                (43.3, 32.7, -26.0),
                (-28.9, -28.9, -24.1),
                (28.9, -28.9, -24.1),
            ],
            dtype=np.float64,
        )
        focal_length = float(frame_width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, frame_width / 2.0],
                [0.0, focal_length, frame_height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, _ = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, roll = angles
        return float(pitch), float(yaw), float(roll)


def _landmark_xy(landmark, frame_width: int, frame_height: int) -> tuple[float, float]:
    return float(landmark.x * frame_width), float(landmark.y * frame_height)
