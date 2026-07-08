"""
Eye State Detection using Eye Aspect Ratio (EAR).
Uses MediaPipe Face Landmarker to get facial landmarks, then computes EAR
from specific eye landmark indices to determine open/closed state.
"""

import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from drivemind.common.interfaces import Detector, FeaturePacket

MODEL_PATH = "models/face_landmarker.task"

# MediaPipe Face Mesh landmark indices for eyes (standard, well-documented indices)
LEFT_EYE = [33, 160, 158, 133, 153, 144]    # p1, p2, p3, p4, p5, p6
RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # p1, p2, p3, p4, p5, p6

EAR_THRESHOLD = 0.2  # below this -> considered "closed"


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _calculate_ear(landmarks, eye_indices):
    p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    vertical_1 = _distance(p2, p6)
    vertical_2 = _distance(p3, p5)
    horizontal = _distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class EyeStateDetector(Detector):
    def __init__(self, model_path: str = MODEL_PATH, ear_threshold: float = EAR_THRESHOLD):
        self.model_path = model_path
        self.ear_threshold = ear_threshold
        self._detector = None

    def load(self) -> None:
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame) -> FeaturePacket:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return FeaturePacket(
                source_module="eye_state_detector",
                data={"eyes_detected": False},
                confidence=None,
            )

        landmarks = result.face_landmarks[0]

        left_ear = _calculate_ear(landmarks, LEFT_EYE)
        right_ear = _calculate_ear(landmarks, RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0

        eye_state = "closed" if avg_ear < self.ear_threshold else "open"

        return FeaturePacket(
            source_module="eye_state_detector",
            data={
                "eyes_detected": True,
                "left_ear": round(left_ear, 3),
                "right_ear": round(right_ear, 3),
                "avg_ear": round(avg_ear, 3),
                "eye_state": eye_state,
            },
            confidence=avg_ear,
        )