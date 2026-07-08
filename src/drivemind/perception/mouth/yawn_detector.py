"""
Yawning Detection using Mouth Aspect Ratio (MAR).
Uses the same MediaPipe Face Landmarker model as eye detection.
Also tracks a rolling yawn count for basic fatigue signaling.
"""

import math
import time
from collections import deque

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from drivemind.common.interfaces import Detector, FeaturePacket

MODEL_PATH = "models/face_landmarker.task"

# MediaPipe Face Mesh landmark indices for the mouth
# Outer lip points used for vertical and horizontal measurement
MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP_1 = 81
MOUTH_BOTTOM_1 = 178
MOUTH_TOP_2 = 13
MOUTH_BOTTOM_2 = 14

MAR_THRESHOLD = 0.6
YAWN_WINDOW_SECONDS = 120  # 2-minute rolling window
YAWN_FATIGUE_COUNT = 3     # 3+ yawns in window -> fatigue signal


def _distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def _calculate_mar(landmarks):
    left = landmarks[MOUTH_LEFT]
    right = landmarks[MOUTH_RIGHT]
    top1 = landmarks[MOUTH_TOP_1]
    bottom1 = landmarks[MOUTH_BOTTOM_1]
    top2 = landmarks[MOUTH_TOP_2]
    bottom2 = landmarks[MOUTH_BOTTOM_2]

    vertical_1 = _distance(top1, bottom1)
    vertical_2 = _distance(top2, bottom2)
    horizontal = _distance(left, right)

    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


class YawnDetector(Detector):
    def __init__(self, model_path: str = MODEL_PATH, mar_threshold: float = MAR_THRESHOLD):
        self.model_path = model_path
        self.mar_threshold = mar_threshold
        self._detector = None
        self._yawn_timestamps = deque()
        self._was_yawning = False  # to detect yawn start (rising edge), not every frame

    def load(self) -> None:
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    def _update_yawn_window(self, is_yawning: bool):
        now = time.time()
        # New yawn event only counted on the transition from not-yawning -> yawning
        if is_yawning and not self._was_yawning:
            self._yawn_timestamps.append(now)
        self._was_yawning = is_yawning

        # Drop timestamps older than the rolling window
        while self._yawn_timestamps and now - self._yawn_timestamps[0] > YAWN_WINDOW_SECONDS:
            self._yawn_timestamps.popleft()

        return len(self._yawn_timestamps)

    def process(self, frame) -> FeaturePacket:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return FeaturePacket(
                source_module="yawn_detector",
                data={"mouth_detected": False},
                confidence=None,
            )

        landmarks = result.face_landmarks[0]
        mar = _calculate_mar(landmarks)
        is_yawning = mar > self.mar_threshold

        yawn_count_in_window = self._update_yawn_window(is_yawning)
        fatigue_signal = yawn_count_in_window >= YAWN_FATIGUE_COUNT

        return FeaturePacket(
            source_module="yawn_detector",
            data={
                "mouth_detected": True,
                "mar": round(mar, 3),
                "is_yawning": is_yawning,
                "yawn_count_last_2min": yawn_count_in_window,
                "fatigue_signal": fatigue_signal,
            },
            confidence=mar,
        )