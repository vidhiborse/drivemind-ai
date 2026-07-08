"""
Head Pose Estimation using facial landmarks + OpenCV solvePnP.
Determines yaw (left/right), pitch (up/down), and roll (tilt) angles
by matching known 2D facial landmarks to a generic 3D face model.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from drivemind.common.interfaces import Detector, FeaturePacket

MODEL_PATH = "models/face_landmarker.task"

# Landmark indices for key facial points (MediaPipe Face Mesh numbering)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# Generic 3D model points (approximate, in arbitrary units) for an average face
# These are standard reference points used widely for head pose estimation
MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye corner
    (225.0, 170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

YAW_THRESHOLD = 20.0
PITCH_THRESHOLD = 15.0


class HeadPoseEstimator(Detector):
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._detector = None

    def load(self) -> None:
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    def _classify_direction(self, yaw, pitch):
        if yaw > YAW_THRESHOLD:
            return "looking_right"
        if yaw < -YAW_THRESHOLD:
            return "looking_left"
        if pitch > PITCH_THRESHOLD:
            return "looking_down"
        if pitch < -PITCH_THRESHOLD:
            return "looking_up"
        return "forward"

    def process(self, frame) -> FeaturePacket:
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return FeaturePacket(
                source_module="head_pose_estimator",
                data={"face_detected": False},
                confidence=None,
            )

        landmarks = result.face_landmarks[0]
        indices = [NOSE_TIP, CHIN, LEFT_EYE_CORNER, RIGHT_EYE_CORNER,
                   LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER]

        image_points = np.array([
            (landmarks[i].x * w, landmarks[i].y * h) for i in indices
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

        success, rotation_vector, _ = cv2.solvePnP(
            MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return FeaturePacket(
                source_module="head_pose_estimator",
                data={"face_detected": True, "pose_solved": False},
                confidence=None,
            )

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        sy = (rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2) ** 0.5

        pitch = np.degrees(np.arctan2(-rotation_matrix[2, 0], sy))
        yaw = np.degrees(np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        roll = np.degrees(np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))

        direction = self._classify_direction(yaw, pitch)

        return FeaturePacket(
            source_module="head_pose_estimator",
            data={
                "face_detected": True,
                "pose_solved": True,
                "yaw": round(float(yaw), 1),
                "pitch": round(float(pitch), 1),
                "roll": round(float(roll), 1),
                "direction": direction,
            },
            confidence=1.0,
        )