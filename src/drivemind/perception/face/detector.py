"""
Face Detection module — uses MediaPipe's Tasks API (BlazeFace short-range model).
Note: MediaPipe 0.10.x+ removed the old `mp.solutions` API for many builds,
so we use the newer, officially supported Tasks API instead.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from drivemind.common.interfaces import Detector, FeaturePacket

MODEL_PATH = "models/blaze_face_short_range.tflite"


class FaceDetector(Detector):
    def __init__(self, model_path: str = MODEL_PATH, min_confidence: float = 0.5):
        self.model_path = model_path
        self.min_confidence = min_confidence
        self._detector = None

    def load(self) -> None:
        base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=self.min_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(options)

    def process(self, frame) -> FeaturePacket:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self._detector.detect(mp_image)

        faces = []
        for detection in result.detections:
            bbox = detection.bounding_box
            confidence = detection.categories[0].score if detection.categories else 0.0
            faces.append({
                "x": bbox.origin_x,
                "y": bbox.origin_y,
                "width": bbox.width,
                "height": bbox.height,
                "confidence": confidence,
            })

        return FeaturePacket(
            source_module="face_detector",
            data={"faces": faces, "face_count": len(faces)},
            confidence=faces[0]["confidence"] if faces else None,
        )