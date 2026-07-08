"""
Distraction Detection using YOLOv8 object detection.
Detects distraction-related objects: cell phone, cup (drinking), etc.
Uses the pretrained YOLOv8-nano model (COCO classes) — no custom training needed yet.
"""

from ultralytics import YOLO

from drivemind.common.interfaces import Detector, FeaturePacket

# COCO class names relevant to driver distraction
DISTRACTION_CLASSES = {
    "cell phone": "phone_usage",
    "cup": "drinking",
    "bottle": "drinking",
}

CONFIDENCE_THRESHOLD = 0.5


class DistractionDetector(Detector):
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = CONFIDENCE_THRESHOLD):
        self.model_name = model_name
        self.confidence = confidence
        self._model = None

    def load(self) -> None:
        # First run downloads yolov8n.pt automatically into the working directory
        self._model = YOLO(self.model_name)

    def process(self, frame) -> FeaturePacket:
        results = self._model.predict(
            frame, conf=self.confidence, verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self._model.names[class_id]

            if class_name in DISTRACTION_CLASSES:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                confidence = float(box.conf[0])

                detections.append({
                    "object": class_name,
                    "distraction_type": DISTRACTION_CLASSES[class_name],
                    "confidence": round(confidence, 3),
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                })

        return FeaturePacket(
            source_module="distraction_detector",
            data={
                "distractions": detections,
                "distraction_detected": len(detections) > 0,
            },
            confidence=detections[0]["confidence"] if detections else None,
        )