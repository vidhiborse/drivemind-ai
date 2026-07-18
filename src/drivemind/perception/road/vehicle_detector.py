"""
Vehicle & Pedestrian Detection with monocular Time-To-Collision (TTC) estimation.
Runs on road-facing camera footage.

Uses YOLOv8 for detection, and a simple centroid-based tracker (not full
ByteTrack yet) to match objects across frames so bounding-box width growth
rate can be measured for TTC approximation.

TTC formula (monocular approximation, no LIDAR/radar):
    TTC = current_width / (current_width - previous_width) * time_delta

A small/negative denominator (object not approaching or barely growing)
is treated as "not approaching" rather than a division error.
"""

import time
from ultralytics import YOLO

from drivemind.common.interfaces import Detector, FeaturePacket

RELEVANT_CLASSES = {"car", "truck", "bus", "motorcycle", "person", "bicycle"}
CONFIDENCE_THRESHOLD = 0.4
TTC_DANGER_SECONDS = 3.0
MATCH_DISTANCE_THRESHOLD = 80  # pixels; how close centers must be to be "the same object"


class VehicleDetector(Detector):
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = CONFIDENCE_THRESHOLD):
        self.model_name = model_name
        self.confidence = confidence
        self._model = None
        self._previous_objects = []  # list of {"center": (x,y), "width": w, "class": name}
        self._previous_time = None

    def load(self) -> None:
        self._model = YOLO(self.model_name)

    def _match_previous(self, center, width, class_name):
        """Find the closest previous-frame object of the same class, if any."""
        best_match = None
        best_dist = MATCH_DISTANCE_THRESHOLD
        for prev in self._previous_objects:
            if prev["class"] != class_name:
                continue
            dist = ((prev["center"][0] - center[0]) ** 2 + (prev["center"][1] - center[1]) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_match = prev
        return best_match

    def _estimate_ttc(self, current_width, previous_width, time_delta):
        growth = current_width - previous_width
        if growth <= 0 or time_delta <= 0:
            return None  # not approaching, or no time elapsed
        ttc = (current_width / growth) * time_delta
        return round(ttc, 2)

    def process(self, frame) -> FeaturePacket:
        now = time.time()
        results = self._model.predict(frame, conf=self.confidence, verbose=False)[0]

        current_objects = []
        for box in results.boxes:
            class_id = int(box.cls[0])
            class_name = self._model.names[class_id]

            if class_name not in RELEVANT_CLASSES:
                continue

            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
            width = x2 - x1
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            confidence = float(box.conf[0])

            time_delta = (now - self._previous_time) if self._previous_time else None
            matched_prev = self._match_previous(center, width, class_name)

            ttc = None
            approaching = False
            if matched_prev and time_delta:
                ttc = self._estimate_ttc(width, matched_prev["width"], time_delta)
                if ttc is not None:
                    approaching = True

            danger = approaching and ttc is not None and ttc < TTC_DANGER_SECONDS

            current_objects.append({
                "class": class_name,
                "confidence": round(confidence, 3),
                "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "center": center,
                "width": width,
                "ttc_seconds": ttc,
                "approaching": approaching,
                "collision_danger": danger,
            })

        self._previous_objects = [
            {"center": obj["center"], "width": obj["width"], "class": obj["class"]}
            for obj in current_objects
        ]
        self._previous_time = now

        any_danger = any(obj["collision_danger"] for obj in current_objects)

        return FeaturePacket(
            source_module="vehicle_detector",
            data={
                "objects": current_objects,
                "object_count": len(current_objects),
                "collision_danger_detected": any_danger,
            },
            confidence=1.0 if current_objects else None,
        )