"""
Seatbelt Detection — RULE-BASED PLACEHOLDER.

LIMITATION / TODO (Phase 7): This is a heuristic implementation using edge
detection + Hough line transform to find a diagonal strap pattern across
the chest area. It is NOT a trained ML model and will have false
positives/negatives with patterned clothing or poor lighting.

Planned upgrade: train a custom YOLO model on a labeled seatbelt dataset
(e.g., Kaggle seatbelt datasets) once we reach the MLOps/training phase.
"""

import cv2
import numpy as np

from drivemind.common.interfaces import Detector, FeaturePacket

MIN_LINE_LENGTH = 60
MAX_LINE_GAP = 15
DIAGONAL_ANGLE_RANGE = (20, 70)  # degrees, roughly diagonal (not fully vertical/horizontal)


class SeatbeltDetector(Detector):
    def __init__(self):
        pass

    def load(self) -> None:
        # No model to load for the rule-based version
        pass

    def _get_chest_roi(self, frame):
        h, w = frame.shape[:2]
        # Rough chest region: lower-middle portion of the frame
        y1, y2 = int(h * 0.45), int(h * 0.9)
        x1, x2 = int(w * 0.2), int(w * 0.8)
        return frame[y1:y2, x1:x2]

    def process(self, frame) -> FeaturePacket:
        roi = self._get_chest_roi(frame)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=50,
            minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP,
        )

        seatbelt_detected = False
        diagonal_line_count = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = np.array(line).flatten()
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if DIAGONAL_ANGLE_RANGE[0] <= angle <= DIAGONAL_ANGLE_RANGE[1]:
                    diagonal_line_count += 1

        if diagonal_line_count >= 2:
            seatbelt_detected = True

        return FeaturePacket(
            source_module="seatbelt_detector",
            data={
                "seatbelt_detected": seatbelt_detected,
                "diagonal_line_count": diagonal_line_count,
                "method": "rule_based_placeholder",
            },
            confidence=min(diagonal_line_count / 5.0, 1.0),
        )