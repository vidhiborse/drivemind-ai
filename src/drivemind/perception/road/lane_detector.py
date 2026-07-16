"""
Lane Detection using classical computer vision (Canny edges + Hough Transform).
Works on road-facing camera footage. Detects left/right lane lines and
estimates whether the vehicle appears centered or deviating.
"""

import cv2
import numpy as np

from drivemind.common.interfaces import Detector, FeaturePacket

CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 20
MIN_LINE_LENGTH = 20
MAX_LINE_GAP = 300
SLOPE_THRESHOLD = 0.3  # ignore near-horizontal lines


class LaneDetector(Detector):
    def __init__(self):
        pass

    def load(self) -> None:
        # No model to load — pure classical CV pipeline
        pass

    def _region_of_interest(self, edges):
        h, w = edges.shape
        mask = np.zeros_like(edges)
        # Trapezoid covering the lower half of the frame (where the road is)
        polygon = np.array([[
            (int(w * 0.1), h),
            (int(w * 0.4), int(h * 0.6)),
            (int(w * 0.6), int(h * 0.6)),
            (int(w * 0.9), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        return cv2.bitwise_and(edges, mask)

    def _classify_lines(self, lines):
        left_lines, right_lines = [], []
        if lines is None:
            return left_lines, right_lines

        for line in lines:
            x1, y1, x2, y2 = np.array(line).flatten()
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < SLOPE_THRESHOLD:
                continue  # too horizontal, not a lane line
            if slope < 0:
                left_lines.append((x1, y1, x2, y2))
            else:
                right_lines.append((x1, y1, x2, y2))
        return left_lines, right_lines

    def process(self, frame) -> FeaturePacket:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, CANNY_LOW, CANNY_HIGH)
        roi_edges = self._region_of_interest(edges)

        lines = cv2.HoughLinesP(
            roi_edges, 1, np.pi / 180, threshold=HOUGH_THRESHOLD,
            minLineLength=MIN_LINE_LENGTH, maxLineGap=MAX_LINE_GAP,
        )

        left_lines, right_lines = self._classify_lines(lines)

        left_detected = len(left_lines) > 0
        right_detected = len(right_lines) > 0

        # Simple lane-deviation heuristic: compare average x-position
        # of left vs right lines to the frame center
        lane_status = "unknown"
        if left_detected and right_detected:
            h, w = frame.shape[:2]
            avg_left_x = np.mean([l[0] for l in left_lines])
            avg_right_x = np.mean([l[0] for l in right_lines])
            lane_center = (avg_left_x + avg_right_x) / 2
            frame_center = w / 2
            deviation = lane_center - frame_center

            if abs(deviation) < w * 0.05:
                lane_status = "centered"
            elif deviation > 0:
                lane_status = "drifting_right"
            else:
                lane_status = "drifting_left"

        return FeaturePacket(
            source_module="lane_detector",
            data={
                "left_lane_detected": left_detected,
                "right_lane_detected": right_detected,
                "left_lines": left_lines,
                "right_lines": right_lines,
                "lane_status": lane_status,
            },
            confidence=1.0 if (left_detected and right_detected) else 0.5,
        )