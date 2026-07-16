"""
Standalone test script for Lane Detection.
Runs on a sample road/dashcam video file (not webcam, since this needs
a road-facing view, not a driver-facing one).
Press 'q' to quit.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.road.lane_detector import LaneDetector

VIDEO_PATH = "data/raw/test_road_video.mp4"

detector = LaneDetector()
detector.load()

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"ERROR: Could not open video file at {VIDEO_PATH}")
    sys.exit(1)

print("Video started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    packet = detector.process(frame)
    data = packet.data

    for x1, y1, x2, y2 in data["left_lines"]:
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 4)   # blue = left lane
    for x1, y1, x2, y2 in data["right_lines"]:
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)   # red = right lane

    status_text = f"Lane status: {data['lane_status']}"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("DriveMind AI - Lane Detection Test", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()