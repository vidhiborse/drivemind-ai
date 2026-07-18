"""
Standalone test script for Vehicle/Pedestrian Detection + TTC estimation.
Runs on the same road video used for lane detection testing.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.road.vehicle_detector import VehicleDetector

VIDEO_PATH = "data/raw/test_road_video.mp4"

detector = VehicleDetector()
print("Loading YOLO model...")
detector.load()

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Could not open video at {VIDEO_PATH}")
    sys.exit(1)

print("Video started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    packet = detector.process(frame)
    data = packet.data

    for obj in data["objects"]:
        box = obj["box"]
        color = (0, 0, 255) if obj["collision_danger"] else (0, 255, 0)
        label = f"{obj['class']} {obj['confidence']}"
        if obj["ttc_seconds"] is not None:
            label += f" TTC:{obj['ttc_seconds']}s"

        cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), color, 2)
        cv2.putText(frame, label, (box["x1"], box["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    status = f"Objects: {data['object_count']}"
    if data["collision_danger_detected"]:
        status += " | COLLISION DANGER!"
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if data["collision_danger_detected"] else (0, 255, 0), 2)

    cv2.imshow("DriveMind AI - Vehicle Detection + TTC Test", frame)

    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()