"""
Standalone test script for Distraction Detection (YOLO).
Press 'q' to quit. Try holding your phone or a cup in front of the camera.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.distraction.object_detector import DistractionDetector

detector = DistractionDetector()
print("Loading YOLO model (first run downloads it automatically)...")
detector.load()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

print("Webcam started. Press 'q' to quit. Hold up your phone!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    packet = detector.process(frame)
    data = packet.data

    for det in data["distractions"]:
        box = det["box"]
        label = f"{det['object']} ({det['distraction_type']}) {det['confidence']}"
        cv2.rectangle(frame, (box["x1"], box["y1"]), (box["x2"], box["y2"]), (0, 0, 255), 2)
        cv2.putText(frame, label, (box["x1"], box["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    status = "DISTRACTION DETECTED!" if data["distraction_detected"] else "No distraction"
    color = (0, 0, 255) if data["distraction_detected"] else (0, 255, 0)
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("DriveMind AI - Distraction Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()