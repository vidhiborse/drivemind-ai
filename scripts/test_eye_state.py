"""
Standalone test script for Eye State Detection (EAR).
Press 'q' to quit. Try blinking to see EAR value and state change.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.eyes.ear_calculator import EyeStateDetector

detector = EyeStateDetector()
detector.load()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

print("Webcam started. Press 'q' to quit. Try blinking!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    packet = detector.process(frame)
    data = packet.data

    if data.get("eyes_detected"):
        color = (0, 0, 255) if data["eye_state"] == "closed" else (0, 255, 0)
        text = f"EAR: {data['avg_ear']} | State: {data['eye_state'].upper()}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("DriveMind AI - Eye State Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()