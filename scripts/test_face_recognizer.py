"""
Standalone test script for Face Recognition.
Press 'q' to quit. Requires at least one enrolled driver profile
(run scripts/enroll_driver.py first).
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.face.face_recognizer import FaceRecognizer

detector = FaceRecognizer()
detector.load()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

print("Webcam started. Press 'q' to quit.")
print("(Recognition runs every 15 frames, so it may take a moment to update.)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    packet = detector.process(frame)
    data = packet.data

    if data.get("driver_identified"):
        if data["is_known_driver"]:
            text = f"Driver: {data['driver_name'].upper()} (known)"
            color = (0, 255, 0)
        else:
            text = "UNKNOWN DRIVER"
            color = (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    else:
        cv2.putText(frame, "Identifying...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("DriveMind AI - Face Recognition Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()