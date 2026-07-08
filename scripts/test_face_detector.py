"""
Standalone test script — run this to see face detection live on your webcam.
Press 'q' to quit.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.face.detector import FaceDetector

detector = FaceDetector()
detector.load()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam. Check camera permissions/index.")
    sys.exit(1)

print("Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    packet = detector.process(frame)

    for face in packet.data["faces"]:
        x, y = face["x"], face["y"]
        w, h = face["width"], face["height"]
        conf = face["confidence"]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f"{conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Faces detected: {packet.data['face_count']}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("DriveMind AI - Face Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()