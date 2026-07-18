"""
Standalone test script for Emotion Detection.
Press 'q' to quit. Try different facial expressions (smile, frown, neutral).
First run will download DeepFace's emotion model — may take a minute.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.emotion.emotion_detector import EmotionDetector

detector = EmotionDetector()
detector.load()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

print("Webcam started. Press 'q' to quit. Try smiling, frowning, etc.")
print("(First analysis may take a few seconds to download the model.)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    packet = detector.process(frame)
    data = packet.data

    if data.get("emotion_detected"):
        text = f"Emotion: {data['dominant_emotion'].upper()}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Analyzing...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    cv2.imshow("DriveMind AI - Emotion Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()