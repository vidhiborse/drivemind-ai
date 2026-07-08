"""
Standalone test script for Yawning Detection (MAR).
Press 'q' to quit. Try yawning (or opening mouth wide) to see MAR spike.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.mouth.yawn_detector import YawnDetector

detector = YawnDetector()
detector.load()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

print("Webcam started. Press 'q' to quit. Try opening your mouth wide!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    packet = detector.process(frame)
    data = packet.data

    if data.get("mouth_detected"):
        color = (0, 0, 255) if data["is_yawning"] else (0, 255, 0)
        text = f"MAR: {data['mar']} | Yawning: {data['is_yawning']}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        count_text = f"Yawns (last 2min): {data['yawn_count_last_2min']}"
        cv2.putText(frame, count_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        if data["fatigue_signal"]:
            cv2.putText(frame, "FATIGUE SIGNAL!", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("DriveMind AI - Yawn Detection Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()