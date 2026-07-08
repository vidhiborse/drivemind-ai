"""
Standalone test script for Seatbelt Detection (rule-based placeholder).
Press 'q' to quit. Results will be noisy/approximate — this is expected,
see the LIMITATION note in seatbelt_detector.py.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.seatbelt.seatbelt_detector import SeatbeltDetector

detector = SeatbeltDetector()
detector.load()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

print("Webcam started. Press 'q' to quit.")
print("NOTE: This is a rule-based placeholder — expect rough/noisy results.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    packet = detector.process(frame)
    data = packet.data

    status = "SEATBELT DETECTED (heuristic)" if data["seatbelt_detected"] else "No seatbelt (heuristic)"
    color = (0, 255, 0) if data["seatbelt_detected"] else (0, 0, 255)

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Diagonal lines found: {data['diagonal_line_count']}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("DriveMind AI - Seatbelt Detection Test (placeholder)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()