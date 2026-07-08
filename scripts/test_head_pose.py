"""
Standalone test script for Head Pose Estimation.
Press 'q' to quit. Try turning your head left/right/up/down.
"""

import sys
sys.path.insert(0, "src")

import cv2
from drivemind.perception.head_pose.head_pose_estimator import HeadPoseEstimator

detector = HeadPoseEstimator()
detector.load()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

print("Webcam started. Press 'q' to quit. Turn your head around!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    packet = detector.process(frame)
    data = packet.data

    if data.get("pose_solved"):
        text1 = f"Yaw: {data['yaw']} Pitch: {data['pitch']} Roll: {data['roll']}"
        text2 = f"Direction: {data['direction'].upper()}"
        color = (0, 255, 0) if data["direction"] == "forward" else (0, 0, 255)

        cv2.putText(frame, text1, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, text2, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        cv2.putText(frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("DriveMind AI - Head Pose Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()