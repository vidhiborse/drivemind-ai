"""
DriveMind AI — Full Pipeline Integration Demo (Phase 4 Milestone)
Runs all driver-facing perception modules together on a single webcam feed,
aggregates their outputs, and shows a live risk decision.

Press 'q' to quit.

NOTE: Loads Face Landmarker model twice internally (once for eyes, once for
mouth, once for head pose) — this is intentional for Phase 4 clarity and will
be optimized in the MLOps/performance-optimization phase (Phase 7) to share
a single Face Landmarker instance across modules.
"""

import sys
sys.path.insert(0, "src")

import cv2

from drivemind.perception.face.detector import FaceDetector
from drivemind.perception.eyes.ear_calculator import EyeStateDetector
from drivemind.perception.mouth.yawn_detector import YawnDetector
from drivemind.perception.head_pose.head_pose_estimator import HeadPoseEstimator
from drivemind.perception.distraction.object_detector import DistractionDetector
from drivemind.perception.seatbelt.seatbelt_detector import SeatbeltDetector

from drivemind.cognition.decision_engine.feature_aggregator import FeatureAggregator
from drivemind.cognition.decision_engine.decision_engine import DecisionEngine

print("Loading all perception modules... this may take a moment.")

face_detector = FaceDetector()
eye_detector = EyeStateDetector()
yawn_detector = YawnDetector()
head_pose_detector = HeadPoseEstimator()
distraction_detector = DistractionDetector()
seatbelt_detector = SeatbeltDetector()

for module in [face_detector, eye_detector, yawn_detector,
               head_pose_detector, distraction_detector, seatbelt_detector]:
    module.load()

aggregator = FeatureAggregator()
decision_engine = DecisionEngine()

print("All modules loaded. Starting webcam...")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    sys.exit(1)

RISK_COLORS = {
    "LOW": (0, 255, 0),
    "MEDIUM": (0, 255, 255),
    "HIGH": (0, 128, 255),
    "CRITICAL": (0, 0, 255),
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    packets = [
        face_detector.process(frame),
        eye_detector.process(frame),
        yawn_detector.process(frame),
        head_pose_detector.process(frame),
        distraction_detector.process(frame),
        seatbelt_detector.process(frame),
    ]

    state = aggregator.update(packets)
    decision = decision_engine.decide(state)

    risk = decision["risk_level"]
    color = RISK_COLORS.get(risk, (255, 255, 255))

    cv2.putText(frame, f"RISK: {risk}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    y_offset = 60
    for reason in decision["reasons"][:4]:  # show up to 4 reasons
        cv2.putText(frame, f"- {reason}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        y_offset += 25

    cv2.imshow("DriveMind AI - Full Pipeline Demo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Session ended.")