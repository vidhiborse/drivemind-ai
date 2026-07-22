"""
Headless Pipeline Runner — runs the full perception + decision pipeline
in a background thread, without any cv2.imshow() window (no display
available/needed on a backend server).

This is what the API's /session/start and /session/stop endpoints control.
The live webcam demo script (run_full_pipeline_demo.py) remains separate,
for local visual testing.
"""

import sys
import threading
import time

sys.path.insert(0, "src")

import cv2

from drivemind.perception.face.detector import FaceDetector
from drivemind.perception.eyes.ear_calculator import EyeStateDetector
from drivemind.perception.mouth.yawn_detector import YawnDetector
from drivemind.perception.head_pose.head_pose_estimator import HeadPoseEstimator
from drivemind.perception.distraction.object_detector import DistractionDetector
from drivemind.perception.seatbelt.seatbelt_detector import SeatbeltDetector
from drivemind.perception.emotion.emotion_detector import EmotionDetector
from drivemind.perception.face.face_recognizer import FaceRecognizer

from drivemind.cognition.decision_engine.feature_aggregator import FeatureAggregator
from drivemind.cognition.decision_engine.decision_engine import DecisionEngine
from drivemind.cognition.fatigue_predictor.fatigue_predictor import FatiguePredictor

from drivemind.database.models import init_db, get_session
from drivemind.database.repositories.decision_repository import DecisionRepository
from drivemind.database.repositories.alert_repository import AlertRepository
from drivemind.action.alert_engine.alert_engine import AlertEngine


class PipelineSession:
    """Represents one running (or stopped) headless pipeline session."""

    def __init__(self, trip_id: int):
        self.trip_id = trip_id
        self._thread = None
        self._stop_flag = threading.Event()
        self._latest_decision = None
        self._latest_fatigue = None
        self._is_running = False

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._is_running = True
        self._thread.start()

    def stop(self):
        self._stop_flag.set()
        self._is_running = False

    def get_status(self):
        return {
            "trip_id": self.trip_id,
            "is_running": self._is_running,
            "latest_decision": self._latest_decision,
            "latest_fatigue": self._latest_fatigue,
        }

    def _run(self):
        engine = init_db()
        db_session = get_session(engine)
        decision_repo = DecisionRepository(db_session)
        alert_repo = AlertRepository(db_session)
        alert_engine = AlertEngine(alert_repo, self.trip_id)

        face_detector = FaceDetector()
        eye_detector = EyeStateDetector()
        yawn_detector = YawnDetector()
        head_pose_detector = HeadPoseEstimator()
        distraction_detector = DistractionDetector()
        seatbelt_detector = SeatbeltDetector()
        emotion_detector = EmotionDetector()
        face_recognizer = FaceRecognizer()

        for module in [face_detector, eye_detector, yawn_detector,
                       head_pose_detector, distraction_detector, seatbelt_detector,
                       emotion_detector, face_recognizer]:
            module.load()

        aggregator = FeatureAggregator()
        decision_engine = DecisionEngine()
        fatigue_predictor = FatiguePredictor()
        fatigue_predictor.load()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._is_running = False
            return

        while not self._stop_flag.is_set():
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
                emotion_detector.process(frame),
                face_recognizer.process(frame),
            ]

            state = aggregator.update(packets)
            decision = decision_engine.decide(state)

            fatigue_input = {
                "timestamp": state.get("timestamp"),
                "avg_ear": state.get("eye_state_detector", {}).get("avg_ear", 0.3),
                "mar": state.get("yawn_detector", {}).get("mar", 0.2),
                "yawn_count_2min": state.get("yawn_detector", {}).get("yawn_count_last_2min", 0),
            }
            fatigue_result = fatigue_predictor.predict(fatigue_input)

            decision_repo.log_decision(self.trip_id, decision, state)
            alert_engine.process_decision(decision)

            self._latest_decision = decision
            self._latest_fatigue = fatigue_result

            time.sleep(0.03)  # roughly cap loop rate, avoid pegging CPU

        cap.release()
        decision_repo.end_trip(self.trip_id)
        self._is_running = False


# In-memory registry of active sessions (single-process; fine for local/dev use)
_active_sessions: dict = {}


def start_new_session() -> int:
    engine = init_db()
    db_session = get_session(engine)
    repo = DecisionRepository(db_session)
    trip_id = repo.create_trip()

    session = PipelineSession(trip_id)
    session.start()
    _active_sessions[trip_id] = session
    return trip_id


def stop_session(trip_id: int) -> bool:
    session = _active_sessions.get(trip_id)
    if not session:
        return False
    session.stop()
    return True


def get_session_status(trip_id: int):
    session = _active_sessions.get(trip_id)
    if not session:
        return None
    return session.get_status()