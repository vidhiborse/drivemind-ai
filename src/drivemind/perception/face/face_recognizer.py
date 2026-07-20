"""
Face Recognition module — identifies WHICH registered driver is present,
or flags an unknown driver. Uses DeepFace.find() to compare the live frame
against enrolled driver profile images in data/driver_profiles/.

Design note: DeepFace.find() re-scans the profiles folder each call, which
is fine for a small number of drivers (a handful) but would need a proper
vector database (e.g., FAISS) for a large fleet — noted as a future
optimization, not needed at this project's current scale.
"""

import os
from deepface import DeepFace

from drivemind.common.interfaces import Detector, FeaturePacket

PROFILES_DIR = "data/driver_profiles"
DISTANCE_THRESHOLD = 0.6  # lower = stricter match; DeepFace cosine distance default range
RECOGNIZE_EVERY_N_FRAMES = 15  # face recognition is expensive; don't run every frame


class FaceRecognizer(Detector):
    def __init__(self, profiles_dir: str = PROFILES_DIR,
                 recognize_every_n_frames: int = RECOGNIZE_EVERY_N_FRAMES):
        self.profiles_dir = profiles_dir
        self.recognize_every_n_frames = recognize_every_n_frames
        self._frame_count = 0
        self._last_result = {
            "driver_identified": False,
            "driver_name": None,
            "is_known_driver": False,
        }

    def load(self) -> None:
        if not os.path.isdir(self.profiles_dir) or not os.listdir(self.profiles_dir):
            print(f"[WARNING] No enrolled driver profiles found in {self.profiles_dir}. "
                  f"Run scripts/enroll_driver.py first.")

    def process(self, frame) -> FeaturePacket:
        self._frame_count += 1

        if self._frame_count % self.recognize_every_n_frames != 0:
            return FeaturePacket(
                source_module="face_recognizer",
                data=self._last_result,
                confidence=None,
            )

        try:
            results = DeepFace.find(
                img_path=frame,
                db_path=self.profiles_dir,
                enforce_detection=False,
                silent=True,
            )

            # DeepFace.find returns a list of DataFrames (one per detected face)
            if results and len(results) > 0 and not results[0].empty:
                best_match = results[0].iloc[0]
                identity_path = best_match["identity"]
                distance = float(best_match.get("distance", 1.0))

                driver_name = os.path.splitext(os.path.basename(identity_path))[0]
                is_known = distance <= DISTANCE_THRESHOLD
               
                self._last_result = {
                    "driver_identified": True,
                    "driver_name": driver_name if is_known else "unknown",
                    "is_known_driver": is_known,
                    "match_distance": round(distance, 3),
                }
            else:
                self._last_result = {
                    "driver_identified": True,
                    "driver_name": "unknown",
                    "is_known_driver": False,
                    "match_distance": None,
                }

        except Exception:
            # No face clear enough to recognize this frame — keep last known result
            pass

        return FeaturePacket(
            source_module="face_recognizer",
            data=self._last_result,
            confidence=self._last_result.get("match_distance"),
        )