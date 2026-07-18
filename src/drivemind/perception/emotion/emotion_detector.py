"""
Emotion Detection using DeepFace.
Analyzes facial expression to classify emotion (happy, sad, angry, neutral, etc.).

Design note: emotion doesn't change frame-to-frame like eye state does, so
this module only runs full analysis every N frames (frame-skipping) to save
compute — matching the "Performance Optimization" strategy from the architecture doc.
"""

from deepface import DeepFace

from drivemind.common.interfaces import Detector, FeaturePacket

ANALYZE_EVERY_N_FRAMES = 10


class EmotionDetector(Detector):
    def __init__(self, analyze_every_n_frames: int = ANALYZE_EVERY_N_FRAMES):
        self.analyze_every_n_frames = analyze_every_n_frames
        self._frame_count = 0
        self._last_result = {
            "emotion_detected": False,
            "dominant_emotion": None,
            "emotion_scores": {},
        }

    def load(self) -> None:
        # DeepFace loads its models lazily on first analyze() call.
        # No explicit load step needed here, but the interface requires it.
        pass

    def process(self, frame) -> FeaturePacket:
        self._frame_count += 1

        # Only run the (relatively expensive) emotion analysis every N frames
        if self._frame_count % self.analyze_every_n_frames != 0:
            return FeaturePacket(
                source_module="emotion_detector",
                data=self._last_result,
                confidence=None,
            )

        try:
            analysis = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False,  # don't crash if face isn't perfectly aligned
                silent=True,
            )
            # DeepFace returns a list (one entry per detected face); take the first
            result = analysis[0] if isinstance(analysis, list) else analysis

            dominant_emotion = result["dominant_emotion"]
            emotion_scores = {k: round(float(v), 1) for k, v in result["emotion"].items()}

            self._last_result = {
                "emotion_detected": True,
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores,
            }

        except Exception:
            # Face not clear enough for emotion analysis this frame — keep last known result
            pass

        return FeaturePacket(
            source_module="emotion_detector",
            data=self._last_result,
            confidence=self._last_result["emotion_scores"].get(
                self._last_result["dominant_emotion"], None
            ) if self._last_result["emotion_detected"] else None,
        )