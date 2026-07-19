"""
Fatigue Predictor — LIMITATION-DOCUMENTED VERSION.

TODO (Phase 7 / production): Replace with an actual trained LSTM once
long, continuous trip logs are available (30-60+ minute sessions across
multiple real drives). LSTMs need sequential depth that short 20-30 second
test clips (used for the Risk Engine in Module 10) cannot provide.

Current implementation: a rolling-window trend estimator. It tracks recent
EAR/MAR/yawn-count history and uses linear regression slope to estimate
whether fatigue indicators are worsening, then projects a rough
"probability of fatigue within the next N minutes" from that trend.
This is a reasonable, honest placeholder — not a trained predictive model.
"""

import time
from collections import deque
import numpy as np

from drivemind.common.interfaces import Predictor

WINDOW_SECONDS = 60
PROJECTION_MINUTES = 10


class FatiguePredictor(Predictor):
    def __init__(self, window_seconds: int = WINDOW_SECONDS):
        self.window_seconds = window_seconds
        self._history = deque()  # list of (timestamp, avg_ear, mar, yawn_count)

    def load(self) -> None:
        # No model weights to load for this trend-based version
        pass

    def _update_history(self, timestamp, avg_ear, mar, yawn_count):
        self._history.append((timestamp, avg_ear, mar, yawn_count))
        while self._history and timestamp - self._history[0][0] > self.window_seconds:
            self._history.popleft()

    def _compute_slope(self, values):
        if len(values) < 3:
            return 0.0
        x = np.arange(len(values))
        y = np.array(values)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)

    def predict(self, feature_vector: dict) -> dict:
        timestamp = feature_vector.get("timestamp", time.time())
        avg_ear = feature_vector.get("avg_ear", 0.3)
        mar = feature_vector.get("mar", 0.2)
        yawn_count = feature_vector.get("yawn_count_2min", 0)

        self._update_history(timestamp, avg_ear, mar, yawn_count)

        if len(self._history) < 5:
            return {
                "fatigue_probability_10min": None,
                "trend": "insufficient_data",
                "method": "rolling_window_trend_placeholder",
            }

        ear_values = [h[1] for h in self._history]
        yawn_values = [h[3] for h in self._history]

        ear_slope = self._compute_slope(ear_values)   # negative slope = EAR dropping = worsening
        yawn_slope = self._compute_slope(yawn_values)  # positive slope = more yawning = worsening

        # Simple heuristic scoring: combine the two slopes into a 0-1 probability.
        # Not a calibrated probability from a trained model — an approximation.
        worsening_score = max(0.0, -ear_slope * 5) + max(0.0, yawn_slope * 2)
        fatigue_probability = min(1.0, worsening_score)

        trend = "worsening" if fatigue_probability > 0.3 else "stable"

        return {
            "fatigue_probability_10min": round(fatigue_probability, 2),
            "trend": trend,
            "ear_slope": round(ear_slope, 4),
            "yawn_slope": round(yawn_slope, 4),
            "method": "rolling_window_trend_placeholder",
        }