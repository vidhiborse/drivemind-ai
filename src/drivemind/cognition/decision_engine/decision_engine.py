"""
Decision Engine v1 — RULE-BASED (Phase 4, per project roadmap).
Combines aggregated perception features into a single risk assessment.

TODO (Phase 5): Replace with a trained ML-based Risk Engine once enough
logged decision data is available from this rule-based system.
"""

import time
from collections import deque
from typing import Dict, Any


EYES_CLOSED_ALERT_SECONDS = 2.0   # eyes closed continuously -> concern
LOOK_AWAY_ALERT_SECONDS = 3.0     # looking away continuously -> concern


class DecisionEngine:
    def __init__(self):
        self._eyes_closed_since = None
        self._looking_away_since = None

    def _check_drowsiness(self, state: Dict[str, Any]) -> Dict[str, Any]:
        eye_data = state.get("eye_state_detector", {})
        yawn_data = state.get("yawn_detector", {})

        eyes_closed = eye_data.get("eye_state") == "closed"
        now = time.time()

        if eyes_closed:
            if self._eyes_closed_since is None:
                self._eyes_closed_since = now
        else:
            self._eyes_closed_since = None

        closed_duration = (now - self._eyes_closed_since) if self._eyes_closed_since else 0.0
        microsleep = closed_duration >= EYES_CLOSED_ALERT_SECONDS
        yawning = yawn_data.get("is_yawning", False)
        fatigue_signal = yawn_data.get("fatigue_signal", False)

        return {
            "eyes_closed_duration": round(closed_duration, 1),
            "microsleep_detected": microsleep,
            "yawning": yawning,
            "yawn_fatigue_signal": fatigue_signal,
        }

    def _check_distraction(self, state: Dict[str, Any]) -> Dict[str, Any]:
        head_pose = state.get("head_pose_estimator", {})
        distraction = state.get("distraction_detector", {})

        direction = head_pose.get("direction", "forward")
        now = time.time()

        if direction != "forward":
            if self._looking_away_since is None:
                self._looking_away_since = now
        else:
            self._looking_away_since = None

        away_duration = (now - self._looking_away_since) if self._looking_away_since else 0.0
        sustained_look_away = away_duration >= LOOK_AWAY_ALERT_SECONDS

        return {
            "head_direction": direction,
            "look_away_duration": round(away_duration, 1),
            "sustained_look_away": sustained_look_away,
            "object_distraction": distraction.get("distraction_detected", False),
            "distraction_details": distraction.get("distractions", []),
        }

    def _check_compliance(self, state: Dict[str, Any]) -> Dict[str, Any]:
        seatbelt = state.get("seatbelt_detector", {})
        return {
            "seatbelt_detected": seatbelt.get("seatbelt_detected", False),
        }

    def decide(self, state: Dict[str, Any]) -> Dict[str, Any]:
        drowsiness = self._check_drowsiness(state)
        distraction = self._check_distraction(state)
        compliance = self._check_compliance(state)

        reasons = []
        risk_level = "LOW"

        if drowsiness["microsleep_detected"]:
            reasons.append("Eyes closed for extended period (possible microsleep)")
            risk_level = "CRITICAL"

        if drowsiness["yawn_fatigue_signal"]:
            reasons.append("Frequent yawning detected (fatigue building up)")
            risk_level = "HIGH" if risk_level != "CRITICAL" else risk_level

        if distraction["object_distraction"]:
            reasons.append("Distraction object detected (phone/cup)")
            risk_level = "HIGH" if risk_level not in ("CRITICAL",) else risk_level

        if distraction["sustained_look_away"]:
            reasons.append(f"Looking away from road ({distraction['head_direction']})")
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level

        if not compliance["seatbelt_detected"]:
            reasons.append("Seatbelt not detected (heuristic)")
            risk_level = "MEDIUM" if risk_level == "LOW" else risk_level

        if not reasons:
            reasons.append("No risk factors detected")

        return {
            "timestamp": state.get("timestamp"),
            "risk_level": risk_level,
            "reasons": reasons,
            "drowsiness": drowsiness,
            "distraction": distraction,
            "compliance": compliance,
        }