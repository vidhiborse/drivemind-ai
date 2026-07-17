"""
Alert Engine — reacts to Decision Engine output.
Prevents alert spam using a per-alert-type cooldown, plays a beep sound
for CRITICAL alerts, and logs every alert to the database.
"""

import time
import winsound  # Windows-only; beep sound for critical alerts

COOLDOWN_SECONDS = {
    "LOW": 999999,       # effectively never alert for LOW
    "MEDIUM": 8.0,
    "HIGH": 5.0,
    "CRITICAL": 2.0,
}


class AlertEngine:
    def __init__(self, alert_repository, trip_id: int):
        self.alert_repository = alert_repository
        self.trip_id = trip_id
        self._last_alert_time: dict = {}  # risk_level -> last alert timestamp

    def _should_alert(self, risk_level: str) -> bool:
        if risk_level == "LOW":
            return False

        now = time.time()
        last_time = self._last_alert_time.get(risk_level, 0)
        cooldown = COOLDOWN_SECONDS.get(risk_level, 5.0)

        if now - last_time >= cooldown:
            self._last_alert_time[risk_level] = now
            return True
        return False

    def process_decision(self, decision: dict):
        risk_level = decision["risk_level"]

        if not self._should_alert(risk_level):
            return None

        message = "; ".join(decision["reasons"])
        timestamp = decision["timestamp"] or time.time()

        self.alert_repository.log_alert(
            trip_id=self.trip_id,
            timestamp=timestamp,
            alert_type="safety_alert",
            severity=risk_level,
            message=message,
        )

        if risk_level == "CRITICAL":
            try:
                winsound.Beep(1000, 300)  # 1000 Hz, 300 ms
            except RuntimeError:
                pass  # in case audio device isn't available

        return {"severity": risk_level, "message": message}