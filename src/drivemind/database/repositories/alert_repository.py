"""
Repository for alert persistence.
"""

from drivemind.database.models import AlertLog


class AlertRepository:
    def __init__(self, session):
        self.session = session

    def log_alert(self, trip_id: int, timestamp: float, alert_type: str,
                   severity: str, message: str):
        entry = AlertLog(
            trip_id=trip_id,
            timestamp=timestamp,
            alert_type=alert_type,
            severity=severity,
            message=message,
        )
        self.session.add(entry)
        self.session.commit()