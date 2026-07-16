"""
Repository pattern for decision log persistence.
Keeps raw SQL/ORM details out of the cognition/pipeline layer.
"""

from drivemind.database.models import Trip, DecisionLog


class DecisionRepository:
    def __init__(self, session):
        self.session = session

    def create_trip(self) -> int:
        trip = Trip()
        self.session.add(trip)
        self.session.commit()
        return trip.id

    def end_trip(self, trip_id: int):
        import datetime
        trip = self.session.get(Trip, trip_id)
        if trip:
            trip.end_time = datetime.datetime.utcnow()
            self.session.commit()

    def log_decision(self, trip_id: int, decision: dict, raw_state: dict):
        entry = DecisionLog(
            trip_id=trip_id,
            timestamp=decision["timestamp"] or 0.0,
            risk_level=decision["risk_level"],
            reasons=decision["reasons"],
            raw_state=raw_state,
        )
        self.session.add(entry)
        self.session.commit()