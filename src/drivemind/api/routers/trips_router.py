"""
Trip-related API endpoints — read-only access to logged trips, decisions,
and alerts stored in the database by the pipeline.
"""

from fastapi import APIRouter, HTTPException
from typing import List

from drivemind.database.models import init_db, get_session, Trip, DecisionLog, AlertLog
from drivemind.api.schemas.trip_schemas import TripSummary, DecisionEntry, AlertEntry

router = APIRouter(prefix="/api/v1/trips", tags=["trips"])


def get_db_session():
    engine = init_db()
    return get_session(engine)


@router.get("/", response_model=List[TripSummary])
def list_trips():
    """List all recorded trips."""
    session = get_db_session()
    trips = session.query(Trip).order_by(Trip.id.desc()).all()
    return trips


@router.get("/{trip_id}", response_model=TripSummary)
def get_trip(trip_id: int):
    """Get details for a specific trip."""
    session = get_db_session()
    trip = session.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")
    return trip


@router.get("/{trip_id}/risk-history", response_model=List[DecisionEntry])
def get_trip_risk_history(trip_id: int):
    """Get all logged risk decisions for a trip, in chronological order."""
    session = get_db_session()
    trip = session.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

    decisions = (
        session.query(DecisionLog)
        .filter(DecisionLog.trip_id == trip_id)
        .order_by(DecisionLog.timestamp.asc())
        .all()
    )
    return decisions


@router.get("/{trip_id}/alerts", response_model=List[AlertEntry])
def get_trip_alerts(trip_id: int):
    """Get all alerts triggered during a trip."""
    session = get_db_session()
    trip = session.get(Trip, trip_id)
    if not trip:
        raise HTTPException(status_code=404, detail=f"Trip {trip_id} not found")

    alerts = (
        session.query(AlertLog)
        .filter(AlertLog.trip_id == trip_id)
        .order_by(AlertLog.timestamp.asc())
        .all()
    )
    return alerts