"""
Session control endpoints — start/stop a headless live pipeline session,
and check its current live status/decision.
"""

from fastapi import APIRouter, HTTPException

from drivemind.api.pipeline_runner import start_new_session, stop_session, get_session_status

router = APIRouter(prefix="/api/v1/session", tags=["session"])


@router.post("/start")
def start_session():
    """Start a new headless pipeline session (opens the local webcam)."""
    trip_id = start_new_session()
    return {"message": "Session started", "trip_id": trip_id}


@router.post("/{trip_id}/stop")
def stop_session_endpoint(trip_id: int):
    """Stop a running session."""
    success = stop_session(trip_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"No active session found for trip_id {trip_id}")
    return {"message": f"Session {trip_id} stop requested"}


@router.get("/{trip_id}/status")
def session_status(trip_id: int):
    """Get the latest status, risk decision, and fatigue estimate for a running session."""
    status = get_session_status(trip_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"No session found for trip_id {trip_id}")
    return status