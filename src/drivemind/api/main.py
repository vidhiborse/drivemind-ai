"""
DriveMind AI — FastAPI Backend Entry Point.

Exposes the system's logged trip/decision/alert data over a REST API,
so any frontend (dashboard, mobile app, or another service) can consume it
without touching the pipeline code directly.

Run with: uvicorn drivemind.api.main:app --reload
"""

from fastapi import FastAPI

from drivemind.api.routers import trips_router

from drivemind.api.routers import trips_router, session_router

app = FastAPI(
    title="DriveMind AI API",
    description="Intelligent Driver Safety, Risk Prediction & Driver Intelligence Platform",
    version="0.1.0",
)

app.include_router(trips_router.router)
app.include_router(session_router.router)


@app.get("/health", tags=["system"])
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "service": "DriveMind AI API"}


@app.get("/", tags=["system"])       
def root():
    return {"message": "DriveMind AI API is running. Visit /docs for API documentation."}