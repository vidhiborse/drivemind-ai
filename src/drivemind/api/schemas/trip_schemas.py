"""
Pydantic schemas define the shape of data going in/out of the API.
FastAPI uses these for automatic validation and documentation.
"""

from pydantic import BaseModel
from typing import Optional, List, Any
from datetime import datetime


class TripSummary(BaseModel):
    id: int
    start_time: datetime
    end_time: Optional[datetime] = None

    class Config:
        from_attributes = True


class DecisionEntry(BaseModel):
    id: int
    timestamp: float
    risk_level: str
    reasons: List[str]

    class Config:
        from_attributes = True


class AlertEntry(BaseModel):
    id: int
    timestamp: float
    alert_type: str
    severity: str
    message: str

    class Config:
        from_attributes = True