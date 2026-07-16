"""
SQLAlchemy ORM models for DriveMind AI.
Simplified version of the full schema (Section 5 of architecture doc) —
starting with just what's needed to log decisions from the current pipeline.
More tables (drivers, driver_profiles, performance_metrics) will be added
as later phases need them.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.orm import declarative_base, sessionmaker
import datetime

Base = declarative_base()


class Trip(Base):
    __tablename__ = "trips"

    id = Column(Integer, primary_key=True, autoincrement=True)
    start_time = Column(DateTime, default=datetime.datetime.utcnow)
    end_time = Column(DateTime, nullable=True)


class DecisionLog(Base):
    __tablename__ = "decision_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trip_id = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    reasons = Column(JSON, nullable=False)
    raw_state = Column(JSON, nullable=False)  # full aggregated feature state


def get_engine(db_url: str = "sqlite:///drivemind.db"):
    return create_engine(db_url, echo=False)


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()


def init_db(db_url: str = "sqlite:///drivemind.db"):
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)
    return engine