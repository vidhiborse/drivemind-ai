"""
Core abstract interfaces for DriveMind AI.
Every perception model (face, eyes, objects, etc.) implements Detector.
Every ML predictor (risk, fatigue) implements Predictor.
This is what makes modules swappable without touching the rest of the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time


@dataclass
class FeaturePacket:
    """Standard output format every perception module must produce."""
    source_module: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None


class Detector(ABC):
    """Base interface for any perception module (face, eyes, objects, lanes, etc.)."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights / initialize the backend (PyTorch, ONNX, TensorRT)."""
        raise NotImplementedError

    @abstractmethod
    def process(self, frame) -> FeaturePacket:
        """Run inference on a single frame and return a FeaturePacket."""
        raise NotImplementedError


class Predictor(ABC):
    """Base interface for ML models that reason over aggregated features
    (Risk Engine, Fatigue Predictor) rather than raw frames."""

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, feature_vector: Dict[str, Any]) -> Dict[str, Any]:
        """Return a prediction dict, e.g. {'risk_score': 72, 'reason': [...]}"""
        raise NotImplementedError