"""
Feature Aggregator — collects FeaturePackets from all perception modules
into a single, timestamped state dictionary that the Decision Engine can reason over.
"""

import time
from typing import Dict, Any, List

from drivemind.common.interfaces import FeaturePacket


class FeatureAggregator:
    def __init__(self):
        self._latest_state: Dict[str, Any] = {}

    def update(self, packets: List[FeaturePacket]) -> Dict[str, Any]:
        """Merge multiple FeaturePackets into one state snapshot."""
        state = {"timestamp": time.time()}
        for packet in packets:
            state[packet.source_module] = packet.data
        self._latest_state = state
        return state

    def get_latest_state(self) -> Dict[str, Any]:
        return self._latest_state