"""
Factory Monitor — continuously observes Factory telemetry and reports
aggregate coherence state to the Studio UI.
"""

from __future__ import annotations
import time
from elementfold.core.control.factory import Factory


class FactoryMonitor:
    """Periodic monitor for Factory status."""

    def __init__(self, factory: Factory, interval: float = 1.0) -> None:
        self.factory = factory
        self.interval = interval
        self.running = False

    def start(self) -> None:
        """Begin monitoring loop."""
        self.running = True
        while self.running:
            snapshot = self.factory.snapshot()
            print(f"[monitor] {len(snapshot)} cores synchronized. δ★={self.factory.synchronizer.delta_star:.6f}")
            time.sleep(self.interval)

    def stop(self) -> None:
        """Stop monitoring loop."""
        self.running = False
