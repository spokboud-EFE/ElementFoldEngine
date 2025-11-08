"""
core/telemetry/metrics.py — Telemetry Event Schema 📏
──────────────────────────────────────────────────────────────────────
Purpose
  • Define a minimal, explicit schema for all telemetry events.
  • Provide helpers to validate and normalize event dictionaries.
  • Guarantee that panels, brains, and recorders see consistent keys.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict


# ------------------------------------------------------------------ #
# 🧱 Base schema for telemetry events
# ------------------------------------------------------------------ #
@dataclass
class TelemetryEvent:
    """
    Standard telemetry record shared across the system.
    """

    event: str
    payload: Dict[str, Any]
    timestamp: float = None
    iso: str = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.iso is None:
            self.iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ #
# 🧮 Utility functions
# ------------------------------------------------------------------ #
def make_event(event: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a standardized telemetry event dictionary.
    """
    return TelemetryEvent(event=event, payload=payload).to_dict()


def validate_event(data: Dict[str, Any]) -> bool:
    """
    Validate that an object conforms to the TelemetryEvent schema.
    """
    required = {"event", "payload", "timestamp", "iso"}
    if not all(k in data for k in required):
        return False
    if not isinstance(data["event"], str):
        return False
    if not isinstance(data["payload"], dict):
        return False
    if not isinstance(data["timestamp"], (float, int)):
        return False
    if not isinstance(data["iso"], str):
        return False
    return True


def normalize_event(event: Any) -> Dict[str, Any]:
    """
    Convert arbitrary telemetry data into a proper event dict.
    """
    if isinstance(event, TelemetryEvent):
        return event.to_dict()
    if isinstance(event, dict) and validate_event(event):
        return event
    # try to infer from minimal keys
    if isinstance(event, dict) and "event" in event:
        return make_event(event["event"], event.get("payload", {}))
    # fallback: wrap as generic notice
    return make_event("⚠️ unknown", {"data": str(event)})
