"""
elementfold.core.telemetry — voices, traces, and whispers 📡

──────────────────────────────────────────────────────────────────────
This package carries ElementFold’s live telemetry streams.
It defines how Runtime and Ledger speak to the Studio,
and how the Factory listens to the heartbeat of each core.
──────────────────────────────────────────────────────────────────────
Exports
--------
TelemetryBus  → main event bus (publish/subscribe, non-blocking)
TelemetryMessage → structured telemetry record
"""

# elementfold/core/telemetry/__init__.py
from .bus import TelemetryBus
from .metrics import TelemetryEvent, make_event, validate_event, normalize_event


__all__ = ["TelemetryBus", "TelemetryMessage"]
