"""
core/telemetry/bus.py — The Voice 📡 of ElementFold

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The Bus carries every heartbeat message through the system.
• It whispers to Studio dashboards, writes to logs, and mirrors to recorders.
• Nothing blocks: telemetry is soft real-time — never stops physics.
• Each message can be spoken, stored, or silently observed.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple


# ====================================================================== #
# 📨 TelemetryMessage — atomic unit of the Bus
# ====================================================================== #
@dataclass
class TelemetryMessage:
    """Immutable telemetry event with timestamp and payload."""
    event: str
    payload: Dict[str, Any]
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "timestamp": self.timestamp,
            "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.timestamp)),
            "payload": self.payload,
        }

    def to_json(self) -> str:
        """Serialize as JSON line."""
        return json.dumps(self.to_dict(), ensure_ascii=False)


# ====================================================================== #
# 📡 TelemetryBus — broadcast hub for all runtime / ledger events
# ====================================================================== #
class TelemetryBus:
    """
    Thread-safe pub/sub bus for lightweight telemetry.

    Subscribers are callables:  fn(event: str, payload: dict, timestamp: float) -> None
    """

    def __init__(self, *, mirror_stdout: bool = True, keep_last: int = 100) -> None:
        self._subs: List[Callable[[str, Dict[str, Any], float], None]] = []
        self._mirror_stdout = mirror_stdout
        self._lock = RLock()
        self._keep_last = max(1, keep_last)
        self._history: List[TelemetryMessage] = []
        self._emit_count: int = 0

    # ------------------------------------------------------------------ #
    # 🧭 Subscription control
    # ------------------------------------------------------------------ #
    def subscribe(self, fn: Callable[[str, Dict[str, Any], float], None]) -> None:
        """Add a subscriber to the bus."""
        with self._lock:
            if fn not in self._subs:
                self._subs.append(fn)
                self._safe_print(f"📡 new subscriber: {fn.__name__}")

    def unsubscribe(self, fn: Callable[[str, Dict[str, Any], float], None]) -> None:
        """Remove a subscriber from the bus."""
        with self._lock:
            try:
                self._subs.remove(fn)
                self._safe_print(f"🕊️ unsubscribed: {fn.__name__}")
            except ValueError:
                self._safe_print(f"⚠️ tried to remove unknown subscriber: {fn}")

    # ------------------------------------------------------------------ #
    # 🗞️ Emit events
    # ------------------------------------------------------------------ #
    def emit(self, event: str, **payload: Any) -> None:
        """
        Publish an event to all subscribers.
        Never blocks or raises; errors are logged locally.
        """
        now = time.time()
        msg = TelemetryMessage(event, payload, now)

        # store to history (ring buffer)
        with self._lock:
            self._emit_count += 1
            self._history.append(msg)
            if len(self._history) > self._keep_last:
                self._history.pop(0)
            subs_snapshot = list(self._subs)

        # mirror to stdout for visibility
        if self._mirror_stdout:
            self._safe_print(f"{msg.to_json()}")

        # dispatch to subscribers
        for fn in subs_snapshot:
            try:
                fn(event, payload, now)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                # We catch individual subscriber errors but keep the bus alive.
                self._safe_print(f"[bus] subscriber {fn.__name__} error: {exc}")

    # ------------------------------------------------------------------ #
    # 📖 History and inspection
    # ------------------------------------------------------------------ #
    def last(self, n: int = 1) -> List[TelemetryMessage]:
        """Return the last n messages."""
        with self._lock:
            return list(self._history[-n:])

    def stats(self) -> Dict[str, Any]:
        """Return quick metrics about bus usage."""
        with self._lock:
            return {
                "emit_count": self._emit_count,
                "subscribers": len(self._subs),
                "history": len(self._history),
            }

    # ------------------------------------------------------------------ #
    # 🪶 Utilities
    # ------------------------------------------------------------------ #
    def _safe_print(self, msg: str) -> None:
        """Write to stdout safely, never crashing the bus."""
        try:
            sys.stdout.write(msg + "\n")
            sys.stdout.flush()
        except (OSError, IOError) as exc:
            # console might be closed; ignore
            sys.stderr.write(f"[bus-print-error] {exc}\n")
        except Exception as exc:
            sys.stderr.write(f"[bus-unknown-error] {exc}\n")

    # ------------------------------------------------------------------ #
    # 🧹 Maintenance
    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        """Erase stored history (does not affect subscribers)."""
        with self._lock:
            self._history.clear()
            self._emit_count = 0
        self._safe_print("🧹 telemetry history cleared")

    # ------------------------------------------------------------------ #
    # 🕊️ Context manager
    # ------------------------------------------------------------------ #
    def __enter__(self) -> "TelemetryBus":
        self._safe_print("📡 TelemetryBus online")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._safe_print("📡 TelemetryBus offline")
        self.clear()

    # ------------------------------------------------------------------ #
    # 🔍 Representation
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"<TelemetryBus subs={s['subscribers']} "
            f"history={s['history']} emits={s['emit_count']}>"
        )
