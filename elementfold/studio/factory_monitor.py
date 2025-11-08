"""
studio/factory_monitor.py — The Listener 🎧 of ElementFold

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The FactoryMonitor sits beside the Factory and listens to its heart.
• It subscribes to the TelemetryBus and prints rhythm, phase, and mood.
• It never blocks; it only listens, translates, and paints.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict

from elementfold.core.telemetry import TelemetryBus


# ---------------------------------------------------------------------- #
# 🎨 Terminal color helpers (soft cross-platform)
# ---------------------------------------------------------------------- #
class _Color:
    RESET = "\033[0m"
    DIM = "\033[2m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BOLD = "\033[1m"


def _colorize(msg: str, color: str) -> str:
    return f"{color}{msg}{_Color.RESET}"


# ====================================================================== #
# 🎧 FactoryMonitor — listens to the telemetry bus and narrates
# ====================================================================== #
class FactoryMonitor:
    """
    Subscribe to a TelemetryBus and print live updates.

    Usage
    -----
    >>> bus = TelemetryBus()
    >>> monitor = FactoryMonitor(bus)
    >>> monitor.start()
    """

    def __init__(self, bus: TelemetryBus, *, interval: float = 0.2, quiet: bool = False) -> None:
        self.bus = bus
        self.interval = interval
        self.quiet = quiet
        self._thread: threading.Thread | None = None
        self._stop_flag = threading.Event()

        # subscribe immediately
        self.bus.subscribe(self._on_event)

        # internal buffers
        self._last_event: str = ""
        self._last_time: float = 0.0
        self._counter: int = 0

    # ------------------------------------------------------------------ #
    # 📡 Subscription callback
    # ------------------------------------------------------------------ #
    def _on_event(self, event: str, payload: Dict[str, Any], ts: float) -> None:
        """Handle incoming telemetry events."""
        self._counter += 1
        self._last_event = event
        self._last_time = ts

        if self.quiet:
            return

        # choose color & tone
        color = _Color.CYAN
        if "error" in event.lower() or "💥" in event:
            color = _Color.RED
        elif "stop" in event.lower() or "⛔" in event:
            color = _Color.MAGENTA
        elif "ledger" in event.lower():
            color = _Color.GREEN
        elif "sync" in event.lower() or "entangle" in event.lower():
            color = _Color.YELLOW

        # printable summary
        msg = f"{_Color.DIM}{time.strftime('%H:%M:%S')}{_Color.RESET} {event} {json.dumps(payload, ensure_ascii=False)}"
        print(_colorize(msg, color))

    # ------------------------------------------------------------------ #
    # ▶️ Threaded monitoring
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Launch background thread that keeps the monitor alive."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(_colorize("🎧 FactoryMonitor started — listening to the bus…", _Color.BOLD))

    def _loop(self) -> None:
        """Internal loop showing heartbeat summaries."""
        while not self._stop_flag.is_set():
            time.sleep(self.interval)
            if self.quiet:
                continue
            stats = self.bus.stats()
            msg = (
                f"{_Color.DIM}Δt={self.interval}s | "
                f"events={stats['emit_count']} | "
                f"subs={stats['subscribers']} | "
                f"history={stats['history']}{_Color.RESET}"
            )
            print(_colorize(msg, _Color.DIM))

    def stop(self) -> None:
        """Stop monitoring."""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print(_colorize("🛑 FactoryMonitor stopped.", _Color.BOLD))

    # ------------------------------------------------------------------ #
    # 🔍 Representation
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        return f"<FactoryMonitor events={self._counter} last='{self._last_event}'>"
