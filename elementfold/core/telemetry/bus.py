"""
core/telemetry/bus.py — Non-blocking Telemetry Bus 📡
──────────────────────────────────────────────────────────────────────
Purpose
  • Publish/subscribe system for all ElementFold components.
  • Non-blocking: publishers never hang.
  • Safe subscribe/unsubscribe even after close().
  • Graceful shutdown via close(); drains & clears queues.
  • Works in idle mode (no background churn).
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ====================================================================== #
# 🧩 TelemetryBus
# ====================================================================== #
@dataclass
class TelemetryBus:
    """
    Thread-safe, non-blocking event bus.
    """

    name: str = "default"
    max_queue: int = 1024
    _subscribers: Dict[str, Callable[[str, Dict[str, Any]], None]] = field(default_factory=dict)
    _queue: "queue.Queue[tuple[str, Dict[str, Any]]]" = field(default_factory=lambda: queue.Queue(maxsize=1024))
    _thread: Optional[threading.Thread] = None
    _stop_flag: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _closed: bool = False
    _published_count: int = 0

    # ------------------------------------------------------------------ #
    # ▶️  Start & Stop
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start dispatch loop (idempotent)."""
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            if self._closed:
                raise RuntimeError("[bus] cannot start a closed bus")
            self._stop_flag.clear()
            self._thread = threading.Thread(target=self._dispatch_loop, daemon=True)
            self._thread.start()
            print(f"[bus:{self.name}] started.")

    def close(self) -> None:
        """Stop the bus and drain queue."""
        with self._lock:
            if self._closed:
                return
            self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        # Drain queue without blocking
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        with self._lock:
            self._subscribers.clear()
            self._closed = True
        print(f"[bus:{self.name}] closed after {self._published_count} events.")

    # ------------------------------------------------------------------ #
    # 📨  Subscribe / Unsubscribe
    # ------------------------------------------------------------------ #
    def subscribe(self, name: str, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Register a subscriber callback."""
        if self._closed:
            print(f"[bus:{self.name}] subscribe() ignored — closed.")
            return
        with self._lock:
            self._subscribers[name] = callback
        print(f"[bus:{self.name}] new subscriber: {name}")

    def unsubscribe(self, name: str) -> None:
        """Remove a subscriber safely."""
        with self._lock:
            if name in self._subscribers:
                del self._subscribers[name]
                print(f"[bus:{self.name}] unsubscribed: {name}")

    # ------------------------------------------------------------------ #
    # 📢  Publish
    # ------------------------------------------------------------------ #
    def publish(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        """Non-blocking publish."""
        if self._closed:
            return
        payload = payload or {}
        try:
            self._queue.put_nowait((event, payload))
            self._published_count += 1
        except queue.Full:
            # Drop oldest item to make room
            try:
                _ = self._queue.get_nowait()
                self._queue.put_nowait((event, payload))
            except Exception:
                pass  # if even that fails, silently skip
        # Lazy-start dispatcher
        if not (self._thread and self._thread.is_alive()):
            self.start()

    # ------------------------------------------------------------------ #
    # 🔁  Dispatcher loop
    # ------------------------------------------------------------------ #
    def _dispatch_loop(self) -> None:
        """Continuously deliver events to subscribers."""
        while not self._stop_flag.is_set():
            try:
                event, payload = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            with self._lock:
                subs = list(self._subscribers.items())
            for name, cb in subs:
                try:
                    cb(event, payload)
                except Exception as exc:
                    # Print once per error to avoid noisy floods
                    print(f"[bus:{self.name}] subscriber '{name}' error: {exc}")

    # ------------------------------------------------------------------ #
    # 🧭  Diagnostics
    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        """Return bus diagnostics."""
        with self._lock:
            return {
                "name": self.name,
                "closed": self._closed,
                "subscribers": list(self._subscribers.keys()),
                "queued": self._queue.qsize(),
                "published": self._published_count,
            }

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"<TelemetryBus {self.name} subs={len(self._subscribers)} {state}>"
