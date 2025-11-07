# ElementFold · server_brain_loop.py
# ============================================================
# Auto-loop Brain Controller for the ElementFold server
# ------------------------------------------------------------
# Extends the /brain/step logic with:
#   POST /brain/loop/start  → run background thread (LocalBrain.step)
#   POST /brain/loop/stop   → stop background loop
#
# The loop runs safely in its own thread, calling LocalBrain.step()
# every few seconds, until stopped or the process exits.
# ============================================================

from __future__ import annotations
import threading, time, traceback
from typing import Optional

from .experience.local_brain import LocalBrain
from .server_api import to_json, pack_error
from .server_brain import _brain, _BRAIN_LOCK  # reuse brain singleton

# Shared loop state
_LOOP_THREAD: Optional[threading.Thread] = None
_LOOP_STOP = threading.Event()
_LOOP_LOCK = threading.Lock()


# ============================================================
# Background worker
# ============================================================

def _loop_worker(brain: LocalBrain, interval: float = 5.0) -> None:
    """Call brain.step() repeatedly until _LOOP_STOP is set."""
    while not _LOOP_STOP.is_set():
        try:
            decision = brain.step()
            brain.engine.apply_control(**(decision.get("params") or {}))
        except Exception as e:
            print(f"[BrainLoop] error: {e}\n{traceback.format_exc()}")
        # interval sleep with early stop check
        for _ in range(int(interval * 10)):
            if _LOOP_STOP.is_set():
                break
            time.sleep(0.1)
    print("[BrainLoop] stopped.")


# ============================================================
# Handler mixin
# ============================================================

class BrainLoopHandlerMixin:
    """
    Mixin for BaseHTTPRequestHandler to add /brain/loop control routes.
    Requires _engine() to be defined (like in server.Handler).
    """

    def _handle_brain_loop_start(self, data: dict) -> None:
        global _LOOP_THREAD
        with _LOOP_LOCK:
            if _LOOP_THREAD and _LOOP_THREAD.is_alive():
                return self._send_json(200, {"status": "running"})
            eng = self._engine() if hasattr(self, "_engine") else None
            if eng is None:
                return self._send_json(500, pack_error("no_engine", "Engine not available."))

            brain = _brain(eng)
            _LOOP_STOP.clear()
            _LOOP_THREAD = threading.Thread(target=_loop_worker, args=(brain,), daemon=True)
            _LOOP_THREAD.start()
        return self._send_json(200, {"status": "started"})

    def _handle_brain_loop_stop(self, data: dict) -> None:
        global _LOOP_THREAD
        _LOOP_STOP.set()
        with _LOOP_LOCK:
            if _LOOP_THREAD and _LOOP_THREAD.is_alive():
                _LOOP_THREAD.join(timeout=2.0)
            _LOOP_THREAD = None
        return self._send_json(200, {"status": "stopped"})
