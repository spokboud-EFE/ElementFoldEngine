# ElementFold · server_brain_loop.py 
# ============================================================
# Brain auto-loop controller + unified /brain/loop/status
# ------------------------------------------------------------
# Adds:
#   POST /brain/loop/start  → begin autonomous reasoning loop
#   POST /brain/loop/stop   → stop background loop
#   GET  /brain/loop/status → live loop + telemetry snapshot
# ============================================================

from __future__ import annotations
import threading, time, traceback
from typing import Optional, Dict, Any

from .experience.local_brain import LocalBrain
# use the new telemetry summary API
from .core.telemetry import summary as measure_telemetry
from .server_api import pack_error
from .server_brain import _brain, _BRAIN_LOCK

# Shared loop state
_LOOP_THREAD: Optional[threading.Thread] = None
_LOOP_STOP = threading.Event()
_LOOP_LOCK = threading.Lock()
_LAST_DECISION: Optional[Dict[str, Any]] = None
_LAST_STEP_TIME: Optional[float] = None


# ============================================================
# Background worker
# ============================================================

def _loop_worker(brain: LocalBrain, interval: float = 5.0) -> None:
    """Continuously run brain.step() until stopped."""
    global _LAST_DECISION, _LAST_STEP_TIME
    while not _LOOP_STOP.is_set():
        try:
            _LAST_DECISION = brain.step()
            _LAST_STEP_TIME = time.time()
        except Exception as e:
            print(f"[BrainLoop] error: {e}\n{traceback.format_exc()}")
        # Wait interval with early stop checks
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
    Mixin for BaseHTTPRequestHandler to add /brain/loop routes.
    Requires _engine() and _send_json() from server.Handler.
    """

    # -------------------- start loop --------------------
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

    # -------------------- stop loop --------------------
    def _handle_brain_loop_stop(self, data: dict) -> None:
        global _LOOP_THREAD
        _LOOP_STOP.set()
        with _LOOP_LOCK:
            if _LOOP_THREAD and _LOOP_THREAD.is_alive():
                _LOOP_THREAD.join(timeout=2.0)
            _LOOP_THREAD = None
        return self._send_json(200, {"status": "stopped"})

    # -------------------- unified status --------------------
    def _handle_brain_loop_status(self) -> None:
        """Return loop state, last brain decision, and live telemetry snapshot."""
        active = _LOOP_THREAD is not None and _LOOP_THREAD.is_alive()
        now = time.time()
        last_step_age = None if _LAST_STEP_TIME is None else (now - _LAST_STEP_TIME)

        # Try to collect telemetry
        telemetry: Dict[str, Any] = {}
        try:
            eng = self._engine() if hasattr(self, "_engine") else None
            if eng and eng.model is not None:
                _, X = eng.infer(x=None, strategy="greedy")
                telemetry = measure_telemetry(X, eng.cfg.delta)
                telemetry["delta"] = eng.cfg.delta
        except Exception as e:
            telemetry = {"error": str(e)}

        payload = {
            "status": "running" if active else "stopped",
            "last_decision": _LAST_DECISION,
            "last_step_time": _LAST_STEP_TIME,
            "seconds_since_last_step": last_step_age,
            "telemetry": telemetry,
        }
        return self._send_json(200, payload)
