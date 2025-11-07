# ElementFold · server_brain.py
# ============================================================
# Brain ⇄ Runtime bridge for the HTTP server
# ------------------------------------------------------------
# Adds one endpoint:
#   POST /brain/step  →  runs LocalBrain.step()
#                         returns telemetry + decision JSON
#
# Designed to be imported in elementfold/server.py.
# ============================================================

from __future__ import annotations
import json
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse
import traceback
import threading

from .experience.local_brain import LocalBrain
from .core.runtime import Engine
from .server_api import to_json, pack_error

# shared brain singleton (thread-safe)
_BRAIN_LOCK = threading.Lock()
_BRAIN: LocalBrain | None = None

def _brain(engine: Engine) -> LocalBrain:
    global _BRAIN
    with _BRAIN_LOCK:
        if _BRAIN is None:
            _BRAIN = LocalBrain(engine)
        return _BRAIN


# ============================================================
# Handler mixin (attach to existing BaseHTTPRequestHandler)
# ============================================================

class BrainHandlerMixin:
    """
    Mixin providing a /brain/step route for any BaseHTTPRequestHandler.
    Assumes _engine() accessor exists (same as in server.py).
    """

    def _handle_brain_step(self, data: dict) -> None:
        try:
            eng = self._engine() if hasattr(self, "_engine") else None
            if eng is None:
                return self._send_json(500, pack_error("no_engine", "Engine not available."))

            brain = _brain(eng)
            decision = brain.step()

            payload = {
                "decision": decision,
                "timestamp": time.time(),
                "engine_state": {
                    "lambda": getattr(eng.cfg, "lambda_", None),
                    "D": getattr(eng.cfg, "D", None),
                    "phi_inf": getattr(eng.cfg, "phi_inf", None),
                    "dt": getattr(eng.cfg, "dt", None),
                },
            }
            return self._send_json(200, payload)

        except Exception as e:
            self.log_message("brain_step error: %s\n%s", repr(e), traceback.format_exc())
            return self._send_json(500, pack_error("internal_error", str(e)))
