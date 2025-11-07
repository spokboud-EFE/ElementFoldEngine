# ElementFold · server_brain.py
# ============================================================
# Brain ⇄ Runtime bridge for the HTTP server
# ------------------------------------------------------------
# Adds one endpoint:
#   POST /brain/step  → runs LocalBrain.step()
#                        returns telemetry + decision JSON
#
# Designed to be imported in elementfold/server.py.
# Compatible with the new RelaxationModel-based runtime.
# ============================================================

from __future__ import annotations
import json, time, traceback, threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse

# Core runtime and brain
from .experience.local_brain import LocalBrain
from .core.model import RelaxationModel
from .server_api import to_json, pack_error

# Shared singleton (thread-safe)
_BRAIN_LOCK = threading.Lock()
_BRAIN: LocalBrain | None = None


# ============================================================
# Brain management helpers
# ============================================================

def _brain(engine: RelaxationModel) -> LocalBrain:
    """Get or create the shared LocalBrain instance."""
    global _BRAIN
    with _BRAIN_LOCK:
        if _BRAIN is None:
            _BRAIN = LocalBrain(engine)
        return _BRAIN


# ============================================================
# HTTP Handler Mixin
# ============================================================

class BrainHandlerMixin:
    """
    Mixin providing a /brain/step route for any BaseHTTPRequestHandler.
    The parent Handler must define:
        • _engine() → returns the active RelaxationModel instance
        • _send_json(status, payload)
    """

    def _handle_brain_step(self, data: dict) -> None:
        """Perform one reasoning step via LocalBrain and return JSON output."""
        try:
            eng = self._engine() if hasattr(self, "_engine") else None
            if eng is None:
                return self._send_json(500, pack_error("no_engine", "Engine not available."))

            brain = _brain(eng)
            decision = brain.step()

            # Collect current configuration safely
            cfg = getattr(eng, "background", None)
            state = {
                "lambda": getattr(cfg, "lambda_", None),
                "D": getattr(cfg, "D", None),
                "phi_inf": getattr(cfg, "phi_inf", None),
                "dt": getattr(eng, "dt", None),
            }

            payload = {
                "decision": decision,
                "timestamp": time.time(),
                "engine_state": state,
            }
            return self._send_json(200, payload)

        except Exception as e:
            self.log_message("brain_step error: %s\n%s", repr(e), traceback.format_exc())
            return self._send_json(500, pack_error("internal_error", str(e)))
