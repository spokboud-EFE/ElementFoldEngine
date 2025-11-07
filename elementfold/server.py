# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ElementFold · server.py                                                      ║
# ║──────────────────────────────────────────────────────────────────────────────║
# ║  Lightweight HTTP interface to the relaxation core and Studio telemetry.     ║
# ║                                                                              ║
# ║  Endpoints (JSON):                                                           ║
# ║   • GET  /health        → minimal readiness check                            ║
# ║   • GET  /diag          → full diagnostic snapshot (mode + narrative)        ║
# ║   • POST /mode          → switch between "shaping" / "forcing"               ║
# ║   • POST /simulate      → evolve Φ for N steps                               ║
# ║   • GET  /telemetry     → raw live physics metrics                           ║
# ║   • GET  /adapters      → list registered adapters                           ║
# ║                                                                              ║
# ║  Narrative telemetry preserved:                                              ║
# ║   β, γ, ⛔, κ, λ, D, ∇Φ — all accompanied by short textual descriptions.     ║
# ║                                                                              ║
# ║  Uses only Python stdlib http.server.                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations
import json, argparse, time, traceback, logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from elementfold.core import runtime, control, telemetry
from elementfold.experience.adapters.base import AdapterRegistry

# ────────────────────────────────────────────────────────────────────────────────
# Global initialization
# ────────────────────────────────────────────────────────────────────────────────
_ENGINE = runtime.init_engine()
_START_TIME = time.time()

LOG = logging.getLogger("elementfold.server")
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ────────────────────────────────────────────────────────────────────────────────
# Custom lightweight exceptions
# ────────────────────────────────────────────────────────────────────────────────
class BadRequest(ValueError): ...
class NotFound(LookupError): ...
class ModeError(ValueError): ...


# ────────────────────────────────────────────────────────────────────────────────
# HTTP Handler
# ────────────────────────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    """HTTP surface of the ElementFold runtime — minimal, human, narrative."""

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, code: str, msg: str) -> None:
        self._json(status, {"error": {"code": code, "message": msg}})

    def _safe_json(self) -> dict:
        """Parse JSON safely with clear error messages."""
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError as e:
            raise BadRequest(f"Invalid JSON: {e.msg}")

    # ────────────────────────────────────────────────────────────────────────────
    # GET routes
    # ────────────────────────────────────────────────────────────────────────────
    def do_GET(self) -> None:
        try:
            if self.path == "/health":
                return self._json(200, {"status": "ok", "message": "ElementFold core responsive."})

            if self.path == "/telemetry":
                snap = telemetry.snapshot()
                snap["narrative"] = telemetry.narrate(snap)
                return self._json(200, snap)

            if self.path == "/adapters":
                return self._json(200, {"adapters": AdapterRegistry.list_all()})

            if self.path == "/diag":
                physics = telemetry.snapshot()
                physics["narrative"] = telemetry.narrate(physics)
                payload = {
                    "studio": {
                        "mode": control.get_mode(),
                        "uptime_sec": round(time.time() - _START_TIME, 2),
                        "symbol": "🎛️"
                    },
                    "physics": physics,
                    "adapters": {
                        "registered": AdapterRegistry.list_all(),
                        "active": AdapterRegistry.active()
                    },
                    "brains": telemetry.brain_status(),
                    "env": telemetry.env_status(),
                    "version": telemetry.version_info(),
                }
                return self._json(200, payload)

            raise NotFound(self.path)

        except NotFound as nf:
            return self._error(404, "not_found", f"unknown path: {nf}")
        except Exception as e:
            LOG.error("Unhandled GET error: %s", e)
            LOG.debug(traceback.format_exc())
            return self._error(500, "internal_error", "unexpected server error")

    # ────────────────────────────────────────────────────────────────────────────
    # POST routes
    # ────────────────────────────────────────────────────────────────────────────
    def do_POST(self) -> None:
        try:
            data = self._safe_json()

            if self.path == "/mode":
                mode = data.get("mode")
                if mode not in ("shaping", "forcing"):
                    raise ModeError("mode must be 'shaping' or 'forcing'")
                control.set_mode(mode)
                phrase = (
                    "gentle shaping — coherence guided."
                    if mode == "shaping"
                    else "direct forcing — field depinned for manual intervention."
                )
                return self._json(200, {
                    "ok": True,
                    "mode": control.get_mode(),
                    "narrative": f"⚙️ Mode switched to {mode}: {phrase}"
                })

            if self.path == "/simulate":
                result = runtime.step(data)
                result["narrative"] = telemetry.narrate(result)
                return self._json(200, result)

            raise NotFound(self.path)

        except BadRequest as e:
            return self._error(400, "bad_request", str(e))
        except ModeError as e:
            return self._error(400, "bad_mode", str(e))
        except NotFound as nf:
            return self._error(404, "not_found", f"unknown path: {nf}")
        except Exception as e:
            LOG.error("Unhandled POST error: %s", e)
            LOG.debug(traceback.format_exc())
            return self._error(500, "internal_error", "unexpected server error")


# ────────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="ElementFold relaxation server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  🧠  ElementFold Server active on  http://{args.host}:{args.port:<5}            ║")
    print("║  Use  /diag  for full system state,  /mode  to toggle forcing/shaping.║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[server] Shutting down gracefully.")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
