# ElementFold · server.py
# ============================================================
# Lightweight HTTP server exposing the relaxation model.
# ------------------------------------------------------------
# Endpoints (JSON):
#   • GET  /health
#   • POST /simulate   → evolve Φ for N steps
#   • POST /folds      → integrate ℱ
#   • POST /redshift   → compute 1+z = e^ℱ − 1
#   • POST /brightness → apply brightness tilt
#   • POST /bend       → compute color-dependent deflection
#
# Uses only stdlib http.server + core.server_api helpers.
# ============================================================

from __future__ import annotations
import argparse, json, traceback
from http.server import BaseHTTPRequestHandler
from .server_brain import BrainHandlerMixin
try:
    from http.server import ThreadingHTTPServer as HTTPServer
except Exception:
    from http.server import HTTPServer
from typing import Any

from .server_api import (
    parse_json, to_json,
    coerce_simulate_request, coerce_path_request,
    pack_simulate_response, pack_error,
    SimulateResponse, FoldsResponse,
    RedshiftResponse, BrightnessResponse, BendResponse,
)
from .core.model import RelaxationModel
from .core.data import default_background, default_optics, FieldState
from .core.runtime import simulate_once
from .core.fgn import folds, redshift_from_F, brightness_tilt, bend


# ============================================================
# Global model singleton (lazy init)
# ============================================================

_MODEL: RelaxationModel | None = None

def _model() -> RelaxationModel:
    global _MODEL
    if _MODEL is None:
        bg = default_background()
        optics = default_optics()
        _MODEL = RelaxationModel(background=bg, optics=optics)
    return _MODEL


# ============================================================
# HTTP handler
# ============================================================

class Handler(BaseHTTPRequestHandler, BrainHandlerMixin):

    def _set_headers(self, status: int = 200, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")

    def _send_json(self, status: int, payload: Any) -> None:
        body = to_json(payload)
        self._set_headers(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, code: str, message: str) -> None:
        self._send_json(status, pack_error(code, message))

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        return parse_json(body)

    def do_OPTIONS(self) -> None:
        self._set_headers(204)
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/health":
            payload = {"status": "ok", "model_ready": True}
            return self._send_json(200, payload)
        return self._send_error(404, "not_found", f"unknown GET path: {self.path}")

    def do_POST(self) -> None:
        try:
            path = self.path
            data = self._read_json()

            if path == "/simulate":
                req = coerce_simulate_request(data)
                bg = default_background(req.lambda_, req.D, req.phi_inf)
                optics = default_optics()
                model = RelaxationModel(background=bg, optics=optics)
                phi0 = FieldState(phi=np.zeros(req.shape, float),
                                  t=0.0,
                                  spacing=tuple(req.spacing),
                                  bc=req.bc)
                out = simulate_once(model, phi0, steps=req.steps, dt=req.dt)
                resp = pack_simulate_response(out["state"].phi,
                                              out["state"].t,
                                              out["metrics"])
                return self._send_json(200, resp)

            if path == "/folds":
                req = coerce_path_request(data)
                model = _model()
                F = folds([seg for seg in req.path], model.optics.eta)
                return self._send_json(200, FoldsResponse(F=F))

            if path == "/redshift":
                req = coerce_path_request(data)
                model = _model()
                F = folds([seg for seg in req.path], model.optics.eta)
                z = redshift_from_F(F)
                return self._send_json(200, RedshiftResponse(z=z))

            if path == "/brightness":
                req = coerce_path_request(data)
                model = _model()
                F = folds([seg for seg in req.path], model.optics.eta)
                I_emit = float(data.get("I_emit", 1.0))
                d_geom = float(data.get("d_geom", 1.0))
                I_obs = I_emit / (4.0 * 3.14159 * d_geom ** 2) * float(brightness_tilt(F))
                return self._send_json(200, BrightnessResponse(I_obs=I_obs))

            if path == "/bend":
                req = coerce_path_request(data)
                model = _model()
                dtheta = bend([seg for seg in req.path], model.optics.n)
                return self._send_json(200, BendResponse(dtheta=dtheta))
            
            if path == "/brain/step":
                return self._handle_brain_step(data)

            return self._send_error(404, "not_found", f"unknown POST path: {path}")

        except ValueError as e:
            return self._send_error(400, "bad_json", str(e))
        except Exception as e:
            self.log_message("error: %s\n%s", repr(e), traceback.format_exc())
            return self._send_error(500, "internal_error", str(e))


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    p = argparse.ArgumentParser(description="ElementFold relaxation HTTP server")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    args = p.parse_args()

    srv = HTTPServer((args.host, args.port), Handler)
    print(f"⟲ ElementFold physics server ready at http://{args.host}:{args.port}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()

if __name__ == "__main__":
    main()
