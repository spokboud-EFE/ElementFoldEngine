# ElementFold · server.py
# A small, dependency‑free HTTP server that exposes the engine as JSON endpoints.
# Endpoints (all JSON):
#   • GET    /health           → HealthResponse
#   • POST   /infer            → InferResponse   (tokens/ledger[/text])
#   • POST   /steer            → SteerResponse   (adapter output [+ applied params])
#   • POST   /train            → TrainResponse   (run a short training loop)
#
# Design choices (plain words):
#   1) Lazy engine: the first request boots the Engine; the first infer/steer may train.
#   2) Thread‑safe: a module‑level lock guards Engine creation; handler methods are kept small.
#   3) Robust JSON: we validate/clean incoming payloads and return structured ErrorResponse.
#   4) CORS: permissive defaults for quick UX experiments from a browser (OPTIONS supported).
#   5) No external deps: stdlib http.server + our tiny server_api helpers.

from __future__ import annotations

import argparse                                  # ✴ CLI flags (host/port)
import threading                                 # ✴ a lock for lazy Engine
from http.server import BaseHTTPRequestHandler
try:
    # Python 3.7+ has ThreadingHTTPServer; fall back to HTTPServer if unavailable.
    from http.server import ThreadingHTTPServer as _HTTPServer
except Exception:
    from http.server import HTTPServer as _HTTPServer

from urllib.parse import urlparse                # ✴ route parsing
import traceback                                 # ✴ pretty errors for logs
import torch                                     # ✴ tensors for input ids

from . import __version__                        # ✴ project version string
from .runtime import Engine                      # ✴ orchestration spine
from .tokenizer import SimpleTokenizer           # ✴ tiny byte tokenizer

# API schemas + JSON helpers
from .server_api import (
    # dataclasses
    InferRequest, InferResponse,
    SteerRequest, SteerResponse,
    HealthResponse, TrainRequest, TrainResponse,
    ErrorResponse,
    # JSON helpers
    parse_json, to_json,
    # request validation & coercion
    coerce_infer_request, validate_infer_request, resolve_tokens,
    # packers
    pack_infer_response, pack_error,
)

# ————————————————————————————————————————————————
# Lazy Engine singleton (protected by a small lock)
# ————————————————————————————————————————————————

_ENGINE_LOCK = threading.Lock()
_ENGINE: Engine | None = None

def _engine() -> Engine:
    """Create or fetch the singleton Engine lazily (thread‑safe)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Engine()                    # lazy boot; may lazy‑train at first infer/steer
        return _ENGINE


# ————————————————————————————————————————————————
# HTTP handler (one instance per request)
# ————————————————————————————————————————————————

class Handler(BaseHTTPRequestHandler):
    # — Low‑level I/O helpers —

    def _set_cors(self) -> None:
        # Permissive CORS for quick experiments (can be tightened if needed).
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, status: int, payload) -> None:
        """Serialize payload as JSON and send with common headers."""
        body = to_json(payload)                   # dataclasses → dict → JSON bytes
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, code: str, message: str, details=None) -> None:
        """Send a structured ErrorResponse with an HTTP error status."""
        self._send(status, pack_error(code=code, message=message, details=details))

    def _read_json(self) -> dict:
        """Read and parse request body into a dict (empty object on no body)."""
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(length) if length > 0 else b""
        try:
            return parse_json(body)
        except Exception as e:
            self.log_message("bad json: %s", repr(e))
            raise

    # — HTTP verbs —

    def do_OPTIONS(self) -> None:
        # CORS preflight
        self.send_response(204)
        self._set_cors()
        self.end_headers()

    def do_GET(self) -> None:
        try:
            path = urlparse(self.path).path
            if path == "/health":
                eng = _engine()
                payload = HealthResponse(
                    status="ok",
                    version=__version__,
                    device=str(eng.device),
                    model_ready=bool(eng.model is not None),
                )
                return self._send(200, payload)
            # Unknown route
            return self._send_error(404, code="not_found", message="unknown GET path")
        except Exception as e:
            self.log_message("GET error: %s\n%s", repr(e), traceback.format_exc())
            return self._send_error(500, code="internal_error", message="unhandled GET exception")

    def do_POST(self) -> None:
        try:
            path = urlparse(self.path).path
            data = self._read_json()

            if path == "/infer":
                return self._handle_infer(data)

            if path == "/steer":
                return self._handle_steer(data)

            if path == "/train":
                return self._handle_train(data)

            return self._send_error(404, code="not_found", message="unknown POST path")
        except ValueError as e:
            # Typically JSON parse errors end up here from _read_json()
            return self._send_error(400, code="bad_json", message=str(e))
        except Exception as e:
            self.log_message("POST error: %s\n%s", repr(e), traceback.format_exc())
            return self._send_error(500, code="internal_error", message="unhandled POST exception")

    # — Route handlers —

    def _handle_infer(self, data: dict):
        # 1) Clean + validate request
        req = coerce_infer_request(data)          # tolerant shaping → InferRequest‑like obj
        err = validate_infer_request(req)         # returns ErrorResponse or None
        if err is not None:
            return self._send(400, err)

        # 2) Ensure engine exists (and train lazily if needed in infer())
        eng = _engine()

        # 3) Resolve tokens from (tokens | text) using our tiny tokenizer
        tok = SimpleTokenizer(vocab=eng.cfg.vocab)
        ids = resolve_tokens(req, tokenizer=tok, vocab=eng.cfg.vocab, seq_len=eng.cfg.seq_len)

        # 4) Build tensor input (B=1) and run inference with chosen decoding strategy
        x = torch.tensor(ids, dtype=torch.long)
        out = eng.infer(
            x=x,
            strategy=req.strategy,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )

        # 5) Pack response (also provide detokenized text for convenience)
        resp = pack_infer_response(out["tokens"], out["ledger"], tokenizer=tok)
        return self._send(200, resp)

    def _handle_steer(self, data: dict):
        # 1) Coerce into dataclass with minimal fields (prompt/modality)
        try:
            req = SteerRequest(**data)
        except TypeError:
            return self._send_error(400, code="bad_request",
                                    message="expected {'prompt': str, 'modality': 'language'|...}")
        if not req.prompt:
            return self._send_error(400, code="bad_request", message="missing 'prompt'")

        # 2) Ensure engine/model exist; Engine.steer will lazy‑train if needed
        eng = _engine()

        # 3) Return adapter output; also expose the mapped control (β,γ,⛔) we applied
        #    We mirror Engine.steer’s mapping so client UIs can show gauges.
        try:
            from .experience.steering import SteeringController
            from .experience.adapters.base import AdapterRegistry
            if eng.model is None:
                _ = eng.infer(x=None)  # lazy materialize/train
            ctrl = SteeringController.load_default(eng.cfg.delta)
            v = ctrl(req.prompt)
            params = SteeringController.to_params(v)  # {'beta','gamma','clamp','style'}
            runner = AdapterRegistry.get(req.modality)()
            output = runner(eng.model, req.prompt, v)
            # Prepare a minimal, JSON‑friendly params view
            pview = {"beta": float(params["beta"]),
                     "gamma": float(params["gamma"]),
                     "clamp": float(params["clamp"])}
            resp = SteerResponse(output=output, params=pview)
        except KeyError as e:
            return self._send_error(400, code="bad_modality",
                                    message=f"unknown modality: {req.modality!r}")
        except Exception as e:
            self.log_message("steer error: %s\n%s", repr(e), traceback.format_exc())
            return self._send_error(500, code="internal_error", message="steer failed")
        return self._send(200, resp)

    def _handle_train(self, data: dict):
        # Small, synchronous training endpoint (useful for demos/tests).
        try:
            req = TrainRequest(**data)
        except TypeError:
            return self._send_error(400, code="bad_request", message="expected {'steps': int}")

        eng = _engine()
        # Run a short training loop; Engine.fit() uses cfg.steps, so we respect the
        # requested steps by temporarily shadowing cfg.steps for this call.
        steps_orig = eng.cfg.steps
        try:
            eng.cfg.steps = int(max(1, req.steps))
            eng.fit()
        finally:
            eng.cfg.steps = steps_orig

        return self._send(200, TrainResponse(trained=True, steps=req.steps))


# ————————————————————————————————————————————————
# CLI entry‑point
# ————————————————————————————————————————————————

def main() -> None:
    p = argparse.ArgumentParser(description="ElementFold HTTP server")
    p.add_argument("--host", type=str, default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="TCP port (default: 8080)")
    args = p.parse_args()

    srv = _HTTPServer((args.host, args.port), Handler)  # ThreadingHTTPServer when available
    print(f"⟲ ElementFold server ⟲  http://{args.host}:{args.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
