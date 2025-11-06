# ElementFold · server.py
# A small, dependency‑free HTTP server that exposes the engine as JSON endpoints.
# Endpoints (all JSON unless otherwise noted):
#   • GET    /                  → tiny HTML index (human friendly)
#   • GET    /favicon.ico       → 204 (silence browser noise)
#   • GET    /health            → HealthResponse
#   • GET    /config            → engine/config snapshot
#   • GET    /adapters          → {"adapters": ["language","audio","multimodal", ...]}
#   • GET    /telemetry         → recent console lines (mirror of utils.display.recent_json)
#   • GET    /telemetry/state   → one live telemetry snapshot (does a tiny infer with relax)
#   • GET    /telemetry/counsel → interpreter summary (numbers → narrative + next actions)
#   • GET    /pilot/health      → BackgroundModel health (local + remote), if available
#   • POST   /pilot/generate    → BackgroundModel.generate(messages=...) with fallback
#   • POST   /infer             → InferResponse (tokens/ledger[/text][+folds][+relax_meta])
#   • POST   /steer             → SteerResponse (adapter output [+ applied params])
#   • POST   /train             → TrainResponse (run a short training loop)
#
# Design choices:
#   1) Lazy engine: first request boots the Engine; first infer/steer may train.
#   2) Thread‑safe: module‑level locks guard singletons.
#   3) Robust JSON: payloads are validated/coerced; errors are structured.
#   4) CORS: permissive defaults; OPTIONS supported.
#   5) No external deps: stdlib http.server + tiny server_api helpers.

from __future__ import annotations

import argparse
import threading
import os
from http.server import BaseHTTPRequestHandler
try:
    from http.server import ThreadingHTTPServer as _HTTPServer  # Py3.7+
except Exception:
    from http.server import HTTPServer as _HTTPServer

from urllib.parse import urlparse, parse_qs
import traceback
from dataclasses import asdict
import json
import torch
import sys
import time

from . import __version__
from .runtime import Engine
from .tokenizer import SimpleTokenizer

# Optional brain bootstrap (ram‑only env). If missing, we silently skip.
try:
    from .utils.bootstrap import bootstrap_brain_env  # type: ignore
except Exception:  # pragma: no cover
    def bootstrap_brain_env(*_, **__):  # type: ignore
        return None

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
    # normalized decode args (includes optional 'relax')
    normalized_decode_args,
)

# Optional: adapter list for /adapters endpoint
try:
    from .experience.adapters.base import AdapterRegistry
except Exception:  # pragma: no cover
    AdapterRegistry = None  # graceful degrade if not importable

# Telemetry mirror (console → web)
try:
    from .utils.display import recent_json
except Exception:  # pragma: no cover
    def recent_json(n: int = 200):
        return []

# Optional: BackgroundModel (pilot LLM). Import lazily/optionally.
try:
    from .experience.background_model import BackgroundModel, PilotConfig, as_messages
except Exception:  # pragma: no cover
    BackgroundModel = None     # type: ignore
    PilotConfig = None         # type: ignore
    as_messages = None         # type: ignore

# Telemetry Interpreter (numbers → narrative)
try:
    from .experience.interpreter import TelemetryInterpreter, InterpreterConfig
except Exception:  # pragma: no cover
    TelemetryInterpreter = None  # type: ignore
    InterpreterConfig = None     # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Lazy singletons (Engine, Pilot, Interpreter), protected by small locks
# ──────────────────────────────────────────────────────────────────────────────

_ENGINE_LOCK = threading.Lock()
_ENGINE: Engine | None = None

def _engine() -> Engine:
    """Create or fetch the singleton Engine lazily (thread‑safe)."""
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = Engine()  # lazy boot; may lazy‑train at first infer/steer
        return _ENGINE

_PILOT_LOCK = threading.Lock()
_PILOT: "BackgroundModel | None" = None

def _pilot_config_from_env() -> "PilotConfig":
    if PilotConfig is None:
        raise RuntimeError("BackgroundModel not available (import failed)")

    def _get(name: str, default: str | None = None) -> str | None:
        v = os.environ.get(name)
        return v if v is not None else default

    cfg = PilotConfig()
    # Local
    cfg.local_kind = (_get("PILOT_LOCAL_KIND", cfg.local_kind) or cfg.local_kind)  # type: ignore
    cfg.local_model = (_get("PILOT_LOCAL_MODEL", cfg.local_model) or cfg.local_model)  # type: ignore
    cfg.local_device = _get("PILOT_LOCAL_DEVICE", cfg.local_device or None)
    cfg.local_ollama_url = _get("PILOT_LOCAL_OLLAMA_URL", cfg.local_ollama_url) or cfg.local_ollama_url
    # Remote
    rk = _get("PILOT_REMOTE_KIND", cfg.remote_kind or "")
    cfg.remote_kind = rk if rk else None  # type: ignore
    cfg.remote_url = _get("PILOT_REMOTE_URL", cfg.remote_url or None)
    cfg.remote_model = _get("PILOT_REMOTE_MODEL", cfg.remote_model or None)
    cfg.remote_api_key = _get("PILOT_REMOTE_API_KEY", cfg.remote_api_key or None)
    # Decoding + limits
    try: cfg.temperature = float(_get("PILOT_TEMPERATURE", str(cfg.temperature)) or cfg.temperature)
    except Exception: pass
    try: cfg.top_p = float(_get("PILOT_TOP_P", str(cfg.top_p)) or cfg.top_p)
    except Exception: pass
    try: cfg.top_k = int(_get("PILOT_TOP_K", str(cfg.top_k)) or cfg.top_k)
    except Exception: pass
    try: cfg.max_new_tokens = int(_get("PILOT_MAX_NEW", str(cfg.max_new_tokens)) or cfg.max_new_tokens)
    except Exception: pass
    # Heartbeat + request timeouts
    try: cfg.heartbeat_every_s = float(_get("PILOT_HEARTBEAT_S", str(cfg.heartbeat_every_s)) or cfg.heartbeat_every_s)
    except Exception: pass
    try: cfg.request_timeout_s = float(_get("PILOT_TIMEOUT_S", str(cfg.request_timeout_s)) or cfg.request_timeout_s)
    except Exception: pass
    # Prefer remote
    pr = _get("PILOT_PREFER_REMOTE")
    if pr is not None:
        cfg.prefer_remote = pr.strip() not in {"0", "false", "False", ""}

    return cfg

def _pilot() -> "BackgroundModel":
    """Create or fetch the singleton BackgroundModel lazily (thread‑safe)."""
    if BackgroundModel is None:
        raise RuntimeError("BackgroundModel not available (import failed)")
    global _PILOT
    with _PILOT_LOCK:
        if _PILOT is None:
            cfg = _pilot_config_from_env()
            _PILOT = BackgroundModel(cfg)
        return _PILOT

_INTERP_LOCK = threading.Lock()
_INTERP: "TelemetryInterpreter | None" = None

def _interpreter() -> "TelemetryInterpreter":
    """Create/fetch a TelemetryInterpreter (numbers→narrative)."""
    if TelemetryInterpreter is None:
        raise RuntimeError("TelemetryInterpreter not available (module import failed)")
    global _INTERP
    with _INTERP_LOCK:
        if _INTERP is None:
            pilot = None
            if BackgroundModel is not None:
                try:
                    pilot = _pilot()
                except Exception:
                    pilot = None
            cfg = InterpreterConfig(enable_llm=bool(pilot))
            _INTERP = TelemetryInterpreter(pilot=pilot, cfg=cfg)
        return _INTERP


# ──────────────────────────────────────────────────────────────────────────────
# Small HTML index (human friendly)
# ──────────────────────────────────────────────────────────────────────────────

_INDEX_HTML = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<title>ElementFold · Server</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem; line-height: 1.45; }
  code, pre { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
  a { text-decoration: none; }
  .chip { display:inline-block; padding:.25rem .5rem; border-radius:999px; border:1px solid #8884; margin:.2rem .3rem; font-size:.9rem; }
  .get { background:#2b8a3e20; border-color:#2b8a3e55; }
  .post{ background:#1c7ed620; border-color:#1c7ed655; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
  footer { margin-top:2rem; opacity:.7; font-size:.9rem; }
</style>
<h1>⟲ ElementFold • Server</h1>
<p>This is a small, dependency‑free JSON server for ElementFold.</p>
<section>
  <h3>Endpoints</h3>
  <div>
    <span class="chip get">GET</span> <a href="/health">/health</a><br>
    <span class="chip get">GET</span> <a href="/config">/config</a><br>
    <span class="chip get">GET</span> <a href="/adapters">/adapters</a><br>
    <span class="chip get">GET</span> <a href="/telemetry?n=200">/telemetry</a><br>
    <span class="chip get">GET</span> <a href="/telemetry/state">/telemetry/state</a><br>
    <span class="chip get">GET</span> <a href="/telemetry/counsel">/telemetry/counsel</a><br>
    <span class="chip get">GET</span> <a href="/pilot/health">/pilot/health</a><br>
    <span class="chip post">POST</span> <code>/pilot/generate</code><br>
    <span class="chip post">POST</span> <code>/infer</code><br>
    <span class="chip post">POST</span> <code>/steer</code><br>
    <span class="chip post">POST</span> <code>/train</code>
  </div>
</section>
<section>
  <h3>Quick POST examples</h3>
  <pre>{
  "text": "hello fold", "strategy": "greedy"
}</pre>
  <pre>{
  "text": "hello fold",
  "strategy": "sample", "temperature": 0.9, "top_p": 0.95,
  "relax": {"eta": 0.02, "eta_path_weight": 0.5, "rho": 0.2, "D": 0.05}
}</pre>
  <pre>{
  "messages":[{"role":"user","content":"pilot, explain β/γ/clamp"}],
  "temperature": 0.2, "top_p": 0.9, "max_new_tokens": 200
}</pre>
</section>
<footer class="mono">ElementFold v__VERSION__ · ready</footer>
"""

# ──────────────────────────────────────────────────────────────────────────────
# Defaults for quick telemetry snapshots
# ──────────────────────────────────────────────────────────────────────────────

_DEFAULT_RELAX = {
    "eta": 0.02,
    "eta_path_weight": 0.5,
    "rho": 0.20,
    "lambda": 0.0,
    "D": 0.05,
    "phi_inf": 0.0,
    "steps": 1,
    "dt": 1.0,
}

def _first_row_list(x) -> list:
    """
    Tensor/list → python list; for 2‑D we take the first row.
    """
    # torch path
    try:
        import torch as _t
        if isinstance(x, _t.Tensor):
            t = x.detach().cpu()
            if t.dim() == 2:
                return t[0].tolist()
            return t.tolist()
    except Exception:
        pass
    # generic
    try:
        arr = x.tolist()
        return arr[0] if (arr and isinstance(arr[0], (list, tuple))) else arr
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return list(x[0]) if (x and isinstance(x[0], (list, tuple))) else list(x)
    return []

def _quick_state_snapshot(eng: Engine) -> dict:
    """
    Produce a small, self‑contained telemetry dict by running a tiny greedy infer
    with relax enabled. This keeps /telemetry/state and /telemetry/counsel simple
    without requiring Engine.telemetry().
    """
    # Ensure model exists (may train on first call)
    if eng.model is None:
        _ = eng.infer(x=None)

    out = eng.infer(
        x=None,  # random quick batch
        strategy="greedy",
        temperature=1.0,
        top_k=None,
        top_p=None,
        relax=_DEFAULT_RELAX,
    )

    tele: dict = {
        "delta": float(getattr(eng.cfg, "delta", 0.0)),
    }

    # folds + relax_meta, if present
    if "folds" in out:
        tele["folds"] = _first_row_list(out["folds"])
    if "relax_meta" in out and isinstance(out["relax_meta"], dict):
        tele["relax_meta"] = out["relax_meta"]

    # Simple motion proxy from ledger (optional)
    try:
        import numpy as _np
        L = _first_row_list(out.get("ledger", []))
        if len(L) >= 3:
            d1 = _np.diff(_np.array(L, dtype="float64"))
            tele["resid_std"] = float(max(0.0, float(_np.std(d1))))
    except Exception:
        pass

    # Placeholders for advanced metrics (adapters may fill these later)
    tele["kappa"] = None
    tele["p_half"] = None
    tele["margin_min"] = None
    tele["margin_mean"] = None
    tele["phase_mean"] = None

    return tele


# ──────────────────────────────────────────────────────────────────────────────
# HTTP handler (one instance per request)
# ──────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    # — Low‑level I/O helpers —

    def _set_cors(self) -> None:
        # Permissive CORS for quick experiments (can be tightened if needed).
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_json(self, status: int, payload) -> None:
        """Serialize payload as JSON and send with common headers."""
        body = to_json(payload)  # dataclasses → dict → JSON bytes
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, code: str, message: str, details=None) -> None:
        """Send a structured ErrorResponse with an HTTP error status."""
        self._send_json(status, pack_error(code=code, message=message, details=details))

    def _send_html(self, status: int, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self._set_cors()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

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
            url = urlparse(self.path)
            path = url.path
            q = parse_qs(url.query)

            if path == "/" or path == "/index.html":
                html = _INDEX_HTML.replace("__VERSION__", str(__version__))
                return self._send_html(200, html)

            if path == "/favicon.ico":
                self.send_response(204)
                self._set_cors()
                self.end_headers()
                return

            if path == "/health":
                eng = _engine()
                payload = HealthResponse(
                    status="ok",
                    version=__version__,
                    device=str(eng.device),
                    model_ready=bool(eng.model is not None),
                )
                return self._send_json(200, payload)

            if path == "/config":
                eng = _engine()
                cfg = getattr(eng, "cfg", None)
                view = {
                    "version": __version__,
                    "device": str(eng.device),
                    "model_ready": bool(eng.model is not None),
                    "config": cfg.to_kwargs() if cfg else {},
                }
                return self._send_json(200, view)

            if path == "/adapters":
                names = []
                try:
                    if AdapterRegistry is not None:
                        names = list(AdapterRegistry.names())
                except Exception:
                    names = []
                return self._send_json(200, {"adapters": names})

            if path == "/telemetry":
                try:
                    n = int(q.get("n", ["200"])[0])
                except Exception:
                    n = 200
                n = max(1, min(2000, n))
                logs = recent_json(n)
                self.send_response(200)
                self._set_cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                body = json.dumps({"lines": logs}).encode("utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if path == "/telemetry/state":
                eng = _engine()
                tele = _quick_state_snapshot(eng)
                return self._send_json(200, {"telemetry": tele, "ts": time.time()})

            if path == "/telemetry/counsel":
                if TelemetryInterpreter is None:
                    return self._send_error(501, code="interpreter_unavailable",
                                            message="TelemetryInterpreter not available (module import failed)")
                eng = _engine()
                tele = _quick_state_snapshot(eng)
                try:
                    interp = _interpreter()
                except Exception as e:
                    # If we cannot init interpreter with pilot, fall back to rules‑only instance
                    interp = TelemetryInterpreter(pilot=None, cfg=InterpreterConfig(enable_llm=False))  # type: ignore
                out = interp.summarize(telemetry=tele, status=None, context=None)
                return self._send_json(200, out)

            if path == "/pilot/health":
                if BackgroundModel is None:
                    return self._send_error(501, code="pilot_unavailable",
                                            message="BackgroundModel not available (module import failed)")
                try:
                    pilot = _pilot()
                except Exception as e:
                    return self._send_error(500, code="pilot_init_error", message=str(e))
                return self._send_json(200, pilot.health())

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

            if path == "/pilot/generate":
                return self._handle_pilot_generate(data)

            return self._send_error(404, code="not_found", message="unknown POST path")

        except ValueError as e:
            return self._send_error(400, code="bad_json", message=str(e))
        except Exception as e:
            self.log_message("POST error: %s\n%s", repr(e), traceback.format_exc())
            return self._send_error(500, code="internal_error", message="unhandled POST exception")

    # — Route handlers —

    def _handle_infer(self, data: dict):
        req = coerce_infer_request(data)
        err = validate_infer_request(req)
        if err is not None:
            return self._send_json(400, err)

        eng = _engine()

        tok = SimpleTokenizer(vocab=eng.cfg.vocab)
        ids = resolve_tokens(req, tokenizer=tok, vocab=eng.cfg.vocab, seq_len=eng.cfg.seq_len)

        x = torch.tensor(ids, dtype=torch.long)

        decode_args = normalized_decode_args(req)

        out = eng.infer(
            x=x,
            **decode_args,
        )

        base = pack_infer_response(out["tokens"], out["ledger"], tokenizer=tok)
        payload = asdict(base)

        try:
            if "folds" in out:
                t = out["folds"].detach().cpu()
                payload["folds"] = t.tolist() if t.dim() == 1 else t[0].tolist()
            if "relax_meta" in out and isinstance(out["relax_meta"], dict):
                payload["relax_meta"] = out["relax_meta"]
        except Exception:
            pass

        return self._send_json(200, payload)

    def _handle_steer(self, data: dict):
        try:
            req = SteerRequest(**data)
        except TypeError:
            return self._send_error(400, code="bad_request",
                                    message="expected {'prompt': str, 'modality': 'language'|...}")
        if not req.prompt:
            return self._send_error(400, code="bad_request", message="missing 'prompt'")

        eng = _engine()

        try:
            from .experience.steering import SteeringController
            from .experience.adapters.base import AdapterRegistry as _AReg
            if eng.model is None:
                _ = eng.infer(x=None)  # lazy materialize/train
            ctrl = SteeringController.load_default(eng.cfg.delta)
            v = ctrl(req.prompt)
            params = SteeringController.to_params(v)  # {'beta','gamma','clamp','style'}
            runner = _AReg.get(req.modality)()
            output = runner(eng.model, req.prompt, v)
            pview = {"beta": float(params["beta"]),
                     "gamma": float(params["gamma"]),
                     "clamp": float(params["clamp"])}
            resp = SteerResponse(output=output, params=pview)
        except KeyError:
            return self._send_error(400, code="bad_modality",
                                    message=f"unknown modality: {req.modality!r}")
        except Exception:
            self.log_message("steer error:\n%s", traceback.format_exc())
            return self._send_error(500, code="internal_error", message="steer failed")
        return self._send_json(200, resp)

    def _handle_train(self, data: dict):
        try:
            req = TrainRequest(**data)
        except TypeError:
            return self._send_error(400, code="bad_request", message="expected {'steps': int}")

        eng = _engine()
        steps_orig = eng.cfg.steps
        try:
            eng.cfg.steps = int(max(1, req.steps))
            eng.fit()
        finally:
            eng.cfg.steps = steps_orig

        return self._send_json(200, TrainResponse(trained=True, steps=req.steps))

    def _handle_pilot_generate(self, data: dict):
        if BackgroundModel is None:
            return self._send_error(501, code="pilot_unavailable",
                                    message="BackgroundModel not available (module import failed)")

        msgs = data.get("messages")
        if msgs is None:
            sys_txt = data.get("system")
            user_txt = data.get("user")
            if not (isinstance(sys_txt, str) or isinstance(user_txt, str)):
                return self._send_error(400, code="bad_request",
                                        message="provide 'messages':[...], or 'system'/'user' strings")
            if as_messages is None:
                return self._send_error(500, code="pilot_init_error",
                                        message="message helper not available")
            msgs = as_messages(system=sys_txt or None, user=user_txt or None)
        if not isinstance(msgs, list):
            return self._send_error(400, code="bad_request", message="'messages' must be a list")

        temp = data.get("temperature")
        top_p = data.get("top_p")
        top_k = data.get("top_k")
        max_new = data.get("max_new_tokens")
        timeout_s = data.get("timeout_s")

        try:
            pilot = _pilot()
        except Exception as e:
            return self._send_error(500, code="pilot_init_error", message=str(e))

        try:
            out = pilot.generate(
                messages=msgs,
                temperature=float(temp) if temp is not None else None,
                top_p=float(top_p) if top_p is not None else None,
                top_k=int(top_k) if top_k is not None else None,
                max_new_tokens=int(max_new) if max_new is not None else None,
                timeout_s=float(timeout_s) if timeout_s is not None else None,
            )
        except Exception as e:
            return self._send_error(502, code="pilot_error", message=str(e))

        payload = {
            "text": out.text,
            "backend": out.backend,
            "latency_ms": out.latency_ms,
            "finish_reason": out.finish_reason,
            "tokens_new": out.tokens_new,
            "meta": out.meta or {},
        }
        return self._send_json(200, payload)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry‑point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="ElementFold HTTP server")
    p.add_argument("--host", type=str, default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="TCP port (default: 8080)")
    args = p.parse_args()

    # Prompt only if TTY; otherwise skip quietly (you can set env before launching)
    bootstrap_brain_env(interactive=sys.stdin.isatty())

    srv = _HTTPServer((args.host, args.port), Handler)  # ThreadingHTTPServer when available
    print(f"⟲ ElementFold server ⟲  http://{args.host}:{args.port}   (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()

# Ensure module can be run with `python -m elementfold.server`
if __name__ == "__main__":
    main()
