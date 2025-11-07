# ElementFold · utils/bootstrap.py
# ──────────────────────────────────────────────────────────────────────────────
# RAM‑only "brain" bootstrap:
#   • If remote LLM ("brain") env vars are not set, interactively ask for them.
#   • Verify connectivity (Ollama or OpenAI‑compatible) with a tiny "ping".
#   • Store only in os.environ for this process (no files written).
#
# Env keys we set (if confirmed / reachable):
#   PILOT_REMOTE_KIND       = "ollama" | "openai"
#   PILOT_REMOTE_URL        = "http://<host>:<port>"
#   PILOT_REMOTE_MODEL      = "<model-id-or-tag>"
#   PILOT_REMOTE_API_KEY    = "<secret or empty>"
#   PILOT_PREFER_REMOTE     = "1"
#
# Optional local defaults (only set if unset):
#   PILOT_LOCAL_KIND        (default "transformers")
#   PILOT_LOCAL_MODEL       (default "tinyllama/TinyLlama-1.1B-Chat-v1.0")
#
# Notes:
#   • If stdin is not a TTY or PILOT_AUTOCONFIG=0, we do nothing.
#   • If env is already set, we don’t prompt (unless force_prompt=True).
#   • Uses only the stdlib (urllib) and short timeouts.

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from urllib.parse import urlsplit, urlunsplit
from typing import Dict, Optional, Tuple

# Prefer our styled display if available; else fall back to print.
try:
    from .display import info, warn, success  # type: ignore
except Exception:  # pragma: no cover
    def info(msg: str) -> None: print(msg)
    def warn(msg: str) -> None: print(msg)
    def success(msg: str) -> None: print(msg)

PING_TIMEOUT = 6.0

# ──────────────────────────────────────────────────────────────────────────────
# Small HTTP helpers
# ──────────────────────────────────────────────────────────────────────────────

def _post_json(
    url: str, payload: Dict, timeout: float, headers: Optional[Dict[str, str]] = None
) -> Tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec B310 (stdlib)
        code = r.getcode()
        body = r.read().decode("utf-8", errors="replace")
    return code, body


def _ping_ollama(base_url: str, model: str, timeout: float = PING_TIMEOUT) -> Tuple[bool, str]:
    """
    POST /api/chat expecting a minimal JSON reply. Success on HTTP 2xx + JSON parse.
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": "ping"}],
        "options": {"temperature": 0.0, "num_predict": 1},
    }
    try:
        code, body = _post_json(url, payload, timeout)
        if not (200 <= code < 300):
            return False, f"http {code}"
        _ = json.loads(body)  # any JSON is fine; shapes vary across versions
        return True, "ok"
    except urllib.error.HTTPError as e:  # pragma: no cover
        return False, f"http {e.code}"
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"


def _ping_openai(
    base_url: str, model: str, api_key: Optional[str], timeout: float = PING_TIMEOUT
) -> Tuple[bool, str]:
    """
    POST /v1/chat/completions. Success on HTTP 2xx + choices[0] present.
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0,
        "max_tokens": 1,
    }
    headers: Dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        code, body = _post_json(url, payload, timeout, headers=headers)
        if code == 401 or code == 403:
            return False, f"http {code} (auth)"
        if not (200 <= code < 300):
            return False, f"http {code}"
        obj = json.loads(body)
        choices = obj.get("choices") or []
        if not choices:
            return False, "no choices in response"
        # Touch typical path but tolerate alternate shapes
        _ = choices[0].get("message", {}).get("content", choices[0].get("text", ""))
        return True, "ok"
    except urllib.error.HTTPError as e:  # pragma: no cover
        return False, f"http {e.code}"
    except Exception as e:  # pragma: no cover
        return False, f"{type(e).__name__}: {e}"

# ──────────────────────────────────────────────────────────────────────────────
# Interactive prompt helpers
# ──────────────────────────────────────────────────────────────────────────────

def _isatty() -> bool:
    try:
        return sys.stdin.isatty()
    except Exception:
        return False


def _ask(prompt: str, default: Optional[str] = None) -> str:
    if not _isatty():
        return default or ""
    try:
        s = input(prompt).strip()
        return s if s else (default or "")
    except EOFError:
        return default or ""


def _confirm(prompt: str, default: bool = False) -> bool:
    if not _isatty():
        return default
    suffix = " [Y/n]:" if default else " [y/N]:"
    s = _ask(prompt + suffix, None).lower()
    if s == "" and default:
        return True
    return s in {"y", "yes"}

# ──────────────────────────────────────────────────────────────────────────────
# URL normalization (robust to host[:port], http(s)://host[:port], IPv6)
# ──────────────────────────────────────────────────────────────────────────────

def _format_host_for_netloc(hostname: str) -> str:
    # Bracket IPv6 literals for netloc if not already bracketed
    if ":" in hostname and not (hostname.startswith("[") and hostname.endswith("]")):
        return f"[{hostname}]"
    return hostname


def _normalize_base_url(kind: str, host_or_url: str, port: Optional[str]) -> str:
    """
    Accepts:
      • "10.0.0.101" or "10.0.0.101:11434"
      • "http://10.0.0.101:11434"
      • "https://host:port"
      • "[2001:db8::1]" or "http://[2001:db8::1]:11434"
    Returns a full base URL with scheme and port, without duplicating ports.
    """
    s = (host_or_url or "").strip()
    default_port = "11434" if kind == "ollama" else "8000"

    # Ensure a scheme for parsing when the input is bare host[:port]
    candidate = s if s.startswith(("http://", "https://")) else ("http://" + s)
    parts = urlsplit(candidate)

    scheme = parts.scheme or "http"
    # If original had https://, keep it
    if s.startswith("https://"):
        scheme = "https"

    hostname = parts.hostname or ""
    parsed_port = parts.port  # may be None
    username = parts.username or ""
    password = parts.password or ""

    # If user typed something odd like just "http://", fall back to default host
    if not hostname:
        hostname = "localhost"

    # Decide the effective port:
    #   1) If URL had an explicit port, use it.
    #   2) Else if a separate `port` arg was provided, use it.
    #   3) Else use the kind default.
    effective_port = str(parsed_port) if parsed_port is not None else (str(port) if port else default_port)

    # Rebuild netloc, preserving userinfo if present, and bracketing IPv6
    host_for_netloc = _format_host_for_netloc(hostname)
    auth = ""
    if username:
        auth = username
        if password:
            auth += f":{password}"
        auth += "@"

    netloc = f"{auth}{host_for_netloc}:{effective_port}"

    # We only want the base; drop path/query/fragment
    return urlunsplit((scheme, netloc, "", "", ""))

# ──────────────────────────────────────────────────────────────────────────────
# Public entrypoint
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_brain_env(*, interactive: bool = True, force_prompt: bool = False) -> None:
    """
    If remote env is missing and interactive is True, prompt user for:
      • kind: ollama | openai
      • host/ip (or full URL) and port
      • model id / tag
      • api key (optional; used for openai-kind endpoints)
    Then verify and set envs in-process only (no file writes).
    """
    # Respect global opt‑out or non‑interactive contexts.
    if os.environ.get("PILOT_AUTOCONFIG", "1") in {"0", "false", "False"}:
        return
    if interactive and not _isatty():
        interactive = False

    # Already configured? Skip unless explicitly forced.
    have_remote = all(
        os.environ.get(k)
        for k in ("PILOT_REMOTE_KIND", "PILOT_REMOTE_URL", "PILOT_REMOTE_MODEL")
    )
    if have_remote and not force_prompt:
        return

    # Always ensure local defaults exist (they don't force usage).
    os.environ.setdefault("PILOT_LOCAL_KIND", "transformers")
    os.environ.setdefault("PILOT_LOCAL_MODEL", "tinyllama/TinyLlama-1.1B-Chat-v1.0")

    if not interactive:
        # Non‑interactive: nothing else to do.
        return

    info("No remote brain configured.")
    if not _confirm("Connect a remote brain now?", default=True):
        info("Continuing with local LLM only.")
        return

    # Defaults tailored to your LAN / Jetsons
    default_hosts = ["10.0.0.101", "10.0.0.100"]

    kind = _ask("Remote kind (ollama/openai) [ollama]: ", "ollama").lower()
    while kind not in {"ollama", "openai"}:
        kind = _ask("Please enter 'ollama' or 'openai' [ollama]: ", "ollama").lower()

    host_in = _ask(f"Brain host/IP or URL [{default_hosts[0]}]: ", default_hosts[0])
    base_url = _normalize_base_url(kind, host_in, None)

    # Suggested models; user can type any compatible tag/id.
    model_default = "llama3:70b-instruct" if kind == "ollama" else "mixtral-8x7b-instruct"
    model = _ask(f"Model tag/id [{model_default}]: ", model_default) or model_default

    api_key = ""
    if kind == "openai":
        api_key = _ask("API key (enter to skip): ", "")

    info(f"Verifying remote brain at {base_url} ({kind}:{model}) …")
    if kind == "ollama":
        ok, msg = _ping_ollama(base_url, model, timeout=PING_TIMEOUT)
    else:
        ok, msg = _ping_openai(base_url, model, api_key or None, timeout=PING_TIMEOUT)

    if not ok:
        warn(f"Remote verification failed: {msg}")
        if not _confirm("Use local only and continue?", default=True):
            warn("Keeping current session without remote config.")
        return

    # Success: set envs RAM‑only for this process
    os.environ["PILOT_REMOTE_KIND"] = kind
    os.environ["PILOT_REMOTE_URL"] = base_url
    os.environ["PILOT_REMOTE_MODEL"] = model
    os.environ["PILOT_PREFER_REMOTE"] = "1"
    if api_key:
        os.environ["PILOT_REMOTE_API_KEY"] = api_key

    # Friendly summary with copy‑ready exports (redacting the API key)
    success(f"🧩 Connected to brain at {base_url}  ({kind}:{model})")
    exports = [
        f'export PILOT_REMOTE_KIND="{kind}"',
        f'export PILOT_REMOTE_URL="{base_url}"',
        f'export PILOT_REMOTE_MODEL="{model}"',
        'export PILOT_PREFER_REMOTE="1"',
    ]
    if api_key:
        exports.append('export PILOT_REMOTE_API_KEY="***"  # (redacted)')
    info("Session env set (RAM only). To persist in your shell later, you can copy:")
    for line in exports:
        info("  " + line)
