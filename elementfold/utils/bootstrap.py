# ElementFold · utils/bootstrap.py
# ──────────────────────────────────────────────────────────────────────────────
# RAM-only "brain" bootstrap:
#   • If remote LLM ("brain") env vars are not set, interactively ask for them.
#   • Verify connectivity (Ollama or OpenAI-compatible) with a tiny "ping".
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
#   • Uses stdlib urllib only; timeouts are short; no heavy imports.
#
from __future__ import annotations

import os
import sys
import json
import urllib.request
import urllib.error
from typing import Optional, Tuple

# Prefer our narrative display if available
try:
    from .display import info, warn, success
except Exception:  # fallback: plain print
    def info(msg: str) -> None:    print(msg)
    def warn(msg: str) -> None:    print(msg)
    def success(msg: str) -> None: print(msg)

# ————————————————————————————————————————————————————————————————
# Small HTTP helpers
# ————————————————————————————————————————————————————————————————

def _post_json(url: str, payload: dict, timeout: float, headers: Optional[dict] = None) -> Tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.getcode(), r.read().decode("utf-8")

def _ping_ollama(base_url: str, model: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """
    POST /api/chat with a 1-token reply. Success on 2xx and parsable JSON.
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
        ok = (200 <= code < 300)
        if not ok:
            return False, f"http {code}"
        _ = json.loads(body)  # ensure JSON
        return True, "ok"
    except Exception as e:
        return False, str(e)

def _ping_openai(base_url: str, model: str, api_key: Optional[str], timeout: float = 5.0) -> Tuple[bool, str]:
    """
    POST /v1/chat/completions. Success on 2xx and parsable JSON with choices[0].
    """
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0,
        "max_tokens": 1,
    }
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        code, body = _post_json(url, payload, timeout, headers=headers)
        ok = (200 <= code < 300)
        if not ok:
            return False, f"http {code}"
        obj = json.loads(body)
        _ = obj["choices"][0]["message"]["content"]
        return True, "ok"
    except Exception as e:
        return False, str(e)

# ————————————————————————————————————————————————————————————————
# Interactive prompt helpers
# ————————————————————————————————————————————————————————————————

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
    if s == "" and default: return True
    return s in {"y", "yes"}

# ————————————————————————————————————————————————————————————————
# Public entrypoint
# ————————————————————————————————————————————————————————————————

def bootstrap_brain_env(*, interactive: bool = True, force_prompt: bool = False) -> None:
    """
    If remote env is missing and interactive is True, prompt user for:
      • kind: ollama | openai
      • host/ip and port
      • model id / tag
      • api key (optional; used for openai-kind endpoints)
    Then verify and set envs in-process only.
    """
    # Respect an opt-out switch or non-tty contexts
    if os.environ.get("PILOT_AUTOCONFIG", "1") in {"0", "false", "False"}:
        return
    if interactive and not _isatty():
        interactive = False

    # Already configured? skip unless forced
    has_remote = all(os.environ.get(k) for k in ("PILOT_REMOTE_KIND", "PILOT_REMOTE_URL", "PILOT_REMOTE_MODEL"))
    if has_remote and not force_prompt:
        return

    if not interactive:
        # Non-interactive: ensure local defaults exist; nothing else to do
        os.environ.setdefault("PILOT_LOCAL_KIND", "transformers")
        os.environ.setdefault("PILOT_LOCAL_MODEL", "tinyllama/TinyLlama-1.1B-Chat-v1.0")
        return

    info("No remote brain configured.")
    if not _confirm("Connect a remote brain now?", default=True):
        info("Continuing with local LLM only.")
        os.environ.setdefault("PILOT_LOCAL_KIND", "transformers")
        os.environ.setdefault("PILOT_LOCAL_MODEL", "tinyllama/TinyLlama-1.1B-Chat-v1.0")
        return

    # Defaults tailored to your Jetsons
    default_hosts = ["10.0.0.101", "10.0.0.100"]
    kind = _ask("Remote kind (ollama/openai) [ollama]: ", "ollama").lower()
    while kind not in {"ollama", "openai"}:
        kind = _ask("Please enter 'ollama' or 'openai' [ollama]: ", "ollama").lower()

    host = _ask(f"Brain host/IP [{default_hosts[0]}]: ", default_hosts[0]) or default_hosts[0]
    if ":" in host:
        base_host, _, port = host.partition(":")
    else:
        base_host, port = host, ""
    if not port:
        port = "11434" if kind == "ollama" else "8000"
    base_url = f"http://{base_host}:{port}"

    # Model suggestions: you can type any tag/id here
    model_default = "llama3:70b-instruct" if kind == "ollama" else "mixtral-8x7b-instruct"
    model = _ask(f"Model tag/id [{model_default}]: ", model_default) or model_default

    api_key = ""
    if kind == "openai":
        api_key = _ask("API key (enter to skip): ", "")

    info(f"Verifying remote brain at {base_url} ({kind}:{model}) …")
    if kind == "ollama":
        ok, msg = _ping_ollama(base_url, model, timeout=6.0)
    else:
        ok, msg = _ping_openai(base_url, model, api_key or None, timeout=6.0)

    if not ok:
        warn(f"Remote verification failed: {msg}")
        if not _confirm("Use local only and continue?", default=True):
            warn("Keeping current session without remote config.")
        # Ensure local defaults exist
        os.environ.setdefault("PILOT_LOCAL_KIND", "transformers")
        os.environ.setdefault("PILOT_LOCAL_MODEL", "tinyllama/TinyLlama-1.1B-Chat-v1.0")
        return

    # Success: set envs RAM-only for this process
    os.environ["PILOT_REMOTE_KIND"] = kind
    os.environ["PILOT_REMOTE_URL"] = base_url
    os.environ["PILOT_REMOTE_MODEL"] = model
    os.environ["PILOT_PREFER_REMOTE"] = "1"
    if api_key:
        os.environ["PILOT_REMOTE_API_KEY"] = api_key

    # Ensure local defaults (only if absent)
    os.environ.setdefault("PILOT_LOCAL_KIND", "transformers")
    os.environ.setdefault("PILOT_LOCAL_MODEL", "tinyllama/TinyLlama-1.1B-Chat-v1.0")

    success(f"🧩 Connected to brain {base_host}:{port}  ({kind}:{model})")
    # Show copy/paste exports (not written to disk)
    exports = [
        f'export PILOT_REMOTE_KIND="{kind}"',
        f'export PILOT_REMOTE_URL="{base_url}"',
        f'export PILOT_REMOTE_MODEL="{model}"',
        'export PILOT_PREFER_REMOTE="1"',
    ]
    if api_key:
        exports.append('export PILOT_REMOTE_API_KEY="***"  # (redacted)')
    info("Session env (RAM only). If you want to persist in your shell later, you can copy:")
    for line in exports:
        info("  " + line)
