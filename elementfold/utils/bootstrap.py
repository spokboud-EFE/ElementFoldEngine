# ElementFold · utils/bootstrap.py
# ============================================================
# RAM-only "brain" bootstrap
# ------------------------------------------------------------
#  • Prompts for remote brain credentials (Ollama / OpenAI)
#  • Verifies connectivity with a quick JSON ping
#  • Stores values in os.environ (session only)
# ============================================================

from __future__ import annotations
import json, os, urllib.request, urllib.error
from typing import Dict, Optional, Tuple

PING_TIMEOUT = 6.0

# ------------------------------------------------------------
# HTTP helper
# ------------------------------------------------------------
def _post_json(url: str, payload: Dict, timeout: float,
               headers: Optional[Dict[str, str]] = None) -> Tuple[int, str]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:  # nosec
        return r.getcode(), r.read().decode("utf-8", errors="replace")

# ------------------------------------------------------------
# Ping helpers
# ------------------------------------------------------------
def _ping_ollama(base_url: str, model: str) -> bool:
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model, "stream": False,
        "messages": [{"role": "user", "content": "ping"}],
        "options": {"temperature": 0.0, "num_predict": 1}
    }
    try:
        code, _ = _post_json(url, payload, PING_TIMEOUT)
        return 200 <= code < 300
    except Exception:
        return False

def _ping_openai(base_url: str, model: str, api_key: Optional[str]) -> bool:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ping"}],
        "temperature": 0.0, "max_tokens": 1
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        code, _ = _post_json(url, payload, PING_TIMEOUT, headers=headers)
        return 200 <= code < 300
    except Exception:
        return False

# ------------------------------------------------------------
# Interactive helpers
# ------------------------------------------------------------
def _ask(prompt: str, default: Optional[str] = None) -> str:
    try:
        s = input(prompt).strip()
        return s or (default or "")
    except EOFError:
        return default or ""

def _confirm(prompt: str, default: bool = False) -> bool:
    suffix = " [Y/n]:" if default else " [y/N]:"
    s = _ask(prompt + suffix, "").lower()
    if not s:
        return default
    return s in {"y", "yes"}

# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------
def bootstrap_brain_env(*, interactive: bool = True, force_prompt: bool = False) -> None:
    """Ask for or auto-set remote brain credentials, stored in os.environ only."""
    # skip if already set
    if not force_prompt and all(
        os.environ.get(k) for k in ("PILOT_REMOTE_KIND", "PILOT_REMOTE_URL", "PILOT_REMOTE_MODEL")
    ):
        return

    if not interactive:
        return

    print("No remote brain configured.")
    if not _confirm("Connect a remote brain now?", default=True):
        print("Continuing with local brain only.")
        return

    kind = _ask("Remote kind (ollama/openai) [ollama]: ", "ollama").lower()
    while kind not in {"ollama", "openai"}:
        kind = _ask("Please enter 'ollama' or 'openai' [ollama]: ", "ollama").lower()

    host = _ask("Brain host or base URL [http://127.0.0.1:11434]: ", "http://127.0.0.1:11434")
    model = _ask("Model name/tag [llama3:8b-instruct]: ", "llama3:8b-instruct")
    api_key = ""
    if kind == "openai":
        api_key = _ask("API key (press Enter to skip): ", "")

    print(f"Verifying {kind} brain at {host} ...")
    ok = _ping_ollama(host, model) if kind == "ollama" else _ping_openai(host, model, api_key)
    if not ok:
        print("⚠ Connection failed. Continuing without remote brain.")
        return

    os.environ["PILOT_REMOTE_KIND"] = kind
    os.environ["PILOT_REMOTE_URL"] = host
    os.environ["PILOT_REMOTE_MODEL"] = model
    os.environ["PILOT_PREFER_REMOTE"] = "1"
    if api_key:
        os.environ["PILOT_REMOTE_API_KEY"] = api_key

    print(f"🧩 Connected to brain at {host} ({kind}:{model}) — session env set in RAM.")
