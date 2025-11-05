# ElementFold · server_api.py
# Minimal but complete HTTP API schema + helpers.
# Goals (plain words):
#   • Define small dataclasses for requests/responses (infer/steer/health/error).
#   • Provide robust JSON (de)serialization with defaults and validation.
#   • Offer tiny utilities to resolve text→tokens (without importing the model),
#     and to clamp/clean decode knobs (temperature/top‑k/top‑p).
#
# Extras (quality-of-life):
#   • JSON is sanitized so NaN/±Inf never leak to clients (RFC‑clean).
#   • Strategy synonyms accepted ('argmax'→'greedy', 'sampling'→'sample').
#   • Token lists accept tuples/iterables in addition to lists.

from __future__ import annotations                # ✴ forward type hints
from dataclasses import dataclass, asdict         # ✴ tiny, typed records
from typing import Optional, List, Dict, Any, Iterable
import json                                      # ✴ JSON encode/decode
import math                                      # ✴ finite checks


# ———————————————————————————————————————————————————————————
# Request / Response schemas (small, explicit dataclasses)
# ———————————————————————————————————————————————————————————

@dataclass
class InferRequest:                               # ✴ /infer input
    tokens: Optional[List[int]] = None            # optional explicit token ids (wins over text)
    text: Optional[str] = None                    # optional prompt; used if tokens is None
    strategy: str = "greedy"                      # 'greedy' | 'sample'
    temperature: float = 1.0                      # >0 when sampling
    top_k: Optional[int] = None                   # keep K tokens (sampling)
    top_p: Optional[float] = None                 # nucleus prob in (0,1) (sampling)
    max_len: Optional[int] = None                 # optional hard cap for sequence length

@dataclass
class InferResponse:                              # ✴ /infer output
    tokens: List[int]                             # decoded token ids
    ledger: List[float]                           # flattened ledger coordinates
    text: Optional[str] = None                    # optional detokenized text

@dataclass
class SteerRequest:                                # ✴ /steer input
    prompt: str                                    # human intent
    modality: str = "language"                     # adapter key (e.g., 'language', 'vision')

@dataclass
class SteerResponse:                                # ✴ /steer output
    output: Any                                     # adapter‑specific (often a string)
    params: Optional[Dict[str, Any]] = None         # optional applied {beta,gamma,clamp,...}

@dataclass
class HealthResponse:                               # ✴ /health
    status: str = "ok"                              # 'ok'|'degraded'...
    version: str = "unknown"                        # project version string (server fills this)
    device: str = "cpu"                             # 'cpu'|'cuda'
    model_ready: bool = False                       # whether weights are live

@dataclass
class TrainRequest:                                  # ✴ /train input (optional API)
    steps: int = 200                                 # number of optimization steps

@dataclass
class TrainResponse:                                 # ✴ /train output
    trained: bool = True                             # whether training ran
    steps: int = 0                                   # steps actually performed

@dataclass
class ErrorResponse:                                  # ✴ unified errors
    code: str                                        # short code (e.g., 'bad_request')
    message: str                                     # human‑readable description
    details: Optional[Dict[str, Any]] = None         # optional payload for debugging


# ———————————————————————————————————————————————————————————
# JSON helpers (bytes⇄dict) with dataclass support
# ———————————————————————————————————————————————————————————

def parse_json(body: bytes) -> Dict[str, Any]:
    """Decode bytes→dict; empty body → {}. Invalid JSON raises ValueError (caught by server)."""
    return json.loads(body.decode("utf-8")) if body else {}


def _json_sanitize(x: Any) -> Any:
    """
    Walk an object and replace non‑finite floats (NaN/±Inf) with finite surrogates.
    This guarantees RFC‑valid JSON (we also set allow_nan=False in dumps).
    """
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]       # tuples → lists (JSON has no tuples)
    if isinstance(x, float):
        if not math.isfinite(x):
            return 0.0                               # conservative finite replacement
        return float(x)
    return x


def to_json(obj: Any) -> bytes:
    """Encode Python objects → bytes. Dataclasses are converted via asdict()."""
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)                           # ✴ serialize @dataclass
    obj = _json_sanitize(obj)                       # ✴ ensure RFC‑clean numbers
    return json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")


# ———————————————————————————————————————————————————————————
# Light validation + coercion utilities for /infer
# ———————————————————————————————————————————————————————————

_ALLOWED_STRATEGIES = {"greedy", "sample"}         # ✴ whitelist for decode
_STRATEGY_ALIASES = {                              # ✴ friendly synonyms
    "argmax": "greedy",
    "deterministic": "greedy",
    "sampling": "sample",
}

def _as_int_list(x: Any) -> Optional[List[int]]:
    """Best‑effort convert to list[int] or None (accepts lists/tuples/iterables of numbers)."""
    if x is None:
        return None
    if isinstance(x, (list, tuple)):
        seq = x
    elif isinstance(x, Iterable) and not isinstance(x, (str, bytes, bytearray)):
        seq = list(x)
    else:
        return None
    out: List[int] = []
    for v in seq:
        try:
            out.append(int(v))
        except Exception:
            return None
    return out


def _clean_strategy(s: Any) -> str:
    """Normalize strategy value; fall back to 'greedy'. Accepts common aliases."""
    try:
        s = str(s).lower().strip()
    except Exception:
        return "greedy"
    s = _STRATEGY_ALIASES.get(s, s)
    return s if s in _ALLOWED_STRATEGIES else "greedy"


def _clamp_temperature(t: Any) -> float:
    """Map to a safe positive temperature (≥1e‑8)."""
    try:
        t = float(t)
    except Exception:
        return 1.0
    return max(1e-8, t)


def _clean_top_k(k: Any) -> Optional[int]:
    """Map to None or positive int."""
    if k is None:
        return None
    try:
        k = int(k)
    except Exception:
        return None
    return k if k > 0 else None


def _clean_top_p(p: Any) -> Optional[float]:
    """Map to None or float in (0,1)."""
    if p is None:
        return None
    try:
        p = float(p)
    except Exception:
        return None
    return p if 0.0 < p < 1.0 else None


def _clean_max_len(m: Any) -> Optional[int]:
    """Map to None or positive int."""
    if m is None:
        return None
    try:
        m = int(m)
    except Exception:
        return None
    return m if m > 0 else None


def coerce_infer_request(payload: Dict[str, Any]) -> InferRequest:
    """
    Convert a raw dict (from JSON) into an InferRequest with cleaned fields.
    """
    tokens = _as_int_list(payload.get("tokens"))
    text = payload.get("text")
    strategy = _clean_strategy(payload.get("strategy", "greedy"))
    temperature = _clamp_temperature(payload.get("temperature", 1.0))
    top_k = _clean_top_k(payload.get("top_k"))
    top_p = _clean_top_p(payload.get("top_p"))
    max_len = _clean_max_len(payload.get("max_len"))
    return InferRequest(tokens=tokens, text=text, strategy=strategy,
                        temperature=temperature, top_k=top_k, top_p=top_p, max_len=max_len)


def validate_infer_request(req: InferRequest) -> Optional[ErrorResponse]:
    """
    Sanity checks:
      • at least one of (tokens, text) must be provided,
      • strategy must be in the allowed set.
    Returns None when valid, or an ErrorResponse.
    """
    if (req.tokens is None) and (not req.text):
        return ErrorResponse(code="bad_request",
                             message="provide either 'tokens' or 'text' in request body")
    if req.strategy not in _ALLOWED_STRATEGIES:
        return ErrorResponse(code="bad_request",
                             message=f"unknown strategy: {req.strategy!r}")
    return None


# ———————————————————————————————————————————————————————————
# Tokenization helpers (pure‑Python; server wires a tokenizer)
# ———————————————————————————————————————————————————————————

def resolve_tokens(req: InferRequest, tokenizer, vocab: int, seq_len: int) -> List[int]:
    """
    Decide which token sequence to use:
      • if req.tokens is set → clamp ids into [0, vocab) and truncate to max_len/seq_len,
      • else encode req.text with the provided tokenizer, then truncate.
    Returns a 1‑D list[int].
    """
    limit = req.max_len if (req.max_len is not None) else seq_len
    limit = int(max(1, min(limit, seq_len)))                 # ✴ hard cap

    if req.tokens is not None:
        ids = [max(0, min(int(t), vocab - 1)) for t in (req.tokens or [0])]
        return ids[:limit]

    # Fallback to text encoding
    s = req.text if (req.text is not None) else ""
    try:
        ids = tokenizer.encode(s)                            # project‑local SimpleTokenizer
    except Exception:
        ids = []
    ids = [max(0, min(int(t), vocab - 1)) for t in (ids or [0])]
    return ids[:limit]


# ———————————————————————————————————————————————————————————
# Response packers
# ———————————————————————————————————————————————————————————

def pack_infer_response(tokens_tensor, ledger_tensor, tokenizer=None) -> InferResponse:
    """
    Convert model outputs to an InferResponse:
      • tokens_tensor: (B,T) int64 (we use the first row),
      • ledger_tensor: (B,T) float (we flatten the first row),
      • tokenizer (optional): if provided, we detokenize to .text.
    """
    # Defensive shape handling
    tt = tokens_tensor.detach().cpu()
    lt = ledger_tensor.detach().cpu()
    if tt.dim() == 1:
        tokens = tt.tolist()
    else:
        tokens = tt[0].tolist()
    if lt.dim() == 1:
        ledger = lt.tolist()
    else:
        ledger = lt[0].tolist()

    text = None
    if tokenizer is not None:
        try:
            text = tokenizer.decode(tokens)
        except Exception:
            text = None

    return InferResponse(tokens=tokens, ledger=ledger, text=text)


def pack_error(code: str, message: str, details: Dict[str, Any] | None = None) -> ErrorResponse:
    """Small convenience to build ErrorResponse consistently."""
    return ErrorResponse(code=code, message=message, details=details)
