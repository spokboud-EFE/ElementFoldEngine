# ElementFold · experience/adapters/audio.py
# Audio adapter = bridge from (model, prompt, style) → JSON payload suitable for the UI.
#
# Contract with the registry:
#   factory = AdapterRegistry.get("audio")
#   runner  = factory()                                # zero‑arg → callable
#   out     = runner(model, prompt, style)             # returns a dict (JSON‑serializable)
#
# Behavior (plain words):
#   • Optionally apply steering controls (β exposure, γ damping, ⛔ clamp) from a raw ℝ⁸ style vector or a dict.
#   • Accept prompt as str or dict; parse generation hints (sr / seconds / len) safely.
#   • Tokenize the prompt to build a seed batch of length T (clamped to model.seq_len).
#   • Decode tokens via the shared infer_loop ('greedy' by default; 'sample' is available).
#   • Convert tokens into unsigned 8‑bit PCM WAV (mono) in memory and return both base64 and a data URL.
#   • Make predictions visible: include a one‑line caption summarizing β/γ/⛔ and decode knobs.
#
# Why 8‑bit PCM?  It matches the project’s byte‑level tokenization in datasets/audio_folder.py:
#   training maps floats in [−1,1] → uint8 [0,255]. We reverse that for simple synthesis.

from __future__ import annotations
from typing import Any, Dict, Tuple, Optional

import base64           # base64 encoding for JSON transport
import io               # in‑memory WAV buffers
import re               # parse sr=, seconds=, len= from string prompts
import wave             # stdlib WAV writer

import torch

from .base import AdapterRegistry                         # 🗂 adapter registry
from elementfold.core.tokenizer import SimpleTokenizer         # ✴ tiny byte tokenizer
from elementfold.core.infer import infer_loop                  # ✴ unified decoding path across adapters

# Optional steering support: map raw ℝ⁸ → {'beta','gamma','clamp','style'}; optional caption via describe()
try:
    from ..steering import SteeringController             # 🎚 intent → control vector (and to_params)
    _HAS_STEER = True
except Exception:
    _HAS_STEER = False


# ──────────────────────────────────────────────────────────────────────────────
# Style → model control (β, γ, ⛔); accept dicts or raw vectors
# ──────────────────────────────────────────────────────────────────────────────

def _map_style_to_params(style: Any) -> Optional[Dict[str, float]]:
    """
    Accept either:
      • dict with {'beta','gamma','clamp'} floats,
      • raw ℝ⁸ vector (Tensor/list/tuple) from SteeringController → map via to_params(),
    and return a clean params dict or None.
    """
    if isinstance(style, dict) and all(k in style for k in ("beta", "gamma", "clamp")):
        try:
            return {
                "beta": float(style["beta"]),
                "gamma": float(style["gamma"]),
                "clamp": float(style["clamp"]),
            }
        except Exception:
            return None

    if _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        try:
            v = torch.as_tensor(style, dtype=torch.float32)
            mapped = SteeringController.to_params(v)  # → {'beta','gamma','clamp','style'}
            return {"beta": float(mapped["beta"]),
                    "gamma": float(mapped["gamma"]),
                    "clamp": float(mapped["clamp"])}
        except Exception:
            return None

    return None


def _apply_style_to_model(model, style) -> Dict[str, float]:
    """
    Determine controls from `style` and, if the model supports it, apply:
        model.apply_control(beta=?, gamma=?, clamp=?)
    Returns the dict actually applied (or {}).
    """
    params = _map_style_to_params(style)
    if params and hasattr(model, "apply_control"):
        try:
            model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        except Exception:
            pass  # Non‑fatal: model might not implement the exact signature
    return params or {}


def _summarize_style(style: Any, params: Dict[str, float]) -> str:
    """
    Human‑friendly single line. Prefer SteeringController.describe(raw ℝ⁸) when available;
    otherwise fall back to the applied params. Returns "" if nothing to say.
    """
    if _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        try:
            v = torch.as_tensor(style, dtype=torch.float32)
            # SteeringController.describe(...) is optional; fall back on failure.
            if hasattr(SteeringController, "describe"):
                return SteeringController.describe(v)  # e.g., "β=1.26  γ=0.43  ⛔=5.7  |  style≈[...]"
        except Exception:
            pass

    if params:
        return f"β={params['beta']:.2f}  γ={params['gamma']:.2f}  ⛔={params['clamp']:.1f}"

    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt hints: "sr=16000", "seconds=1.0", "len=16000" in str or dict
# ──────────────────────────────────────────────────────────────────────────────

_HINT_RE = re.compile(
    r"(sr|sample_rate|seconds|sec|len|length)\s*=\s*([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)

def _parse_prompt_hints(s_or_dict: Any, default_sr: int, default_len: int) -> Tuple[int, int]:
    """
    Extract simple generation hints from either a string prompt *or* a dict prompt:
      • sr / sample_rate = integer Hz (e.g., 16000)
      • seconds / sec    = float seconds → length = sr * seconds
      • len / length     = integer number of samples (tokens)
    Returns (sr, length_tokens) clamped to safe ranges.
    """
    sr = int(default_sr)
    length = int(default_len)

    # Dict path (structured)
    if isinstance(s_or_dict, dict):
        def _num(x):
            try:
                return None if x is None else float(x)
            except Exception:
                return None

        sr_from_dict = _num(s_or_dict.get("sr")) or _num(s_or_dict.get("sample_rate"))
        if sr_from_dict is not None:
            sr = int(sr_from_dict)

        seconds_from_dict = _num(s_or_dict.get("seconds")) or _num(s_or_dict.get("sec"))
        if seconds_from_dict is not None:
            length = int(max(1, round(sr * float(seconds_from_dict))))

        len_from_dict = _num(s_or_dict.get("len")) or _num(s_or_dict.get("length"))
        if len_from_dict is not None:
            length = int(len_from_dict)

    # String path (back‑compat / quick experiments)
    elif isinstance(s_or_dict, str):
        for key, val in _HINT_RE.findall(s_or_dict):
            key_l = key.lower()
            if key_l in ("sr", "sample_rate"):
                try:
                    sr = int(float(val))
                except Exception:
                    pass
            elif key_l in ("seconds", "sec"):
                try:
                    length = int(max(1, round(sr * float(val))))
                except Exception:
                    pass
            elif key_l in ("len", "length"):
                try:
                    length = int(float(val))
                except Exception:
                    pass

    # Rails: keep memory stable and WAV sane
    sr = int(max(8000, min(48000, sr)))          # 8 kHz … 48 kHz
    length = int(max(1, min(4 * sr, length)))    # up to ~4 seconds by default
    return sr, length


# ──────────────────────────────────────────────────────────────────────────────
# Tokens → WAV (unsigned 8‑bit mono), Base64 for transport
# ──────────────────────────────────────────────────────────────────────────────

def _tokens_to_wav_b64(tokens: torch.Tensor, sample_rate: int) -> Tuple[str, float]:
    """
    Take a 1‑D LongTensor (values 0..255), write an unsigned 8‑bit mono WAV into memory,
    and return (base64_string, duration_sec).
    """
    q = tokens.to(torch.uint8).contiguous().view(-1).cpu()   # bytes in [0,255]
    duration = float(q.numel()) / float(sample_rate)

    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)               # mono
        w.setsampwidth(1)               # 1 byte = 8‑bit unsigned PCM
        w.setframerate(int(sample_rate))
        w.writeframes(q.numpy().tobytes())

    wav_b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return wav_b64, duration


# ──────────────────────────────────────────────────────────────────────────────
# Core adapter runner (unified with infer_loop)
# ──────────────────────────────────────────────────────────────────────────────

def _run(model, prompt, style):
    """
    1) Optionally apply steering controls to the model (β, γ, ⛔).
    2) Parse prompt hints (sr=…, seconds=…, len=…) from str or dict.
    3) Build an input token batch (seed) of requested length T, clamped to model.seq_len.
    4) Decode via infer_loop (strategy='greedy' unless overridden in prompt dict['decode']).
    5) Pack tokens into a base64 WAV and return a JSON‑friendly dict with a human caption.
    """
    # 1) Steering (no‑op if style is None or model lacks apply_control)
    applied = _apply_style_to_model(model, style)
    caption = _summarize_style(style, applied)

    # 2) Determine target length and sample rate from prompt hints
    T_default = int(getattr(model, "seq_len", 128))  # model’s configured max length
    sr, T_req = _parse_prompt_hints(prompt, default_sr=16000, default_len=T_default)
    T = int(min(T_req, T_default))                   # always respect model’s max

    # 3) Prepare seed token batch from prompt text (for dict: use prompt.get("text", ""))
    tok = SimpleTokenizer()
    if isinstance(prompt, dict):
        text_seed = str(prompt.get("text", ""))
    else:
        text_seed = str(prompt or "")
    ids = tok.encode(text_seed) or [0]               # ensure at least one id

    # Device and batch pack
    try:
        dev = next(model.parameters()).device
    except Exception:
        dev = torch.device("cpu")
    x = torch.tensor(ids[:T], dtype=torch.long, device=dev).unsqueeze(0)  # (1,≤T)

    # Pad to T if needed (right‑pad zeros → neutral byte)
    if x.size(1) < T:
        pad = torch.zeros(1, T - x.size(1), dtype=torch.long, device=dev)
        x = torch.cat([x, pad], dim=1)

    # 4) Decode via the shared inference path for consistency
    # Allow dict prompts to pass decoding knobs: {'decode': {'strategy': 'sample', 'temperature': 0.8, ...}}
    strategy = "greedy"
    temperature = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    if isinstance(prompt, dict):
        dec = prompt.get("decode", {})
        if isinstance(dec, dict):
            strategy = str(dec.get("strategy", strategy))
            try:
                temperature = float(dec.get("temperature", temperature))
            except Exception:
                pass
            try:
                tk = dec.get("top_k", top_k)
                top_k = int(tk) if tk is not None else None
            except Exception:
                top_k = None
            try:
                tp = dec.get("top_p", top_p)
                tp = float(tp) if tp is not None else None
                top_p = tp if (tp is None or 0.0 < tp < 1.0) else None
            except Exception:
                top_p = None

    out = infer_loop(
        model,
        x=x,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    y = out["tokens"].squeeze(0)                 # (T,) int64 in [0..V−1]

    # Map decoded ids into 0..255 for audio (wrap if vocab>256; expand range if vocab<256)
    q = (y.to(torch.int64) % 256).to(torch.uint8)

    # 5) Pack into WAV and return payload
    wav_b64, duration = _tokens_to_wav_b64(q, sample_rate=sr)
    return {
        "wav_b64": wav_b64,                              # base64 WAV (mono, 8‑bit)
        "data_url": f"data:audio/wav;base64,{wav_b64}",  # handy for browsers
        "tokens": q.tolist(),                            # decoded token sequence (ints 0..255)
        "sr": int(sr),                                   # sample rate
        "duration_sec": float(duration),                 # seconds
        "applied": applied,                              # {'beta','gamma','clamp'} if steering was applied
        "decode": {                                      # echo knobs for traceability
            "strategy": strategy,
            "temperature": float(temperature),
            "top_k": (int(top_k) if top_k is not None else None),
            "top_p": (float(top_p) if top_p is not None else None),
        },
        "caption": caption,                              # human‑readable summary (β/γ/⛔, style sketch)
    }


# — Registry wiring: decorator form keeps registration concise and consistent —
@AdapterRegistry.register_fn("audio")
def make_audio_adapter():
    # Zero‑arg factory → runner(model, prompt, style) → dict
    return _run
