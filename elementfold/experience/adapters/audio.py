# ElementFold · experience/adapters/audio.py
# Audio adapter = bridge from (model, prompt, style) → {wav_b64, tokens, sr, duration_sec}.
# Contract with the registry:
#   factory = AdapterRegistry.get("audio")
#   runner  = factory()                                # zero‑arg → callable
#   out     = runner(model, prompt, style)             # returns a dict (JSON‑serializable)
#
# Behavior:
#   • Optionally apply steering controls (β, γ, ⛔) from a raw ℝ⁸ style vector.
#   • Tokenize the prompt (fallback to a neutral seed if empty).
#   • Run a forward pass and greedy‑decode tokens (0..255).
#   • Convert tokens into mono 8‑bit PCM WAV in memory (base64 string).
#   • Return a compact dict: {'wav_b64','tokens','sr','duration_sec','applied'}.
#
# Why 8‑bit PCM?  It matches our simple byte‑level tokenization:
#   dataset/audio_folder.py maps floats in [-1,1] → uint8 [0,255].
# For synthesis, we reverse that choice and write an unsigned 8‑bit PCM WAV (portable, dependency‑free).

from __future__ import annotations

import io                 # ✴ in‑memory buffers (BytesIO)
import re                 # ✴ tiny prompt hint parsing (sr=, seconds=, len=)
import base64             # ✴ base64 encode WAV bytes for JSON transport
import wave               # ✴ write a minimal WAV header and frames (stdlib)
import torch              # ✴ tensors
from .base import AdapterRegistry  # 🗂 registry for modality adapters
from ...tokenizer import SimpleTokenizer  # ✴ prompt → seed tokens

# Optional: interpret a raw ℝ⁸ style vector → (beta,gamma,clamp,style₅)
try:
    from ..steering import SteeringController
    _HAS_STEER = True
except Exception:
    _HAS_STEER = False


# ———————————————————————————————————————————————————————————
# Style → model control (β, γ, ⛔); accept dicts or raw vectors
# ———————————————————————————————————————————————————————————

def _apply_style_to_model(model, style):
    """
    If `style` is a dict with keys beta/gamma/clamp, apply them directly.
    If `style` looks like the raw ℝ⁸ vector from SteeringController, map it via to_params().
    If the model exposes .apply_control(beta=?, gamma=?, clamp=?), push controls in.
    Return the dict actually applied (or {} if nothing applied).
    """
    params = None

    # Case A: explicit dictionary
    if isinstance(style, dict) and all(k in style for k in ("beta", "gamma", "clamp")):
        params = {"beta": float(style["beta"]), "gamma": float(style["gamma"]), "clamp": float(style["clamp"])}

    # Case B: raw vector from SteeringController (ℝ⁸)
    elif _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        v = torch.as_tensor(style, dtype=torch.float32)
        params = SteeringController.to_params(v)  # → {'beta','gamma','clamp','style'}

    # Push into the model if supported
    if params and hasattr(model, "apply_control"):
        model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])

    return params or {}


# ———————————————————————————————————————————————————————————
# Prompt hints: allow "sr=16000", "seconds=1.0", or "len=16000"
# ———————————————————————————————————————————————————————————

_HINT_RE = re.compile(r"(sr|sample_rate|seconds|len|length)\s*=\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def _parse_prompt_hints(s: str, default_sr: int, default_len: int) -> tuple[int, int]:
    """
    Extract simple generation hints from the prompt:
      • sr / sample_rate = integer Hz (e.g., 16000)
      • seconds         = float seconds → length = sr * seconds
      • len / length    = integer number of samples (tokens)
    Returns (sr, length_tokens) clamped to reasonable ranges.
    """
    sr = int(default_sr)
    length = int(default_len)

    if not s:
        return sr, length

    for key, val in _HINT_RE.findall(s):
        key_l = key.lower()
        if key_l in ("sr", "sample_rate"):
            try:
                sr = int(float(val))
            except Exception:
                pass
        elif key_l == "seconds":
            try:
                sec = float(val)
                length = int(max(1, round(sr * sec)))
            except Exception:
                pass
        elif key_l in ("len", "length"):
            try:
                length = int(float(val))
            except Exception:
                pass

    # Clamp to safe bounds (keeps memory stable and WAV sane)
    sr = int(max(8000, min(48000, sr)))          # 8 kHz … 48 kHz
    length = int(max(1, min(4 * sr, length)))    # up to 4 seconds by default
    return sr, length


# ———————————————————————————————————————————————————————————
# Tokens → WAV (unsigned 8‑bit mono), Base64 for transport
# ———————————————————————————————————————————————————————————

def _tokens_to_wav_b64(tokens: torch.Tensor, sample_rate: int) -> tuple[str, float]:
    """
    Take a 1‑D LongTensor (values 0..255), write an unsigned 8‑bit mono WAV into memory,
    and return (base64_string, duration_sec).
    """
    # Ensure contiguous 1‑D uint8 for wave.writeframes
    q = tokens.to(torch.uint8).contiguous().view(-1)          # bytes in [0,255]
    duration = float(q.numel()) / float(sample_rate)          # seconds = samples / sr

    # Build an in‑memory WAV file (mono, 8‑bit)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)               # mono
        w.setsampwidth(1)               # 1 byte = 8‑bit unsigned PCM
        w.setframerate(int(sample_rate))
        w.writeframes(q.numpy().tobytes())

    wav_b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return wav_b64, duration


# ———————————————————————————————————————————————————————————
# Core adapter runner
# ———————————————————————————————————————————————————————————

def _run(model, prompt, style):
    """
    1) Optionally apply steering controls to the model (β, γ, ⛔).
    2) Parse prompt hints (sr=…, seconds=…, len=…).
    3) Build an input token batch (seed) and clip/pad to desired length.
    4) Forward once, greedy‑decode tokens.
    5) Pack tokens into a base64 WAV and return a JSON‑friendly dict.
    """
    # 1) Steering (no‑op if style is None or model lacks apply_control)
    applied = _apply_style_to_model(model, style)

    # 2) Generation hints (sample rate and length)
    T_default = int(getattr(model, "seq_len", 128))    # model’s configured max length
    sr, T = _parse_prompt_hints(prompt or "", default_sr=16000, default_len=T_default)

    # 3) Seed tokens from the prompt (byte tokenizer); ensure at least one id
    tok = SimpleTokenizer()
    ids = tok.encode(prompt or "")
    if len(ids) == 0:
        ids = [0]                                      # neutral seed if prompt empty

    # Build a (1,T) tensor on the model’s device; pad or truncate to the requested T
    dev = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)    # (1,L)
    if x.size(1) < T:
        pad = torch.zeros(1, T - x.size(1), dtype=torch.long, device=dev)
        x = torch.cat([x, pad], dim=1)
    elif x.size(1) > T:
        x = x[:, :T]

    # 4) Forward pass and greedy decode tokens in [0..V-1]; we then clamp to [0..255] for audio
    model_was_training = getattr(model, "training", False)
    model.eval()
    with torch.inference_mode():
        logits, _X = model(x)                          # (1,T,V), (1,T)
        y = logits.argmax(dim=-1).squeeze(0)          # (T,)
    if model_was_training:
        model.train()# ElementFold · experience/adapters/audio.py
# Audio adapter = bridge from (model, prompt, style) → {wav_b64, tokens, sr, duration_sec}.
# Contract with the registry:
#   factory = AdapterRegistry.get("audio")
#   runner  = factory()                                # zero‑arg → callable
#   out     = runner(model, prompt, style)             # returns a dict (JSON‑serializable)
#
# Behavior:
#   • Optionally apply steering controls (β, γ, ⛔) from a raw ℝ⁸ style vector or a dict.
#   • Accept prompt as str or dict; parse generation hints (sr / seconds / len) safely.
#   • Tokenize the prompt to build a seed batch of length T.
#   • Decode tokens via the shared infer_loop ('greedy' by default; supports sampling if requested).
#   • Convert tokens into unsigned 8‑bit PCM WAV (mono) in memory and return base64.
#
# Why 8‑bit PCM?  It matches the project’s byte‑level tokenization in datasets/audio_folder.py:
#   training maps floats in [-1,1] → uint8 [0,255]. We reverse that for simple synthesis.

from __future__ import annotations

import io           # in‑memory WAV buffers
import re           # parsing sr=, seconds=, len= from string prompts
import base64       # base64 encoding for JSON transport
import wave         # stdlib WAV writer
from typing import Any, Dict, Tuple

import torch

from .base import AdapterRegistry
from ...tokenizer import SimpleTokenizer
from ...infer import infer_loop  # unify decoding path across adapters

# Optional steering support: map raw ℝ⁸ → {'beta','gamma','clamp','style'}
try:
    from ..steering import SteeringController
    _HAS_STEER = True
except Exception:
    _HAS_STEER = False


# ———————————————————————————————————————————————————————————
# Style → model control (β, γ, ⛔); accept dicts or raw vectors
# ———————————————————————————————————————————————————————————

def _apply_style_to_model(model, style) -> Dict[str, Any]:
    """
    If `style` is a dict with keys beta/gamma/clamp, apply them directly.
    If `style` looks like the raw ℝ⁸ vector from SteeringController, map via to_params().
    If the model exposes .apply_control(beta=?, gamma=?, clamp=?), push controls in.
    Return the dict actually applied (or {} if nothing applied).
    """
    params = None

    # Case A: explicit dictionary
    if isinstance(style, dict) and all(k in style for k in ("beta", "gamma", "clamp")):
        params = {"beta": float(style["beta"]), "gamma": float(style["gamma"]), "clamp": float(style["clamp"])}

    # Case B: raw vector from SteeringController (ℝ⁸)
    elif _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        v = torch.as_tensor(style, dtype=torch.float32)
        params = SteeringController.to_params(v)  # → {'beta','gamma','clamp','style'}

    # Push into the model if supported
    if params and hasattr(model, "apply_control"):
        model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])

    return params or {}


# ———————————————————————————————————————————————————————————
# Prompt hints: "sr=16000", "seconds=1.0", "len=16000" in str or dict
# ———————————————————————————————————————————————————————————

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

    # Dict path (preferred when caller can structure the request)
    if isinstance(s_or_dict, dict):
        def _get_num(key):
            v = s_or_dict.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except Exception:
                return None

        sr_from_dict = _get_num("sr") or _get_num("sample_rate")
        if sr_from_dict is not None:
            sr = int(sr_from_dict)

        seconds_from_dict = _get_num("seconds") or _get_num("sec")
        if seconds_from_dict is not None:
            length = int(max(1, round(sr * float(seconds_from_dict))))

        len_from_dict = _get_num("len") or _get_num("length")
        if len_from_dict is not None:
            length = int(len_from_dict)

    # String path (back‑compat, quick experiments)
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
                    sec = float(val)
                    length = int(max(1, round(sr * sec)))
                except Exception:
                    pass
            elif key_l in ("len", "length"):
                try:
                    length = int(float(val))
                except Exception:
                    pass

    # Clamp to safe bounds (keeps memory stable and WAV sane)
    sr = int(max(8000, min(48000, sr)))          # 8 kHz … 48 kHz
    length = int(max(1, min(4 * sr, length)))    # up to ~4 seconds by default
    return sr, length


# ———————————————————————————————————————————————————————————
# Tokens → WAV (unsigned 8‑bit mono), Base64 for transport
# ———————————————————————————————————————————————————————————

def _tokens_to_wav_b64(tokens: torch.Tensor, sample_rate: int) -> Tuple[str, float]:
    """
    Take a 1‑D LongTensor (values 0..255), write an unsigned 8‑bit mono WAV into memory,
    and return (base64_string, duration_sec).
    """
    # Ensure contiguous 1‑D uint8 on CPU for wave.writeframes
    q = tokens.to(torch.uint8).contiguous().view(-1).cpu()   # bytes in [0,255]
    duration = float(q.numel()) / float(sample_rate)         # seconds = samples / sr

    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)               # mono
        w.setsampwidth(1)               # 1 byte = 8‑bit unsigned PCM
        w.setframerate(int(sample_rate))
        w.writeframes(q.numpy().tobytes())

    wav_b64 = base64.b64encode(bio.getvalue()).decode("ascii")
    return wav_b64, duration


# ———————————————————————————————————————————————————————————
# Core adapter runner (unified with infer_loop)
# ———————————————————————————————————————————————————————————

def _run(model, prompt, style):
    """
    1) Optionally apply steering controls to the model (β, γ, ⛔).
    2) Parse prompt hints (sr=…, seconds=…, len=…) from str or dict.
    3) Build an input token batch (seed) of length T on the model’s device.
    4) Decode via infer_loop (strategy='greedy' unless overridden).
    5) Pack tokens into a base64 WAV and return a JSON‑friendly dict.
    """
    # 1) Steering (no‑op if style is None or model lacks apply_control)
    applied = _apply_style_to_model(model, style)

    # 2) Determine target length and sample rate from prompt hints
    T_default = int(getattr(model, "seq_len", 128))  # model’s configured max length
    sr, T = _parse_prompt_hints(prompt, default_sr=16000, default_len=T_default)

    # 3) Prepare seed token batch from prompt text (for dict: use prompt.get("text", ""))
    tok = SimpleTokenizer()
    if isinstance(prompt, dict):
        text_seed = str(prompt.get("text", ""))
    else:
        text_seed = str(prompt or "")
    ids = tok.encode(text_seed) or [0]               # ensure at least one id

    dev = next(model.parameters()).device
    x = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)  # (1,L)
    # Pad/trim to requested T
    if x.size(1) < T:
        pad = torch.zeros(1, T - x.size(1), dtype=torch.long, device=dev)
        x = torch.cat([x, pad], dim=1)
    elif x.size(1) > T:
        x = x[:, :T]

    # 4) Decode via the shared inference path for consistency with other adapters
    # Allow dict prompts to pass decoding knobs: {'decode': {'strategy': 'sample', 'temperature': 0.8, ...}}
    strategy = "greedy"
    temperature = 1.0
    top_k = None
    top_p = None
    if isinstance(prompt, dict):
        dec = prompt.get("decode", {})
        if isinstance(dec, dict):
            strategy = str(dec.get("strategy", strategy))
            temperature = float(dec.get("temperature", temperature))
            top_k = dec.get("top_k", top_k)
            top_p = dec.get("top_p", top_p)

    out = infer_loop(
        model,
        x=x,
        strategy=strategy,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    y = out["tokens"].squeeze(0)                 # (T,) int64 in [0..V-1]
    # Map decoded ids into 0..255 for audio (wrap if vocab>256; expand range if vocab<256)
    q = (y.to(torch.int64) % 256).to(torch.uint8)

    # 5) Pack into WAV and return payload
    wav_b64, duration = _tokens_to_wav_b64(q, sample_rate=sr)
    return {
        "wav_b64": wav_b64,                             # base64 WAV (mono, 8‑bit)
        "data_url": f"data:audio/wav;base64,{wav_b64}", # handy for browsers
        "tokens": q.tolist(),                           # decoded token sequence (ints 0..255)
        "sr": int(sr),                                  # sample rate
        "duration_sec": float(duration),                # seconds
        "applied": applied,                             # {'beta','gamma','clamp'} if steering was applied
    }


# — registry wiring: zero‑arg factory that returns the runner —
AdapterRegistry.register("audio", lambda: _runner)

def _runner(model, prompt, style):
    return _run(model, prompt, style)


    # Map decoded ids into 0..255 byte range (if vocab > 256, wrap; if < 256, pad range)
    q = (y.to(torch.int64) % 256).to(torch.uint8)      # audio tokens (uint8)

    # 5) Pack into a WAV and return payload
    wav_b64, duration = _tokens_to_wav_b64(q, sample_rate=sr)
    return {
        "wav_b64": wav_b64,                            # base64 WAV (mono, 8‑bit)
        "data_url": f"data:audio/wav;base64,{wav_b64}",# useful for browsers
        "tokens": q.tolist(),                          # decoded token sequence (ints 0..255)
        "sr": int(sr),                                 # sample rate
        "duration_sec": float(duration),               # seconds
        "applied": applied,                            # {'beta','gamma','clamp'} if steering was applied
    }


# — registry wiring: zero‑arg factory that returns the runner —
AdapterRegistry.register("audio", lambda: _runner)

def _runner(model, prompt, style):
    return _run(model, prompt, style)
