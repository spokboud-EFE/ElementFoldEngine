# ElementFold · experience/adapters/multimodal.py
# Multimodal adapter = tiny orchestrator that can accept TEXT / IMAGE / AUDIO
# in a single "prompt" and route each part through the appropriate pathway.
#
# Contract with the registry:
#   factory = AdapterRegistry.get("multimodal")
#   runner  = factory()                           # zero‑arg → callable
#   out     = runner(model, prompt, style)        # returns a dict
#
# Prompt formats we accept (choose the one that’s most convenient):
#   1) Dict:
#        {
#          "text": "hello",
#          "image": "path/to.jpg" | (3,H,W) uint8|float,
#          "audio": "path/to.wav" | 1‑D waveform float | 1‑D token ids int,
#          "decode": {"strategy":"sample","temperature":0.9,"top_k":50,"top_p":0.95},
#          "simulate": false,    # ← optional; allow synthetic fallbacks if true
#          "strict": true        # ← optional; if true (default), we *wait* for real data
#        }
#
#   2) Inline string (mini‑DSL; '|' as separator):
#        "text: hello world | image: ./cat.png | audio: ./clip.wav | simulate: on"
#      If you pass a plain string with no "kind:", we treat it as text.
#
# Style handling:
#   • If `style` is a raw 8‑D vector from SteeringController, we map it into
#     ranges (β, γ, ⛔) and apply to the model when available.
#   • If `style` is a dict with {'beta','gamma','clamp'}, we use it directly.
#
# Notes:
#   • Our base Model is token‑centric (vocab≈256). For non‑text inputs we
#     map image/audio bytes into that space to produce a ledger reading; it’s
#     a demonstration path, not a full vision/audio model.

from __future__ import annotations

import os, io, wave, struct, re
from typing import Any, Dict, Tuple, Optional

import torch
from .base import AdapterRegistry
from ...core.tokenizer import SimpleTokenizer
from ...core.infer import infer_loop                # ⟲ unified decoding

# Reuse the language adapter runner for text parts (keeps behavior identical).
# We prefer late lookup via AdapterRegistry to avoid import order traps,
# but we also try a direct import with graceful fallback.
def _get_language_runner():
    try:
        # If already registered, prefer the registry
        if AdapterRegistry.has("language"):
            return AdapterRegistry.get("language")()
    except Exception:
        pass
    try:
        from .language import _run as _lang_run  # correct symbol in language.py
        return _lang_run
    except Exception:
        # Minimal fallback: simple greedy detokenize using model directly
        def _fallback(model, prompt: str, _style=None) -> str:
            tok = SimpleTokenizer()
            ids = tok.encode(prompt or "") or [0]
            try:
                dev = next(model.parameters()).device
            except Exception:
                dev = torch.device("cpu")
            T = int(getattr(model, "seq_len", len(ids)))
            x = torch.tensor(ids[:T], dtype=torch.long, device=dev).unsqueeze(0)
            model.eval()
            with torch.inference_mode():
                logits, _X = model(x)
                y = logits.argmax(dim=-1).squeeze(0).detach().cpu().tolist()
            return tok.decode(y)
        return _fallback

# Optional: steering param mapping (for style → β,γ,⛔) and captioning support.
try:
    from ..steering import SteeringController
    _HAS_STEER = True
except Exception:
    _HAS_STEER = False

# Optional: Pillow for image decode (portable)
try:
    from PIL import Image
    _HAS_PIL = True
except Exception:
    Image = None
    _HAS_PIL = False


# ──────────────────────────────────────────────────────────────────────────────
# Style application (β, γ, ⛔) and human caption
# ──────────────────────────────────────────────────────────────────────────────

def _map_style_to_params(style: Any) -> Optional[Dict[str, float]]:
    """Accept dict {'beta','gamma','clamp'} or raw ℝ⁸ → params dict; else None."""
    if isinstance(style, dict) and all(k in style for k in ("beta", "gamma", "clamp")):
        try:
            return {"beta": float(style["beta"]), "gamma": float(style["gamma"]), "clamp": float(style["clamp"])}
        except Exception:
            return None
    if _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        try:
            v = torch.as_tensor(style, dtype=torch.float32)
            p = SteeringController.to_params(v)
            return {"beta": float(p["beta"]), "gamma": float(p["gamma"]), "clamp": float(p["clamp"])}
        except Exception:
            return None
    return None

def _apply_style_to_model(model, style) -> Dict[str, float]:
    """Apply params into model if available; return params actually applied (or {})."""
    params = _map_style_to_params(style)
    if params and hasattr(model, "apply_control"):
        try:
            model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        except Exception:
            pass
    return params or {}

def _caption(style: Any, params: Dict[str, float]) -> str:
    """Friendly one‑liner; prefer SteeringController.describe(raw) if present."""
    if _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        try:
            v = torch.as_tensor(style, dtype=torch.float32)
            if hasattr(SteeringController, "describe"):
                return SteeringController.describe(v)
        except Exception:
            pass
    if params:
        return f"β={params['beta']:.2f}  γ={params['gamma']:.2f}  ⛔={params['clamp']:.1f}"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Prompt parsing helpers (dict, mini‑DSL, tolerant)
# ──────────────────────────────────────────────────────────────────────────────

_FLAG_RE = re.compile(r"\b(simulate|strict)\s*:\s*(on|off|true|false|1|0)\b", re.IGNORECASE)

def _parse_prompt(prompt: Any) -> Dict[str, Any]:
    """
    Normalize the incoming 'prompt' into a dict possibly containing keys:
      'text': str, 'image': path|tensor, 'audio': path|tensor, 'decode': dict,
      'simulate': bool, 'strict': bool
    """
    # Dict path — pass through and we’ll sanitize flags below.
    if isinstance(prompt, dict):
        out = dict(prompt)
        # Normalize booleans if provided as strings
        for key in ("simulate", "strict"):
            if key in out and isinstance(out[key], str):
                out[key] = out[key].strip().lower() in {"on", "true", "1", "yes"}
        return out

    # List/tuple of "kind:value" chunks → fold into a dict.
    if isinstance(prompt, (list, tuple)):
        out: Dict[str, Any] = {}
        for item in prompt:
            if isinstance(item, str) and ":" in item:
                k, v = item.split(":", 1)
                out.setdefault(k.strip().lower(), v.strip())
        # Extract flags if present as separate items
        flags = " ".join([str(it) for it in prompt if isinstance(it, str)])
        for m in _FLAG_RE.finditer(flags):
            out[m.group(1).lower()] = m.group(2).lower() in {"on", "true", "1"}
        return out

    # String mini‑DSL: "text: hi | image: ./cat.png | simulate: on"
    if isinstance(prompt, str):
        s = prompt.strip()
        if not s:
            return {}
        out: Dict[str, Any] = {}
        parts = [c.strip() for c in s.split("|") if c.strip()]
        has_kind = any(":" in c for c in parts)
        if not has_kind:
            return {"text": s}
        for chunk in parts:
            if ":" not in chunk:
                continue
            k, v = chunk.split(":", 1)
            key = k.strip().lower()
            val = v.strip()
            if key in ("text", "image", "audio"):
                out.setdefault(key, val)
            elif key in ("simulate", "strict"):
                out[key] = val.lower() in {"on", "true", "1", "yes"}
        return out

    # Fallback: anything else → stringified text.
    return {"text": str(prompt)}


# ──────────────────────────────────────────────────────────────────────────────
# Image helpers (portable; Pillow optional)
# ──────────────────────────────────────────────────────────────────────────────

def _load_image_as_uint8(path: str, size: int | Tuple[int, int] = 64) -> torch.Tensor:
    """
    Return (3,H,W) uint8 tensor. If decoding fails or Pillow missing, raise.
    """
    H = size if isinstance(size, int) else int(size[1])
    W = size if isinstance(size, int) else int(size[0])
    if _HAS_PIL:
        im = Image.open(path).convert("RGB")
        im = im.resize((W, H), resample=Image.BICUBIC)
        buf = im.tobytes()
        return torch.frombuffer(buf, dtype=torch.uint8).view(H, W, 3).permute(2, 0, 1).contiguous()
    raise RuntimeError("Pillow not available for image decode")

def _ensure_image_tensor(x: torch.Tensor) -> torch.Tensor:
    """Coerce to (3,H,W) uint8; accept float in [0,1] or various channel orders."""
    if x.dim() == 3 and x.shape[0] in (1, 3):  # (C,H,W)
        pass
    elif x.dim() == 3 and x.shape[-1] == 3:    # (H,W,3) → (3,H,W)
        x = x.permute(2, 0, 1).contiguous()
    else:
        flat = x.flatten()
        n = int((flat.numel() // 3) ** 0.5)
        x = flat[: 3 * n * n].view(3, n, n)
    if x.dtype.is_floating_point:
        x = (x.clamp(0, 1) * 255.0).to(torch.uint8)
    else:
        x = x.to(torch.uint8)
    return x

def _image_to_token_ids(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Flatten image bytes → token ids in [0..255], clip/pad to seq_len."""
    flat = x.flatten()
    if flat.numel() < seq_len:
        out = torch.zeros(seq_len, dtype=torch.uint8)
        out[: flat.numel()] = flat
        flat = out
    else:
        flat = flat[:seq_len]
    return flat.to(torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Audio helpers (stdlib wave → mono PCM → token ids)
# ──────────────────────────────────────────────────────────────────────────────

def _read_wav_mono(path: str, max_samples: int = 10_000) -> torch.Tensor:
    """Read a WAV file using stdlib wave. Return int16 mono PCM as float32."""
    with wave.open(path, "rb") as w:
        nchan, sampwidth, fr, nframes, _, _ = w.getparams()
        frames = w.readframes(min(max_samples, nframes))
    if sampwidth == 1:
        data = torch.tensor(list(frames), dtype=torch.int16) - 128
    elif sampwidth == 2:
        data = torch.tensor(list(struct.iter_unpack("<h", frames)), dtype=torch.int16).squeeze(-1)
    else:
        data = torch.tensor(list(frames), dtype=torch.int16)
    if nchan > 1:
        data = data.view(-1, nchan).mean(dim=1).to(torch.int16)
    return data.to(torch.float32)

def _pcm_to_token_ids(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Map mono PCM (float32) → token ids in [0..255] with simple linear companding."""
    if x.numel() == 0:
        return torch.zeros(seq_len, dtype=torch.long)
    x = x / (x.abs().max().clamp(min=1.0))
    ids = (((x + 1.0) * 0.5) * 255.0).clamp(0, 255).to(torch.uint8)
    if ids.numel() < seq_len:
        out = torch.zeros(seq_len, dtype=torch.uint8)
        out[: ids.numel()] = ids
        ids = out
    else:
        ids = ids[:seq_len]
    return ids.to(torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# Decode knob extraction (greedy vs. sample) from prompt dict
# ──────────────────────────────────────────────────────────────────────────────

def _decode_knobs(prompt_dict: Dict[str, Any]) -> Tuple[str, float, Optional[int], Optional[float]]:
    strategy = "greedy"
    temperature = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    dec = prompt_dict.get("decode", {}) if isinstance(prompt_dict, dict) else {}
    if isinstance(dec, dict):
        try:
            strategy = str(dec.get("strategy", strategy))
        except Exception:
            pass
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
    return strategy, temperature, top_k, top_p


# ──────────────────────────────────────────────────────────────────────────────
# Core multimodal runner
# ──────────────────────────────────────────────────────────────────────────────

def _run(model, prompt, style):
    """
    1) Apply optional steering (β,γ,⛔) to the model.
    2) Parse prompt into parts (text/image/audio) + flags (simulate/strict).
    3) For each part, produce a small, meaningful output (or a clear ‘waiting’ note).
    4) Echo decode knobs and add a human caption so predictions are visible and explained.
    """
    # 1) Controls
    applied = _apply_style_to_model(model, style)
    caption = _caption(style, applied)

    # 2) Parts + flags
    parts = _parse_prompt(prompt)
    simulate = bool(parts.get("simulate", False))
    strict = bool(parts.get("strict", True))

    # 3) Shared decode & model knobs
    strategy, temperature, top_k, top_p = _decode_knobs(parts)
    try:
        dev = next(model.parameters()).device
    except Exception:
        dev = torch.device("cpu")
    T = int(getattr(model, "seq_len", 128))

    # 4) Results container
    result: Dict[str, Any] = {
        "applied": applied,                         # {'beta','gamma','clamp'} if applied
        "caption": caption,                         # human summary of controls
        "decode": {                                 # echo decode knobs
            "strategy": strategy,
            "temperature": float(temperature),
            "top_k": (int(top_k) if top_k is not None else None),
            "top_p": (float(top_p) if top_p is not None else None),
        },
    }

    # (A) TEXT → via language adapter
    if "text" in parts and parts["text"] is not None:
        try:
            lang_run = _get_language_runner()
            txt = str(parts["text"])
            result["text"] = lang_run(model, txt, style)
        except Exception as e:
            result["text"] = {"error": f"text failed: {e}"}

    # (B) IMAGE → require real data unless simulate==True or strict==False
    if "image" in parts and parts["image"] is not None:
        img_val = parts["image"]
        have_tensor = isinstance(img_val, torch.Tensor)
        have_path = isinstance(img_val, (str, os.PathLike))
        path_exists = have_path and os.path.exists(str(img_val))

        if not have_tensor and not path_exists:
            if strict and not simulate:
                result["image"] = {
                    "status": "waiting",
                    "waiting_for": "image (path or (3,H,W) tensor)",
                    "hint": "Provide prompt.image as an existing filepath or a (3,H,W) uint8/float tensor.",
                }
            else:
                # synthetic fallback to keep demos flowing
                x = torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8)
                ids = _image_to_token_ids(x, seq_len=T).to(dev).unsqueeze(0)
                out = infer_loop(model, x=ids, strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p)
                y = out["tokens"].squeeze(0).tolist()
                result["image"] = {
                    "status": "simulated",
                    "shape": [3, 64, 64],
                    "tokens_in_len": int(ids.size(1)),
                    "tokens_out_head": y[: min(16, len(y))],
                    "ledger_mean": float(out["ledger"].mean().item()),
                }
        else:
            try:
                if have_tensor:
                    img = _ensure_image_tensor(img_val.detach().cpu())
                else:
                    img = _load_image_as_uint8(str(img_val), size=64)
                ids = _image_to_token_ids(img, seq_len=T).to(dev).unsqueeze(0)  # (1,T)
                out = infer_loop(model, x=ids, strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p)
                y = out["tokens"].squeeze(0).tolist()
                result["image"] = {
                    "status": "ok",
                    "source": ("tensor" if have_tensor else str(img_val)),
                    "shape": [int(s) for s in img.shape],
                    "tokens_in_len": int(ids.size(1)),
                    "tokens_out_head": y[: min(16, len(y))],
                    "ledger_mean": float(out["ledger"].mean().item()),
                }
            except Exception as e:
                result["image"] = {"error": f"image failed: {e}"}

    # (C) AUDIO → require real data unless simulate==True or strict==False
    if "audio" in parts and parts["audio"] is not None:
        aud_val = parts["audio"]
        have_tensor = isinstance(aud_val, torch.Tensor)
        have_path = isinstance(aud_val, (str, os.PathLike))
        path_exists = have_path and os.path.exists(str(aud_val))

        if not have_tensor and not path_exists:
            if strict and not simulate:
                result["audio"] = {
                    "status": "waiting",
                    "waiting_for": "audio (WAV path, 1‑D float PCM, or token ids)",
                    "hint": "Provide prompt.audio as a .wav filepath, a 1‑D float waveform, or a 1‑D int token tensor.",
                }
            else:
                # synthetic fallback
                ids = torch.randint(0, 256, (T,), dtype=torch.long).to(dev).unsqueeze(0)
                out = infer_loop(model, x=ids, strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p)
                y = out["tokens"].squeeze(0).tolist()
                result["audio"] = {
                    "status": "simulated",
                    "tokens_in_len": int(ids.size(1)),
                    "tokens_out_head": y[: min(16, len(y))],
                    "ledger_mean": float(out["ledger"].mean().item()),
                }
        else:
            try:
                if have_tensor:
                    a = aud_val.detach().cpu().flatten()
                    if a.dtype.is_floating_point:
                        ids = _pcm_to_token_ids(a.to(torch.float32), seq_len=T)
                    else:
                        ids = (a.to(torch.int64) % 256).to(torch.long)
                        if ids.numel() < T:
                            pad = torch.zeros(T - ids.numel(), dtype=torch.long)
                            ids = torch.cat([ids, pad], dim=0)
                        else:
                            ids = ids[:T]
                else:
                    pcm = _read_wav_mono(str(aud_val))
                    ids = _pcm_to_token_ids(pcm, seq_len=T)
                ids = ids.to(dev).unsqueeze(0)  # (1,T)
                out = infer_loop(model, x=ids, strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p)
                y = out["tokens"].squeeze(0).tolist()
                result["audio"] = {
                    "status": "ok",
                    "source": ("tensor" if have_tensor else str(aud_val)),
                    "tokens_in_len": int(ids.size(1)),
                    "tokens_out_head": y[: min(16, len(y))],
                    "ledger_mean": float(out["ledger"].mean().item()),
                }
            except Exception as e:
                result["audio"] = {"error": f"audio failed: {e}"}

    # If nothing matched and we had no explicit text, offer a friendly nudge.
    if ("text" not in parts) and ("image" not in parts) and ("audio" not in parts):
        result["note"] = (
            "Provide a dict like {'text': 'hi', 'image': 'a.png', 'audio': 'a.wav'} "
            "or 'text: hi | image: a.png'. Set 'simulate: on' to allow synthetic fallback."
        )

    return result


# — Registry wiring (concise and consistent) —
@AdapterRegistry.register_fn("multimodal")
def make_multimodal_adapter():
    # Zero‑arg factory → runner(model, prompt, style) → dict
    return _run
