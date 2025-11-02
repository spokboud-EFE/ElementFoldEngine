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
#        {"text": "hello", "image": "path/to.jpg", "audio": "path/to.wav",
#         "decode": {"strategy":"sample","temperature":0.9,"top_k":50,"top_p":0.95}}
#      Values can also be tensors:
#        {"image": (3,H,W) uint8|float, "audio": 1‑D waveform (float) or token ids (int)}
#
#   2) Inline string (mini‑DSL; '|' as separator):
#        "text: hello world | image: ./cat.png | audio: ./clip.wav"
#      If you pass a plain string with no "kind:", we treat it as text.
#
# Style handling:
#   • If `style` is a raw 8‑D vector from SteeringController, we map it into
#     ranges (β, γ, ⛔) and apply to the model when available.
#   • If `style` is a dict with {'beta','gamma','clamp'}, we use it directly.
#
# Notes:
#   • Our base Model is token‑centric (vocab=256). For non‑text inputs we
#     map image/audio bytes into that space to produce a ledger reading; it’s
#     a demonstration path, not a full vision/audio model.

from __future__ import annotations

import os, io, wave, struct
from typing import Any, Dict, Tuple

import torch
from .base import AdapterRegistry
from ...tokenizer import SimpleTokenizer
from ...infer import infer_loop                # ⟲ unified decoding

# Reuse the language adapter runner for text parts (keeps behavior identical).
from .language import _runner as _lang_run

# Optional: steering param mapping (for style → β,γ,⛔).
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


# ———————————————————————————————————————————————————————————
# Style application (β, γ, ⛔) — unified semantics
# ———————————————————————————————————————————————————————————

def _apply_style_to_model(model, style) -> Dict[str, float]:
    """
    If `style` looks like SteeringController’s raw ℝ⁸ vector, map to ranges and
    apply (beta, gamma, clamp) onto the model (if it supports .apply_control).
    If `style` is already a dict with those keys, use it directly.
    """
    params = None

    if isinstance(style, dict) and all(k in style for k in ("beta", "gamma", "clamp")):
        params = {"beta": float(style["beta"]), "gamma": float(style["gamma"]), "clamp": float(style["clamp"])}
    elif _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        v = torch.as_tensor(style, dtype=torch.float32)
        params = SteeringController.to_params(v)

    if params and hasattr(model, "apply_control"):
        model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])

    return params or {}


# ———————————————————————————————————————————————————————————
# Prompt parsing helpers
# ———————————————————————————————————————————————————————————

def _parse_prompt(prompt: Any) -> Dict[str, Any]:
    """
    Normalize the incoming 'prompt' into a dict possibly containing keys:
      'text': str, 'image': path|tensor, 'audio': path|tensor, 'decode': dict
    """
    # Dict → keep as‑is (we’ll lightly sanitize below).
    if isinstance(prompt, dict):
        return dict(prompt)

    # List/tuple of "kind:value" chunks → fold into a dict.
    if isinstance(prompt, (list, tuple)):
        out: Dict[str, Any] = {}
        for item in prompt:
            if isinstance(item, str) and ":" in item:
                k, v = item.split(":", 1)
                out.setdefault(k.strip().lower(), v.strip())
        return out

    # String mini‑DSL: "text: hi | image: ./cat.png | audio: ./clip.wav"
    if isinstance(prompt, str):
        s = prompt.strip()
        if not s:
            return {}
        # If the string contains no "kind:" markers, treat the whole thing as text.
        if ":" not in s:
            return {"text": s}
        out: Dict[str, Any] = {}
        for chunk in [c.strip() for c in s.split("|") if c.strip()]:
            if ":" not in chunk:
                continue
            k, v = chunk.split(":", 1)
            k = k.strip().lower()
            v = v.strip()
            if k in ("text", "image", "audio"):
                out.setdefault(k, v)
        return out

    # Fallback: anything else → stringified text.
    return {"text": str(prompt)}


# ———————————————————————————————————————————————————————————
# Image helpers (portable; Pillow optional)
# ———————————————————————————————————————————————————————————

def _load_image_as_uint8(path: str, size: int | Tuple[int, int] = 64) -> torch.Tensor:
    """
    Return (3,H,W) uint8 tensor. If decoding fails or Pillow missing,
    produce a synthetic tensor for pipeline continuity.
    """
    H = W = size if isinstance(size, int) else int(size[1])  # (W,H) → H
    W = size if isinstance(size, int) else int(size[0])
    if _HAS_PIL:
        try:
            im = Image.open(path).convert("RGB")
            im = im.resize((W, H), resample=Image.BICUBIC)
            buf = im.tobytes()
            x = torch.frombuffer(buf, dtype=torch.uint8).view(H, W, 3).permute(2, 0, 1).contiguous()
            return x
        except Exception:
            pass
    # Synthetic fallback
    return torch.randint(0, 256, (3, H, W), dtype=torch.uint8)

def _image_to_token_ids(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Flatten image bytes into token ids in [0..255] and clip/pad to seq_len.
    Accepts uint8 (preferred) or float in [0,1].
    """
    if x.dtype != torch.uint8:
        xf = x
        if xf.dtype.is_floating_point:
            xf = (xf.clamp(0, 1) * 255.0).to(torch.uint8)
        else:
            xf = xf.to(torch.uint8)
        x = xf
    flat = x.flatten()
    if flat.numel() < seq_len:
        out = torch.zeros(seq_len, dtype=torch.uint8)
        out[: flat.numel()] = flat
        flat = out
    else:
        flat = flat[:seq_len]
    return flat.to(torch.long)


# ———————————————————————————————————————————————————————————
# Audio helpers (stdlib wave → mono PCM → token ids)
# ———————————————————————————————————————————————————————————

def _read_wav_mono(path: str, max_samples: int = 10_000) -> torch.Tensor:
    """
    Read a WAV file using stdlib wave. Return int16 mono PCM as float32.
    """
    try:
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
    except Exception:
        return torch.zeros(0, dtype=torch.float32)

def _pcm_to_token_ids(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    Map mono PCM (float32) to token ids in [0..255] with simple linear companding.
    """
    if x.numel() == 0:
        return torch.randint(0, 256, (seq_len,), dtype=torch.long)
    x = x / (x.abs().max().clamp(min=1.0))
    ids = (((x + 1.0) * 0.5) * 255.0).clamp(0, 255).to(torch.uint8)
    if ids.numel() < seq_len:
        out = torch.zeros(seq_len, dtype=torch.uint8)
        out[: ids.numel()] = ids
        ids = out
    else:
        ids = ids[:seq_len]
    return ids.to(torch.long)


# ———————————————————————————————————————————————————————————
# Decode knob extraction (greedy vs. sample) from prompt dict
# ———————————————————————————————————————————————————————————

def _decode_knobs(prompt_dict: Dict[str, Any]) -> Tuple[str, float, Any, Any]:
    strategy = "greedy"
    temperature = 1.0
    top_k = None
    top_p = None
    dec = prompt_dict.get("decode", {}) if isinstance(prompt_dict, dict) else {}
    if isinstance(dec, dict):
        strategy = str(dec.get("strategy", strategy))
        temperature = float(dec.get("temperature", temperature))
        top_k = dec.get("top_k", top_k)
        top_p = dec.get("top_p", top_p)
    return strategy, temperature, top_k, top_p


# ———————————————————————————————————————————————————————————
# Core multimodal runner
# ———————————————————————————————————————————————————————————

def _run(model, prompt, style):
    """
    1) Apply optional steering (β,γ,⛔) to the model.
    2) Parse prompt into parts (text/image/audio).
    3) For each part, produce a small, meaningful output:
         • text   → language adapter (deterministic or sampled via infer_loop in language adapter)
         • image  → bytes→tokens→model→return tokens_out_head + ledger summary
         • audio  → wav/pcm/tokens→model→return tokens_out_head + ledger summary
    """
    # 1) Apply style to model if it’s a raw vector or a param dict.
    applied = _apply_style_to_model(model, style)

    # 2) Normalize the prompt into a dict with possible keys: text, image, audio, decode.
    parts = _parse_prompt(prompt)

    # 3) Decode knobs shared by image/audio flows (language runner has its own).
    strategy, temperature, top_k, top_p = _decode_knobs(parts)

    # 4) Prepare result container and model/device knobs.
    result: Dict[str, Any] = {"applied": applied}
    dev = next(model.parameters()).device
    T = int(getattr(model, "seq_len", 128))
    V = int(getattr(model, "vocab", 256))

    # (A) TEXT → reuse language adapter to keep behavior perfectly aligned.
    if "text" in parts and parts["text"] is not None:
        try:
            txt = str(parts["text"])
            result["text"] = _lang_run(model, txt, style)
        except Exception as e:
            result["text"] = {"error": f"text failed: {e}"}

    # (B) IMAGE → flatten bytes → (1,T) ids → infer_loop → summary
    if "image" in parts and parts["image"] is not None:
        img_val = parts["image"]
        try:
            # Ingest as (3,H,W) uint8
            if isinstance(img_val, (str, os.PathLike)) and os.path.exists(str(img_val)):
                img = _load_image_as_uint8(str(img_val), size=T)  # crude tie to seq_len for demo
            elif isinstance(img_val, torch.Tensor):
                img = img_val.detach().cpu()
                if img.dim() == 3 and img.shape[0] in (1, 3):
                    pass  # assume (C,H,W)
                elif img.dim() == 3 and img.shape[-1] == 3:
                    img = img.permute(2, 0, 1).contiguous()
                else:
                    flat = img.flatten()
                    n = int((flat.numel() // 3) ** 0.5)
                    img = flat[: 3 * n * n].view(3, n, n)
                img = img.to(dtype=torch.uint8 if img.dtype != torch.uint8 else img.dtype)
            else:
                img = _load_image_as_uint8("")  # synthetic fallback

            ids = _image_to_token_ids(img, seq_len=T).to(dev).unsqueeze(0)  # (1,T)
            out = infer_loop(model, x=ids, strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p)
            y = out["tokens"].squeeze(0).tolist()
            ledger_mean = float(out["ledger"].mean().item())
            result["image"] = {
                "shape": [int(s) for s in img.shape],
                "tokens_in_len": int(ids.size(1)),
                "tokens_out_head": y[: min(16, len(y))],
                "ledger_mean": ledger_mean,
            }
        except Exception as e:
            result["image"] = {"error": f"image failed: {e}"}

    # (C) AUDIO → wav/pcm/tokens → (1,T) ids → infer_loop → summary
    if "audio" in parts and parts["audio"] is not None:
        aud_val = parts["audio"]
        try:
            # Accept path to .wav, 1‑D float PCM, or 1‑D byte‑level token ids (ints).
            if isinstance(aud_val, (str, os.PathLike)) and os.path.exists(str(aud_val)):
                pcm = _read_wav_mono(str(aud_val))
                ids = _pcm_to_token_ids(pcm, seq_len=T)
            elif isinstance(aud_val, torch.Tensor):
                a = aud_val.detach().cpu().flatten()
                if a.dtype.is_floating_point:
                    ids = _pcm_to_token_ids(a.to(torch.float32), seq_len=T)
                else:
                    # Assume these are already token ids; normalize to [0,255].
                    ids = (a.to(torch.int64) % 256).to(torch.long)
                    if ids.numel() < T:
                        pad = torch.zeros(T - ids.numel(), dtype=torch.long)
                        ids = torch.cat([ids, pad], dim=0)
                    else:
                        ids = ids[:T]
            else:
                ids = torch.randint(0, 256, (T,), dtype=torch.long)

            ids = ids.to(dev).unsqueeze(0)  # (1,T)
            out = infer_loop(model, x=ids, strategy=strategy, temperature=temperature, top_k=top_k, top_p=top_p)
            y = out["tokens"].squeeze(0).tolist()
            ledger_mean = float(out["ledger"].mean().item())
            result["audio"] = {
                "tokens_in_len": int(ids.size(1)),
                "tokens_out_head": y[: min(16, len(y))],
                "ledger_mean": ledger_mean,
            }
        except Exception as e:
            result["audio"] = {"error": f"audio failed: {e}"}

    # If nothing matched and we had no explicit text, offer a friendly nudge.
    if ("text" not in parts) and ("image" not in parts) and ("audio" not in parts):
        result["note"] = "Provide a dict like {'text': 'hi', 'image': 'a.png', 'audio': 'a.wav'} or 'text: hi | image: a.png'"

    return result


# — registry wiring —
AdapterRegistry.register("multimodal", lambda: _runner)

def _runner(model, prompt, style):
    return _run(model, prompt, style)
