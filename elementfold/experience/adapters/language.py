# ElementFold · experience/adapters/language.py
# Language adapter = tiny bridge from (model, prompt, style) → text.
#
# Contract with the registry:
#   factory = AdapterRegistry.get("language")
#   runner  = factory()                          # zero‑arg → callable
#   out     = runner(model, prompt, style)       # returns a string
#
# What it does (plain words):
#   1) (Optional) Apply steering controls (β exposure, γ damping, ⛔ clamp)
#      to the model when a raw ℝ⁸ style vector is provided.
#   2) Tokenize the prompt into byte‑ids (vocab≈256).
#   3) Run a single forward pass and greedy‑decode logits to tokens.
#   4) Detokenize back to text and return it.
#
# The “style” input is usually the raw ℝ⁸ from SteeringController:
#     v = [β̂, γ̂, ⛔̂, style₅]
# We map it into meaningful ranges and forward (β,γ,⛔) to the model.
# Adapters are intentionally tiny so they’re easy to read and replace.

from __future__ import annotations
from typing import Any, Dict

import torch

from .base import AdapterRegistry                     # 🗂 adapter registry
from elementfold.tokenizer import SimpleTokenizer     # ✴ tiny byte tokenizer

# Optional import: used to map raw ℝ⁸ into (beta, gamma, clamp).
try:
    from ..steering import SteeringController         # 🎚 intent → control vector (and to_params)
    _HAS_STEER = True
except Exception:                                     # Keep adapter usable even if steering isn’t present
    _HAS_STEER = False


def _apply_style_to_model(model: Any, style: Any) -> Dict[str, float]:
    """
    If `style` looks like a raw ℝ⁸ vector from the SteeringController, map it to
    parameters and apply to the model (if it supports .apply_control). If `style`
    is already a dict with β/γ/⛔, use it directly. Otherwise, do nothing.

    Returns the parameters that were applied (or an empty dict when none).
    """
    params: Dict[str, float] | None = None

    # Case A: explicit dict {beta,gamma,clamp}
    if isinstance(style, dict) and all(k in style for k in ("beta", "gamma", "clamp")):
        try:
            params = {
                "beta": float(style["beta"]),
                "gamma": float(style["gamma"]),
                "clamp": float(style["clamp"]),
            }
        except Exception:
            params = None  # fall through to no‑op if values aren’t clean floats

    # Case B: raw ℝ⁸ (Tensor/list/tuple) from SteeringController
    elif _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        v = torch.as_tensor(style, dtype=torch.float32)  # normalize dtype/device‑agnostic
        try:
            mapped = SteeringController.to_params(v)     # {'beta','gamma','clamp','style'}
            params = {
                "beta": float(mapped["beta"]),
                "gamma": float(mapped["gamma"]),
                "clamp": float(mapped["clamp"]),
            }
        except Exception:
            params = None

    # Apply if possible
    if params is not None and hasattr(model, "apply_control"):
        try:
            model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        except Exception:
            # Non‑fatal: ignore if model lacks the expected hook signature
            pass

    return params or {}


def _run(model, prompt: str, style: Any) -> str:
    """
    Core language adapter runner:
      1) optionally apply steering (β,γ,⛔),
      2) encode prompt → ids,
      3) forward once (inference mode),
      4) greedy decode,
      5) decode ids → text.
    """
    # 1) Steering (safe no‑op if style is None/unsupported)
    _apply_style_to_model(model, style)

    # 2) Tokenize the prompt; ensure at least one token for empty strings
    tok = SimpleTokenizer()
    ids = tok.encode(prompt or "")
    if not ids:
        ids = [0]  # neutral byte

    # 3) Pack batch and clip to model’s expected sequence length
    try:
        dev = next(model.parameters()).device
    except Exception:
        dev = torch.device("cpu")
    T = int(getattr(model, "seq_len", len(ids)))
    x = torch.tensor(ids[:T], dtype=torch.long, device=dev).unsqueeze(0)  # (1,T')

    # 4) Forward once and greedy‑decode (deterministic)
    was_training = getattr(model, "training", False)
    model.eval()  # polite: disable dropout etc. if present
    with torch.inference_mode():
        logits, _X = model(x)                       # (1,T',V), (1,T')
        y = logits.argmax(dim=-1).squeeze(0)        # (T',)
        out_ids = y.detach().cpu().tolist()

    # Restore training state if needed
    if was_training:
        try:
            model.train()
        except Exception:
            pass

    # 5) Detokenize to text
    return tok.decode(out_ids)


# — Registry wiring: decorator form keeps definitions concise and thread‑safe —
@AdapterRegistry.register_fn("language")
def make_language_adapter():
    # Zero‑arg factory → runner(model, prompt, style) → str
    return _run
