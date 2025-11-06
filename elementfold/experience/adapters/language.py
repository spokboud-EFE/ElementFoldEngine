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
#      to the model when a raw ℝ⁸ style vector is provided, or a dict with
#      those controls is given.
#   2) Tokenize the prompt into byte‑ids (vocab≈256).
#   3) Run a single forward pass and greedy‑decode logits to tokens.
#   4) Detokenize back to text and return it.
#
# Visibility: when steering is used, we also prepend a single‑line summary:
#   "β=1.26  γ=0.43  ⛔=5.7  |  style≈[+0.31, −0.12, +0.04, +0.77, −0.05]"
# so predictions are “visible and explained” in Studio logs/UX.
#
# Adapters are intentionally tiny so they’re easy to read and replace.

from __future__ import annotations
from typing import Any, Dict, Optional

import torch

from .base import AdapterRegistry                     # 🗂 adapter registry
from elementfold.tokenizer import SimpleTokenizer     # ✴ tiny byte tokenizer

# Optional import: used to map raw ℝ⁸ into (beta, gamma, clamp) and summarize.
try:
    from ..steering import SteeringController         # 🎚 intent → control vector (and to_params/describe)
    _HAS_STEER = True
except Exception:                                     # Keep adapter usable even if steering isn’t present
    _HAS_STEER = False


# ———————————————————————————————————————————————————————————
# Steering helpers
# ———————————————————————————————————————————————————————————

def _map_style_to_params(style: Any) -> Optional[Dict[str, float]]:
    """
    Accept either:
      • dict with {'beta','gamma','clamp'} floats,
      • raw ℝ⁸ vector (Tensor/list/tuple) from SteeringController → map via to_params(),
    and return a clean params dict or None.
    """
    # Case A: explicit dict {beta,gamma,clamp}
    if isinstance(style, dict) and all(k in style for k in ("beta", "gamma", "clamp")):
        try:
            return {
                "beta": float(style["beta"]),
                "gamma": float(style["gamma"]),
                "clamp": float(style["clamp"]),
            }
        except Exception:
            return None

    # Case B: raw ℝ⁸ (Tensor/list/tuple) from SteeringController
    if _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        try:
            v = torch.as_tensor(style, dtype=torch.float32)
            mapped = SteeringController.to_params(v)     # {'beta','gamma','clamp','style',...}
            return {"beta": float(mapped["beta"]),
                    "gamma": float(mapped["gamma"]),
                    "clamp": float(mapped["clamp"])}
        except Exception:
            return None

    # Otherwise: unknown style form
    return None


def _apply_params_to_model(model: Any, params: Optional[Dict[str, float]]) -> None:
    """Best‑effort application of β/γ/⛔ to models that support .apply_control(...)."""
    if not params:
        return
    if hasattr(model, "apply_control"):
        try:
            model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        except Exception:
            # Non‑fatal: ignore if the model’s hook signature differs
            pass


def _summarize_style(style: Any, params: Optional[Dict[str, float]]) -> str:
    """
    Human‑friendly single line. Prefer SteeringController.describe(raw ℝ⁸) when available;
    otherwise fall back to the applied params. Returns "" if nothing to say.
    """
    # Prefer a raw vector summary using SteeringController (shows style preview nicely)
    if _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        try:
            v = torch.as_tensor(style, dtype=torch.float32)
            return SteeringController.describe(v)
        except Exception:
            pass

    # Fall back to the applied controls only
    if params:
        return f"β={params['beta']:.2f}  γ={params['gamma']:.2f}  ⛔={params['clamp']:.1f}"

    return ""


# ———————————————————————————————————————————————————————————
# Core runner
# ———————————————————————————————————————————————————————————

def _run(model, prompt: str, style: Any) -> str:
    """
    Core language adapter runner:
      1) optionally apply steering (β,γ,⛔),
      2) encode prompt → ids,
      3) forward once (inference mode),
      4) greedy decode,
      5) decode ids → text,
      6) if steering used → prepend a friendly one‑line summary.
    """
    # 1) Steering (safe no‑op if style is None/unsupported)
    params = _map_style_to_params(style)
    _apply_params_to_model(model, params)
    summary = _summarize_style(style, params)

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
        logits, _ledger = model(x)                    # (1,T',V), (1,T')
        y = logits.argmax(dim=-1).squeeze(0)         # (T',)
        out_ids = y.detach().cpu().tolist()

    # Restore training state if needed
    if was_training:
        try:
            model.train()
        except Exception:
            pass

    # 5) Detokenize to text
    text_out = tok.decode(out_ids)

    # 6) If we have a steering summary, make the output self‑describing
    if summary:
        return f"{summary}\n{text_out}"
    return text_out


# — Registry wiring: decorator form keeps definitions concise and thread‑safe —
@AdapterRegistry.register_fn("language")
def make_language_adapter():
    # Zero‑arg factory → runner(model, prompt, style) → str
    return _run
