# ElementFold · experience/adapters/language.py
# Language adapter = tiny bridge from (model, prompt, style) → text.
# Contract with the registry:
#   factory = AdapterRegistry.get("language")
#   runner  = factory()                          # zero‑arg → callable
#   out     = runner(model, prompt, style)       # returns a string
#
# Behavior:
#   • Tokenize the prompt with SimpleTokenizer (vocab≈256).
#   • Optionally apply steering controls (β, γ, ⛔) to the model if a raw style vector is provided.
#   • Run a single forward pass and greedy‑decode logits to tokens.
#   • Detokenize tokens back to text.
#
#   The “style” input may be the raw ℝ⁸ vector produced by the SteeringController:
#     v = [β̂, γ̂, ⛔̂, style₅]
#   We map it into meaningful ranges and apply (β, γ, ⛔) to the model’s Fold–Gate–Norm blocks.
#   Adapters are intentionally small so they’re easy to reason about and replace.

import torch                                    # ✴ tensors
from .base import AdapterRegistry               # 🗂 adapter registry
from ...tokenizer import SimpleTokenizer        # ✴ tokenizer
try:
    # Optional import: used only to map raw style vectors into (beta, gamma, clamp).
    from ..steering import SteeringController   # 🎚 intent → control vector (and to_params mapping)
    _HAS_STEER = True
except Exception:
    _HAS_STEER = False


def _apply_style_to_model(model, style):
    """
    If `style` looks like a raw ℝ⁸ vector from SteeringController, map it to parameters
    and apply to the model (if it implements .apply_control). If `style` is already a dict
    with beta/gamma/clamp, use it directly. Otherwise, do nothing.

    Returns a dict of the parameters that were (or would be) applied.
    """
    params = None

    # case A: dict with explicit params
    if isinstance(style, dict) and all(k in style for k in ("beta", "gamma", "clamp")):
        params = {"beta": float(style["beta"]), "gamma": float(style["gamma"]), "clamp": float(style["clamp"])}

    # case B: Tensor / list that looks like ℝ⁸ from SteeringController
    elif _HAS_STEER and isinstance(style, (torch.Tensor, list, tuple)) and len(style) >= 3:
        v = torch.as_tensor(style, dtype=torch.float32)         # normalize type
        params = SteeringController.to_params(v)                # map raw → ranges

    # Apply if supported
    if params and hasattr(model, "apply_control"):
        model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])

    return params or {}


def _run(model, prompt, style):
    """
    Core language adapter runner:
      1) optional steering → apply (β, γ, ⛔),
      2) tokenize prompt,
      3) forward once,
      4) greedy decode,
      5) detokenize to string.
    """
    # 1) Optionally apply steering controls to the model (no‑op if not provided/unsupported).
    _apply_style_to_model(model, style)

    # 2) Tokenize the prompt; ensure at least one token for empty strings.
    tok = SimpleTokenizer()                                   # ✴ tokenizer instance
    ids = tok.encode(prompt or "")                            # ↦ token ids (list[int])
    if len(ids) == 0:
        ids = [0]                                             # neutral token if prompt is empty

    # 3) Build a batch tensor and clip to the model's sequence length.
    dev = next(model.parameters()).device                     # 🖥 model device
    T = int(getattr(model, "seq_len", len(ids)))              # max tokens the model expects
    x = torch.tensor(ids[:T], dtype=torch.long, device=dev).unsqueeze(0)  # (1,T')

    # 4) Forward pass in no‑grad mode; decode greedily.
    with torch.no_grad():                                     # ≡ eval path
        logits, _X = model(x)                                 # ⟲ forward → (1,T',V),(1,T')
        y = logits.argmax(dim=-1).squeeze(0).tolist()         # greedy decode ids

    # 5) Detokenize to a human‑readable string and return.
    return tok.decode(y)                                      # ↤ text


# — registry wiring: provide a zero‑arg factory that returns the runner —
AdapterRegistry.register("language", lambda: _runner)

def _runner(model, prompt, style):
    return _run(model, prompt, style)
