# ElementFold · experience/adapters/vision.py
# Vision adapter — a light, dependency‑free bridge from (model, prompt, style) → a
# “coherence glyph” rendered in Unicode blocks. Since our core model is token‑based,
# we map the prompt’s bytes to a small square grid and run the model to obtain the
# ledger field X. We then render a per‑row coherence bar using Unicode levels.
#
# Contract with the registry:
#   factory = AdapterRegistry.get("vision")
#   runner  = factory()                          # zero‑arg → callable
#   out     = runner(model, prompt, style)       # returns a string (glyph)
#
# Notes:
#   • We optionally apply steering controls (β, γ, ⛔) when `style` looks like a raw ℝ⁸
#     vector from the SteeringController or a dict {beta,gamma,clamp}.
#   • No external image deps (PIL etc.). This adapter produces a deterministic
#     “coherence fingerprint” from text inputs, useful for demos and tests.

from __future__ import annotations
import math
import torch
from .base import AdapterRegistry
from ...tokenizer import SimpleTokenizer

# Optional import for mapping raw style → {beta,gamma,clamp}
try:
    from ..steering import SteeringController
    _HAS_STEER = True
except Exception:
    _HAS_STEER = False


def _apply_style_to_model(model, style):
    """
    If `style` is a dict with beta/gamma/clamp, apply it; if it’s a Tensor/list/tuple of length≥3,
    interpret it via SteeringController.to_params(...). Silently no‑op if unsupported.
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


# Small glyph palette from light to dark (Unicode block elements).
_PALETTE = " ▁▂▃▄▅▆▇█"  # leading space + 7 levels


def _row_bar(values, width=32):
    """
    Render a single row of coherence values (0..1) into a Unicode bar.
    We resample the row into `width` cells and pick a block level per cell.
    """
    if values.numel() == 0:
        return ""
    v = values.float().clamp(0, 1)
    # Resample to fixed width (nearest): index proportional positions along the row.
    idx = torch.linspace(0, v.numel() - 1, steps=width, device=v.device)
    vi = v[idx.long()]
    # Map to discrete levels
    levels = (vi * (len(_PALETTE) - 1)).round().long().clamp(0, len(_PALETTE) - 1)
    return "".join(_PALETTE[l] for l in levels.tolist())


def _prompt_to_square_ids(prompt: str, vocab: int = 256) -> torch.Tensor:
    """
    Convert text → token ids and arrange into a near‑square grid (H×W) flattened row‑major.
    This gives a deterministic “pseudo‑image” without external I/O.
    """
    tok = SimpleTokenizer(vocab=vocab)
    ids = tok.encode(prompt or "")
    if len(ids) == 0:
        ids = [0]
    n = len(ids)
    s = max(8, int(round(math.sqrt(n))))  # at least 8×8 for a visible glyph
    total = s * s
    # Repeat or truncate to fill s*s
    tiled = (ids * ((total + len(ids) - 1) // len(ids)))[:total]
    return torch.tensor(tiled, dtype=torch.long), s  # (H*W,), side length


def _run(model, prompt, style):
    """
    1) Optionally apply steering (β, γ, ⛔) to the model.
    2) Map text → square grid of token ids.
    3) Forward through the model to get ledger field X (B,T).
    4) Compute a per‑row coherence metric and render Unicode bars per row.
    5) Return the multi‑line glyph.
    """
    _apply_style_to_model(model, style)

    # Build pseudo‑image tokens
    dev = next(model.parameters()).device
    ids_flat, side = _prompt_to_square_ids(prompt, vocab=getattr(model, "vocab", 256))
    # Respect model.seq_len limit; if too small, reduce side accordingly.
    T = int(getattr(model, "seq_len", ids_flat.numel()))
    total = min(T, ids_flat.numel())
    side = int(math.sqrt(total)) or 1
    total = side * side
    ids_flat = ids_flat[:total].to(dev)
    x = ids_flat.view(1, total)  # (1, H*W)

    with torch.no_grad():
        _logits, X = model(x)     # X: (1, H*W)
        X = X.view(1, side, side) # reshape to (1, H, W)

    # Compute a simple per‑row coherence: normalize each row to [0,1] by min‑max on that row.
    rows = []
    for r in range(side):
        row = X[0, r, :]
        # Normalize safely: if flat row, fallback to zeros.
        minv, maxv = row.min(), row.max()
        denom = (maxv - minv).abs().clamp_min(1e-9)
        norm = (row - minv) / denom
        rows.append(_row_bar(norm, width=32))

    header = f"⟲ Vision Coherence Glyph  {side}×{side}"
    return header + "\n" + "\n".join(rows)


# Register adapter: returns a zero‑arg factory that yields the runner.
AdapterRegistry.register("vision", lambda: _runner)

def _runner(model, prompt, style):
    return _run(model, prompt, style)
