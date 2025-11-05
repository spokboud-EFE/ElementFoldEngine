# ElementFold · infer.py
# ──────────────────────────────────────────────────────────────────────────────
# Inference is the calm twin of training: we take a model and (optionally) a token
# batch, run a forward pass, and turn logits into tokens. The ledger X comes along
# for telemetry. This file keeps things simple but robust:
#   • Auto‑select device + eval/inference mode.
#   • Greedy decoding by default (argmax).
#   • Optional temperature / top‑k / top‑p sampling for stochastic outputs.
#
# Notation (unicode used sparingly for clarity):
#   • β — “exposure” (lives inside the model; not touched here),
#   • γ — “damping”   (ditto),
#   • ⛔ — clamp       (ditto).
#
# We only *decode* here (logits → tokens); controlling β/γ/⛔ happens elsewhere.

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple

import torch


# ————————————————————————————————————————————————————————————————
# Sampling utilities
# ————————————————————————————————————————————————————————————————

def _apply_topk(logits: torch.Tensor, k: Optional[int]) -> torch.Tensor:
    """
    Keep the top‑k logits per position, set the rest to −∞ (so softmax drops them).
    Shape: logits ∈ ℝ^{B×T×V}  →  returns a new tensor with same shape.
    """
    if k is None or k <= 0:
        return logits
    V = logits.size(-1)
    k = int(k)
    if k >= V:
        return logits  # nothing to do
    topk_vals, topk_idx = torch.topk(logits, k, dim=-1)
    pruned = torch.full_like(logits, float("-inf"))
    return pruned.scatter(-1, topk_idx, topk_vals)


def _apply_topp(logits: torch.Tensor, p: Optional[float]) -> torch.Tensor:
    """
    Nucleus (top‑p) filtering: keep the smallest prefix of tokens whose
    probability mass ≥ p at each (B,T) position; drop the rest (−∞).
    """
    if p is None or not (0.0 < float(p) < 1.0):
        return logits

    # Sort by logit for stable nucleus selection.
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)  # (B,T,V)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = probs.cumsum(dim=-1)

    # Mask everything AFTER the cutoff. Ensure the best token always survives.
    cutoff = cumprobs > float(p)
    cutoff[..., 0] = False

    kept = sorted_logits.masked_fill(cutoff, float("-inf"))
    # Scatter back to original token order.
    out = torch.full_like(logits, float("-inf"))
    return out.scatter(-1, sorted_idx, kept)


def _sample_from_logits(
    logits: torch.Tensor,           # (B,T,V) unnormalized scores
    temperature: float = 1.0,       # >0; lower → sharper, higher → flatter
    top_k: Optional[int] = None,    # keep K most likely tokens per position
    top_p: Optional[float] = None,  # nucleus sampling: keep minimal set with cumprob ≥ p
) -> torch.Tensor:
    """
    Turn logits into token ids using temperature + optional top‑k/top‑p filters.
    We operate position‑wise; the model predicts all positions in one pass.
    """
    # 0) Deterministic fallback: temperature ≤ 0 behaves like greedy (common UI edge case).
    if temperature <= 0:
        return logits.argmax(dim=-1)

    # 1) Temperature scaling (never divide by 0).
    logits = logits / float(temperature)

    # 2) Optional pruning. Order: top‑k then top‑p (common practice; either order is defensible).
    logits = _apply_topk(logits, top_k)
    logits = _apply_topp(logits, top_p)

    # 3) Sample from categorical distribution at each (B,T) position.
    probs = torch.softmax(logits, dim=-1)                      # (B,T,V)
    B, T, V = probs.shape
    flat = probs.reshape(B * T, V)                             # (B·T, V)

    # Guard against numerical edge cases (shouldn’t trigger with our filters):
    # If any row sums to ~0 due to all −∞, softmax would be NaNs. As a belt‑and‑suspenders,
    # replace NaNs by uniform mass over V (rare; mainly defensive during bring‑up).
    nan_rows = torch.isnan(flat).any(dim=1)
    if nan_rows.any():
        flat[nan_rows] = 1.0 / V

    idx = torch.multinomial(flat, num_samples=1)               # (B·T,1)
    return idx.view(B, T)                                      # (B,T) int64


# ————————————————————————————————————————————————————————————————
# Main inference entry
# ————————————————————————————————————————————————————————————————

def _model_device(model) -> torch.device:
    """Best‑effort device discovery even for exotic models."""
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def infer_loop(
    model,
    x: Optional[torch.Tensor] = None,   # optional input tokens: (B,T) int64
    *,
    strategy: str = "greedy",           # 'greedy' or 'sample'
    temperature: float = 1.0,           # used when strategy='sample'
    top_k: Optional[int] = None,        # used when strategy='sample'
    top_p: Optional[float] = None,      # used when strategy='sample'
) -> Dict[str, Any]:
    """
    Run a forward pass and decode tokens with the chosen strategy.

    Inputs
    ------
    model:
        ElementFold model whose forward returns (logits, X), where:
          • logits ∈ ℝ^{B×T×V} (unnormalized scores over vocab),
          • X      ∈ ℝ^{B×T}   (ledger: latent coordinate along rungs).
    x:
        Optional input tokens (B,T), dtype long. If None, we synthesize a random
        (1,T) batch using model.vocab and model.seq_len (handy for demos).
    strategy:
        'greedy' (argmax; deterministic) or 'sample' (stochastic).
    temperature, top_k, top_p:
        Sampling knobs when strategy='sample' (ignored for greedy).

    Returns
    -------
    dict with:
      • 'tokens' : (B,T) int64 — decoded tokens,
      • 'ledger' : (B,T) float — ledger X for telemetry/visualization.
    """
    if model is None:
        raise ValueError("infer_loop: model is None")

    device = _model_device(model)

    # Prepare input: ensure (B,T) on the correct device.
    if x is None:
        # Rely on model attributes (ElementFold Model exposes .vocab and .seq_len).
        vocab = getattr(model, "vocab", None)
        seq_len = getattr(model, "seq_len", None)
        if vocab is None or seq_len is None:
            raise AttributeError("model must expose .vocab and .seq_len when x is None")
        x = torch.randint(0, int(vocab), (1, int(seq_len)), device=device)
    else:
        if x.dim() == 1:
            x = x.unsqueeze(0)  # → (1,T)
        x = x.to(device)

    was_training = getattr(model, "training", False)
    model.eval()
    with torch.inference_mode():
        logits, X = model(x)  # (B,T,V), (B,T)

        if strategy == "greedy":
            y = logits.argmax(dim=-1)  # (B,T)
        elif strategy == "sample":
            y = _sample_from_logits(
                logits,
                temperature=float(temperature),
                top_k=(int(top_k) if top_k is not None else None),
                top_p=(float(top_p) if top_p is not None else None),
            )
        else:
            raise ValueError(f"unknown strategy: {strategy!r}; use 'greedy' or 'sample'")

    if was_training:
        model.train()

    return {"tokens": y, "ledger": X}
