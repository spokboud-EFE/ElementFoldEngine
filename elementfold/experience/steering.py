# ElementFold · experience/steering.py
# ──────────────────────────────────────────────────────────────────────────────
# The SteeringController turns *intent text* into a compact control vector:
#   v ∈ ℝ⁸ = [β, γ, ⛔, style₅]
#
# Meanings:
#   • β (beta)    — gate exposure (how strongly FGN exposes novelty),
#   • γ (gamma)   — normalization damping (how hard FGN calms energy),
#   • ⛔ (clamp)  — gate cap (how deep negative gate values can go before clipping),
#   • style₅      — five free “style” scalars adapters can interpret (tone, tempo, etc.).
#
# Design goals:
#   • Minimal & fast: tokenizer → ids → embedding → mean‑pool → 2‑layer MLP → ℝ⁸.
#   • Trainable: see steering_train.py; defaults work out‑of‑the‑box.
#   • Safe ranges: a helper maps raw outputs into Supervisor‑aligned bounds.
#
# Contract with Studio:
#   ctrl = SteeringController.load_default(cfg.delta)
#   v    = ctrl("gentle, coherent")     # → ℝ⁸
#   p    = SteeringController.to_params(v)  # → {'beta','gamma','clamp','style'}
#
from __future__ import annotations

from typing import Dict
import torch
import torch.nn as nn

from ..tokenizer import SimpleTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sigmoid_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    Map ℝ → (lo, hi) smoothly via σ. Works element‑wise and is differentiable.
    """
    return torch.sigmoid(x) * (hi - lo) + lo


class SteeringController(nn.Module):  # 🎚 intent → (β, γ, ⛔, style₅)
    """
    A tiny intent→control head. Forward returns a raw ℝ⁸ vector; use .to_params()
    to map into meaningful ranges.

    Notes:
      • δ⋆ (delta) is cached for convenience—some adapters may want to read it.
      • Tokenizer is intentionally tiny (vocab≈256); mean‑pooling is robust for short prompts.
    """

    # For very long prompts, we can cap length to keep latency predictable.
    MAX_TOKENS: int = 512

    def __init__(self, delta: float = 0.030908106561043047):
        super().__init__()
        self.delta = float(delta)               # δ⋆ cached (read‑only convenience)

        # — Embedding —
        # Keep in sync with SimpleTokenizer (vocab size = 256). Each token → ℝ⁶⁴.
        self.emb = nn.Embedding(256, 64)

        # — Head (MLP) —
        # A tiny two‑layer perceptron: ℝ⁶⁴ → ℝ⁸ = [β̂, γ̂, ⛔̂, style₅].
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

    # ————————————————————————————————————————————————
    # Core forward (kept trainable for fine‑tuning)
    # ————————————————————————————————————————————————
    def forward(self, s: str) -> torch.Tensor:
        """
        Turn a text prompt into a raw control vector v ∈ ℝ⁸.

        Steps:
          1) tokenize prompt → ids,
          2) embed → (1, L, 64),
          3) mean‑pool across tokens → (1, 64),
          4) MLP → (1, 8),
          5) squeeze batch → (8,).

        Returns:
            torch.Tensor shape (8,), dtype float32 by default.
        """
        tok = SimpleTokenizer()
        ids = tok.encode(s)

        # Guard: allow empty input, cap extremely long inputs to bound latency.
        if not ids:
            ids = [0]
        if len(ids) > self.MAX_TOKENS:
            ids = ids[: self.MAX_TOKENS]

        dev = self.emb.weight.device
        x = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)  # (1, L)
        e = self.emb(x).mean(dim=1)                                       # (1, 64)
        v = self.fc(e).squeeze(0)                                         # (8,)
        return v

    # ————————————————————————————————————————————————
    # Helpers: map raw vector → meaningful ranges
    # ————————————————————————————————————————————————
    @staticmethod
    def to_params(v: torch.Tensor) -> Dict[str, object]:
        """
        Convert a raw ℝ⁸ vector (as returned by forward) into interpretable parameters
        with ranges aligned to the Supervisor’s defaults:

            β    ∈ [0.5, 2.0]
            γ    ∈ [0.0, 0.9]
            ⛔   ∈ [1.0, 10.0]
            style ∈ ℝ⁵  (left unconstrained; adapters can interpret freely)

        Returns:
            {'beta': float, 'gamma': float, 'clamp': float, 'style': torch.Tensor(5,)}
        """
        # Ensure predictable dtype/device; drop grad to avoid leaking graphs into the UI.
        with torch.no_grad():
            v = v.to(dtype=torch.float32)

            # Map first three controls into safe physical ranges via σ.
            beta  = _sigmoid_range(v[0], 0.5, 2.0).item()
            gamma = _sigmoid_range(v[1], 0.0, 0.9).item()
            clamp = _sigmoid_range(v[2], 1.0, 10.0).item()

            # Style left unconstrained for adapters (they may tanh/normalize if desired).
            style = v[3:8].detach()

        return {"beta": beta, "gamma": gamma, "clamp": clamp, "style": style}

    # ————————————————————————————————————————————————
    # Convenience factories (untrained vs. checkpoint)
    # ————————————————————————————————————————————————
    @classmethod
    def load_default(cls, delta: float = 0.030908106561043047) -> "SteeringController":
        """
        Create a fresh, untrained controller (useful for prototyping).
        Training lives in steering_train.py.
        """
        return cls(delta)

    @classmethod
    def load(cls, path: str, delta: float = 0.030908106561043047) -> "SteeringController":
        """
        Load weights from a state_dict checkpoint at `path`. Returns the controller in eval mode.
        """
        m = cls(delta)
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd)
        m.eval()
        return m

    # ————————————————————————————————————————————————
    # Optional: apply controls to a model directly
    # ————————————————————————————————————————————————
    def apply_to_model(self, model, s: str | None = None, v: torch.Tensor | None = None) -> Dict[str, object]:
        """
        Produce controls (from a prompt `s` or raw vector `v`) and push them
        into any model that implements `.apply_control(beta=?, gamma=?, clamp=?)`.

        Returns:
            The parameter dict actually applied (useful for logging/UX).
        """
        if v is None:
            if s is None:
                raise ValueError("either `s` (prompt) or `v` (raw ℝ⁸ vector) must be provided")
            v = self.forward(s)
        params = self.to_params(v)
        if hasattr(model, "apply_control"):
            model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        return params
