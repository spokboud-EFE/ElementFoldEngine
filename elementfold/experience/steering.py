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
#   v    = ctrl("gentle, coherent")               # → ℝ⁸
#   p    = SteeringController.to_params(v)        # → {'beta','gamma','clamp','style', 'style_norm'}
#
from __future__ import annotations

from typing import Dict, Tuple, Optional
import math
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


def _tanh_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    Map ℝ → [lo, hi] with tanh; handy for style normalization into [-1,1] or a narrow band.
    """
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + half * torch.tanh(x)


class SteeringController(nn.Module):  # 🎚 intent → (β, γ, ⛔, style₅)
    """
    A tiny intent→control head. Forward returns a raw ℝ⁸ vector; use .to_params()
    to map into meaningful ranges.

    Notes:
      • δ⋆ (delta) is cached for convenience—some adapters may want to read it.
      • Tokenizer is intentionally tiny (vocab≈256); mean‑pooling is robust for short prompts.
      • Ranges can match Supervisor/RungController rails by passing `rails=...` to to_params().
    """

    # For very long prompts, we cap length to keep latency predictable.
    MAX_TOKENS: int = 512

    def __init__(self, delta: float = 0.030908106561043047, vocab: int = 256):
        super().__init__()
        self.delta = float(delta)               # δ⋆ cached (read‑only convenience)

        # — Embedding —
        # Keep in sync with SimpleTokenizer (vocab size ~ 256). Each token → ℝ⁶⁴.
        self.emb = nn.Embedding(vocab, 64)

        # — Head (MLP) —
        # A tiny two‑layer perceptron: ℝ⁶⁴ → ℝ⁸ = [β̂, γ̂, ⛔̂, style₅].
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
        )

        # Cache a tokenizer instance to avoid per‑call construction overhead.
        self._tok = SimpleTokenizer(vocab=vocab) if hasattr(SimpleTokenizer, "__call__") else SimpleTokenizer()

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
        # Tokenize (allow empty input; cap length)
        try:
            ids = self._tok.encode(s or "")
        except Exception:
            ids = []
        if not ids:
            ids = [0]
        if len(ids) > self.MAX_TOKENS:
            ids = ids[: self.MAX_TOKENS]

        dev = self.emb.weight.device
        x = torch.as_tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)  # (1, L)
        e = self.emb(x).mean(dim=1)                                          # (1, 64)
        v = self.fc(e).squeeze(0)                                            # (8,)
        return v

    # ————————————————————————————————————————————————
    # Helpers: map raw vector → meaningful ranges
    # ————————————————————————————————————————————————
    @staticmethod
    def to_params(
        v: torch.Tensor,
        rails: Optional[Dict[str, Tuple[float, float]]] = None,
        include_style_norm: bool = True,
    ) -> Dict[str, object]:
        """
        Convert a raw ℝ⁸ vector (as returned by forward) into interpretable parameters.

        Ranges:
            By default (conservative):
                β    ∈ [0.5, 2.0]
                γ    ∈ [0.0, 0.9]
                ⛔   ∈ [1.0, 10.0]
            To fully match RungController rails (see RungTuning), pass:
                rails={'beta':(0.5,3.0), 'gamma':(0.0,1.5), 'clamp':(1.0,12.0)}

        Returns:
            {
              'beta': float, 'gamma': float, 'clamp': float,
              'style': List[float],                 # raw style (length 5)
              'style_norm': List[float] (optional)  # tanh‑normalized to [-1,1] (length 5)
            }
        """
        # Rails (lo, hi) per control
        r = rails or {"beta": (0.5, 2.0), "gamma": (0.0, 0.9), "clamp": (1.0, 10.0)}

        with torch.no_grad():
            v = v.to(dtype=torch.float32)

            # Map first three controls into safe physical ranges via σ.
            beta  = float(_sigmoid_range(v[0], *r["beta"]))
            gamma = float(_sigmoid_range(v[1], *r["gamma"]))
            clamp = float(_sigmoid_range(v[2], *r["clamp"]))

            # Style block (unconstrained raw + normalized helper for UI sliders)
            style_vec = v[3:8].detach().cpu()
            style = [float(x) for x in style_vec.tolist()]
            out: Dict[str, object] = {"beta": beta, "gamma": gamma, "clamp": clamp, "style": style}

            if include_style_norm:
                # normalized to [-1,1] for immediate UI use
                s_norm = _tanh_range(style_vec, -1.0, 1.0).tolist()
                out["style_norm"] = [float(x) for x in s_norm]

        return out

    @staticmethod
    def describe(v: torch.Tensor, rails: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
        """
        Human‑readable one‑liner for logs/Studio:
            "β=1.26  γ=0.43  ⛔=5.7  |  style≈[+0.31, −0.12, +0.04, +0.77, −0.05]"
        """
        p = SteeringController.to_params(v, rails=rails, include_style_norm=True)
        s = p.get("style_norm") or p.get("style") or []
        fmt = lambda x: f"{x:+.2f}"
        s_preview = ", ".join(fmt(x) for x in (s[:5] if isinstance(s, list) else []))
        return f"β={p['beta']:.2f}  γ={p['gamma']:.2f}  ⛔={p['clamp']:.1f}  |  style≈[{s_preview}]"

    # ————————————————————————————————————————————————
    # Convenience factories (untrained vs. checkpoint)
    # ————————————————————————————————————————————————
    @classmethod
    def load_default(cls, delta: float = 0.030908106561043047, vocab: int = 256) -> "SteeringController":
        """
        Create a fresh, untrained controller (useful for prototyping).
        Training lives in steering_train.py.
        """
        return cls(delta, vocab=vocab)

    @classmethod
    def load(cls, path: str, delta: float = 0.030908106561043047, vocab: int = 256) -> "SteeringController":
        """
        Load weights from a state_dict checkpoint at `path`. Returns the controller in eval mode.
        """
        m = cls(delta, vocab=vocab)
        sd = torch.load(path, map_location="cpu")
        m.load_state_dict(sd)
        m.eval()
        return m

    # ————————————————————————————————————————————————
    # Optional: apply controls to a model directly
    # ————————————————————————————————————————————————
    def apply_to_model(self, model, s: str | None = None, v: torch.Tensor | None = None,
                       rails: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, object]:
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

        params = self.to_params(v, rails=rails or {"beta": (0.5, 2.0), "gamma": (0.0, 0.9), "clamp": (1.0, 10.0)})
        if hasattr(model, "apply_control"):
            model.apply_control(beta=params["beta"], gamma=params["gamma"], clamp=params["clamp"])
        return params
