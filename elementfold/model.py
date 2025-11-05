# ElementFold · model.py
# ──────────────────────────────────────────────────────────────────────────────
# Overview (plain words)
# ----------------------
# The model does three simple things end‑to‑end:
#   1) Embed tokens into ℝᴰ, then apply a tiny, deterministic “rotor” per time
#      step that depends on the click size δ⋆. This is the RotaryClick.
#   2) Run a stack of Fold–Gate–Norm (FGN) blocks that:
#        • fold information,
#        • expose novelty via a gate (strength ≈ β),
#        • calm energy via a normalizer (damping ≈ γ),
#        • and keep identity through a residual path,
#      optionally clamped by ⛔ to keep gates safe.
#   3) Read out:
#        • logits over tokens (language head),
#        • a scalar ledger coordinate X per time (ledger head).
#
# Public contract
# ---------------
#   forward(x: (B,T) int64) → (logits: (B,T,V), X: (B,T))
#   apply_control(beta=?, gamma=?, clamp=?)  # propagate (β, γ, ⛔) into all FGN blocks
#
# Notes
# -----
# • The “heads” knob is kept for config parity with attention models; FGN doesn’t
#   use it directly but other parts of the code expect it to exist.
# • RotaryClick uses θ⋆ = 2π·δ⋆ so ~1/δ⋆ steps make a full turn; δ⋆≈0.03 → ~32 steps.
# • The ledger head is intentionally simple (linear) so training + regularizers
#   (AlignHead, VariationalLedger) shape it cleanly.

from __future__ import annotations

import math
import torch
import torch.nn as nn

from .fgn import FGNBlock  # Fold–Gate–Norm building block (must expose either .apply_control(...) or gate/norm params)


# ──────────────────────────────────────────────────────────────────────────────
# Rotary phase: a tiny, deterministic “turn” per time step
# ──────────────────────────────────────────────────────────────────────────────
class RotaryClick(nn.Module):
    """
    Apply a phase rotation across feature lanes that advances by a fixed angle per
    time index. For each adjacent feature pair (A,B) we rotate:
        [A'; B'] = [ cos θ_t  −sin θ_t ] [A; B]
                   [ sin θ_t   cos θ_t ]
    with θ_t = t · (2π·δ⋆). Odd tails (if D is odd) are passed through unchanged.
    """
    def __init__(self, dim: int, delta: float = 0.030908106561043047):
        super().__init__()
        self.dim   = int(dim)
        self.theta = 2.0 * math.pi * float(delta)  # θ⋆ = 2π·δ⋆

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B,T,D) features
        Returns:
            (B,T,D) rotated features
        """
        B, T, D = x.shape
        # If we have fewer than 2 lanes, rotation is a no‑op.
        if D < 2:
            return x

        # Use the largest even slice for paired rotation; carry any odd tail.
        even = (D // 2) * 2
        i0 = torch.arange(0, even, 2, device=x.device)
        i1 = torch.arange(1, even, 2, device=x.device)

        # θ_t for t = 0..T−1 (match dtype/device to x for clean math)
        angles = torch.arange(T, device=x.device, dtype=x.dtype) * self.theta
        c = torch.cos(angles).view(1, T, 1)   # (1,T,1)
        s = torch.sin(angles).view(1, T, 1)   # (1,T,1)

        A = x[:, :, i0]                       # (B,T,even/2)
        Bp = x[:, :, i1]                      # (B,T,even/2)

        Ar = A * c - Bp * s                   # rotate pairs in place…
        Br = A * s + Bp * c

        out = x.new_empty(B, T, D)
        out[:, :, 0:even:2] = Ar
        out[:, :, 1:even:2] = Br
        if D > even:
            out[:, :, even:] = x[:, :, even:]  # pass odd tail unchanged
        return out


# ──────────────────────────────────────────────────────────────────────────────
# The end‑to‑end ElementFold model
# ──────────────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    """
    Stem → RotaryClick → [FGN]×L → LayerNorm → {lm head, ledger head}

    forward(x) returns (logits, X), where:
      • logits ∈ ℝ^{B×T×V} for token prediction,
      • X      ∈ ℝ^{B×T}   is the ledger scalar (anchored log) per time.
    """
    def __init__(
        self,
        vocab:   int   = 256,                     # vocabulary size V
        d:       int   = 128,                     # feature width D
        layers:  int   = 4,                       # number of FGN blocks
        heads:   int   = 4,                       # kept for config parity (unused here)
        seq_len: int   = 128,                     # max sequence length (for helpers/CLI)
        fold:    str   = "grid",                  # stylistic knob for FGN families
        delta:   float = 0.030908106561043047,    # δ⋆ click size
    ):
        super().__init__()
        # Expose a few attributes (other modules read these)
        self.vocab   = int(vocab)
        self.d       = int(d)
        self.layers  = int(layers)
        self.heads   = int(heads)
        self.seq_len = int(seq_len)
        self.fold    = str(fold)
        self.delta   = float(delta)

        # — Stem: token embedding + deterministic rotary phase —
        self.emb = nn.Embedding(self.vocab, self.d)
        self.rot = RotaryClick(self.d, self.delta)

        # — Core: a tidy stack of FGN blocks —
        self.blocks = nn.ModuleList([FGNBlock(self.d) for _ in range(self.layers)])

        # — Heads: calm output scale then project to logits and ledger —
        self.norm   = nn.LayerNorm(self.d)
        self.lm     = nn.Linear(self.d, self.vocab)   # (B,T,D) → (B,T,V)
        self.ledger = nn.Linear(self.d, 1)            # (B,T,D) → (B,T,1)

        # Track the last applied (β, γ, ⛔) for UX/telemetry readback
        self._last_control = {"beta": None, "gamma": None, "clamp": None}

        self._init_weights()

    # ————————————————————————————————————————————————
    # Forward: (B,T) → (B,T,V) & (B,T)
    # ————————————————————————————————————————————————
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B,T) int64 token ids
        Returns:
            logits: (B,T,V)
            X:      (B,T)
        """
        h = self.emb(x)          # (B,T,D)
        h = self.rot(h)          # inject δ⋆ rhythm
        for blk in self.blocks:  # refine through Fold–Gate–Norm stacks
            h = blk(h)
        h = self.norm(h)         # calm scale for stable heads

        logits = self.lm(h)                  # (B,T,V)
        X      = self.ledger(h).squeeze(-1)  # (B,T,1) → (B,T)
        return logits, X

    # ————————————————————————————————————————————————
    # Slow‑control hook: propagate (β, γ, ⛔) to all FGN blocks
    # ————————————————————————————————————————————————
    def apply_control(
        self,
        beta:  float | None = None,   # β — gate exposure
        gamma: float | None = None,   # γ — normalizer damping
        clamp: float | None = None,   # ⛔ — gate clamp cap
    ) -> None:
        """
        Tries the block‑level `.apply_control(...)` if present. Otherwise, falls back to
        writing into known attributes:
            • blk.gate.beta, blk.gate.clamp
            • blk.norm.gamma  (or blk.norm.weight as a universal fallback)
        """
        self._last_control = {"beta": beta, "gamma": gamma, "clamp": clamp}

        for blk in self.blocks:
            # Preferred: block knows how to apply controls to its internals.
            if hasattr(blk, "apply_control") and callable(blk.apply_control):
                blk.apply_control(beta=beta, gamma=gamma, clamp=clamp)
                continue

            # Defensive fallbacks (avoid breaking if FGN internals differ slightly)
            with torch.no_grad():
                # Gate exposure β and clamp ⛔
                gate = getattr(blk, "gate", None)
                if gate is not None:
                    if beta  is not None and hasattr(gate, "beta"):
                        gate.beta.fill_(float(beta))
                    if clamp is not None and hasattr(gate, "clamp"):
                        gate.clamp.fill_(float(clamp))

                # Norm damping γ (try .gamma, else fall back to LayerNorm.weight)
                norm = getattr(blk, "norm", None)
                if norm is not None and gamma is not None:
                    if hasattr(norm, "gamma"):
                        norm.gamma.fill_(float(gamma))
                    elif hasattr(norm, "weight"):
                        norm.weight.fill_(float(gamma))

    def last_control(self) -> dict:
        """Return the most recently applied (β, γ, ⛔) values (handy for status UIs)."""
        return dict(self._last_control)

    # ————————————————————————————————————————————————
    # Initialization (small, stable scales)
    # ————————————————————————————————————————————————
    def _init_weights(self) -> None:
        # Keep embeddings and blocks default‑init (PyTorch does a sensible job).
        # Heads: small normal for logits; near‑zero ledger so training sculpts X gently.
        nn.init.normal_(self.lm.weight, std=0.02)
        if self.lm.bias is not None:
            nn.init.zeros_(self.lm.bias)

        nn.init.zeros_(self.ledger.weight)
        if self.ledger.bias is not None:
            nn.init.zeros_(self.ledger.bias)
