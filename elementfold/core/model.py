# ElementFold · model.py
# ============================================================
# ElementFold Model — from tokens to ledger.
#
# Pipeline
# --------
#   Stem (Embedding)
#   → RotaryClick (per‑timestep phase rotation)
#   → [FGNBlock] × L (Fold–Gate–Norm coherence engine)
#   → LayerNorm
#   → Heads: {lm logits, ledger scalar}
#
# Forward:
#   (B,T) int64  →  logits:(B,T,V),  X:(B,T)
#
# Control passthrough:
#   apply_control(beta=?, gamma=?, clamp=?)
# ============================================================

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn

from .fgn import FGNBlock  # Fold–Gate–Norm unit

__all__ = ["RotaryClick", "Model"]


# ============================================================
# 1) RotaryClick — deterministic per‑time phase rotation
# ============================================================

class RotaryClick(nn.Module):
    """
    Rotate feature lanes in pairs with an angle that depends only on time step t.

    For each t (0..T-1) and each feature pair (A,B):
        [A'; B'] = [ cos θ_t  -sin θ_t ] [A; B]
                   [ sin θ_t   cos θ_t ]
    with θ_t = t * (2π * δ).

    Notes
    -----
    * Operates in float32 for the trig, then casts back to x.dtype (safe for fp16/bf16).
    * If D is odd, the last channel is passed through unchanged.
    """
    def __init__(self, dim: int, delta: float = 0.03):
        super().__init__()
        self.dim = int(dim)
        self.delta = float(delta)
        self._theta = 2.0 * math.pi * self.delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        x : (B, T, D)

        Returns
        -------
        (B, T, D) rotated in feature pairs per timestep.
        """
        B, T, D = x.shape
        if D < 2 or T == 0:
            return x

        even = (D // 2) * 2
        # Time-dependent angles computed in float32 for stability.
        t = torch.arange(T, dtype=torch.float32, device=x.device)
        c = torch.cos(t * self._theta).view(1, T, 1, 1)
        s = torch.sin(t * self._theta).view(1, T, 1, 1)
        if x.dtype != torch.float32:
            c = c.to(torch.float32)
            s = s.to(torch.float32)

        # Rotate pairs [A,B] along the last dimension.
        x_head = x[:, :, :even].reshape(B, T, even // 2, 2).to(torch.float32)
        A, Bp = x_head[..., 0:1], x_head[..., 1:2]
        Ar = A * c - Bp * s
        Br = A * s + Bp * c
        rotated = torch.cat([Ar, Br], dim=-1).reshape(B, T, even).to(x.dtype)

        if even == D:
            return rotated
        # Pass through last (odd) channel unchanged.
        tail = x[:, :, even:]
        return torch.cat([rotated, tail], dim=-1)


# ============================================================
# 2) ElementFold Model — tokens → (logits, ledger)
# ============================================================

class Model(nn.Module):
    """
    Stem → RotaryClick → [FGN]×L → LayerNorm → {lm head, ledger head}

    Outputs
    -------
    logits : (B, T, V)
    X      : (B, T)
    """
    def __init__(
        self,
        vocab: int = 256,
        d: int = 128,
        layers: int = 4,
        heads: int = 4,         # kept for interface symmetry; not used here
        seq_len: int = 128,
        fold: str = "grid",     # retained for config visibility
        delta: float = 0.03,
    ):
        super().__init__()
        # Public config (int-cast to avoid surprises when loading from yaml/json)
        self.vocab = int(vocab)
        self.d = int(d)
        self.layers = int(layers)
        self.heads = int(heads)
        self.seq_len = int(seq_len)
        self.fold = str(fold)
        self.delta = float(delta)

        # --- Stem ---
        self.emb = nn.Embedding(self.vocab, self.d)
        self.rot = RotaryClick(self.d, self.delta)

        # --- Core ---
        self.blocks = nn.ModuleList([FGNBlock(self.d) for _ in range(self.layers)])

        # --- Heads ---
        self.norm = nn.LayerNorm(self.d)
        self.lm = nn.Linear(self.d, self.vocab)    # token logits
        self.ledger = nn.Linear(self.d, 1)         # scalar per token
        self._last_control = {"beta": None, "gamma": None, "clamp": None}

        self._init_weights()

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : LongTensor (B, T)
            Token ids.

        Returns
        -------
        logits : FloatTensor (B, T, V)
        X      : FloatTensor (B, T)
        """
        h = self.emb(x)              # (B,T,D)
        h = self.rot(h)              # (B,T,D)
        for blk in self.blocks:
            h = blk(h)               # (B,T,D)
        h = self.norm(h)             # (B,T,D)

        logits = self.lm(h)          # (B,T,V)
        X = self.ledger(h).squeeze(-1)  # (B,T)
        return logits, X

    # --------------------------------------------------------
    # Control interface — propagate β, γ, ⛔ to all FGN blocks
    # --------------------------------------------------------
    def apply_control(self, beta=None, gamma=None, clamp=None) -> None:
        """
        Push external coherence controls into internal FGN blocks.
        """
        self._last_control = {"beta": beta, "gamma": gamma, "clamp": clamp}
        for blk in self.blocks:
            if hasattr(blk, "apply_control") and callable(blk.apply_control):
                blk.apply_control(beta=beta, gamma=gamma, clamp=clamp)
            else:
                # Best-effort fill for compatible attributes
                with torch.no_grad():
                    gate = getattr(blk, "gate", None)
                    if gate is not None:
                        if beta is not None and hasattr(gate, "beta"):
                            gate.beta.fill_(float(beta))
                        if clamp is not None and hasattr(gate, "clamp"):
                            gate.clamp.fill_(float(clamp))
                    norm = getattr(blk, "norm", None)
                    if norm is not None and gamma is not None:
                        if hasattr(norm, "gamma"):
                            norm.gamma.fill_(float(gamma))
                        elif hasattr(norm, "weight"):
                            norm.weight.fill_(float(gamma))

    def last_control(self) -> dict:
        """Return most recently applied β, γ, ⛔ (for dashboards/telemetry)."""
        return dict(self._last_control)

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    def _init_weights(self) -> None:
        # Embedding & heads: small-norm init; ledger starts near 0 signal.
        nn.init.normal_(self.emb.weight, std=0.02)

        nn.init.normal_(self.lm.weight, std=0.02)
        if self.lm.bias is not None:
            nn.init.zeros_(self.lm.bias)

        nn.init.zeros_(self.ledger.weight)
        if self.ledger.bias is not None:
            nn.init.zeros_(self.ledger.bias)
