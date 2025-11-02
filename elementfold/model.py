# ElementFold · model.py
# This file defines the end‑to‑end model the engine trains and serves.
# It has two ideas:
#   1) RotaryClick — apply a clean, deterministic rotation per time step using the click δ⋆.
#      This “turns” feature pairs like a tiny 2‑D rotor so depth can feel the ledger rhythm.
#   2) A stack of Fold–Gate–Norm (FGN) blocks — each block collects → exposes → normalizes,
#      then we keep identity through a residual lane and read out logits (for tokens) and X (ledger scalar).

import torch, torch.nn as nn, math        # ✴ PyTorch tensors & modules; math for π
from .fgn import FGNBlock                 # ⟲ Fold–Gate–Norm building block


class RotaryClick(nn.Module):
    """
    RotaryClick applies a *deterministic phase rotation* across the feature dimension
    that advances one fixed step per time index. It is controlled by the click δ⋆:

        θ⋆ = 2π · δ⋆
        angle at step t: θ_t = t · θ⋆

    We split the feature dimension D into interleaved 2‑component lanes (A,B) and
    rotate each pair by (cos θ_t, sin θ_t). If D is odd, the last feature is left unchanged.
    """
    def __init__(self, dim, delta=0.030908106561043047):       # ✴ receive feature size and δ⋆
        super().__init__()                                     # ✴ standard nn.Module init
        self.dim = int(dim)                                    # ≡ keep D as a plain int
        self.theta = 2 * math.pi * float(delta)                # θ⋆ = 2π·δ⋆ (precompute)

    def forward(self, x):                                      # x: (B,T,D) batch × time × features
        b, t, d = x.shape                                      # ✴ read shapes
        half = (d // 2) * 2                                    # ≡ largest even ≤ D (pairs need even count)
        idx = torch.arange(half, device=x.device)              # [0..half−1] on the right device
        i0 = idx[0::2]                                         # even lanes → A components
        i1 = idx[1::2]                                         # odd  lanes → B components

        angles = torch.arange(t, device=x.device).float() * self.theta  # θ_t = t·θ⋆
        c = torch.cos(angles).unsqueeze(0).unsqueeze(-1)       # cos θ → (1,T,1) for broadcast
        s = torch.sin(angles).unsqueeze(0).unsqueeze(-1)       # sin θ → (1,T,1)

        x0 = x[:, :, i0]                                       # (B,T,D/2) → A lanes
        x1 = x[:, :, i1]                                       # (B,T,D/2) → B lanes
        xro0 = x0 * c - x1 * s                                 # A' =  A·cos − B·sin
        xro1 = x0 * s + x1 * c                                 # B' =  A·sin + B·cos

        xr = x.new_zeros(b, t, half)                           # buffer for the rotated even part
        xr[:, :, 0::2] = xro0                                  # interleave A'
        xr[:, :, 1::2] = xro1                                  # interleave B'

        if d > half:                                           # if D is odd, carry the tail unchanged
            xr = torch.cat([xr, x[:, :, half:]], dim=-1)

        return xr                                              # ✴ rotated features (B,T,D)


class Model(nn.Module):
    """
    ElementFold model:
      • Token embedding → (B,T,D)
      • RotaryClick to inject δ⋆ rhythm across time
      • L stacked FGN blocks (each collects, gates, normalizes, preserves identity)
      • LayerNorm for a calm output scale
      • Two heads:
          - lm:     D → vocab logits (language modeling / token prediction)
          - ledger: D → scalar X per step (the learned ledger coordinate)
    """
    def __init__(
        self,
        vocab=256,                            # vocabulary size
        d=128,                                # feature width D
        layers=4,                             # number of FGN blocks
        heads=4,                              # kept for config parity with attention models
        seq_len=128,                          # max sequence length for helpers/CLI
        fold='grid',                          # placeholder knob (current FGN is grid‑style)
        delta=0.030908106561043047,           # click δ⋆ (controls rotary angle θ⋆)
    ):
        super().__init__()                    # ✴ base init

        # — Hyper‑parameters (exposed as attributes for UX/tools) —
        self.vocab = int(vocab)
        self.d = int(d)
        self.seq_len = int(seq_len)
        self.delta = float(delta)

        # — Stem: embedding + rotary phase —
        self.emb = nn.Embedding(self.vocab, self.d)            # ids → dense features
        self.rot = RotaryClick(self.d, self.delta)             # deterministic δ⋆ rotation

        # — Core: stack of Fold–Gate–Norm blocks —
        self.blocks = nn.ModuleList([FGNBlock(self.d) for _ in range(int(layers))])

        # — Heads: logits + ledger —
        self.norm = nn.LayerNorm(self.d)                       # calm output scale
        self.lm = nn.Linear(self.d, self.vocab)                # D → V logits
        self.ledger = nn.Linear(self.d, 1)                     # D → 1 ledger scalar

        # last applied controls (useful for UX/telemetry)
        self._last_control = {"beta": None, "gamma": None, "clamp": None}

    def forward(self, x):
        """
        Args:
            x: (B,T) int64 token ids

        Returns:
            logits: (B,T,V) token logits
            X:      (B,T)   ledger scalar per time step
        """
        h = self.emb(x)                                        # (B,T) → (B,T,D)
        h = self.rot(h)                                        # inject δ⋆ rhythm
        for b in self.blocks:                                  # refinement stack
            h = b(h)                                           # fold → gate → norm → residual
        h = self.norm(h)                                       # stabilize for heads

        logits = self.lm(h)                                    # (B,T,D) → (B,T,V)
        X = self.ledger(h).squeeze(-1)                         # (B,T,D) → (B,T)
        return logits, X

    # ————————————————————————————————————————————————
    # External slow‑control: (β exposure, γ damping, ⛔ clamp)
    # ————————————————————————————————————————————————
    def apply_control(self, beta: float | None = None, gamma: float | None = None, clamp: float | None = None):
        """
        Push control into all FGN blocks. We try a block‑level `apply_control` if present,
        otherwise fall back to setting gate/norm parameters directly. Callers can leave any
        argument as None to keep it unchanged.
        """
        self._last_control = {"beta": beta, "gamma": gamma, "clamp": clamp}
        for b in self.blocks:
            # Preferred path: let the block do its own bookkeeping if it exposes apply_control.
            if hasattr(b, "apply_control"):
                b.apply_control(beta=beta, gamma=gamma, clamp=clamp)
                continue
            # Fallback: reach into submodules defensively.
            if beta is not None and hasattr(b, "gate") and hasattr(b.gate, "beta"):
                with torch.no_grad():
                    b.gate.beta.copy_(torch.tensor(float(beta), device=b.gate.beta.device, dtype=b.gate.beta.dtype))
            if clamp is not None and hasattr(b, "gate") and hasattr(b.gate, "clamp"):
                with torch.no_grad():
                    b.gate.clamp.copy_(torch.tensor(float(clamp), device=b.gate.clamp.device, dtype=b.gate.clamp.dtype))
            if gamma is not None and hasattr(b, "norm") and hasattr(b.norm, "gamma"):
                with torch.no_grad():
                    b.norm.gamma.copy_(torch.tensor(float(gamma), device=b.norm.gamma.device, dtype=b.norm.gamma.dtype))

    def last_control(self) -> dict:
        """Return the latest (β, γ, ⛔) controls that were applied."""
        return dict(self._last_control)
