# ElementFold · fgn.py
# ──────────────────────────────────────────────────────────────────────────────
# Fold–Gate–Norm (FGN) is the engine’s heartbeat:
#   1) Fold      — gather local structure along time without blowing up scale.
#   2) Gate      — compute a scalar exposure per step and apply a gain e^{β·g}.
#   3) Normalize — damp energy so depth remains stable.
# Finally, a residual lane keeps identity always available.
#
# Key knobs (steered by Supervisor/RungController):
#   • β (beta):     exposure — how strongly we amplify structure where g>0.
#   • γ (gamma):    damping  — how hard we calm energy after gating.
#   • ⛔ (clamp):   gate cap — soft bound on how negative g may go (safety).
#
# Design notes for coherence and safety
# -------------------------------------
# • Gate centering uses the *per‑sequence mean* so g has positive/negative mass.
#   This lets β behave intuitively: β↑ → stronger amplification where g>0.
# • We clamp g asymmetrically: g ∈ [−clamp, +g_pos_cap] with g_pos_cap≈1.
#   The negative side follows ⛔; the positive side is lightly capped so
#   e^{β·g} cannot explode (e^{2·1}≈7.4 at β≈2), while still allowing
#   meaningful exposure boosts. Normalization then catches any excess.
# • The residual projection is zero‑initialized so each block starts as *near‑identity*,
#   making the first training steps gentle and predictable.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1) Fold: depth‑wise 1‑D aggregation along time (non‑expansive defaults)
# ──────────────────────────────────────────────────────────────────────────────
class FoldGrid(nn.Module):
    """
    Depth‑wise 1D convolution over the time axis.
    Each feature channel is folded independently (groups=d), so we never mix channels here.
    This preserves identity and keeps the fold non‑expansive by default.

    Args:
        d:      number of channels (feature dimension D).
        kind:   'identity' (center tap = 1) or 'avg3' ([1,2,1]/4 smoothing).
        learn:  if True, the kernel is learnable; if False, it stays as initialized.
    """
    def __init__(self, d: int, kind: str = "identity", learn: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False)  # (B,D,T) ↔ (B,D,T)
        with torch.no_grad():
            k = torch.zeros(d, 1, 3)
            if kind == "avg3":
                k[:, :, 0] = 0.25; k[:, :, 1] = 0.50; k[:, :, 2] = 0.25
            else:  # 'identity'
                k[:, :, 1] = 1.0
            self.conv.weight.copy_(k)
        self.conv.weight.requires_grad = bool(learn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D) → y: (B,T,D)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


# ──────────────────────────────────────────────────────────────────────────────
# 2) Gate: scalar potential → exponential gain e^{β·g}
# ──────────────────────────────────────────────────────────────────────────────
class Gate(nn.Module):
    """
    Exponential gate: y ← y * exp(β · g), where g(x) is a learned scalar per time step.

    Centering & clamps
    ------------------
    • We center g by subtracting the per‑sequence mean, so g has both signs.
    • We cap g ∈ [−clamp, +g_pos_cap] with a *light* positive cap (≈1.0) to
      enable amplification but avoid runaway gains. The negative side follows
      ⛔ (clamp) exactly, matching the project’s “gate cap” semantics.

    Args:
        d:         feature dimension D (input to the small probe φ: ℝ^D→ℝ).
        pos_cap:   positive cap for g (safe exposure ceiling, default 1.0).
    """
    def __init__(self, d: int, pos_cap: float = 1.0):
        super().__init__()
        self.lin = nn.Linear(d, 1)                              # φ(x): (B,T,D) → (B,T,1)
        self.beta  = nn.Parameter(torch.tensor(1.0))            # β exposure (learnable; can be steered)
        self.clamp = nn.Parameter(torch.tensor(5.0))            # ⛔ negative cap magnitude
        self._g_pos_cap = float(pos_cap)                        # light positive cap (constant by default)

        # Gentle init: keep φ small so exp(β·g) ≈ 1 at start.
        nn.init.normal_(self.lin.weight, std=0.02)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D)
        returns: (B,T,1) multiplicative gain
        """
        # Raw gate potential and per‑sequence mean centering
        g = self.lin(x).squeeze(-1)                             # (B,T)
        g = g - g.mean(dim=1, keepdim=True)                     # center per sequence (row)

        # Asymmetric safety clamp: negative side follows ⛔, positive side lightly capped
        neg_cap = -self.clamp.item()
        pos_cap =  self._g_pos_cap
        g = g.clamp(min=neg_cap, max=pos_cap)                   # (B,T)

        # Exponential gain; broadcast over channels
        gain = torch.exp(self.beta * g).unsqueeze(-1)           # (B,T,1)
        return gain

    def set_control(self, beta: float | None = None, clamp: float | None = None) -> None:
        """External control hook (Supervisor/RungController)."""
        with torch.no_grad():
            if beta is not None:
                self.beta.copy_(torch.tensor(float(beta), device=self.beta.device, dtype=self.beta.dtype))
            if clamp is not None:
                self.clamp.copy_(torch.tensor(float(clamp), device=self.clamp.device, dtype=self.clamp.dtype))


# ──────────────────────────────────────────────────────────────────────────────
# 3) Normalize: row‑wise damping to keep energy bounded
# ──────────────────────────────────────────────────────────────────────────────
class Norm(nn.Module):
    """
    Row‑wise energy normalization:
        y_norm = y / (‖y‖₁ + ε)^γ
    Larger activations get damped more when γ>0.

    Args:
        d:     feature dimension (kept for symmetry; not used directly).
        gamma: initial damping strength in [0,1].
    """
    def __init__(self, d: int, gamma: float = 0.5):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.eps   = 1e-6

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        g = torch.clamp(self.gamma, 0.0, 1.0)
        scale = (y.abs().sum(dim=-1, keepdim=True) + self.eps).pow(g)
        return y / scale

    def set_control(self, gamma: float | None = None) -> None:
        """External control hook (Supervisor/RungController)."""
        if gamma is not None:
            with torch.no_grad():
                self.gamma.copy_(torch.tensor(float(gamma), device=self.gamma.device, dtype=self.gamma.dtype))


# ──────────────────────────────────────────────────────────────────────────────
# 4) The block: Fold → Gate → Norm → Residual(Proj)
# ──────────────────────────────────────────────────────────────────────────────
class FGNBlock(nn.Module):
    """
    One Fold–Gate–Norm block with a residual projection.

        y = Fold(x)                 # local, depth‑wise aggregation
        y = y * Gate(x)             # multiplicative exposure (β, ⛔)
        y = Norm(y)                 # damping (γ)
        out = x + Proj(y)           # identity lane kept clean

    Args:
        d:            feature dimension.
        fold_kind:    'identity' or 'avg3'.
        fold_learn:   whether the fold kernel is trainable.
        resid_scale:  scale on the residual update before adding back.
        dropout:      optional dropout after projection (regularization).
    """
    def __init__(self, d: int, fold_kind: str = "identity", fold_learn: bool = True,
                 resid_scale: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.fold = FoldGrid(d, kind=fold_kind, learn=fold_learn)          # (B,T,D) → (B,T,D)
        self.gate = Gate(d)                                                # (B,T,D) → (B,T,1) gain
        self.norm = Norm(d)                                                # (B,T,D) → (B,T,D)
        self.proj = nn.Linear(d, d)                                        # (B,T,D) → (B,T,D)
        self.drop = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.resid_scale = float(resid_scale)

        # Make the residual path gentle at init: Proj ≈ 0 ⇒ block ≈ identity initially.
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D) → out: (B,T,D)
        """
        y = self.fold(x)                      # 1) structural fold (depth‑wise)
        gain = self.gate(x)                   # 2) exposure from current state
        y = y * gain
        y = self.norm(y)                      # 3) damping
        y = self.drop(self.proj(y))           # project (and maybe drop)
        return x + self.resid_scale * y       # 4) residual add

    # External slow‑control: (β, γ, ⛔)
    def apply_control(self, beta: float | None = None, gamma: float | None = None, clamp: float | None = None) -> None:
        """
        Update inner gate/norm hyper‑parameters from an external controller.
        """
        self.gate.set_control(beta=beta, clamp=clamp)
        self.norm.set_control(gamma=gamma)
