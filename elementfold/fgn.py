# elementfold/fgn.py
# ──────────────────────────────────────────────────────────────────────────────
# Fold–Gate–Norm (FGN) is the engine’s heartbeat:
#   1) Fold      — gather local structure along time without blowing up scale.
#   2) Gate      — compute a scalar exposure per step and apply a gain e^{β·g}.
#   3) Normalize — damp energy so depth remains stable.
# Finally, a residual lane keeps identity always available.
#
# Knobs (steered via Model.apply_control or external Supervisor):
#   • β (beta):   exposure — how strongly we amplify structure where g>0.
#   • γ (gamma):  damping  — how hard we calm energy after gating.
#   • ⛔ clamp:   gate cap — soft bound on how negative g may go (safety).
#
# Design & safety:
#   • Gate is centered per sequence (mean over T) so g has positive/negative mass.
#   • Asymmetric clamp: g ∈ [−clamp, +pos_cap] with a light positive cap (~1.0)
#     to allow meaningful boost yet avoid runaway exp(β·g). Norm then calms energy.
#   • Residual projection is zero‑init so each block starts near‑identity.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

__all__ = ["FoldGrid", "Gate", "Norm", "FGNBlock"]


# ──────────────────────────────────────────────────────────────────────────────
# 1) Fold: depth‑wise 1‑D aggregation along time (non‑expansive defaults)
# ──────────────────────────────────────────────────────────────────────────────
class FoldGrid(nn.Module):
    """
    Depth‑wise 1D convolution over the time axis.
    Each feature channel is folded independently (groups=d), so we never mix
    channels here. This preserves identity and keeps the fold non‑expansive.

    Args:
        d:      number of channels (feature dimension D).
        kind:   'identity' (center tap = 1) or 'avg3' ([1,2,1]/4 smoothing).
        learn:  if True, the kernel is learnable; otherwise it remains fixed.
    """
    def __init__(self, d: int, kind: str = "identity", learn: bool = True):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d, bias=False)  # (B,D,T) ↔ (B,D,T)
        with torch.no_grad():
            k = torch.zeros(d, 1, 3, dtype=self.conv.weight.dtype, device=self.conv.weight.device)
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
    • g is centered by subtracting the per‑sequence mean over time (T).
    • Asymmetric clamp: g ∈ [−clamp, +pos_cap] with a *light* positive cap (≈1.0)
      to enable amplification but avoid runaway gains. The negative side follows
      ⛔ (clamp) exactly.

    Args:
        d:         feature dimension D (input to the probe φ: ℝ^D→ℝ).
        pos_cap:   positive cap for g (safe exposure ceiling, default 1.0).
    """
    def __init__(self, d: int, pos_cap: float = 1.0):
        super().__init__()
        self.lin = nn.Linear(d, 1)                              # φ(x): (B,T,D) → (B,T,1)
        self.beta  = nn.Parameter(torch.tensor(1.0))            # β exposure (learnable; steerable)
        self.clamp = nn.Parameter(torch.tensor(5.0))            # ⛔ negative cap magnitude
        self._g_pos_cap = float(pos_cap)                        # light positive cap (constant)

        # Gentle init: keep φ small so exp(β·g) ≈ 1 at start.
        nn.init.normal_(self.lin.weight, std=0.02)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B,T,D)
        returns: (B,T,1) multiplicative gain (same device as x)
        """
        # Raw gate potential and per‑sequence mean centering
        g = self.lin(x).squeeze(-1)                             # (B,T)
        g = g - g.mean(dim=1, keepdim=True)                     # center per sequence

        # Asymmetric safety clamp: negative side follows ⛔, positive side lightly capped
        neg_cap = -self.clamp.item()
        pos_cap =  self._g_pos_cap
        g = g.clamp(min=neg_cap, max=pos_cap)                   # (B,T)

        # Mixed‑precision‑safe exponential:
        # - clamp β·g to a numerically safe window
        # - compute exp in fp32 when running in low precision, then cast back
        EXP_CAP = 12.0  # e^{±12} ~ [6e-6, 1.6e5]
        prod = (self.beta.to(g.dtype) * g).clamp(-EXP_CAP, EXP_CAP)
        if prod.dtype in (torch.float16, torch.bfloat16):
            gain = torch.exp(prod.float()).to(dtype=prod.dtype).unsqueeze(-1)  # (B,T,1)
        else:
            gain = torch.exp(prod).unsqueeze(-1)                                # (B,T,1)
        return gain

    def set_control(self, *, beta: Optional[float] = None, clamp: Optional[float] = None) -> None:
        """External slow‑control hook."""
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
        # Clamp gamma and match dtype for mixed‑precision stability
        g = torch.clamp(self.gamma, 0.0, 1.0).to(dtype=y.dtype)
        scale = (y.abs().sum(dim=-1, keepdim=True) + self.eps).pow(g)
        return y / scale

    def set_control(self, *, gamma: Optional[float] = None) -> None:
        """External slow‑control hook."""
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
    def __init__(
        self,
        d: int,
        fold_kind: str = "identity",
        fold_learn: bool = True,
        resid_scale: float = 1.0,
        dropout: float = 0.0,
    ):
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
        gain = self.gate(x).to(dtype=x.dtype) # 2) exposure from current state
        y = y * gain
        y = self.norm(y)                      # 3) damping
        y = self.drop(self.proj(y))           # project (and maybe drop)
        return x + self.resid_scale * y       # 4) residual add

    def apply_control(
        self,
        *,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        clamp: Optional[float] = None,
    ) -> None:
        """
        External slow‑control: update inner gate/norm hyper‑parameters.
        Mirrors Model.apply_control(...) expectations.
        """
        self.gate.set_control(beta=beta, clamp=clamp)
        self.norm.set_control(gamma=gamma)
