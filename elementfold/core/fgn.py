# ElementFold · fgn.py
# ============================================================
# Fold–Gate–Norm (FGN): the coherence engine’s heartbeat.
#
# Overview
# --------
#   1) Fold   — depth-wise local structure (no mixing across channels)
#   2) Gate   — scalar exposure multiplier e^{β·g(y)} (centered, safe)
#   3) Norm   — energy damping (γ) to keep stability through depth
#   4) Residual projection — preserves identity, adds gentle updates
#
# Control knobs
#   β  exposure : how much to amplify structure (gate strength)
#   γ  damping  : how strongly to calm energy
#   ⛔ clamp    : soft bound on negative gate side (safety)
#
# Each block is self-contained, deterministic, and torch-native.
# ============================================================

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

__all__ = ["FoldGrid", "Gate", "Norm", "FGNBlock"]

# ============================================================
# 1. Fold — depth-wise local aggregation (non-expansive)
# ============================================================

class FoldGrid(nn.Module):
    """Depth-wise 1-D convolution along time, preserving identity."""
    def __init__(
        self,
        d: int,
        kind: str = "identity",
        learn: bool = True,
        enforce_l1_nonexpansive: bool = True,
    ):
        super().__init__()
        self.d = d
        self.kind = kind
        self._use_param = bool(learn and enforce_l1_nonexpansive)

        if self._use_param:
            # Simplex kernel via softmax → L1-non-expansive.
            self.w_logits = nn.Parameter(torch.zeros(d, 3))
            with torch.no_grad():
                base = torch.tensor([0.25, 0.50, 0.25]) if kind == "avg3" else torch.tensor([1e-4, 0.9998, 1e-4])
                self.w_logits.copy_(base.log().expand(d, 3))
        else:
            self.conv = nn.Conv1d(d, d, 3, padding=1, groups=d, bias=False)
            with torch.no_grad():
                k = torch.zeros(d, 1, 3)
                if kind == "avg3":
                    k[:, :, 0], k[:, :, 1], k[:, :, 2] = 0.25, 0.50, 0.25
                else:
                    k[:, :, 1] = 1.0
                self.conv.weight.copy_(k)
            self.conv.weight.requires_grad = bool(learn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._use_param:
            w = F.softmax(self.w_logits, dim=-1).view(self.d, 1, 3)
            y = F.conv1d(x.transpose(1, 2), w, padding=1, groups=self.d)
            return y.transpose(1, 2)
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

# ============================================================
# 2. Gate — exponential exposure e^{β·g}
# ============================================================

class Gate(nn.Module):
    """Learned scalar gate per timestep; safe exponential exposure."""
    def __init__(self, d: int, pos_cap: float = 1.0):
        super().__init__()
        self.lin = nn.Linear(d, 1)
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("clamp", torch.tensor(5.0))
        self._pos_cap = float(pos_cap)
        nn.init.normal_(self.lin.weight, std=0.02)
        if self.lin.bias is not None:
            nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        g = self.lin(x).squeeze(-1)
        if mask is not None:
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1)
            g = g - (g * mask).sum(dim=1, keepdim=True) / denom
        else:
            g = g - g.mean(dim=1, keepdim=True)

        g = g.clamp(min=-float(self.clamp), max=self._pos_cap)
        beta = F.softplus(self.beta)        # β ≥ 0
        prod = (beta * g).clamp(-12.0, 12.0)
        if prod.dtype in (torch.float16, torch.bfloat16):
            gain = torch.exp(prod.float()).to(prod.dtype).unsqueeze(-1)
        else:
            gain = torch.exp(prod).unsqueeze(-1)
        return gain

    def set_control(self, *, beta: Optional[float] = None, clamp: Optional[float] = None) -> None:
        with torch.no_grad():
            if beta is not None:
                self.beta.copy_(torch.tensor(float(beta), device=self.beta.device))
            if clamp is not None:
                self.clamp.copy_(torch.tensor(float(clamp), device=self.clamp.device))

# ============================================================
# 3. Norm — damping energy per row
# ============================================================

class Norm(nn.Module):
    """Row-wise normalization: y_norm = y / (‖y‖₁+ε)^γ."""
    def __init__(self, d: int, gamma: float = 0.5):
        super().__init__()
        self.gamma = nn.Parameter(torch.tensor(float(gamma)))
        self.eps = 1e-6

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        g = torch.clamp(self.gamma, 0.0, 1.0).to(y.dtype)
        scale = (y.abs().sum(dim=-1, keepdim=True) + self.eps).pow(g)
        return y / scale

    def set_control(self, *, gamma: Optional[float] = None) -> None:
        if gamma is not None:
            with torch.no_grad():
                self.gamma.copy_(torch.tensor(float(gamma), device=self.gamma.device))

# ============================================================
# 4. FGN Block — Fold → Gate → Norm → Residual
# ============================================================

class FGNBlock(nn.Module):
    """
    Core block:
        y = Fold(x)
        y = y * Gate(y)
        y = Norm(y)
        out = x + Proj(y)
    """
    def __init__(
        self,
        d: int,
        fold_kind: str = "identity",
        fold_learn: bool = True,
        resid_scale: float = 1.0,
        dropout: float = 0.0,
        enforce_fold_nonexpansive: bool = True,
        gate_on: str = "y",  # default: gate derives from folded state
    ):
        super().__init__()
        self.fold = FoldGrid(d, kind=fold_kind, learn=fold_learn, enforce_l1_nonexpansive=enforce_fold_nonexpansive)
        self.gate = Gate(d)
        self.norm = Norm(d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.resid_scale = float(resid_scale)
        self.gate_on = gate_on

        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.fold(x)
        gain_src = y if self.gate_on == "y" else x
        gain = self.gate(gain_src, mask=mask).to(x.dtype)
        y = y * gain
        y = self.norm(y)
        y = self.drop(self.proj(y))
        return x + self.resid_scale * y

    def apply_control(self, *, beta: Optional[float] = None, gamma: Optional[float] = None, clamp: Optional[float] = None) -> None:
        """Propagate β, γ, ⛔ controls to internal subsystems."""
        self.gate.set_control(beta=beta, clamp=clamp)
        self.norm.set_control(gamma=gamma)
