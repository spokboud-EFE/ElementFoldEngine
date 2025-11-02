# ElementFold · domains/grid.py
# Grid‑domain folds, refreshed and aligned with the Fold–Gate–Norm contract:
#   • DepthwiseConv1dFold / DepthwiseConv2dFold  — non‑expansive, per‑channel folds
#   • HeatKernel1dFold / HeatKernel2dFold        — fixed Gaussian (heat) smoothing steps
#   • make_fold(kind, d, **kw)                   — tiny factory
#   • grid_neighbors / neighbors_to_edge_index   — optional graph helpers (1‑D ring/line)
#   • make_edge_index                            — neighbors → edge_index in one call
#
# Design notes (plain words):
#   1) Folds must be calm: we initialize to identity/safe kernels and (optionally) re‑normalize
#      weights so each channel’s kernel has ℓ₁≤1 (non‑expansive in the energy sense).
#   2) Heat kernels are fixed (requires_grad=False): they model a single diffusion “click”.
#   3) Shapes are explicit and boring on purpose: (B,T,D) for 1‑D and (B,H,W,D)|(B,D,H,W) for 2‑D.
#   4) Neighbor helpers let you lift the same idea onto a simple graph if you need message passing.

from __future__ import annotations
import math                                        # ✴ π and exp for Gaussians
from typing import Tuple
import torch                                       # ✴ tensors
import torch.nn as nn                              # ✴ modules
import torch.nn.functional as F                   # ✴ ops


# ————————————————————————————————————————————————————————————
# Utilities — kernel builders and safety renorm
# ————————————————————————————————————————————————————————————

def _gauss_1d(ksize: int, sigma: float) -> torch.Tensor:       # ⟲ normalized 1‑D Gaussian
    x = torch.arange(ksize, dtype=torch.float32) - (ksize - 1) / 2.0
    w = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return w / w.sum().clamp_min(1e-12)

def _gauss_2d(ksize: int, sigma: float) -> torch.Tensor:       # ⟲ separable 2‑D Gaussian = g⊗g
    g = _gauss_1d(ksize, sigma)
    k = torch.outer(g, g)
    return k / k.sum().clamp_min(1e-12)

@torch.no_grad()
def _renorm_depthwise_1d(w: torch.Tensor, positive: bool, l1_cap: float) -> None:
    """
    In‑place safety: per‑channel ℓ₁ clamp (≤ l1_cap). Optionally enforce non‑negativity.
    w shape: (D,1,K)
    """
    if positive:
        w.clamp_(min=0.0)
    # per‑channel ℓ₁ norm
    l1 = w.abs().sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    scale = (l1_cap / l1).clamp(max=1.0)                         # scale ≤ 1 (only shrink)
    w.mul_(scale)

@torch.no_grad()
def _renorm_depthwise_2d(w: torch.Tensor, positive: bool, l1_cap: float) -> None:
    """
    In‑place safety for 2‑D kernels: per‑channel ℓ₁ clamp (≤ l1_cap).
    w shape: (D,1,K,K)
    """
    if positive:
        w.clamp_(min=0.0)
    l1 = w.abs().sum(dim=(1, 2, 3), keepdim=True).clamp_min(1e-12)
    scale = (l1_cap / l1).clamp(max=1.0)
    w.mul_(scale)


# ————————————————————————————————————————————————————————————
# Depthwise (channel‑wise) convolutions as calm folds
# ————————————————————————————————————————————————————————————

class DepthwiseConv1dFold(nn.Module):
    """
    Depth‑wise 1‑D fold over time (groups=D, channels do not mix inside the fold).
    kind:
      • 'identity' → center tap = 1
      • 'avg3'     → [1,2,1]/4 smoother (requires ksize≥3)
      • 'gauss'    → Gaussian window with σ derived from ksize
    Options:
      • learn (bool)              → whether the kernel is trainable
      • enforce_nonexpansive (bool) → clamp per‑channel ℓ₁≤1 during training
      • positive (bool)           → clip negative weights (monotone averaging)
      • taps (odd int)            → kernel size (auto‑odd)
      • l1_cap (float)            → ℓ₁ cap value (default 1.0)
    """
    def __init__(
        self,
        d: int,
        kind: str = "identity",
        learn: bool = True,
        enforce_nonexpansive: bool = True,
        positive: bool = False,
        taps: int = 3,
        l1_cap: float = 1.0,
    ):
        super().__init__()
        ksize = int(taps if taps % 2 == 1 else taps + 1)        # ✴ ensure odd kernel
        pad = ksize // 2                                        # ✴ same‑padding
        self.conv = nn.Conv1d(d, d, ksize, padding=pad, groups=d, bias=False)  # ⟲ depth‑wise conv
        self._init_kernel(d, ksize, kind)                       # ✴ safe init
        self.conv.weight.requires_grad = bool(learn)            # ✴ freeze if not learning
        # safety knobs
        self.enforce_nonexp = bool(enforce_nonexpansive)        # ⚖ on/off clamp
        self.positive = bool(positive)                          # ⊕ enforce ≥0
        self.l1_cap = float(l1_cap)                             # ‖w‖₁ cap

    def _init_kernel(self, d: int, ksize: int, kind: str) -> None:
        with torch.no_grad():
            w = torch.zeros(d, 1, ksize)
            if kind == "avg3" and ksize >= 3:
                base = torch.tensor([0.25, 0.5, 0.25], dtype=torch.float32)
                start = (ksize - 3) // 2
                w[:, 0, start:start + 3] = base
            elif kind == "gauss":
                sigma = max(1.0, 0.25 * ksize)
                base = _gauss_1d(ksize, sigma)
                w[:, 0, :] = base
            else:  # identity
                w[:, 0, ksize // 2] = 1.0
            self.conv.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D) → Conv1d expects (B,D,T)
        if self.training and self.enforce_nonexp:               # ⚖ keep kernels calm while training
            _renorm_depthwise_1d(self.conv.weight, self.positive, self.l1_cap)
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)        # ⟲ fold along time
        return y                                                # (B,T,D)


class DepthwiseConv2dFold(nn.Module):
    """
    Depth‑wise 2‑D fold over (H,W) with per‑channel kernels.
    Accepts either (B,H,W,D) or (B,D,H,W) and preserves input ordering.
    kind: 'identity' | 'avg3' | 'gauss' (same semantics as 1‑D).
    """
    def __init__(
        self,
        d: int,
        kind: str = "identity",
        learn: bool = True,
        enforce_nonexpansive: bool = True,
        positive: bool = False,
        taps: int = 3,
        l1_cap: float = 1.0,
    ):
        super().__init__()
        ksize = int(taps if taps % 2 == 1 else taps + 1)
        pad = ksize // 2
        self.conv = nn.Conv2d(d, d, ksize, padding=pad, groups=d, bias=False)
        self._init_kernel(d, ksize, kind)
        self.conv.weight.requires_grad = bool(learn)
        self.enforce_nonexp = bool(enforce_nonexpansive)
        self.positive = bool(positive)
        self.l1_cap = float(l1_cap)

    def _init_kernel(self, d: int, ksize: int, kind: str) -> None:
        with torch.no_grad():
            w = torch.zeros(d, 1, ksize, ksize)
            if kind == "avg3" and ksize >= 3:
                base = torch.tensor([[0.0625, 0.125, 0.0625],
                                     [0.1250, 0.250, 0.1250],
                                     [0.0625, 0.125, 0.0625]], dtype=torch.float32)
                s = (ksize - 3) // 2
                w[:, 0, s:s + 3, s:s + 3] = base
            elif kind == "gauss":
                sigma = max(1.0, 0.25 * ksize)
                base = _gauss_2d(ksize, sigma)
                w[:, 0, :, :] = base
            else:  # identity
                w[:, 0, ksize // 2, ksize // 2] = 1.0
            self.conv.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.enforce_nonexp:
            _renorm_depthwise_2d(self.conv.weight, self.positive, self.l1_cap)
        if x.dim() != 4:
            raise ValueError("DepthwiseConv2dFold expects (B,H,W,D) or (B,D,H,W)")
        if x.shape[1] == x.shape[-1]:                           # (B,D,H,W)
            return self.conv(x)
        else:                                                    # (B,H,W,D)
            return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


# ————————————————————————————————————————————————————————————
# Heat‑kernel folds (fixed diffusion step)
# ————————————————————————————————————————————————————————————

class HeatKernel1dFold(nn.Module):
    """
    One diffusion “click” along time via a normalized Gaussian window.
    τ sets the smoothness; weights are fixed (requires_grad=False) for stability.
    """
    def __init__(self, d: int, tau: float = 1.0, taps: int = 7):
        super().__init__()
        ksize = int(taps if taps % 2 == 1 else taps + 1)
        pad = ksize // 2
        self.conv = nn.Conv1d(d, d, ksize, padding=pad, groups=d, bias=False)
        with torch.no_grad():
            # σ heuristics: proportional to √τ and window size; wide enough but calm.
            sigma = max(1.0, math.sqrt(max(1e-6, float(tau))) * (ksize / 3.0))
            g = _gauss_1d(ksize, sigma)
            w = torch.zeros(d, 1, ksize)
            w[:, 0, :] = g
            self.conv.weight.copy_(w)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class HeatKernel2dFold(nn.Module):
    """
    One diffusion “click” on images/grids via a separable 2‑D Gaussian.
    We accept (B,H,W,D) and (B,D,H,W) and preserve input ordering.
    """
    def __init__(self, d: int, tau: float = 1.0, taps: int = 7):
        super().__init__()
        ksize = int(taps if taps % 2 == 1 else taps + 1)
        pad = ksize // 2
        self.conv = nn.Conv2d(d, d, ksize, padding=pad, groups=d, bias=False)
        with torch.no_grad():
            sigma = max(1.0, math.sqrt(max(1e-6, float(tau))) * (ksize / 3.0))
            k2 = _gauss_2d(ksize, sigma)
            w = torch.zeros(d, 1, ksize, ksize)
            w[:, 0, :, :] = k2
            self.conv.weight.copy_(w)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError("HeatKernel2dFold expects (B,H,W,D) or (B,D,H,W)")
        if x.shape[1] == x.shape[-1]:                           # (B,D,H,W)
            return self.conv(x)
        else:                                                    # (B,H,W,D)
            return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


# ————————————————————————————————————————————————————————————
# Minimal factory
# ————————————————————————————————————————————————————————————

def make_fold(kind: str, *, d: int, **kw) -> nn.Module:        # ✴ tiny “choose your fold” helper
    k = str(kind).lower()
    if k == "conv1d":
        return DepthwiseConv1dFold(d=d, **kw)
    if k == "conv2d":
        return DepthwiseConv2dFold(d=d, **kw)
    if k == "heat1d":
        return HeatKernel1dFold(d=d, **kw)
    if k == "heat2d":
        return HeatKernel2dFold(d=d, **kw)
    raise KeyError(f"unknown fold kind: {kind!r}")


# ————————————————————————————————————————————————————————————
# Optional graph helpers — 1‑D ring/line neighbors → edge_index
# ————————————————————————————————————————————————————————————

def grid_neighbors(n: int, radius: int = 1, wrap: bool = True) -> torch.LongTensor:
    """
    Neighbor table for a 1‑D grid of length n.
    Returns (n, 2r+1) with indices wrapped (ring) or clamped (line).
    """
    n = int(max(1, n)); r = int(max(0, radius))
    win = 2 * r + 1
    if win <= 1:
        return torch.arange(n, dtype=torch.long).unsqueeze(1)
    centers = torch.arange(n, dtype=torch.long).unsqueeze(1)
    offsets = torch.arange(-r, r + 1, dtype=torch.long)
    table = centers + offsets
    return table.remainder(n) if wrap else table.clamp(0, n - 1)

def neighbors_to_edge_index(
    table: torch.LongTensor, directed: bool = False, self_loops: bool = False
) -> torch.LongTensor:
    """
    Convert (n, win) neighbor table into COO edge_index (2, E).
    """
    n, win = table.shape
    src = torch.arange(n, dtype=torch.long).unsqueeze(1).expand(n, win).reshape(-1)
    dst = table.reshape(-1)
    if not self_loops:
        m = src != dst
        src, dst = src[m], dst[m]
    if not directed:
        src, dst = torch.cat([src, dst], 0), torch.cat([dst, src], 0)
    return torch.stack([src, dst], dim=0)

def make_edge_index(
    n: int, radius: int = 1, wrap: bool = True, directed: bool = False, self_loops: bool = False
) -> torch.LongTensor:
    """
    Convenience: neighbors → edge_index in one call.
    """
    return neighbors_to_edge_index(grid_neighbors(n, radius=radius, wrap=wrap),
                                   directed=directed, self_loops=self_loops)
