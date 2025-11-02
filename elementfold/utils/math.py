# ElementFold · utils/math.py
# Small numerical helpers that keep training/inference stable.

from __future__ import annotations
import math
import torch
from typing import Tuple


# ———————————————————————————————————————————————————————————
# Wrapping / modular arithmetic
# ———————————————————————————————————————————————————————————

def mod(x: torch.Tensor, m: float | int) -> torch.Tensor:
    """x mod m, preserving dtype/device."""
    return torch.remainder(x, float(m))


def wrap_centered(x: torch.Tensor, period: float) -> torch.Tensor:
    """
    Wrap real values into the centered interval (−period/2, +period/2].
    """
    p = float(period)
    x = torch.remainder(x + 0.5 * p, p)
    x = torch.where(x <= 0, x + p, x)
    return x - 0.5 * p


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap radians into (−π, π]."""
    return wrap_centered(x, 2.0 * math.pi)


# ———————————————————————————————————————————————————————————
# Stable norms / normalization
# ———————————————————————————————————————————————————————————

def safe_norm(x: torch.Tensor, p: float = 2.0, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    ‖x‖_p stabilized with ε.
    """
    if p == 2.0:
        return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=keepdim), min=eps))
    if p == 1.0:
        return torch.clamp(torch.sum(torch.abs(x), dim=dim, keepdim=keepdim), min=eps)
    # generic
    return torch.clamp(torch.sum(torch.abs(x) ** p, dim=dim, keepdim=keepdim), min=eps) ** (1.0 / p)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """x / max(‖x‖₂, ε)"""
    n = safe_norm(x, p=2.0, dim=dim, keepdim=True, eps=eps)
    return x / n


def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity between tensors along `dim`.
    """
    a_hat = l2_normalize(a, dim=dim, eps=eps)
    b_hat = l2_normalize(b, dim=dim, eps=eps)
    return torch.sum(a_hat * b_hat, dim=dim)


def rms(x: torch.Tensor, dim: int | None = None, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Root-mean-square with ε for stability.
    """
    if dim is None:
        return torch.sqrt(torch.clamp(torch.mean(x * x), min=eps))
    return torch.sqrt(torch.clamp(torch.mean(x * x, dim=dim, keepdim=keepdim), min=eps))


# ———————————————————————————————————————————————————————————
# Log-sum-exp / soft utilities
# ———————————————————————————————————————————————————————————

def logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Stable log ∑ exp(x)."""
    m = torch.max(x, dim=dim, keepdim=True).values
    z = torch.log(torch.clamp(torch.sum(torch.exp(x - m), dim=dim, keepdim=True), min=1e-20)) + m
    return z if keepdim else z.squeeze(dim)


def softclip(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """Smoothly clamp x into [lo, hi] using tanh."""
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + half * torch.tanh((x - mid) / max(1e-6, half))


# ———————————————————————————————————————————————————————————
# EMA / parameter utilities
# ———————————————————————————————————————————————————————————

@torch.no_grad()
def ema_(target: torch.nn.Module | torch.Tensor, source: torch.nn.Module | torch.Tensor, decay: float = 0.99) -> None:
    """
    In-place exponential moving average:
        target ← decay * target + (1 - decay) * source
    Works for nn.Modules (matches parameters by name) or plain tensors.
    """
    d = float(decay)
    if isinstance(target, torch.nn.Module) and isinstance(source, torch.nn.Module):
        t_params = dict(target.named_parameters())
        s_params = dict(source.named_parameters())
        for k, tp in t_params.items():
            sp = s_params.get(k, None)
            if (sp is None) or (sp.data.shape != tp.data.shape):
                continue
            tp.data.mul_(d).add_(sp.data, alpha=(1.0 - d))
        return
    if isinstance(target, torch.Tensor) and isinstance(source, torch.Tensor):
        target.mul_(d).add_(source, alpha=(1.0 - d))
        return
    raise TypeError("ema_ expects both args to be nn.Module or both to be torch.Tensor")
