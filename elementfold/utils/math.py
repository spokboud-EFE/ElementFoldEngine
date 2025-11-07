# ElementFold · utils/math.py
# ============================================================
# Fundamental numerical helpers for the relaxation engine
# ------------------------------------------------------------
# Contains:
#   • safe_norm / normalize / rms
#   • finite difference Laplacian + gradient
#   • exponential decay / relaxation primitives
#   • smooth clipping and EMA utilities
#
# All operations are NumPy-based; Torch optional for acceleration.
# ============================================================

from __future__ import annotations
import numpy as np
import math
from typing import Tuple, Sequence, Any

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# 1. Norms and normalization
# ============================================================

def safe_norm(x: np.ndarray, eps: float = 1e-12) -> float:
    """Return √(⟨x²⟩) with small ε floor."""
    v = float(np.mean(np.square(x)))
    return math.sqrt(max(v, eps))


def rms(x: np.ndarray) -> float:
    """Root-mean-square amplitude."""
    return math.sqrt(float(np.mean(np.square(x))))


def normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return x / ‖x‖₂ safely."""
    n = safe_norm(x, eps)
    return x / n if n > 0 else x


# ============================================================
# 2. Finite differences (Laplacian, gradient)
# ============================================================

def laplacian(phi: np.ndarray,
              spacing: Tuple[float, ...],
              bc: str = "neumann") -> np.ndarray:
    """
    Compute ∇²Φ using centered differences.
    """
    ndim = phi.ndim
    out = np.zeros_like(phi, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.size != ndim:
        raise ValueError(f"spacing must have {ndim} values")
    for axis in range(ndim):
        dx = spacing[axis]
        roll_f = np.roll(phi, -1, axis=axis)
        roll_b = np.roll(phi, +1, axis=axis)
        if bc == "neumann":
            sl_f = [slice(None)] * ndim
            sl_b = [slice(None)] * ndim
            sl_f[axis] = -1
            sl_b[axis] = 0
            roll_f[tuple(sl_f)] = phi[tuple(sl_f)]
            roll_b[tuple(sl_b)] = phi[tuple(sl_b)]
        out += (roll_f - 2.0 * phi + roll_b) / (dx * dx)
    return out


def grad(phi: np.ndarray,
         spacing: Tuple[float, ...],
         bc: str = "neumann") -> Tuple[np.ndarray, ...]:
    """
    Compute gradient components ∂Φ/∂x_i.
    """
    ndim = phi.ndim
    spacing = np.asarray(spacing, dtype=float)
    grads = []
    for axis in range(ndim):
        dx = spacing[axis]
        roll_f = np.roll(phi, -1, axis=axis)
        roll_b = np.roll(phi, +1, axis=axis)
        if bc == "neumann":
            sl_f = [slice(None)] * ndim
            sl_b = [slice(None)] * ndim
            sl_f[axis] = -1
            sl_b[axis] = 0
            roll_f[tuple(sl_f)] = phi[tuple(sl_f)]
            roll_b[tuple(sl_b)] = phi[tuple(sl_b)]
        grads.append((roll_f - roll_b) / (2.0 * dx))
    return tuple(grads)


# ============================================================
# 3. Exponential decay and relaxation primitives
# ============================================================

def exp_decay(y: float | np.ndarray, rate: float, dt: float) -> np.ndarray:
    """Apply y ← y·exp(−rate·dt)."""
    return np.asarray(y) * math.exp(-float(rate) * float(dt))


def relaxation_step(phi: np.ndarray,
                    lam: float,
                    D: float,
                    phi_inf: float,
                    spacing: Tuple[float, ...],
                    bc: str = "neumann",
                    dt: float = 1.0) -> np.ndarray:
    """
    Perform one explicit relaxation update:
        Φ ← Φ + dt·(−λ(Φ−Φ∞) + D∇²Φ)
    """
    lap = laplacian(phi, spacing, bc)
    return phi + dt * (-lam * (phi - phi_inf) + D * lap)


# ============================================================
# 4. Smooth clipping and soft transitions
# ============================================================

def softclip(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Smooth tanh-based clamp into [lo, hi]."""
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + half * np.tanh((x - mid) / max(1e-6, half))


# ============================================================
# 5. Exponential moving average (EMA)
# ============================================================

def ema_update(target: np.ndarray,
               source: np.ndarray,
               decay: float = 0.99) -> np.ndarray:
    """Exponential moving average update."""
    return decay * target + (1.0 - decay) * source


# ============================================================
# 6. Optional Torch wrappers
# ============================================================

if TORCH_AVAILABLE:
    import torch

    def to_torch(x: np.ndarray) -> "torch.Tensor":
        return torch.as_tensor(x, dtype=torch.float32)

    def from_torch(x: "torch.Tensor") -> np.ndarray:
        return x.detach().cpu().numpy()
else:
    def to_torch(x: Any):  # type: ignore
        raise RuntimeError("Torch not available")
    def from_torch(x: Any):  # type: ignore
        raise RuntimeError("Torch not available")
