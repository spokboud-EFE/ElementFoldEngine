# ElementFold · core/quantize.py
# ============================================================
# Quantization and discretization helpers for the Relaxation engine
# ------------------------------------------------------------
# 1. Finite-difference Laplacian stencils (NumPy default)
# 2. Optional torch-accelerated versions (if torch available)
# 3. INT8 quantization utilities (symmetric range) usable on CPU
#
# Design goals:
#   • Deterministic: same input → same output, CPU/GPU consistent.
#   • Graceful fallback: NumPy first, Torch only if present.
#   • Numerical stability: symmetric quantization avoids bias.
# ============================================================

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Literal, Union, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ============================================================
# Core numeric constants
# ============================================================

INT8_QMIN_SYM = -127
INT8_QMAX_SYM = +127
EPS = 1e-12


# ============================================================
# 1. Laplacian and gradient stencils (NumPy)
# ============================================================

def laplacian(phi: np.ndarray,
              spacing: Tuple[float, ...],
              bc: Literal["neumann", "periodic"] = "neumann") -> np.ndarray:
    """
    Finite-difference Laplacian ∇²φ for 1D, 2D, 3D arrays.
    Boundary conditions:
        'neumann'  → mirror edge (zero-gradient)
        'periodic' → wrap-around
    """
    ndim = phi.ndim
    out = np.zeros_like(phi, dtype=np.float64)
    spacing = np.asarray(spacing, dtype=float)
    if spacing.size != ndim:
        raise ValueError(f"spacing must have {ndim} elements")

    for axis in range(ndim):
        dx = spacing[axis]
        roll_f = np.roll(phi, -1, axis=axis)
        roll_b = np.roll(phi, +1, axis=axis)

        if bc == "neumann":
            # Mirror boundary
            sl_f = [slice(None)] * ndim
            sl_b = [slice(None)] * ndim
            sl_f[axis] = -1
            sl_b[axis] = 0
            roll_f[tuple(sl_f)] = phi[tuple(sl_f)]
            roll_b[tuple(sl_b)] = phi[tuple(sl_b)]

        term = (roll_f - 2.0 * phi + roll_b) / (dx * dx)
        out += term
    return out


def grad_squared(phi: np.ndarray,
                 spacing: Tuple[float, ...],
                 bc: Literal["neumann", "periodic"] = "neumann") -> np.ndarray:
    """
    Return |∇φ|² using centered differences.
    """
    ndim = phi.ndim
    spacing = np.asarray(spacing, dtype=float)
    if spacing.size != ndim:
        raise ValueError(f"spacing must have {ndim} elements")

    g2 = np.zeros_like(phi, dtype=np.float64)
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

        grad_axis = (roll_f - roll_b) / (2.0 * dx)
        g2 += grad_axis ** 2
    return g2


# ============================================================
# 2. Optional torch-accelerated versions
# ============================================================

if TORCH_AVAILABLE:

    def laplacian_torch(phi: torch.Tensor,
                        spacing: Tuple[float, ...],
                        bc: Literal["neumann", "periodic"] = "neumann") -> torch.Tensor:
        ndim = phi.ndim
        out = torch.zeros_like(phi, dtype=torch.float32)
        for axis in range(ndim):
            dx = spacing[axis]
            roll_f = torch.roll(phi, -1, dims=axis)
            roll_b = torch.roll(phi, +1, dims=axis)
            if bc == "neumann":
                # Mirror boundary
                idx_f = [slice(None)] * ndim
                idx_b = [slice(None)] * ndim
                idx_f[axis] = -1
                idx_b[axis] = 0
                roll_f[tuple(idx_f)] = phi[tuple(idx_f)]
                roll_b[tuple(idx_b)] = phi[tuple(idx_b)]
            term = (roll_f - 2.0 * phi + roll_b) / (dx * dx)
            out += term
        return out

    def grad_squared_torch(phi: torch.Tensor,
                           spacing: Tuple[float, ...],
                           bc: Literal["neumann", "periodic"] = "neumann") -> torch.Tensor:
        ndim = phi.ndim
        g2 = torch.zeros_like(phi, dtype=torch.float32)
        for axis in range(ndim):
            dx = spacing[axis]
            roll_f = torch.roll(phi, -1, dims=axis)
            roll_b = torch.roll(phi, +1, dims=axis)
            if bc == "neumann":
                idx_f = [slice(None)] * ndim
                idx_b = [slice(None)] * ndim
                idx_f[axis] = -1
                idx_b[axis] = 0
                roll_f[tuple(idx_f)] = phi[tuple(idx_f)]
                roll_b[tuple(idx_b)] = phi[tuple(idx_b)]
            grad_axis = (roll_f - roll_b) / (2.0 * dx)
            g2 += grad_axis ** 2
        return g2
else:
    laplacian_torch = None
    grad_squared_torch = None


# ============================================================
# 3. Symmetric INT8 quantization (NumPy / Torch)
# ============================================================

def choose_scale_symmetric(x: Union[np.ndarray, "torch.Tensor"]) -> float:
    """Return scalar symmetric scale factor."""
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        maxabs = float(x.detach().abs().max().cpu().item())
    else:
        maxabs = float(np.abs(x).max())
    return max(maxabs / float(INT8_QMAX_SYM), EPS)


def quantize_array(x: Union[np.ndarray, "torch.Tensor"],
                   scale: Optional[float] = None) -> Tuple[Union[np.ndarray, "torch.Tensor"], float]:
    """
    Quantize a float array → int8 using symmetric scaling.
    Returns (q, s)
    """
    if scale is None:
        scale = choose_scale_symmetric(x)
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        q = torch.round(x / scale).clamp(INT8_QMIN_SYM, INT8_QMAX_SYM).to(torch.int8)
    else:
        q = np.clip(np.round(x / scale), INT8_QMIN_SYM, INT8_QMAX_SYM).astype(np.int8)
    return q, scale


def dequantize_array(q: Union[np.ndarray, "torch.Tensor"],
                     scale: float) -> Union[np.ndarray, "torch.Tensor"]:
    """Return float array reconstructed from quantized values."""
    if TORCH_AVAILABLE and isinstance(q, torch.Tensor):
        return q.to(torch.float32) * float(scale)
    return q.astype(np.float32) * float(scale)


# ============================================================
# 4. Diagnostics
# ============================================================

def quantization_error(x: Union[np.ndarray, "torch.Tensor"],
                       scale: Optional[float] = None) -> float:
    """Return mean absolute quantization error."""
    q, s = quantize_array(x, scale)
    x_rec = dequantize_array(q, s)
    if TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        return float((x - x_rec).abs().mean().cpu().item())
    return float(np.mean(np.abs(x - x_rec)))


# ============================================================
# Public exports
# ============================================================

__all__ = [
    "laplacian", "grad_squared",
    "laplacian_torch", "grad_squared_torch",
    "quantize_array", "dequantize_array",
    "choose_scale_symmetric", "quantization_error",
    "INT8_QMIN_SYM", "INT8_QMAX_SYM"
]
