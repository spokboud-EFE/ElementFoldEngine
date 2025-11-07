# ElementFold · core/fgn.py
# ============================================================
# Fold–Gate–Norm (FGN) — Physical interpretation
# ------------------------------------------------------------
# In the relaxation model:
#   • Fold  → cumulative smoothing along a path  (ℱ)
#   • Gate  → exponential mapping  (e^{±ℱ})
#   • Norm  → normalization ensuring energy stability
#
# This module defines scalar/vectorized functions for:
#     folds(path, η)
#     redshift_from_F(F)
#     brightness_tilt(F)
#     time_dilation(F, σ_obs, σ_emit)
#     bend(ray, n, Φ, ν)
#
# NumPy by default; Torch when available.
# ============================================================

from __future__ import annotations
import numpy as np
import math
from typing import Callable, List, Dict, Any, Union

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .data import Path, PathSegment

__all__ = [
    "folds",
    "redshift_from_F",
    "brightness_tilt",
    "time_dilation",
    "bend",
    "folds_vec",
]


# ============================================================
# Fold integration
# ============================================================

def folds(path: Path,
          eta: Callable[[float, float], float]) -> float:
    """
    Integrate cumulative relaxation folds ℱ = ∫ η(Φ,ν) ds along path segments.
    """
    F = 0.0
    for seg in path:
        F += eta(seg.phi, seg.nu) * seg.ds
    return float(F)


def folds_vec(ds: np.ndarray,
              phi: np.ndarray,
              nu: np.ndarray,
              eta: Callable[[float, float], float]) -> np.ndarray:
    """
    Vectorized fold integral: each element corresponds to one segment.
    """
    eta_vals = np.vectorize(eta)(phi, nu)
    return np.cumsum(eta_vals * ds)


# ============================================================
# Observable transformations
# ============================================================

def redshift_from_F(F: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Return redshift 1+z = e^F."""
    return np.exp(F) - 1.0


def brightness_tilt(F: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Surface brightness dimming factor ∝ e^{-2F}."""
    return np.exp(-2.0 * np.asarray(F))


def time_dilation(F: Union[float, np.ndarray],
                  sigma_obs: float,
                  sigma_emit: float) -> Union[float, np.ndarray]:
    """
    Apparent time stretch due to relaxation folds and local clock factors.
        (dτ_obs / dτ_emit) = (σ_obs / σ_emit) · e^{F}
    """
    return (sigma_obs / sigma_emit) * np.exp(np.asarray(F))


def bend(ray: Path,
         n_func: Callable[[float, float], float],
         grad_func: Callable[[float, float], float] | None = None) -> float:
    """
    Approximate total angular deflection Δθ ≈ ∫ ∇⊥ ln n ds.
    If grad_func not provided, finite-difference of log(n) along path is used.
    """
    if len(ray) < 2:
        return 0.0
    nus = np.array([seg.nu for seg in ray], dtype=float)
    phis = np.array([seg.phi for seg in ray], dtype=float)
    ds = np.array([seg.ds for seg in ray], dtype=float)

    n_vals = np.array([n_func(phi, nu) for phi, nu in zip(phis, nus)], dtype=float)
    logn = np.log(np.clip(n_vals, 1e-12, None))
    if grad_func is not None:
        grads = np.array([grad_func(phi, nu) for phi, nu in zip(phis, nus)], dtype=float)
    else:
        grads = np.gradient(logn, ds, edge_order=1)
    return float(np.trapz(np.abs(grads), x=None, dx=np.mean(ds)))


# ============================================================
# Torch variants (optional)
# ============================================================

if TORCH_AVAILABLE:

    def folds_torch(path: Path,
                    eta: Callable[[float, float], float]) -> float:
        F = 0.0
        for seg in path:
            F += eta(float(seg.phi), float(seg.nu)) * float(seg.ds)
        return float(F)

    def redshift_from_F_torch(F: "torch.Tensor") -> "torch.Tensor":
        return torch.exp(F) - 1.0

    def brightness_tilt_torch(F: "torch.Tensor") -> "torch.Tensor":
        return torch.exp(-2.0 * F)

    def time_dilation_torch(F: "torch.Tensor",
                            sigma_obs: float,
                            sigma_emit: float) -> "torch.Tensor":
        return (sigma_obs / sigma_emit) * torch.exp(F)
else:
    folds_torch = redshift_from_F_torch = brightness_tilt_torch = time_dilation_torch = None


# ============================================================
# Pretty / Summary
# ============================================================

def summary(path: Path,
            eta: Callable[[float, float], float],
            sigma_obs: float,
            sigma_emit: float,
            n_func: Callable[[float, float], float]) -> Dict[str, Any]:
    """
    Return a compact summary of observables along a path:
        {'F','1+z','brightness','time_dilation','bend'}
    """
    F = folds(path, eta)
    return {
        "F": F,
        "1+z": redshift_from_F(F),
        "brightness": brightness_tilt(F),
        "time_dilation": time_dilation(F, sigma_obs, sigma_emit),
        "bend": bend(path, n_func),
    }
