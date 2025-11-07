# ElementFold · core/telemetry.py
# ============================================================
# Telemetry for relaxation dynamics
# ------------------------------------------------------------
# Provides live stability and coherence metrics:
#     • variance(Φ)      – how uneven the field is
#     • grad_L2(Φ)       – magnitude of gradients
#     • total_energy(Φ)  – energy functional value
#     • monotone_var_drop – check that variance decreases
#
# All functions operate on NumPy arrays by default and
# use torch when available (for GPU acceleration).
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Tuple, Literal, Dict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .quantize import grad_squared, grad_squared_torch


# ============================================================
# Basic statistics
# ============================================================

def variance(phi: np.ndarray) -> float:
    """Spatial variance of Φ."""
    return float(np.var(phi))


def grad_L2(phi: np.ndarray,
            spacing: Tuple[float, ...],
            bc: Literal["neumann", "periodic"] = "neumann") -> float:
    """Integral of |∇Φ|² (NumPy)."""
    g2 = grad_squared(phi, spacing, bc)
    return float(np.mean(g2))


def total_energy(phi: np.ndarray,
                 lambda_: float,
                 D: float,
                 phi_inf: float,
                 spacing: Tuple[float, ...],
                 bc: Literal["neumann", "periodic"] = "neumann") -> float:
    """
    Energy functional:
        E = ∫ [ λ/2·(Φ−Φ∞)² + D/2·|∇Φ|² ] dV
    Integrated approximately as mean * volume element.
    """
    diff = phi - phi_inf
    local_E = 0.5 * (lambda_ * diff ** 2 + D * grad_squared(phi, spacing, bc))
    return float(np.mean(local_E))


# ============================================================
# Torch versions
# ============================================================

if TORCH_AVAILABLE:

    def variance_torch(phi: torch.Tensor) -> float:
        return float(torch.var(phi).cpu().item())

    def grad_L2_torch(phi: torch.Tensor,
                      spacing: Tuple[float, ...],
                      bc: Literal["neumann", "periodic"] = "neumann") -> float:
        g2 = grad_squared_torch(phi, spacing, bc)
        return float(g2.mean().cpu().item())

    def total_energy_torch(phi: torch.Tensor,
                           lambda_: float,
                           D: float,
                           phi_inf: float,
                           spacing: Tuple[float, ...],
                           bc: Literal["neumann", "periodic"] = "neumann") -> float:
        diff = phi - phi_inf
        local_E = 0.5 * (lambda_ * diff ** 2 + D * grad_squared_torch(phi, spacing, bc))
        return float(local_E.mean().cpu().item())
else:
    variance_torch = grad_L2_torch = total_energy_torch = None


# ============================================================
# Coherence and safety checks
# ============================================================

def monotone_var_drop(phi_t: np.ndarray,
                      phi_t1: np.ndarray,
                      tol: float = 1e-9) -> bool:
    """
    True if variance has not increased beyond tolerance.
    """
    return np.var(phi_t1) <= np.var(phi_t) + tol


def summary(phi: np.ndarray,
            lambda_: float,
            D: float,
            phi_inf: float,
            spacing: Tuple[float, ...],
            bc: Literal["neumann", "periodic"] = "neumann") -> Dict[str, float]:
    """
    Return a compact dictionary for dashboards:
        {'variance','grad_L2','energy'}
    """
    return {
        "variance": variance(phi),
        "grad_L2": grad_L2(phi, spacing, bc),
        "energy": total_energy(phi, lambda_, D, phi_inf, spacing, bc),
    }


# ============================================================
# Pretty print for Studio/CLI
# ============================================================

def pretty(metrics: Dict[str, float]) -> str:
    """Formatted one-line summary."""
    v = metrics.get("variance", 0.0)
    g = metrics.get("grad_L2", 0.0)
    e = metrics.get("energy", 0.0)
    return f"variance={v:.6g}  grad²={g:.6g}  energy={e:.6g}"


# ============================================================
# Exports
# ============================================================

__all__ = [
    "variance", "grad_L2", "total_energy",
    "variance_torch", "grad_L2_torch", "total_energy_torch",
    "monotone_var_drop", "summary", "pretty"
]
