# ElementFold · core/control.py
# ============================================================
# PDE controller — stable explicit update for the relaxation field
# ------------------------------------------------------------
# Evolves:
#     ∂Φ/∂t = -λ(Φ - Φ∞) + D∇²Φ
#
# Responsibilities:
#   • compute safe dt given λ, D, and grid spacing
#   • perform one explicit (or semi-implicit) step
#   • automatically use torch when available, else NumPy
# ============================================================

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Literal, Optional, Dict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .quantize import laplacian, laplacian_torch


# ============================================================
# Utility: compute safe dt
# ============================================================

def safe_dt(lambda_: float,
            D: float,
            spacing: Tuple[float, ...],
            safety: float = 0.8) -> float:
    """
    Return a numerically stable explicit step size for the relaxation PDE.
    CFL-like bound:
        dt ≤ safety / (λ + 2D∑(1/Δx_i²))
    """
    inv_sq = sum((1.0 / (dx * dx)) for dx in spacing)
    denom = max(lambda_ + 2.0 * D * inv_sq, 1e-12)
    return safety / denom


# ============================================================
# Main explicit stepper
# ============================================================

def step(phi: np.ndarray,
         lambda_: float,
         D: float,
         phi_inf: float,
         spacing: Tuple[float, ...],
         bc: Literal["neumann", "periodic"] = "neumann",
         dt: Optional[float] = None,
         safety: float = 0.8) -> np.ndarray:
    """
    Perform one relaxation update step on Φ (NumPy version).
    If dt is None, an automatically safe dt is computed.
    """
    if dt is None:
        dt = safe_dt(lambda_, D, spacing, safety)
    lap = laplacian(phi, spacing, bc)
    phi_next = phi + dt * (-lambda_ * (phi - phi_inf) + D * lap)
    return phi_next


# ============================================================
# Torch variant (auto GPU if available)
# ============================================================

def step_torch(phi,
               lambda_: float,
               D: float,
               phi_inf: float,
               spacing: Tuple[float, ...],
               bc: Literal["neumann", "periodic"] = "neumann",
               dt: Optional[float] = None,
               safety: float = 0.8) -> "torch.Tensor":
    """
    Torch version of the explicit PDE update.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("Torch not available; use step() instead.")
    if dt is None:
        dt = safe_dt(lambda_, D, spacing, safety)
    lap = laplacian_torch(phi, spacing, bc)
    return phi + dt * (-lambda_ * (phi - phi_inf) + D * lap)


# ============================================================
# Semi-implicit (optional) stepper for stiff regimes
# ============================================================

def step_semi_implicit(phi: np.ndarray,
                       lambda_: float,
                       D: float,
                       phi_inf: float,
                       spacing: Tuple[float, ...],
                       bc: Literal["neumann", "periodic"] = "neumann",
                       dt: Optional[float] = None,
                       safety: float = 0.8) -> np.ndarray:
    """
    Crank–Nicolson-like semi-implicit update for stiff λ,D.
    Solves (I - dt·D·∇²)Φ_{t+1} = Φ_t + dt·λ·Φ∞ - dt·λ·Φ_t
    Approximate with one Jacobi iteration (good enough for moderate stiffness).
    """
    if dt is None:
        dt = safe_dt(lambda_, D, spacing, safety)
    lap = laplacian(phi, spacing, bc)
    rhs = phi + dt * (-lambda_ * (phi - phi_inf) + D * lap)
    corr = dt * D * laplacian(rhs, spacing, bc)
    phi_next = rhs + corr
    return phi_next


# ============================================================
# Adaptive controller (deterministic)
# ============================================================

class Controller:
    """
    Adaptive driver that adjusts dt to maintain monotone variance drop.
    """
    def __init__(self,
                 lambda_: float,
                 D: float,
                 phi_inf: float,
                 spacing: Tuple[float, ...],
                 bc: str = "neumann",
                 safety: float = 0.8,
                 growth: float = 1.1,
                 shrink: float = 0.5):
        self.lambda_ = lambda_
        self.D = D
        self.phi_inf = phi_inf
        self.spacing = spacing
        self.bc = bc
        self.safety = safety
        self.growth = growth
        self.shrink = shrink
        self.dt = safe_dt(lambda_, D, spacing, safety)

    def evolve_once(self, phi: np.ndarray,
                    variance_fn: Optional[callable] = None) -> Tuple[np.ndarray, float]:
        """Perform one step and adapt dt if variance increases."""
        phi_next = step(phi, self.lambda_, self.D, self.phi_inf, self.spacing, self.bc, self.dt)
        if variance_fn is None:
            v_ok = np.var(phi_next) <= np.var(phi) + 1e-9
        else:
            v_ok = variance_fn(phi, phi_next)
        if v_ok:
            self.dt *= self.growth
        else:
            self.dt *= self.shrink
        self.dt = min(self.dt, safe_dt(self.lambda_, self.D, self.spacing, self.safety))
        return phi_next, self.dt


# ============================================================
# Exports
# ============================================================

__all__ = [
    "safe_dt",
    "step", "step_torch", "step_semi_implicit",
    "Controller"
]
