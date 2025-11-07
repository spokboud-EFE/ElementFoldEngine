# ElementFold · helpers/qctrl.py
# ============================================================
# Runtime safety controls (numerical "quantum control")
# ------------------------------------------------------------
# Purpose:
#   Maintain stable, monotonic relaxation dynamics by
#   clamping dt, λ, D, and validating variance monotonicity.
#
# These controls are called inside core.runtime loops or
# before/after each PDE update.
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any


# ============================================================
# 1. Clamp utilities
# ============================================================

def clamp_dt(dt: float, min_dt: float = 1e-9, max_dt: float = 1.0) -> float:
    """
    Clamp integration step size to a safe numeric range.
    """
    dt = float(dt)
    if not np.isfinite(dt):
        dt = min_dt
    return max(min_dt, min(max_dt, dt))


def clamp_params(lambda_: float,
                 D: float,
                 limits: Tuple[float, float] = (0.0, 10.0)) -> Tuple[float, float]:
    """
    Clamp λ (letting-go rate) and D (diffusion coefficient)
    into physically reasonable bounds.
    """
    lo, hi = limits
    lam = min(max(lambda_, lo), hi)
    diff = min(max(D, lo), hi)
    return lam, diff


# ============================================================
# 2. Variance-based monotonicity enforcement
# ============================================================

def enforce_monotone(phi_t: np.ndarray,
                     phi_t1: np.ndarray,
                     allow_increase: float = 1e-9) -> bool:
    """
    Enforce that variance has not increased beyond tolerance.
    Returns True if stable, False if a significant increase is detected.
    """
    v0 = float(np.var(phi_t))
    v1 = float(np.var(phi_t1))
    return v1 <= v0 + allow_increase


# ============================================================
# 3. Safety summary (used by controllers)
# ============================================================

def safety_summary(dt: float,
                   lambda_: float,
                   D: float,
                   phi_t: np.ndarray,
                   phi_t1: np.ndarray) -> Dict[str, Any]:
    """
    Produce a concise diagnostic dictionary.
    """
    stable = enforce_monotone(phi_t, phi_t1)
    return {
        "dt": float(dt),
        "lambda": float(lambda_),
        "D": float(D),
        "variance_t": float(np.var(phi_t)),
        "variance_t1": float(np.var(phi_t1)),
        "variance_drop": float(np.var(phi_t) - np.var(phi_t1)),
        "stable": stable,
    }


# ============================================================
# 4. Example self-test
# ============================================================

if __name__ == "__main__":
    phi0 = np.random.normal(0, 0.1, (32, 32))
    phi1 = phi0 * 0.9
    lam, D = clamp_params(0.5, 0.2)
    dt = clamp_dt(0.05)
    print("Safety:", safety_summary(dt, lam, D, phi0, phi1))
