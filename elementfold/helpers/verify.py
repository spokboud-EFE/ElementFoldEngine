# ElementFold · helpers/verify.py
# ============================================================
# Runtime invariants and safety diagnostics
# ------------------------------------------------------------
# Purpose:
#   Verify that the relaxation field Φ evolves calmly:
#     • variance never increases
#     • gradients remain finite
#     • energy decreases or stabilizes
#
# These checks are meant to be lightweight and usable
# inside the runtime loop (no heavy allocations).
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple

from ..core.telemetry import variance, grad_L2, total_energy


# ============================================================
# 1. Monotone variance drop
# ============================================================

def check_monotone_calming(phi_t: np.ndarray,
                           phi_t1: np.ndarray,
                           tol: float = 1e-9) -> bool:
    """
    Verify that variance has not increased beyond tolerance.
    Returns True if stable, False if variance rose.
    """
    v0 = variance(phi_t)
    v1 = variance(phi_t1)
    return v1 <= v0 + tol


# ============================================================
# 2. Gradient and energy finiteness
# ============================================================

def check_finiteness(phi: np.ndarray,
                     lambda_: float,
                     D: float,
                     phi_inf: float,
                     spacing: Tuple[float, ...],
                     bc: str = "neumann") -> Dict[str, bool]:
    """
    Compute variance, grad², and energy; report if each is finite.
    """
    v = variance(phi)
    g = grad_L2(phi, spacing, bc)
    e = total_energy(phi, lambda_, D, phi_inf, spacing, bc)
    return {
        "variance_finite": np.isfinite(v),
        "grad_finite": np.isfinite(g),
        "energy_finite": np.isfinite(e),
    }


# ============================================================
# 3. Combined invariant check
# ============================================================

def check_invariants(phi_t: np.ndarray,
                     phi_t1: np.ndarray,
                     lambda_: float,
                     D: float,
                     phi_inf: float,
                     spacing: Tuple[float, ...],
                     bc: str = "neumann") -> Dict[str, Any]:
    """
    Combine variance, gradient, and energy diagnostics.
    Returns a dict with current metrics and boolean flags.
    """
    v_ok = check_monotone_calming(phi_t, phi_t1)
    fin = check_finiteness(phi_t1, lambda_, D, phi_inf, spacing, bc)
    report = {
        "monotone_calming": bool(v_ok),
        "variance_t": variance(phi_t),
        "variance_t1": variance(phi_t1),
        "grad_L2_t1": grad_L2(phi_t1, spacing, bc),
        "energy_t1": total_energy(phi_t1, lambda_, D, phi_inf, spacing, bc),
    }
    report.update(fin)
    return report


# ============================================================
# 4. Summary diagnostics for dashboards
# ============================================================

def summary(phi_t: np.ndarray,
            phi_t1: np.ndarray,
            lambda_: float,
            D: float,
            phi_inf: float,
            spacing: Tuple[float, ...],
            bc: str = "neumann") -> Dict[str, Any]:
    """
    Compact report used in runtime and Studio displays.
    Returns: {'variance_drop': float, 'grad_L2': float, 'energy': float}
    """
    v0 = variance(phi_t)
    v1 = variance(phi_t1)
    drop = v0 - v1
    return {
        "variance_drop": float(drop),
        "grad_L2": grad_L2(phi_t1, spacing, bc),
        "energy": total_energy(phi_t1, lambda_, D, phi_inf, spacing, bc),
        "stable": drop >= -1e-9,
    }


# ============================================================
# 5. Self-test (manual)
# ============================================================

if __name__ == "__main__":
    import numpy as np
    phi0 = np.random.normal(0, 0.1, (32, 32))
    phi1 = phi0 * 0.9  # relaxed
    res = check_invariants(phi0, phi1, 0.33, 0.15, 0.0, (1.0, 1.0))
    print("Diagnostics:", res)
