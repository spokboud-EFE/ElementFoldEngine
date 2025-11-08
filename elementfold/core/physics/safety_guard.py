"""
core/physics/safety_guard.py — The Safety Guardian 🛡️

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The SafetyGuard watches every number that could hurt stability.
• It clamps, checks, and whispers warnings when parameters drift
  beyond physically meaningful ranges.
• It never interrupts the simulation — it simply restores balance.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from elementfold.core.physics.field import Field, BACKEND

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ====================================================================== #
# ⚙️ Parameter bounds
# ====================================================================== #
_DEFAULT_LIMITS = {
    "beta": (0.0, 5.0),        # damping  γ
    "gamma": (0.0, 5.0),       # damping  γ alt
    "lambda": (0.0, 1.0),      # relaxation λ
    "D": (0.0, 1.0),           # diffusion coefficient
    "kappa": (0.0, 2.0),       # coherence κ
    "rho": (0.0, 1.0),         # coupling strength ρ
    "phi": (-1e3, 1e3),        # potential Φ
}


# ====================================================================== #
# 🧭 SafetyGuard
# ====================================================================== #
@dataclass
class SafetyGuard:
    """
    Enforces physical parameter ranges and clamps fields when needed.
    It does not raise errors — it corrects softly and logs warnings.
    """

    limits: Dict[str, tuple[float, float]] = field(default_factory=lambda: dict(_DEFAULT_LIMITS))
    verbose: bool = True

    # ------------------------------------------------------------------ #
    # 🧮 Clamp scalar parameter
    # ------------------------------------------------------------------ #
    def clamp_param(self, name: str, value: float) -> float:
        """Clamp a single scalar parameter within allowed range."""
        if name not in self.limits:
            return value
        lo, hi = self.limits[name]
        clamped = min(max(value, lo), hi)
        if self.verbose and clamped != value:
            print(f"[guard] ⚠️ {name}={value:.3g} clamped to {clamped:.3g}")
        return clamped

    def clamp_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clamp all known parameters in a dict."""
        return {k: self.clamp_param(k, float(v)) if isinstance(v, (float, int)) else v
                for k, v in params.items()}

    # ------------------------------------------------------------------ #
    # 🌡️ Clamp Field data
    # ------------------------------------------------------------------ #
    def clamp_field(self, field: Field, name: Optional[str] = None) -> Field:
        """
        Clamp a Field’s numeric values to the allowed range.
        If no named range exists, uses ±1e6 as generic bound.
        """
        start_name = name or field.name
        bounds = self.limits.get(start_name, (-1e6, 1e6))
        lo, hi = bounds
        try:
            if field.backend == "torch" and _TORCH_AVAILABLE:
                field.data.clamp_(lo, hi)
            else:
                np.clip(field.data, lo, hi, out=field.data)
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[guard] field clamp failed ({exc}); fallback to numpy.")
            arr = np.asarray(field.data)
            np.clip(arr, lo, hi, out=arr)
            field.data = arr
            field.backend = "numpy"
        if self.verbose:
            fmin, fmax = float(np.min(np.asarray(field.data))), float(np.max(np.asarray(field.data)))
            print(f"[guard] 🧮 {start_name} clamped to range [{fmin:.3g}, {fmax:.3g}]")
        return field

    # ------------------------------------------------------------------ #
    # 🧘 Sanity checks
    # ------------------------------------------------------------------ #
    def check_stability(self, params: Dict[str, Any]) -> bool:
        """
        Quick stability test: λ*dt < 1 and D*dt < 0.5 are safe heuristics.
        Returns True if stable, False otherwise.
        """
        lam = float(params.get("lambda", 0.0))
        D = float(params.get("D", 0.0))
        dt = float(params.get("dt", 1e-3))
        stable = lam * dt < 1.0 and D * dt < 0.5
        if not stable and self.verbose:
            print(f"[guard] ⚠️ stability risk: λΔt={lam*dt:.3g}, DΔt={D*dt:.3g}")
        return stable

    # ------------------------------------------------------------------ #
    # 🗣️ Narrative summary
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """Describe current parameter limits in human terms."""
        lines = [f"{k}: {lo:.3g} ≤ value ≤ {hi:.3g}" for k, (lo, hi) in self.limits.items()]
        return "SafetyGuard limits:\n" + "\n".join(lines)
