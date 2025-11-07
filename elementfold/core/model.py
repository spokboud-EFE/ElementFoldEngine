# ElementFold · core/model.py
# ============================================================
# RelaxationModel — unified physical interface
# ------------------------------------------------------------
# Combines:
#   • Field evolution (Φ)
#   • Fold / redshift / brightness calculations
#   • Energy and variance telemetry
#
# Uses only NumPy (Torch optional for acceleration).
# ============================================================

from __future__ import annotations
import numpy as np
from typing import Callable, Optional, Dict, Any, Tuple

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .data import BackgroundParams, OpticalParams, FieldState, Path
from .control import step, step_torch, safe_dt
from .telemetry import summary as telemetry_summary, monotone_var_drop
from .fgn import folds, redshift_from_F, brightness_tilt, time_dilation, bend


# ============================================================
# RelaxationModel class
# ============================================================

class RelaxationModel:
    """
    Unified numerical model of the relaxation field.

    Components
    ----------
    background : BackgroundParams
    optics     : OpticalParams

    Methods
    -------
    evolve(state, steps|dt)     – integrate PDE
    folds(path)                 – integrate cumulative folds
    redshift(path)              – compute 1+z = e^F − 1
    brightness(path, I_emit, d_geom)
    time_dilation(path)
    bend(path)
    telemetry(state)            – compute variance, grad², energy
    """

    def __init__(self,
                 background: BackgroundParams,
                 optics: Optional[OpticalParams] = None,
                 use_torch: bool = False) -> None:
        self.background = background
        self.optics = optics
        self.use_torch = bool(use_torch and TORCH_AVAILABLE)

    # ------------------------------------------------------------
    # PDE evolution
    # ------------------------------------------------------------
    def evolve(self,
               state: FieldState,
               steps: Optional[int] = None,
               dt: Optional[float] = None,
               hooks: Optional[Dict[str, Callable]] = None) -> FieldState:
        """
        Advance Φ for given number of steps (or until stability).
        Hooks may include 'on_step(state, metrics)'.
        """
        lam, D, phi_inf = self.background.lambda_, self.background.D, self.background.phi_inf
        spacing, bc = state.spacing, state.bc

        # Compute safe dt if none
        if dt is None:
            dt = safe_dt(lam, D, spacing)

        phi = state.phi.copy()
        t = state.t
        n_steps = int(steps or 1)
        on_step = None if not hooks else hooks.get("on_step")

        for _ in range(n_steps):
            if self.use_torch:
                phi_t = torch.as_tensor(phi, dtype=torch.float32)
                phi_next = step_torch(phi_t, lam, D, phi_inf, spacing, bc, dt).cpu().numpy()
            else:
                phi_next = step(phi, lam, D, phi_inf, spacing, bc, dt)
            # Check monotonic variance
            if not monotone_var_drop(phi, phi_next):
                dt *= 0.5  # shrink dt for safety
            phi = phi_next
            t += dt
            if on_step:
                metrics = self.telemetry(FieldState(phi, t, spacing, bc))
                on_step(FieldState(phi, t, spacing, bc), metrics)

        return FieldState(phi=phi, t=t, spacing=spacing, bc=bc)

    # ------------------------------------------------------------
    # Fold and observable wrappers
    # ------------------------------------------------------------
    def folds(self, path: Path) -> float:
        """Integrate ℱ along a path using current optics. Requires optics.eta."""
        if not self.optics:
            raise ValueError("Optical parameters required for folds().")
        return folds(path, self.optics.eta)

    def redshift(self, path: Path) -> float:
        """Compute redshift 1+z = e^ℱ - 1."""
        F = self.folds(path)
        return float(redshift_from_F(F))

    def brightness(self, path: Path, I_emit: float, d_geom: float) -> float:
        """Compute observed brightness including geometric and relaxation terms."""
        F = self.folds(path)
        return float(I_emit / (4.0 * np.pi * d_geom ** 2) * brightness_tilt(F))

    def time_dilation(self, path: Path) -> float:
        """Compute apparent time dilation along a path."""
        if not self.optics:
            raise ValueError("Optical parameters required.")
        F = self.folds(path)
        sigma_obs = self.optics.sigma(0.0)
        sigma_emit = self.optics.sigma(path[0].phi)
        return float(time_dilation(F, sigma_obs, sigma_emit))

    def bend(self, path: Path) -> float:
        """Compute chromatic bending via ∇⊥ ln n."""
        if not self.optics:
            raise ValueError("Optical parameters required.")
        return float(bend(path, self.optics.n))

    # ------------------------------------------------------------
    # Telemetry and diagnostics
    # ------------------------------------------------------------
    def telemetry(self, state: FieldState) -> Dict[str, float]:
        """Return variance, grad², energy summary for the field."""
        lam, D, phi_inf = self.background.lambda_, self.background.D, self.background.phi_inf
        return telemetry_summary(state.phi, lam, D, phi_inf, state.spacing, state.bc)

    # ------------------------------------------------------------
    # Convenience wrappers for simulation
    # ------------------------------------------------------------
    def simulate(self,
                 state: FieldState,
                 steps: int,
                 dt: Optional[float] = None,
                 hooks: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
        """Run simulation and return final state and telemetry."""
        final_state = self.evolve(state, steps=steps, dt=dt, hooks=hooks)
        metrics = self.telemetry(final_state)
        return {"state": final_state, "metrics": metrics}

    # ------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "background": self.background.as_dict(),
            "optics": self.optics.as_dict() if self.optics else None,
            "use_torch": self.use_torch,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RelaxationModel":
        bg = BackgroundParams(**d["background"])
        optics = None
        if d.get("optics"):
            optics = OpticalParams(**d["optics"])
        return cls(background=bg, optics=optics, use_torch=bool(d.get("use_torch", False)))
