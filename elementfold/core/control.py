# ElementFold · control.py
# ============================================================
# Supervisor — feedback controller that keeps Fold–Gate–Norm in tune.
#
# Controls (published):
#   β (beta)   — exposure: amplifies structure/flow
#   γ (gamma)  — damping : calms excess energy
#   ⛔ (clamp) — gate cap : limits negative-side depth
#
# Telemetry (inputs; ASCII or Unicode keys accepted):
#   'kappa' or 'κ'      — phase concentration in [0,1]
#   'p_half' or 'p½'    — proximity to half-click boundary
#   'grad_norm'         — gradient magnitude proxy
#   optional: 'x_mean' + 'delta'/'δ⋆' to derive p½ when missing
#
# Behavior:
#   • Small, EMA-smoothed incremental updates (no bangs or oscillations)
#   • Safety rails clip β,γ,⛔ into stable ranges
#   • Pure stdlib; deterministic and dependency-free
# ============================================================

from __future__ import annotations
import math
from typing import Dict, Optional, Mapping


class Supervisor:
    """
    Stateful feedback controller for the Fold–Gate–Norm engine.

    Usage
    -----
        sup = Supervisor()
        ctrl = sup.update(telemetry)     # -> {'beta','gamma','clamp'}
        sup.apply(model)                 # if model implements .apply_control(...)
    """

    def __init__(self,
                 beta: float = 1.0,
                 gamma: float = 0.5,
                 clamp: float = 5.0,
                 beta_bounds: tuple[float, float] = (0.5, 3.0),
                 gamma_bounds: tuple[float, float] = (0.0, 1.5),
                 clamp_bounds: tuple[float, float] = (1.0, 12.0),
                 step: float = 0.05,
                 ema: float = 0.90):
        # Published control values
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.clamp = float(clamp)

        # Safety rails
        self.beta_bounds = (float(beta_bounds[0]), float(beta_bounds[1]))
        self.gamma_bounds = (float(gamma_bounds[0]), float(gamma_bounds[1]))
        self.clamp_bounds = (float(clamp_bounds[0]), float(clamp_bounds[1]))

        # Update dynamics
        self.step = float(step)
        self.ema = float(ema)

        # Thresholds / hysteresis
        self._kappa_expose = 0.80   # κ below → raise β
        self._kappa_relax  = 0.95   # κ above (and safe) → lower β, loosen ⛔
        self._p_half_safe  = 0.05   # near boundary → raise γ, tighten ⛔
        self._p_half_tiny  = 0.01   # far from boundary → relax
        self._grad_high    = 1.50   # large gradient → tighten ⛔

        # EMA state
        self._ema: Dict[str, float] = {}

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def update(self, telemetry: Mapping[str, float]) -> Dict[str, float]:
        """
        Read telemetry, update β, γ, and ⛔ with gentle, bounded steps.

        Returns:
            {'beta': β, 'gamma': γ, 'clamp': ⛔}  (all floats clipped to rails)
        """
        # 0) Parse tolerant telemetry; derive fallbacks when missing
        kappa_raw, p_half_raw, grad_raw = self._read_telemetry(telemetry)

        # 1) EMA smoothing
        kappa = self._ema_update("kappa", kappa_raw)
        p_half = self._ema_update("p_half", p_half_raw)
        grad = self._ema_update("grad_norm", grad_raw)

        # 2) Replace non-finite with safe defaults
        kappa = self._finite_or(kappa, 1.0)
        p_half = self._finite_or(p_half, 0.0)
        grad = self._finite_or(grad, 0.0)

        # 3) Control logic (small proportional nudges)
        # γ: damping up when near barrier; down when clearly safe
        if p_half > self._p_half_safe:
            self.gamma += self.step
        else:
            self.gamma -= 0.5 * self.step

        # β: increase if coherence is weak; gently decrease if very coherent & safe
        if kappa < self._kappa_expose:
            self.beta += 2.0 * self.step
        elif kappa > self._kappa_relax and p_half < self._p_half_tiny:
            self.beta -= self.step

        # ⛔: tighten when near barrier or gradients high; loosen if very coherent & safe
        if p_half > self._p_half_safe or grad > self._grad_high:
            self.clamp -= 4.0 * self.step
        elif kappa > self._kappa_relax and p_half < self._p_half_tiny:
            self.clamp += 2.0 * self.step

        # 4) Enforce rails
        self.beta = self._clip(self.beta, *self.beta_bounds)
        self.gamma = self._clip(self.gamma, *self.gamma_bounds)
        self.clamp = self._clip(self.clamp, *self.clamp_bounds)

        return {"beta": self.beta, "gamma": self.gamma, "clamp": self.clamp}

    def apply(self, model) -> None:
        """
        Push current controls into a model that implements `.apply_control(beta, gamma, clamp)`.
        Safe no-op if the method is absent.
        """
        if hasattr(model, "apply_control"):
            model.apply_control(beta=self.beta, gamma=self.gamma, clamp=self.clamp)

    def set_control(self,
                    beta: Optional[float] = None,
                    gamma: Optional[float] = None,
                    clamp: Optional[float] = None) -> None:
        """Manually override any of β, γ, ⛔ (values are clipped to rails)."""
        if beta is not None:
            self.beta = self._clip(float(beta), *self.beta_bounds)
        if gamma is not None:
            self.gamma = self._clip(float(gamma), *self.gamma_bounds)
        if clamp is not None:
            self.clamp = self._clip(float(clamp), *self.clamp_bounds)

    def reset(self,
              beta: float = 1.0,
              gamma: float = 0.5,
              clamp: float = 5.0) -> None:
        """Reset controls to defaults and clear EMA memory."""
        self.beta = self._clip(float(beta), *self.beta_bounds)
        self.gamma = self._clip(float(gamma), *self.gamma_bounds)
        self.clamp = self._clip(float(clamp), *self.clamp_bounds)
        self._ema.clear()

    def state(self) -> Dict[str, float]:
        """Snapshot of current controls (for logs/dashboards)."""
        return {"beta": self.beta, "gamma": self.gamma, "clamp": self.clamp}

    # ──────────────────────────────────────────────────────────────────────
    # INTERNALS
    # ──────────────────────────────────────────────────────────────────────

    def _read_telemetry(self, tele: Mapping[str, float]) -> tuple[float, float, float]:
        """
        Parse tolerant telemetry and derive (κ, p½, grad_norm).

        Accepts:
            κ: tele['kappa'] or tele['κ']  (fallback from p½ if needed)
            p½: tele['p_half'] or tele['p½']  (fallback from x_mean & δ⋆)
            grad_norm: tele['grad_norm']  (default 0.0 if missing)
            optional: 'x_mean' + 'delta'/'δ⋆' to derive p½
        """
        # Primary reads (tolerant keys)
        kappa = tele.get("kappa", tele.get("κ"))
        p_half = tele.get("p_half", tele.get("p½"))
        grad = tele.get("grad_norm", 0.0)

        # Derive p½ from x_mean and δ⋆ if missing
        if p_half is None:
            x = tele.get("x_mean")
            delta = tele.get("delta", tele.get("δ⋆"))
            if (x is not None) and (delta is not None):
                try:
                    d = float(delta)
                    if d > 0.0 and math.isfinite(d):
                        k = math.floor((float(x) / d) + 0.5)
                        r = float(x) - k * d
                        # Normalize to proportion of half-click (≥0)
                        p_half = min(1.0, abs(r) / (0.5 * d))
                except Exception:
                    p_half = None

        # Derive κ from p½ if missing (simple complement)
        if kappa is None and p_half is not None:
            try:
                kappa = 1.0 - float(p_half)
            except Exception:
                kappa = None

        # Safe defaults
        kappa = float(kappa) if (kappa is not None and math.isfinite(float(kappa))) else 1.0
        p_half = float(p_half) if (p_half is not None and math.isfinite(float(p_half))) else 0.0
        grad = float(grad) if math.isfinite(float(grad)) else 0.0

        return kappa, p_half, grad

    def _ema_update(self, key: str, value: float) -> float:
        """Exponential moving average: y_t = α·y_{t−1} + (1−α)·x_t."""
        v = float(value)
        if key not in self._ema:
            self._ema[key] = v
        else:
            a = self.ema
            self._ema[key] = a * self._ema[key] + (1.0 - a) * v
        return self._ema[key]

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        """Clamp v into [lo, hi]."""
        return max(lo, min(hi, v))

    @staticmethod
    def _finite_or(v: float, default: float) -> float:
        """Return v if finite; otherwise default."""
        try:
            return v if math.isfinite(float(v)) else float(default)
        except Exception:
            return float(default)
