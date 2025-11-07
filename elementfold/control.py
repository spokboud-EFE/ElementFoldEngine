# ElementFold · control.py
# ============================================================
# Supervisor — feedback controller that keeps Fold–Gate–Norm in tune.
#
# Plain words:
#   The engine behaves like a resonant system with three live knobs:
#       β (beta)   – exposure: amplifies structure / flow
#       γ (gamma)  – damping : calms excess energy
#       ⛔ (clamp) – gate cap: limits how deep the gate can go
#
# The Supervisor reads telemetry values such as:
#       κ (kappa)  – coherence / phase alignment in [0,1]
#       p½ (p_half) – proximity to mid-step barrier (safety margin)
#       grad_norm   – gradient magnitude (stability proxy)
#
# Based on these it makes small, smoothed adjustments to β, γ, and ⛔.
# There are no oscillations or sharp switches: updates are gentle,
# EMA-smoothed, and bounded by rails.
# ============================================================

from __future__ import annotations
import math
from typing import Dict, Optional


class Supervisor:
    """
    Stateful controller for the Fold–Gate–Norm engine.

    Usage:
    -------
        sup = Supervisor()
        ctrl = sup.update(telemetry)    # -> {"beta","gamma","clamp"}
        sup.apply(model)                # optional: push into model.apply_control()

    Tuning goals:
    -------------
    • Simple, dependency-free logic (stdlib only).
    • Robust telemetry parsing (accepts ASCII and Unicode keys).
    • EMA smoothing + hysteresis to avoid hunting or noise amplification.
    • Safety rails aligned with RungController defaults.
    """

    def __init__(self,
                 beta: float = 1.0,
                 gamma: float = 0.5,
                 clamp: float = 5.0,
                 beta_bounds=(0.5, 3.0),
                 gamma_bounds=(0.0, 1.5),
                 clamp_bounds=(1.0, 12.0),
                 step: float = 0.05,
                 ema: float = 0.90):
        # Published control values
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.clamp = float(clamp)

        # Safety rails to prevent runaway parameters
        self.beta_bounds = (float(beta_bounds[0]), float(beta_bounds[1]))
        self.gamma_bounds = (float(gamma_bounds[0]), float(gamma_bounds[1]))
        self.clamp_bounds = (float(clamp_bounds[0]), float(clamp_bounds[1]))

        # Step size for adjustments and EMA factor for smoothing
        self.step = float(step)
        self.ema = float(ema)

        # Behavioral thresholds
        self._kappa_expose = 0.80   # if κ < 0.8 → system losing coherence → boost β
        self._kappa_relax = 0.95    # if κ > 0.95 and safe → reduce β, loosen clamp
        self._p_half_safe = 0.05    # if p½ > 0.05 → near barrier → increase γ
        self._p_half_tiny = 0.01    # if p½ < 0.01 → stable → relax β, loosen clamp
        self._grad_high = 1.50      # large gradient → treat as instability

        # Smoothed telemetry cache: holds EMA state
        self._ema: Dict[str, float] = {}

    # ============================================================
    # PUBLIC API
    # ============================================================

    def update(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """
        Read telemetry and update β, γ, ⛔ accordingly.

        Expected telemetry keys (any subset, ASCII or Unicode):
            'kappa' or 'κ'     : phase concentration in [0,1]
            'p_half' or 'p½'   : proximity to half-step (barrier risk)
            'grad_norm'        : gradient norm
            optionally 'x_mean' + 'delta'/'δ⋆' to derive p½ if missing

        Returns:
            {'beta': β, 'gamma': γ, 'clamp': ⛔}
        """
        # --------------------------------------------------------
        # 0) Read tolerant telemetry and compute fallbacks
        # --------------------------------------------------------
        kappa_raw, p_half_raw, grad_raw = self._read_telemetry(telemetry)

        # --------------------------------------------------------
        # 1) Apply EMA smoothing
        # --------------------------------------------------------
        kappa = self._ema_update("kappa", kappa_raw)
        p_half = self._ema_update("p_half", p_half_raw)
        grad = self._ema_update("grad_norm", grad_raw)

        # --------------------------------------------------------
        # 2) Replace NaN or Inf with safe defaults
        # --------------------------------------------------------
        kappa = self._finite_or(kappa, 1.0)   # assume coherent if bad
        p_half = self._finite_or(p_half, 0.0) # assume safe if bad
        grad = self._finite_or(grad, 0.0)     # assume calm if bad

        # --------------------------------------------------------
        # 3) Control logic: gentle proportional adjustments
        # --------------------------------------------------------

        # γ (damping): increase if near barrier, decrease if safe.
        if p_half > self._p_half_safe:
            self.gamma += self.step
        else:
            self.gamma -= 0.5 * self.step

        # β (exposure): raise if system under-coherent (κ low);
        # lower slightly if strongly coherent and safe.
        if kappa < self._kappa_expose:
            self.beta += 2.0 * self.step
        elif kappa > self._kappa_relax and p_half < self._p_half_tiny:
            self.beta -= self.step

        # ⛔ (clamp): tighten if near barrier or gradients large;
        # loosen if very coherent and far from barriers.
        if p_half > self._p_half_safe or grad > self._grad_high:
            self.clamp -= 4.0 * self.step
        elif kappa > self._kappa_relax and p_half < self._p_half_tiny:
            self.clamp += 2.0 * self.step

        # --------------------------------------------------------
        # 4) Enforce rails to stay within safe ranges
        # --------------------------------------------------------
        self.beta = self._clip(self.beta, *self.beta_bounds)
        self.gamma = self._clip(self.gamma, *self.gamma_bounds)
        self.clamp = self._clip(self.clamp, *self.clamp_bounds)

        # Return current control state
        return {"beta": self.beta, "gamma": self.gamma, "clamp": self.clamp}

    def apply(self, model) -> None:
        """
        Push the current controls into a model that implements `.apply_control()`.
        This allows seamless integration inside training loops:
            ctrl = sup.update(tele)
            sup.apply(model)
        """
        if hasattr(model, "apply_control"):
            model.apply_control(beta=self.beta, gamma=self.gamma, clamp=self.clamp)

    def set_control(self,
                    beta: Optional[float] = None,
                    gamma: Optional[float] = None,
                    clamp: Optional[float] = None) -> None:
        """
        Manually override any of the three controls.
        Values are clipped to safety rails.
        """
        if beta is not None:
            self.beta = self._clip(float(beta), *self.beta_bounds)
        if gamma is not None:
            self.gamma = self._clip(float(gamma), *self.gamma_bounds)
        if clamp is not None:
            self.clamp = self._clip(float(clamp), *self.clamp_bounds)

    def reset(self, beta: float = 1.0,
              gamma: float = 0.5,
              clamp: float = 5.0) -> None:
        """
        Reset controller to default values and clear EMA memory.
        """
        self.beta = self._clip(float(beta), *self.beta_bounds)
        self.gamma = self._clip(float(gamma), *self.gamma_bounds)
        self.clamp = self._clip(float(clamp), *self.clamp_bounds)
        self._ema.clear()

    def state(self) -> Dict[str, float]:
        """Return a snapshot of current controls (for logs or dashboards)."""
        return {"beta": self.beta, "gamma": self.gamma, "clamp": self.clamp}

    # ============================================================
    # INTERNAL HELPERS
    # ============================================================

    def _read_telemetry(self, tele: Dict[str, float]) -> tuple[float, float, float]:
        """
        Parse telemetry dict and derive (κ, p½, grad_norm) safely.
        Supports both ASCII ('kappa','p_half') and Unicode ('κ','p½') keys.
        Derives missing values when possible.
        """
        kappa = tele.get("kappa", tele.get("κ", None))
        p_half = tele.get("p_half", tele.get("p½", None))
        grad = tele.get("grad_norm", 0.0)

        # Derive p_half from x_mean and δ★ if missing
        if p_half is None:
            x = tele.get("x_mean", None)
            delta = tele.get("delta", tele.get("δ⋆", None))
            if (x is not None) and (delta is not None):
                try:
                    delta = float(delta)
                    if delta > 0.0:
                        k = int(math.floor((float(x) / delta) + 0.5))
                        r = float(x) - k * delta
                        p_half = min(1.0, abs(r) / (0.5 * delta))
                except Exception:
                    p_half = None

        # Derive κ from p½ if missing: coherence ≈ 1 − p½
        if kappa is None and p_half is not None:
            try:
                kappa = 1.0 - float(p_half)
            except Exception:
                kappa = None

        # Safe defaults if still missing
        kappa = float(kappa) if kappa is not None else 1.0
        p_half = float(p_half) if p_half is not None else 0.0
        grad = float(grad) if math.isfinite(float(grad)) else 0.0

        return kappa, p_half, grad

    def _ema_update(self, key: str, value: float) -> float:
        """Exponential moving average: y_t = α·y_{t−1} + (1−α)·x_t"""
        v = float(value)
        if key not in self._ema:
            self._ema[key] = v
        else:
            a = self.ema
            self._ema[key] = a * self._ema[key] + (1.0 - a) * v
        return self._ema[key]

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        """Clamp value into [lo, hi]."""
        return max(lo, min(hi, v))

    @staticmethod
    def _finite_or(v: float, default: float) -> float:
        """Return v if finite, otherwise a safe default."""
        return v if math.isfinite(v) else float(default)
