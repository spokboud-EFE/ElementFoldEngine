# ElementFold · control.py
# The Supervisor is a tiny feedback controller that keeps the engine “in tune.”
# It watches telemetry (κ for phase alignment, p½ for boundary hits, ‖∇‖ for instability)
# and gently nudges three knobs used by Fold–Gate–Norm blocks:
#   • β (beta):    exposure — how strongly the gate amplifies structure,
#   • γ (gamma):   damping  — how hard the normalizer calms energy,
#   • ⛔ (clamp):  gate cap — how far negative gate values can go before we cut them off.

from __future__ import annotations  # Allow forward type hints in older Python
from typing import Dict, Optional   # Type hints for readability
import math                        # For finite checks and simple numbers


class Supervisor:
    """
    A small, stateful controller. You call .update(telemetry) each step to
    get a new {beta, gamma, clamp} recommendation. Optionally call .apply(model)
    to push those settings into the model’s FGN blocks (via .apply_control).

    Design choices:
      • Boundaries: prevent runaway (β, γ, clamp) with simple min/max limits.
      • Gentle steps: small increments (step≈0.05) reduce oscillation (“hunting”).
      • EMA telemetry: exponential moving average smooths noisy signals.
      • Hysteresis: only relax when the system is safely coherent to avoid flapping.
    """

    def __init__(
        self,
        beta: float = 1.0,                       # Start with moderate exposure.
        gamma: float = 0.5,                      # Start with moderate damping.
        clamp: float = 5.0,                      # Allow moderate gating depth.
        beta_bounds = (0.5, 2.0),                # Keep β within physical sense.
        gamma_bounds = (0.0, 0.9),               # Keep γ ∈ [0,1); 1.0 can over‑damp.
        clamp_bounds = (1.0, 10.0),              # Clamp cannot be negative; 1..10 is practical.
        step: float = 0.05,                      # Base step for nudges (small = smoother).
        ema: float = 0.9,                        # EMA smoothing for telemetry (closer to 1 → smoother).
    ):
        # — store control state (what we will broadcast) —
        self.beta = float(beta)                  # β exposure
        self.gamma = float(gamma)                # γ damping
        self.clamp = float(clamp)                # ⛔ gate clamp

        # — remember allowed ranges so we never drift into nonsense —
        self.beta_bounds = (float(beta_bounds[0]), float(beta_bounds[1]))     # [β_min, β_max]
        self.gamma_bounds = (float(gamma_bounds[0]), float(gamma_bounds[1]))  # [γ_min, γ_max]
        self.clamp_bounds = (float(clamp_bounds[0]), float(clamp_bounds[1]))  # [c_min, c_max]

        # — controller tuning parameters —
        self.step = float(step)                  # Smallest adjustment unit.
        self.ema = float(ema)                    # EMA factor for telemetry smoothing.

        # — smoothed telemetry cache (initialized empty; filled on first update) —
        self._ema: Dict[str, float] = {}         # Keys: 'kappa', 'p_half', 'grad_norm'

    # ————————————————————————————————————————————————
    # Public API
    # ————————————————————————————————————————————————

    def update(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """
        Update internal β/γ/⛔ recommendations using current telemetry.

        Telemetry keys we read (all optional; default to safe values):
          • 'kappa'     : phase concentration in [0,1], larger is better.
          • 'p_half'    : fraction touching half‑click boundary, smaller is better.
          • 'grad_norm' : gradient norm, larger can signal instability.

        Returns a dict you can log or feed to .apply():
            {'beta': β, 'gamma': γ, 'clamp': ⛔}
        """
        # — read and smooth incoming telemetry —
        kappa = self._ema_update("kappa", float(telemetry.get("kappa", 1.0)))         # default to “good” if absent
        p_half = self._ema_update("p_half", float(telemetry.get("p_half", 0.0)))      # default to “safe”
        grad = self._ema_update("grad_norm", float(telemetry.get("grad_norm", 0.0)))  # default to “calm”

        # — guard against NaNs/inf: fall back to last known good values —
        kappa = self._finite_or(kappa, 1.0)   # if broken, pretend we are coherent
        p_half = self._finite_or(p_half, 0.0) # if broken, pretend we are safe
        grad = self._finite_or(grad, 0.0)     # if broken, pretend gradients are calm

        # — control logic: gently push toward “coherent and safe” regime —
        # 1) Adjust γ (damping): increase when boundary hits are common; relax when safely locked.
        if p_half > 0.05:                               # too many boundary touches → more damping
            self.gamma += self.step
        else:                                           # otherwise, relax slightly
            self.gamma -= 0.5 * self.step

        # 2) Adjust β (exposure): increase when phases are not aligned; decrease when very coherent and safe.
        if kappa < 0.80:                                # low alignment → expose more
            self.beta += 2.0 * self.step               # a bit stronger push (0.10 with step=0.05)
        elif kappa > 0.95 and p_half < 0.01:           # locked and safe → calm exposure
            self.beta -= self.step

        # 3) Adjust clamp (gate cap): tighten when unstable; loosen when very stable.
        if p_half > 0.05 or grad > 1.5:                # unsafe edges or big gradients
            self.clamp -= 4.0 * self.step             # shrink fast (e.g., −0.20)
        elif kappa > 0.95 and p_half < 0.01:           # very stable
            self.clamp += 2.0 * self.step             # expand slowly (+0.10)

        # — enforce bounds so controls remain meaningful —
        self.beta = self._clip(self.beta, *self.beta_bounds)     # clamp β into its range
        self.gamma = self._clip(self.gamma, *self.gamma_bounds)  # clamp γ into [γ_min, γ_max]
        self.clamp = self._clip(self.clamp, *self.clamp_bounds)  # clamp ⛔ into [c_min, c_max]

        # — report the current controller recommendation —
        return {"beta": self.beta, "gamma": self.gamma, "clamp": self.clamp}

    def apply(self, model) -> None:
        """
        Push the current controls into a model if it supports .apply_control().
        This is a convenience so training loops can do:
            sup.update(tele); sup.apply(model)
        """
        # Many ElementFold models implement .apply_control(beta=?, gamma=?, clamp=?).
        if hasattr(model, "apply_control"):
            model.apply_control(beta=self.beta, gamma=self.gamma, clamp=self.clamp)

    def set_control(self, beta: Optional[float] = None, gamma: Optional[float] = None, clamp: Optional[float] = None) -> None:
        """
        Manually override any of the three knobs. Values are clipped to legal ranges.
        Useful when a steering layer or UI wants to take over.
        """
        if beta is not None:
            self.beta = self._clip(float(beta), *self.beta_bounds)
        if gamma is not None:
            self.gamma = self._clip(float(gamma), *self.gamma_bounds)
        if clamp is not None:
            self.clamp = self._clip(float(clamp), *self.clamp_bounds)

    def state(self) -> Dict[str, float]:
        """
        Snapshot the current controller settings (handy for logs and dashboards).
        """
        return {"beta": self.beta, "gamma": self.gamma, "clamp": self.clamp}

    # ————————————————————————————————————————————————
    # Internals (small helpers)
    # ————————————————————————————————————————————————

    def _ema_update(self, key: str, value: float) -> float:
        """
        Exponential moving average: smooths noisy telemetry.
        y_t = α·y_{t−1} + (1−α)·x_t
        """
        if key not in self._ema:
            self._ema[key] = float(value)                # First value initializes the EMA.
        else:
            a = self.ema                                 # α close to 1 favors history (smoother).
            self._ema[key] = a * self._ema[key] + (1 - a) * float(value)
        return self._ema[key]

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        """
        Keep v within [lo, hi].
        """
        return max(lo, min(hi, v))

    @staticmethod
    def _finite_or(v: float, default: float) -> float:
        """
        Replace NaN/Inf with a safe default.
        """
        return v if math.isfinite(v) else float(default)
