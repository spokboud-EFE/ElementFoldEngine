# ElementFold · control.py
# ──────────────────────────────────────────────────────────────────────────────
# The Supervisor is a tiny feedback controller that keeps the engine “in tune.”
# It watches telemetry and gently nudges three knobs used by Fold–Gate–Norm:
#   • β (beta)   — exposure: how strongly the gate amplifies structure
#   • γ (gamma)  — damping : how hard the normalizer calms energy
#   • ⛔ (clamp) — gate cap: ceiling for negative gate excursions
#
# Design goals:
#   • Simple, readable, dependency‑free (pure stdlib).
#   • Forgiving telemetry: accepts ASCII ('kappa','p_half') and Unicode ('κ','p½').
#   • Gentle changes with EMA smoothing + hysteresis to avoid “hunting.”
#   • Rails aligned with RungController so blending stays sane.
#
# Contract:
#   sup = Supervisor()
#   ctrl = sup.update(telemetry)        # -> {"beta","gamma","clamp"}
#   sup.apply(model)                    # optional: calls model.apply_control(...)
#
from __future__ import annotations

from typing import Dict, Optional
import math


class Supervisor:
    """
    A small, stateful controller. Call .update(telemetry) each step to obtain a
    {beta, gamma, clamp} recommendation. Optionally call .apply(model) to push
    those settings into the model (expects model.apply_control(beta=?, gamma=?, clamp=?)).

    Tunables:
      • step: smallest nudge (default 0.05)
      • ema : exponential smoothing factor in [0,1) (closer to 1 → smoother)
      • bounds: safety rails; chosen to match RungController’s defaults
    """

    def __init__(
        self,
        beta: float = 1.0,                       # start with moderate exposure
        gamma: float = 0.5,                      # start with moderate damping
        clamp: float = 5.0,                      # allow moderate gating depth
        beta_bounds = (0.5, 3.0),                # match RungController rails
        gamma_bounds = (0.0, 1.5),
        clamp_bounds = (1.0, 12.0),
        step: float = 0.05,                      # base nudge size
        ema: float = 0.90,                       # telemetry smoothing
    ):
        # Control state (what we “publish”)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.clamp = float(clamp)

        # Safety rails
        self.beta_bounds = (float(beta_bounds[0]), float(beta_bounds[1]))
        self.gamma_bounds = (float(gamma_bounds[0]), float(gamma_bounds[1]))
        self.clamp_bounds = (float(clamp_bounds[0]), float(clamp_bounds[1]))

        # Tuning
        self.step = float(step)
        self.ema = float(ema)

        # Targets + thresholds (gentle heuristics)
        self._kappa_expose = 0.80   # if κ < 0.80 → increase β
        self._kappa_relax  = 0.95   # if κ > 0.95 and p½ small → relax β, loosen ⛔
        self._p_half_safe  = 0.05   # if p½ > 0.05 → increase γ, tighten ⛔
        self._p_half_tiny  = 0.01   # if p½ < 0.01 and κ high → relax β, loosen ⛔
        self._grad_high    = 1.50   # optional: treat big gradients as instability

        # Smoothed telemetry cache
        self._ema: Dict[str, float] = {}  # keys: 'kappa','p_half','grad_norm'

    # ————————————————————————————————————————————————
    # Public API
    # ————————————————————————————————————————————————

    def update(self, telemetry: Dict[str, float]) -> Dict[str, float]:
        """
        Update internal β/γ/⛔ recommendations using current telemetry.

        Telemetry keys we accept (all optional):
          • 'kappa' or 'κ'          — phase concentration in [0,1] (larger is better)
          • 'p_half' or 'p½'        — proximity to half‑step barrier in [0,1] (smaller is better)
          • 'grad_norm'             — gradient norm (larger can signal instability)
          • Optional for fallback: 'x_mean', 'delta' or 'δ⋆'
            (if p_half is missing, we derive it from residual vs. rung grid)

        Returns:
            {'beta': β, 'gamma': γ, 'clamp': ⛔}
        """
        # 0) Read tolerant telemetry (ASCII or Unicode) + derive fallbacks
        kappa_raw, p_half_raw, grad_raw = self._read_telemetry(telemetry)

        # 1) Smooth with EMA
        kappa = self._ema_update("kappa", kappa_raw)
        p_half = self._ema_update("p_half", p_half_raw)
        grad = self._ema_update("grad_norm", grad_raw)

        # 2) Replace NaN/Inf with safe defaults
        kappa = self._finite_or(kappa, 1.0)   # pretend “coherent” if broken
        p_half = self._finite_or(p_half, 0.0) # pretend “safe” if broken
        grad = self._finite_or(grad, 0.0)     # pretend “calm” if broken

        # 3) Control logic (gentle, with hysteresis)
        # — γ (damping): higher when near barriers, lower when safely away
        if p_half > self._p_half_safe:
            self.gamma += self.step
        else:
            self.gamma -= 0.5 * self.step

        # — β (exposure): increase if not well aligned; trim if very coherent and safe
        if kappa < self._kappa_expose:
            self.beta += 2.0 * self.step          # slightly stronger push toward movement
        elif kappa > self._kappa_relax and p_half < self._p_half_tiny:
            self.beta -= self.step

        # — ⛔ (clamp): tighten near barriers or on high gradients; loosen when very stable
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
        Push the current controls into a model if it supports .apply_control().
        This lets training loops do:
            ctrl = sup.update(tele); sup.apply(model)
        """
        if hasattr(model, "apply_control"):
            model.apply_control(beta=self.beta, gamma=self.gamma, clamp=self.clamp)

    def set_control(
        self,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        clamp: Optional[float] = None,
    ) -> None:
        """
        Manually override any of the three knobs (values are clipped to legal ranges).
        Useful when a UI or a higher‑level policy wants to take over temporarily.
        """
        if beta is not None:
            self.beta = self._clip(float(beta), *self.beta_bounds)
        if gamma is not None:
            self.gamma = self._clip(float(gamma), *self.gamma_bounds)
        if clamp is not None:
            self.clamp = self._clip(float(clamp), *self.clamp_bounds)

    def reset(self, beta: float = 1.0, gamma: float = 0.5, clamp: float = 5.0) -> None:
        """
        Reset controller state and clear EMA memory (fresh start).
        """
        self.beta = self._clip(float(beta), *self.beta_bounds)
        self.gamma = self._clip(float(gamma), *self.gamma_bounds)
        self.clamp = self._clip(float(clamp), *self.clamp_bounds)
        self._ema.clear()

    def state(self) -> Dict[str, float]:
        """
        Snapshot the current controller settings (handy for logs and dashboards).
        """
        return {"beta": self.beta, "gamma": self.gamma, "clamp": self.clamp}

    # ————————————————————————————————————————————————
    # Internals
    # ————————————————————————————————————————————————

    def _read_telemetry(self, tele: Dict[str, float]) -> tuple[float, float, float]:
        """
        Tolerant reader for (κ, p½, grad_norm), with fallbacks:
          • κ:  prefer 'kappa' else 'κ'; if missing and p½ present → 1−p½.
          • p½: prefer 'p_half' else 'p½'; if missing and x_mean, δ⋆ present,
                 estimate from residual: p½ ≈ min(1, |x − k·δ⋆| / (0.5·δ⋆)).
          • grad_norm: optional; default 0.0.
        """
        # Raw pulls (accept both ASCII and Unicode keys)
        kappa = tele.get("kappa", tele.get("κ", None))
        p_half = tele.get("p_half", tele.get("p½", None))
        grad = tele.get("grad_norm", 0.0)

        # If p_half missing, try to derive from x_mean and δ⋆
        if p_half is None:
            x = tele.get("x_mean", None)
            delta = tele.get("delta", tele.get("δ⋆", None))
            if (x is not None) and (delta is not None):
                try:
                    delta = float(delta)
                    if delta > 0.0:
                        # nearest rung index
                        k = int(math.floor((float(x) / delta) + 0.5))
                        r = float(x) - k * delta
                        p_half = min(1.0, abs(r) / (0.5 * delta))
                except Exception:
                    p_half = None

        # If κ missing but p½ known, approximate κ ≈ 1 − p½ (coherence proxy)
        if kappa is None and p_half is not None:
            try:
                kappa = 1.0 - float(p_half)
            except Exception:
                kappa = None

        # Safe defaults
        kappa = float(kappa) if kappa is not None else 1.0
        p_half = float(p_half) if p_half is not None else 0.0
        grad = float(grad) if math.isfinite(float(grad)) else 0.0

        return kappa, p_half, grad

    def _ema_update(self, key: str, value: float) -> float:
        """
        Exponential moving average: y_t = α·y_{t−1} + (1−α)·x_t
        """
        v = float(value)
        if key not in self._ema:
            self._ema[key] = v
        else:
            a = self.ema
            self._ema[key] = a * self._ema[key] + (1.0 - a) * v
        return self._ema[key]

    @staticmethod
    def _clip(v: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, v))

    @staticmethod
    def _finite_or(v: float, default: float) -> float:
        return v if math.isfinite(v) else float(default)
