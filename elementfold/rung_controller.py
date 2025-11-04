# elementfold/rung_controller.py
# ──────────────────────────────────────────────────────────────────────────────
# RungController — coherence‑aware control shim for ElementFold
#   • Exports RungIntent enum used by train.py loss shaping
#   • Blends tiny overrides (β, γ, ⛔ clamp) on top of Supervisor output
#   • Optional SEEK mode toward a target rung k_target using a safe FSM
#
# This file is intentionally lean (no heavy deps; pure stdlib).
# It replaces older variants that exposed different RungIntent shapes or ctor args.
#
# MIT‑style tiny utility; zero dependencies outside standard library.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Literal, Union

ModePhase = Literal["LOCKED", "MID", "CROSSING", "CAPTURE"]


# ──────────────────────────────────────────────────────────────────────────────
# Public: RungIntent used both by the controller *and* by train.py for loss shaping
# ──────────────────────────────────────────────────────────────────────────────

class RungIntent(str, Enum):
    """High‑level policy around rungs used by training and control."""
    STABILIZE = "stabilize"  # keep within acceptance band; default
    HOLD      = "hold"       # actively center and damp on the nearest rung
    SEEK      = "seek"       # walk across barriers toward k_target (if provided)


def _as_intent(x: Union[str, "RungIntent", None]) -> RungIntent:
    if isinstance(x, RungIntent):
        return x
    if isinstance(x, str):
        x = x.lower().strip()
        for v in RungIntent:
            if v.value == x:
                return v
    return RungIntent.STABILIZE


# ──────────────────────────────────────────────────────────────────────────────
# Tuning
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RungTuning:
    """
    Gentle defaults that behave well on CPU/GPU smoke tests.
    All values dimensionless. Clamp (⛔) is a magnitude cap on auxiliary updates.
    """
    # Acceptance band half‑width; default ≈ δ⋆/6
    epsilon_scale: float = 1 / 6

    # State machine thresholds (fallbacks if tele['p_half'] unavailable)
    p_half_target: float = 0.55     # push above 0.5 → likely over the ridge
    p_half_lock: float = 0.12       # “locked” once well below this

    # Dwell (steps) for decisions; short for training loops, lengthen in production
    dwell_mid: int = 3              # hold mid‑step band this many ticks
    dwell_lock: int = 3             # hold lock band this many ticks

    # Override magnitudes (targets blended with Supervisor’s outputs)
    beta_boost: float = 2.0         # raise β to encourage movement
    beta_hold: float = 1.2          # gentle β when holding
    gamma_damp_lo: float = 0.05     # low damping while crossing
    gamma_damp_hi: float = 0.60     # more damping for lock/capture
    clamp_safe: float = 5.0         # ⛔ limit used for capture and safety

    # Blend (0..1): 0=keep Supervisor, 1=use target; 0.25–0.6 is gentle
    blend_cross: float = 0.50
    blend_hold: float = 0.35
    blend_capture: float = 0.45

    # Hard rails so we never send wild values
    beta_min: float = 0.5
    beta_max: float = 3.0
    gamma_min: float = 0.0
    gamma_max: float = 1.5
    clamp_min: float = 1.0
    clamp_max: float = 12.0


@dataclass
class _RungState:
    phase: ModePhase = "LOCKED"
    dwell: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────────────

class RungController:
    """
    Non‑intrusive rung control that blends tiny overrides onto the Supervisor.

    Typical use in a training loop:
        ctrl_sup = sup.update(tele)                # Supervisor (baseline policy)
        ctrl_out = rung.update(tele, ctrl_sup)     # small rung‑aware nudge
        model.apply_control(**ctrl_out)

    Arguments
    ---------
    delta: float
        δ⋆ — the nominal rung spacing in the hidden log variable X.
    k_target: Optional[int]
        Target rung index for SEEK/HOLD strategies; ignored for STABILIZE.
    band: Optional[float]
        Acceptance half‑band in X. If None, uses epsilon_scale * δ⋆.
    intent: RungIntent | str
        High‑level policy ("stabilize" | "hold" | "seek"). Default: STABILIZE.
    tuning: Optional[RungTuning]
        Tunable safe defaults for blending and rails.
    """

    def __init__(self,
                 delta: float,
                 k_target: Optional[int] = None,
                 band: Optional[float] = None,
                 intent: Union[RungIntent, str, None] = None,
                 tuning: Optional[RungTuning] = None):
        self.delta = float(delta)
        self.k_target = k_target
        self.band = band  # absolute band in X; if None, we derive from epsilon_scale
        self.intent = _as_intent(intent)
        self.tuning = tuning or RungTuning()
        self.state = _RungState()

    # ——— public API ————————————————————————————————————————————————

    def set_intent(self,
                   intent: Union[RungIntent, str],
                   k_target: Optional[int] = None) -> None:
        self.intent = _as_intent(intent)
        if k_target is not None:
            self.k_target = int(k_target)
        self.state = _RungState(phase="LOCKED", dwell=0)

    def clear(self) -> None:
        """Return to STABILIZE; state machine resets."""
        self.set_intent(RungIntent.STABILIZE, k_target=None)

    def status(self) -> Dict[str, object]:
        """Tiny JSON‑like snapshot (nice to print in Studio)."""
        return {
            "intent": self.intent.value,
            "target_k": self.k_target,
            "phase": self.state.phase,
            "dwell": self.state.dwell,
            "delta": self.delta,
            "band": self._epsilon()
        }

    # ——— main hook ————————————————————————————————————————————————

    def update(self,
               tele: Dict[str, float],
               ctrl_from_supervisor: Optional[Dict[str, float]] = None
               ) -> Dict[str, float]:
        """
        Called once per training/inference step.

        tele — best‑effort keys used:
            'delta'|'δ⋆', 'kappa'|'κ', 'p_half'|'p½', 'x_mean'
        ctrl_from_supervisor — existing {beta,gamma,clamp} suggestion;
            we blend toward our targets; if None, we start from {1.0,0.5,5.0}.
        """
        # 0) Safe defaults
        ctrl = dict(ctrl_from_supervisor or {"beta": 1.0, "gamma": 0.5, "clamp": 5.0})

        # 1) Read telemetry (best‑effort, tolerant to missing keys)
        delta = float(tele.get("delta", tele.get("δ⋆", self.delta)))
        if delta != self.delta:
            self.delta = delta  # follow runtime’s δ⋆ if it changes

        kappa = float(tele.get("kappa", tele.get("κ", 0.0)))
        p_half = tele.get("p_half", tele.get("p½", None))
        x_mean = tele.get("x_mean", None)  # optional but helpful

        # Infer k index & residual when possible
        k_now, r = None, None
        if x_mean is not None:
            k_now = self._nearest_k(x_mean)
            r = x_mean - k_now * self.delta

        # 2) Policy
        if self.intent == RungIntent.STABILIZE:
            # Keep within acceptance band; increase damping near barriers.
            target = self._target_stabilize(p_half=p_half)
            return self._blend(ctrl, target, self.tuning.blend_hold)

        if self.intent == RungIntent.HOLD:
            # Gently center on the nearest rung and keep it quiet.
            target = self._target_hold(p_half=p_half)
            return self._blend(ctrl, target, self.tuning.blend_hold)

        if self.intent == RungIntent.SEEK:
            # Walk toward a target rung if provided; otherwise behave like HOLD.
            if self.k_target is None or k_now is None:
                target = self._target_hold(p_half=p_half)
                return self._blend(ctrl, target, self.tuning.blend_hold)

            dirn = 0
            if self.k_target > k_now:
                dirn = +1
            elif self.k_target < k_now:
                dirn = -1

            target, completed = self._target_step(direction=dirn, p_half=p_half, r=r, kappa=kappa)
            out = self._blend(ctrl, target, target.pop("_blend", self.tuning.blend_cross))

            # If we've arrived, automatically switch to HOLD
            if dirn == 0 and completed:
                self.intent = RungIntent.HOLD
                self.state = _RungState(phase="LOCKED", dwell=0)
            return out

        # Fallback
        return ctrl

    # ——— internal helpers ———————————————————————————————————————————

    def _epsilon(self) -> float:
        if self.band is not None:
            return float(self.band)
        return self.tuning.epsilon_scale * self.delta

    def _nearest_k(self, x_mean: float) -> int:
        from math import floor
        return int(floor((x_mean / self.delta) + 0.5))

    def _clip(self, beta: float, gamma: float, clamp: float) -> Dict[str, float]:
        T = self.tuning
        b = min(max(beta, T.beta_min), T.beta_max)
        g = min(max(gamma, T.gamma_min), T.gamma_max)
        c = min(max(clamp, T.clamp_min), T.clamp_max)
        return {"beta": b, "gamma": g, "clamp": c}

    def _blend(self, base: Dict[str, float], target: Dict[str, float], w: float) -> Dict[str, float]:
        """Linear blend toward a safe target (and clip)."""
        w = float(min(max(w, 0.0), 1.0))
        beta = (1 - w) * base.get("beta", 1.0) + w * target.get("beta", 1.2)
        gamma = (1 - w) * base.get("gamma", 0.5) + w * target.get("gamma", 0.5)
        clamp = (1 - w) * base.get("clamp", 5.0) + w * target.get("clamp", 5.0)
        return self._clip(beta, gamma, clamp)

    # ——— policies ————————————————————————————————————————————————

    def _target_stabilize(self, p_half: Optional[float]) -> Dict[str, float]:
        """
        Stabilize around rungs without forcing tight centering.
        Closer to a barrier (p½≈0.5) → keep damping higher; away → moderate.
        """
        T = self.tuning
        if p_half is None:
            gamma = 0.5 * (T.gamma_damp_lo + T.gamma_damp_hi)
        else:
            # Peak damping near barrier (p½~0.5), taper toward center
            # Map p_half∈[0,0.5]∪[0.5,1] to a “nearness to barrier” score
            d = abs(0.5 - float(p_half))
            nearness = 1.0 - min(max(d / 0.5, 0.0), 1.0)
            gamma = T.gamma_damp_lo + nearness * (T.gamma_damp_hi - T.gamma_damp_lo)
        return {"beta": T.beta_hold, "gamma": gamma, "clamp": T.clamp_safe}

    def _target_hold(self, p_half: Optional[float]) -> Dict[str, float]:
        """
        Gentle, sticky centering on the nearest rung:
          • slightly higher damping (γ) to absorb chatter,
          • modest β to keep responsiveness,
          • clamp at a safe value.
        """
        T = self.tuning
        gamma = T.gamma_damp_hi if (p_half is None or p_half < 0.25) else 0.5 * (T.gamma_damp_lo + T.gamma_damp_hi)
        return {"beta": T.beta_hold, "gamma": gamma, "clamp": T.clamp_safe}

    def _target_step(self,
                     direction: int,
                     p_half: Optional[float],
                     r: Optional[float],
                     kappa: float) -> tuple[Dict[str, float], bool]:
        """
        Cross a barrier in the chosen direction and then re‑lock.
        Returns: (target_control, completed: bool)
        """
        T = self.tuning

        # Decide via p_half if available; otherwise rely on r (residual) and κ (confidence)
        at_mid = (p_half is not None and p_half >= 0.48) or (r is not None and abs(r) >= 0.45 * self.delta)
        well_locked = (p_half is not None and p_half <= T.p_half_lock) or (kappa >= 0.20)

        # Phase machine
        if self.state.phase == "LOCKED":
            if direction == 0:
                # We are already at target; finished
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, True
            # Start leaning toward the barrier in the chosen direction
            self.state.phase = "MID" if at_mid else "LOCKED"
            return {"beta": T.beta_boost, "gamma": T.gamma_damp_lo, "clamp": T.clamp_safe, "_blend": T.blend_cross}, False

        if self.state.phase == "MID":
            self.state.dwell += 1
            # If we can hold mid a few ticks, we’re safe to attempt crossing
            if self.state.dwell >= T.dwell_mid:
                self.state.phase = "CROSSING"
                self.state.dwell = 0
            # Keep nimble while staying around the barrier
            return {"beta": T.beta_boost, "gamma": T.gamma_damp_lo, "clamp": T.clamp_safe, "_blend": T.blend_cross}, False

        if self.state.phase == "CROSSING":
            # Watch for a sign that we’ve gone over:
            crossed = False
            if p_half is not None:
                crossed = p_half >= T.p_half_target
            if (not crossed) and (r is not None):
                # If residual now sits closer to the next rung in the chosen direction
                sign = 1 if direction > 0 else -1
                crossed = (sign * r) > 0  # residual sign matches push direction

            if crossed:
                self.state.phase = "CAPTURE"
                self.state.dwell = 0
                # Switch to capture damping
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_capture}, False

            # Still crossing: stay nimble
            return {"beta": T.beta_boost, "gamma": T.gamma_damp_lo, "clamp": T.clamp_safe, "_blend": T.blend_cross}, False

        if self.state.phase == "CAPTURE":
            self.state.dwell += 1
            if self.state.dwell >= T.dwell_lock and well_locked:
                # Completed capture of the next rung
                self.state.phase = "LOCKED"
                self.state.dwell = 0
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, True
            # Keep damping up during capture
            return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_capture}, False

        # Fallback
        self.state.phase = "LOCKED"
        self.state.dwell = 0
        return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, False
