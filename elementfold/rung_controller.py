# elementfold/rung_controller.py
# ──────────────────────────────────────────────────────────────────────────────
# RungController — coherence‑aware control shim for ElementFold
#
# Goals
#   • Provide a tiny, safe controller that adds rung‑aware nudges on top of a
#     higher‑level Supervisor (β, γ, and ⛔ clamp), without hijacking policy.
#   • Export RungIntent used by training/loss shaping (STABILIZE / HOLD / SEEK).
#   • Offer a *compatible* API with optional “relative steps” for legacy code:
#       set_intent("step_up", steps=2)   # ← works
#       set_intent("step_down", steps=1) # ← works
#
# Key ideas (plain language, with a little math)
#   • ElementFold’s hidden coordinate X forms discrete “rungs” spaced by δ⋆.
#   • The controller watches telemetry (p½ ~ “barrier probability”, κ ~ “lock
#     confidence”, x_mean ~ “where X sits”), then blends tiny overrides:
#         β (responsiveness), γ (damping), and ⛔ (magnitude clamp).
#   • A gentle FSM guides barrier crossing:
#         LOCKED → MID (lean) → CROSSING (over the ridge) → CAPTURE (re‑lock).
#
# Implementation notes
#   • Pure stdlib; no heavy deps. Designed to be easy to read & test.
#   • Back‑compat: accepts step_up/step_down intents with a `steps=` kwarg.
#   • If x_mean is missing, decisions degrade gracefully using p½ and κ.
#
# MIT‑style tiny utility. © 2025 ElementFold authors.

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Literal, Union, Any

# Phase of the finite‑state machine while seeking
ModePhase = Literal["LOCKED", "MID", "CROSSING", "CAPTURE"]

# ──────────────────────────────────────────────────────────────────────────────
# Public: RungIntent (also referenced by training code / loss shaping)
# ──────────────────────────────────────────────────────────────────────────────

class RungIntent(str, Enum):
    """High‑level policy around rungs used by training and control."""
    STABILIZE = "stabilize"  # keep within acceptance band; default safe mode
    HOLD      = "hold"       # actively center & damp on the nearest rung
    SEEK      = "seek"       # walk across barriers toward a target rung (k_target)


# ──────────────────────────────────────────────────────────────────────────────
# Tuning (gentle defaults that behave well in smoke tests)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RungTuning:
    """
    All values are dimensionless. ⛔ = clamp on auxiliary updates.

    🧭 Intuition:
      • Higher γ near the barrier discourages “teetering”.
      • Slight β boost when crossing encourages decisive movement.
      • Blends are small (≤ 0.5), because Supervisor stays in charge.
    """
    # Acceptance band half‑width; default ≈ δ⋆/6
    epsilon_scale: float = 1 / 6

    # State machine thresholds (fallbacks if tele['p_half'] unavailable)
    p_half_target: float = 0.55     # above 0.5 → very likely “over the ridge”
    p_half_lock: float = 0.12       # well under this → “locked” on a rung

    # Dwell (ticks) for decisions; short for training, can be longer in prod
    dwell_mid: int = 3              # linger at MID a few ticks before crossing
    dwell_lock: int = 3             # linger at CAPTURE before declaring lock

    # Override magnitudes (targets we blend toward)
    beta_boost: float = 2.0         # nimble while crossing
    beta_hold: float = 1.2          # modest β when holding
    gamma_damp_lo: float = 0.05     # low damping while crossing
    gamma_damp_hi: float = 0.60     # higher damping when locking/capturing
    clamp_safe: float = 5.0         # safe ⛔ during guidance

    # Blend weights (0..1): 0=keep Supervisor, 1=use target
    blend_cross: float = 0.50
    blend_hold: float = 0.35
    blend_capture: float = 0.45

    # Hard rails: we never send wild values to the model
    beta_min: float = 0.5
    beta_max: float = 3.0
    gamma_min: float = 0.0
    gamma_max: float = 1.5
    clamp_min: float = 1.0
    clamp_max: float = 12.0


# Internal FSM state
@dataclass
class _RungState:
    phase: ModePhase = "LOCKED"
    dwell: int = 0


# Human‑readable snapshot (handy for Studio/CLI)
@dataclass
class RungSnapshot:
    intent: str
    k_target: Optional[int]
    phase: ModePhase
    dwell: int
    delta: float
    band: float
    plan: Optional[Dict[str, Any]]

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


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
        "stabilize" | "hold" | "seek". Default: STABILIZE.
    tuning: Optional[RungTuning]
        Tunable safe defaults for blending and rails.

    Compatibility
    -------------
    This controller understands legacy “relative step” requests via set_intent:
        set_intent("step_up", steps=2)     # seek up by +2 rungs
        set_intent("step_down", steps=1)   # seek down by −1 rung
    These are converted into a *pending plan* that is resolved on the next
    update() once the current rung k_now is observable.
    """

    # ——— construction ————————————————————————————————————————————————

    def __init__(self,
                 delta: float,
                 k_target: Optional[int] = None,
                 band: Optional[float] = None,
                 intent: Union[RungIntent, str, None] = None,
                 tuning: Optional[RungTuning] = None):
        self.delta = float(delta)
        self.k_target = k_target
        self.band = band  # absolute band in X; if None, derive from epsilon_scale
        self.intent = self._as_intent(intent)
        self.tuning = tuning or RungTuning()
        self.state = _RungState()
        # Pending relative step plan, e.g. {"dir": +1, "steps": 2, "armed": True}
        self._plan: Optional[Dict[str, Any]] = None

    # ——— public API ————————————————————————————————————————————————

    def set_intent(self,
                   intent: Union[RungIntent, str],
                   k_target: Optional[int] = None,
                   *,
                   steps: Optional[int] = None,
                   direction: Optional[int] = None) -> None:
        """
        Set high‑level intent. Back‑compatible with “step_up/down” + steps=N.

        Examples
        --------
        set_intent("hold")
        set_intent(RungIntent.SEEK, k_target=7)
        set_intent("step_up", steps=2)      # ← legacy CLI path (resonator)
        set_intent("step_down", steps=1)    # ← legacy CLI path (resonator)
        """
        # Normalize
        if isinstance(intent, str):
            key = intent.lower().strip().replace("-", "_").replace(" ", "_")
        else:
            key = intent.value

        # Handle legacy relative steps explicitly
        if key in {"step_up", "up", "seek_up"} or (key == "seek" and direction in {+1}):
            n = max(int(steps or 1), 1)
            self._arm_relative_plan(dirn=+1, steps=n)
            self.intent = RungIntent.SEEK
            # absolute k_target (if provided) overrides the relative plan
            if k_target is not None:
                self.k_target = int(k_target)
                self._plan = None
            self._reset_fsm()
            return

        if key in {"step_down", "down", "seek_down"} or (key == "seek" and direction in {-1}):
            n = max(int(steps or 1), 1)
            self._arm_relative_plan(dirn=-1, steps=n)
            self.intent = RungIntent.SEEK
            if k_target is not None:
                self.k_target = int(k_target)
                self._plan = None
            self._reset_fsm()
            return

        # Standard intents
        self.intent = self._as_intent(key)
        if k_target is not None:
            self.k_target = int(k_target)
        self._reset_fsm()

    def clear(self) -> None:
        """Return to STABILIZE; reset state and clear plans."""
        self.intent = RungIntent.STABILIZE
        self.k_target = None
        self._plan = None
        self._reset_fsm()

    def status(self) -> Dict[str, object]:
        """Tiny JSON‑like snapshot (handy to print in Studio)."""
        snap = RungSnapshot(
            intent=self.intent.value,
            k_target=self.k_target,
            phase=self.state.phase,
            dwell=self.state.dwell,
            delta=self.delta,
            band=self._epsilon(),
            plan=(dict(self._plan) if self._plan else None),
        )
        return snap.as_dict()

    # Convenience wrappers (optional)
    def hold(self) -> None:
        self.set_intent(RungIntent.HOLD)

    def seek_to(self, k: int) -> None:
        self.set_intent(RungIntent.SEEK, k_target=int(k))

    def step_up(self, n: int = 1) -> None:
        self.set_intent("step_up", steps=n)

    def step_down(self, n: int = 1) -> None:
        self.set_intent("step_down", steps=n)

    # ——— main hook ————————————————————————————————————————————————

    def update(self,
               tele: Dict[str, float],
               ctrl_from_supervisor: Optional[Dict[str, float]] = None
               ) -> Dict[str, float]:
        """
        Called once per training/inference step.

        Telemetry keys (best‑effort; all optional):
          'delta'|'δ⋆'          — current δ⋆ (if Supervisor changes it)
          'kappa'|'κ'           — lock confidence (0..1)
          'p_half'|'p½'         — barrier probability (~0.5 near ridge)
          'x_mean'              — mean position in X (enables precise k detection)

        ctrl_from_supervisor: existing dict with {beta, gamma, clamp}; if None,
        we start from {1.0, 0.5, 5.0} and blend toward our small targets.

        Returns a dict: {"beta": float, "gamma": float, "clamp": float}
        """
        # 0) Safe defaults
        ctrl = dict(ctrl_from_supervisor or {"beta": 1.0, "gamma": 0.5, "clamp": 5.0})

        # 1) Read telemetry (tolerant to missing keys)
        delta = float(tele.get("delta", tele.get("δ⋆", self.delta)))
        if delta != self.delta:
            self.delta = delta  # follow runtime’s δ⋆ if it changes

        kappa = float(tele.get("kappa", tele.get("κ", 0.0)))
        p_half = tele.get("p_half", tele.get("p½", None))
        x_mean = tele.get("x_mean", None)

        # Infer current rung index & residual if possible
        k_now, r = None, None
        if x_mean is not None:
            k_now = self._nearest_k(float(x_mean))
            r = float(x_mean) - k_now * self.delta

        # 2) If there is a pending relative plan, resolve it to an absolute k_target
        if self._plan and self._plan.get("armed") and k_now is not None:
            # “seek N steps in dir” → absolute target
            dirn = int(self._plan.get("dir", 0))
            steps = int(self._plan.get("steps", 0))
            if dirn != 0 and steps > 0:
                self.k_target = k_now + dirn * steps
                self.intent = RungIntent.SEEK
            self._plan = None  # clear plan once translated

        # 3) Policy selection
        if self.intent == RungIntent.STABILIZE:
            # Keep within acceptance band; increase damping near barriers.
            target = self._target_stabilize(p_half=p_half)
            return self._blend(ctrl, target, self.tuning.blend_hold)

        if self.intent == RungIntent.HOLD:
            # Gently center on the nearest rung and keep it quiet.
            target = self._target_hold(p_half=p_half)
            return self._blend(ctrl, target, self.tuning.blend_hold)

        if self.intent == RungIntent.SEEK:
            # Walk toward a target rung; if missing, fall back to HOLD.
            if self.k_target is None or k_now is None:
                target = self._target_hold(p_half=p_half)
                return self._blend(ctrl, target, self.tuning.blend_hold)

            # Direction toward target (+1 up, −1 down, 0 at target)
            dirn = 0
            if self.k_target > k_now:
                dirn = +1
            elif self.k_target < k_now:
                dirn = -1

            target, completed = self._target_step(direction=dirn, p_half=p_half, r=r, kappa=kappa)
            out = self._blend(ctrl, target, target.pop("_blend", self.tuning.blend_cross))

            # If we finished capturing a rung:
            if completed:
                # Are we at the absolute target? (Re‑compute with the latest k_now if available)
                if x_mean is not None:
                    k_now_after = self._nearest_k(float(x_mean))
                    if k_now_after == self.k_target:
                        # Switch to HOLD automatically at destination
                        self.intent = RungIntent.HOLD
                        self._reset_fsm()
                    else:
                        # Keep seeking: next barrier will be handled next tick
                        self.state.phase = "LOCKED"
                        self.state.dwell = 0
                else:
                    # Without x_mean, assume one rung progress; keep seeking if diff remains
                    self.state.phase = "LOCKED"
                    self.state.dwell = 0
            return out

        # Fallback: pass‑through
        return ctrl

    # ——— internal helpers ———————————————————————————————————————————

    @staticmethod
    def _as_intent(x: Union[str, RungIntent, None]) -> RungIntent:
        if isinstance(x, RungIntent):
            return x
        if isinstance(x, str):
            key = x.lower().strip()
            for v in RungIntent:
                if v.value == key:
                    return v
        return RungIntent.STABILIZE

    def _reset_fsm(self) -> None:
        self.state = _RungState()

    def _arm_relative_plan(self, dirn: int, steps: int) -> None:
        """Record a ‘relative step’ plan to be resolved on next update()."""
        self._plan = {"dir": int(dirn), "steps": int(steps), "armed": True}

    def _epsilon(self) -> float:
        """Acceptance half‑band in X."""
        if self.band is not None:
            return float(self.band)
        return self.tuning.epsilon_scale * self.delta

    def _nearest_k(self, x_mean: float) -> int:
        from math import floor
        # Round to nearest integer rung index
        return int(floor((x_mean / self.delta) + 0.5))

    def _clip(self, beta: float, gamma: float, clamp: float) -> Dict[str, float]:
        T = self.tuning
        b = min(max(beta, T.beta_min), T.beta_max)
        g = min(max(gamma, T.gamma_min), T.gamma_max)
        c = min(max(clamp, T.clamp_min), T.clamp_max)
        return {"beta": b, "gamma": g, "clamp": c}

    def _blend(self, base: Dict[str, float], target: Dict[str, float], w: float) -> Dict[str, float]:
        """Linear blend toward a safe target, then clip to rails."""
        w = float(min(max(w, 0.0), 1.0))
        beta = (1 - w) * base.get("beta", 1.0) + w * target.get("beta", 1.2)
        gamma = (1 - w) * base.get("gamma", 0.5) + w * target.get("gamma", 0.5)
        clamp = (1 - w) * base.get("clamp", 5.0) + w * target.get("clamp", 5.0)
        return self._clip(beta, gamma, clamp)

    # ——— policies ————————————————————————————————————————————————

    def _target_stabilize(self, p_half: Optional[float]) -> Dict[str, float]:
        """
        Stabilize around rungs without forcing tight centering.

        Intuition
        ---------
        Closer to a barrier (p½≈0.5) → keep damping higher; away → moderate.
        """
        T = self.tuning
        if p_half is None:
            gamma = 0.5 * (T.gamma_damp_lo + T.gamma_damp_hi)
        else:
            # Map p½ ∈ [0..1] to a “nearness to barrier” score ∈ [0..1]
            d = abs(0.5 - float(p_half))
            nearness = 1.0 - min(max(d / 0.5, 0.0), 1.0)
            gamma = T.gamma_damp_lo + nearness * (T.gamma_damp_hi - T.gamma_damp_lo)
        return {"beta": T.beta_hold, "gamma": gamma, "clamp": T.clamp_safe}

    def _target_hold(self, p_half: Optional[float]) -> Dict[str, float]:
        """
        Sticky centering on the nearest rung:
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
        Cross one barrier in the chosen direction, then re‑lock.
        Returns: (target_control, completed: bool)

        completed=True means “we have captured *a* rung” (one step finished),
        not necessarily that we’ve reached the absolute k_target. The caller
        continues seeking until k_now == k_target.
        """
        T = self.tuning

        # Decide via p_half if available; otherwise rely on r (residual) and κ (confidence)
        at_mid = (p_half is not None and p_half >= 0.48) or (r is not None and abs(r) >= 0.45 * self.delta)
        well_locked = (p_half is not None and p_half <= T.p_half_lock) or (kappa >= 0.20)

        # FSM
        if self.state.phase == "LOCKED":
            if direction == 0:
                # Already at destination rung for this step
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, True
            # Start leaning toward the barrier in the chosen direction
            self.state.phase = "MID" if at_mid else "LOCKED"
            return {"beta": T.beta_boost, "gamma": T.gamma_damp_lo, "clamp": T.clamp_safe, "_blend": T.blend_cross}, False

        if self.state.phase == "MID":
            self.state.dwell += 1
            # If we can maintain MID a few ticks, we’re safe to attempt crossing
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
                sign = 1 if direction > 0 else -1
                crossed = (sign * r) > 0  # residual sign matches push direction

            if crossed:
                self.state.phase = "CAPTURE"
                self.state.dwell = 0
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_capture}, False

            # Still crossing: stay nimble
            return {"beta": T.beta_boost, "gamma": T.gamma_damp_lo, "clamp": T.clamp_safe, "_blend": T.blend_cross}, False

        if self.state.phase == "CAPTURE":
            self.state.dwell += 1
            if self.state.dwell >= T.dwell_lock and well_locked:
                # Completed capture of one rung
                self.state.phase = "LOCKED"
                self.state.dwell = 0
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, True
            # Keep damping up during capture
            return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_capture}, False

        # Fallback (reset)
        self.state.phase = "LOCKED"
        self.state.dwell = 0
        return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, False


# ──────────────────────────────────────────────────────────────────────────────
# Null controller (no‑op). Useful for tests / ablations.
# ──────────────────────────────────────────────────────────────────────────────

class NullRungController(RungController):
    """Pass‑through controller that never changes Supervisor outputs."""
    def __init__(self, delta: float, **_: Any) -> None:
        super().__init__(delta=delta)
        self.intent = RungIntent.STABILIZE

    def update(self, tele: Dict[str, float], ctrl_from_supervisor: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        return dict(ctrl_from_supervisor or {"beta": 1.0, "gamma": 0.5, "clamp": 5.0})
