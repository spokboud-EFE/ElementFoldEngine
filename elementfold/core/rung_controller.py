# ElementFold · rung_controller.py
# ============================================================
# RungController — coherence‑aware shim over the Supervisor
#
# Role
# ----
# Works beside the Supervisor to handle discrete δ⋆ “rungs”.
# Adds small, safe overrides for β, γ, and ⛔ based on the
# system’s phase relative to a click barrier.
#
# Phase state machine
# -------------------
#   LOCKED → MID → CROSSING → CAPTURE → LOCKED
#
# Relaxation awareness
# --------------------
# Optional calmness from {folds ℱ, share‑rate η, letting‑go λ, smoothing D}
# softly reduces β, loosens ⛔, and boosts γ when the medium is calm.
#
# Public API (unchanged)
# ----------------------
#   class RungIntent(Enum)
#   class RungController:
#       update(tele, ctrl_from_supervisor) -> dict(beta,gamma,clamp)
#       set_intent(...), clear(), hold(), seek_to(k), step_up(n), step_down(n)
#       status() -> dict
#   class NullRungController(RungController)
# ============================================================

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Literal, Union, Any

__all__ = ["RungIntent", "RungTuning", "RungController", "NullRungController"]

# Type alias for FSM phase
ModePhase = Literal["LOCKED", "MID", "CROSSING", "CAPTURE"]


# ============================================================
# 1) RungIntent — high‑level control policy
# ============================================================

class RungIntent(str, Enum):
    STABILIZE = "stabilize"  # stay within acceptance band
    HOLD      = "hold"       # center and damp at the current rung
    SEEK      = "seek"       # move toward a target rung


# ============================================================
# 2) Tuning parameters (safe defaults)
# ============================================================

@dataclass
class RungTuning:
    """
    Gentle defaults for stable rung behavior.

    Crossing → higher β, lower γ.
    Holding  → lower β, higher γ.
    All blends ≤ 0.5 so the Supervisor remains dominant.
    """
    # geometry
    epsilon_scale: float = 1.0 / 6.0   # acceptance band: band = epsilon_scale * δ⋆

    # targets and dwell
    p_half_target: float = 0.55
    p_half_lock: float   = 0.12
    dwell_mid: int       = 3
    dwell_lock: int      = 3

    # attitude (exposure / damping / gate cap)
    beta_boost: float    = 2.0
    beta_hold: float     = 1.2
    gamma_damp_lo: float = 0.05
    gamma_damp_hi: float = 0.60
    clamp_safe: float    = 5.0

    # blend weights (how much to nudge over supervisor)
    blend_cross: float   = 0.50
    blend_hold: float    = 0.35
    blend_capture: float = 0.45

    # rails
    beta_min: float  = 0.5
    beta_max: float  = 3.0
    gamma_min: float = 0.0
    gamma_max: float = 1.5
    clamp_min: float = 1.0
    clamp_max: float = 12.0

    # relaxation (calmness) squashes and effects
    relax_F_scale: float      = 1.0
    relax_eta_scale: float    = 0.02
    relax_lambda_scale: float = 0.10
    relax_D_scale: float      = 0.10

    relax_beta_soften: float        = 0.10
    relax_gamma_boost: float        = 0.15
    relax_clamp_soften: float       = 0.25
    relax_blend_cross_suppress: float = 0.20
    relax_blend_hold_boost: float     = 0.10


# ============================================================
# 3) FSM state & snapshot
# ============================================================

@dataclass
class _RungState:
    phase: ModePhase = "LOCKED"
    dwell: int = 0


@dataclass
class RungSnapshot:
    intent: str
    k_target: Optional[int]
    phase: ModePhase
    dwell: int
    delta: float
    band: float
    plan: Optional[Dict[str, Any]]
    relax: Optional[Dict[str, float]] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# 4) Core controller
# ============================================================

class RungController:
    """
    Adds small rung‑aware overrides on top of Supervisor outputs.

    Usage
    -----
        ctrl_sup = sup.update(tele)
        ctrl_out = rung.update(tele, ctrl_sup)  # dict(beta,gamma,clamp)
        model.apply_control(**ctrl_out)
    """

    # -------------------------------
    # Construction & intent handling
    # -------------------------------
    def __init__(self,
                 delta: float,
                 k_target: Optional[int] = None,
                 band: Optional[float] = None,
                 intent: Union[RungIntent, str, None] = None,
                 tuning: Optional[RungTuning] = None):
        self.delta = float(delta)
        self.k_target = k_target
        self.band = band
        self.intent = self._as_intent(intent)
        self.tuning = tuning or RungTuning()
        self.state = _RungState()
        self._plan: Optional[Dict[str, Any]] = None  # relative plan (dir, steps)
        self._last_calm: Optional[float] = None

    def set_intent(self,
                   intent: Union[RungIntent, str],
                   k_target: Optional[int] = None,
                   *, steps: Optional[int] = None,
                   direction: Optional[int] = None) -> None:
        """Set high‑level intent; supports legacy relative plans (step_up/down)."""
        key = intent.value if isinstance(intent, RungIntent) else str(intent).lower().strip().replace("-", "_")

        # Relative SEEK plans
        if key in {"step_up", "up", "seek_up"} or (key == "seek" and direction in {+1}):
            self._arm_relative_plan(+1, int(steps or 1))
            self.intent = RungIntent.SEEK
            if k_target is not None:
                self.k_target = int(k_target); self._plan = None
            self._reset_fsm(); return
        if key in {"step_down", "down", "seek_down"} or (key == "seek" and direction in {-1}):
            self._arm_relative_plan(-1, int(steps or 1))
            self.intent = RungIntent.SEEK
            if k_target is not None:
                self.k_target = int(k_target); self._plan = None
            self._reset_fsm(); return

        # Absolute intents
        self.intent = self._as_intent(key)
        if k_target is not None: self.k_target = int(k_target)
        self._reset_fsm()

    def clear(self) -> None:
        """Return to STABILIZE and clear plans."""
        self.intent = RungIntent.STABILIZE
        self.k_target = None
        self._plan = None
        self._reset_fsm()

    def hold(self) -> None:
        self.set_intent(RungIntent.HOLD)

    def seek_to(self, k: int) -> None:
        self.set_intent(RungIntent.SEEK, k_target=int(k))

    def step_up(self, n: int = 1) -> None:
        self.set_intent("step_up", steps=n)

    def step_down(self, n: int = 1) -> None:
        self.set_intent("step_down", steps=n)

    def status(self) -> Dict[str, object]:
        """Human‑readable snapshot for logs/UI."""
        snap = RungSnapshot(
            intent=self.intent.value,
            k_target=self.k_target,
            phase=self.state.phase,
            dwell=self.state.dwell,
            delta=self.delta,
            band=self._epsilon(),
            plan=(dict(self._plan) if self._plan else None),
            relax=(None if self._last_calm is None else {"calm": float(self._last_calm)}),
        )
        return snap.as_dict()

    # ---------------
    # Main update
    # ---------------
    def update(self,
               tele: Dict[str, float],
               ctrl_from_supervisor: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Read telemetry and gently adjust Supervisor’s control outputs.

        tele: may contain ASCII or Unicode keys (kappa/κ, p_half/p½, delta/δ⋆, etc.)
        """
        # Start from supervisor’s suggestion (or safe defaults)
        base = dict(ctrl_from_supervisor or {"beta": 1.0, "gamma": 0.5, "clamp": 5.0})

        # ---- Telemetry (robust parsing) ----
        delta = float(tele.get("delta", tele.get("δ⋆", self.delta)))
        if delta != self.delta:  # keep in sync
            self.delta = delta

        kappa = float(tele.get("kappa", tele.get("κ", 0.0)))
        p_half_val = tele.get("p_half", tele.get("p½", None))
        p_half = None if p_half_val is None else float(p_half_val)

        x_mean_val = tele.get("x_mean", None)
        x_mean = None if x_mean_val is None else float(x_mean_val)

        # Relaxation calmness in [0,1] (optional)
        calm = self._relaxation_calm(tele)
        self._last_calm = calm

        # Rung index / residual if possible
        k_now, r = None, None
        if x_mean is not None:
            k_now = self._nearest_k(x_mean)
            r = x_mean - k_now * self.delta  # signed residual

        # Resolve relative plan → absolute target rung
        if self._plan and self._plan.get("armed") and k_now is not None:
            dirn = int(self._plan.get("dir", 0))
            steps = int(self._plan.get("steps", 0))
            if dirn != 0 and steps > 0:
                self.k_target = k_now + dirn * steps
                self.intent = RungIntent.SEEK
            self._plan = None

        # ---- Intent behavior ----
        if self.intent == RungIntent.STABILIZE:
            tgt = self._target_stabilize(p_half)
            tgt = self._apply_relaxation(tgt, calm, phase=self.state.phase)
            w = self._relax_blend_weight(self.tuning.blend_hold, calm, phase="LOCKED")
            return self._blend(base, tgt, w)

        if self.intent == RungIntent.HOLD:
            tgt = self._target_hold(p_half)
            tgt = self._apply_relaxation(tgt, calm, phase=self.state.phase)
            w = self._relax_blend_weight(self.tuning.blend_hold, calm, phase="LOCKED")
            return self._blend(base, tgt, w)

        if self.intent == RungIntent.SEEK:
            # Without a position or target, fall back to HOLD‑like behavior.
            if self.k_target is None or k_now is None:
                tgt = self._target_hold(p_half)
                tgt = self._apply_relaxation(tgt, calm, phase=self.state.phase)
                w = self._relax_blend_weight(self.tuning.blend_hold, calm, phase="LOCKED")
                return self._blend(base, tgt, w)

            # Direction toward target rung
            dirn = 0
            if self.k_target > k_now: dirn = +1
            elif self.k_target < k_now: dirn = -1

            tgt, done = self._target_step(dirn, p_half, r, kappa)
            tgt = self._apply_relaxation(tgt, calm, phase=self.state.phase)
            w0 = float(tgt.pop("_blend", self.tuning.blend_cross))
            w = self._relax_blend_weight(w0, calm, phase=self.state.phase)
            out = self._blend(base, tgt, w)

            if done:
                # Confirm lock on target if we can read x_mean; otherwise reset FSM
                if x_mean is not None and self._nearest_k(x_mean) == self.k_target:
                    self.intent = RungIntent.HOLD
                    self._reset_fsm()
                else:
                    self.state.phase, self.state.dwell = "LOCKED", 0
                return out
            return out

        # Fallback: no change
        return base

    # ========================================================
    # Internals
    # ========================================================

    @staticmethod
    def _as_intent(x: Union[str, RungIntent, None]) -> RungIntent:
        if isinstance(x, RungIntent): return x
        if isinstance(x, str):
            key = x.lower().strip()
            for v in RungIntent:
                if v.value == key: return v
        return RungIntent.STABILIZE

    def _reset_fsm(self) -> None:
        self.state = _RungState()

    def _arm_relative_plan(self, dirn: int, steps: int) -> None:
        self._plan = {"dir": int(dirn), "steps": int(steps), "armed": True}

    def _epsilon(self) -> float:
        return float(self.band if self.band is not None else self.tuning.epsilon_scale * self.delta)

    def _nearest_k(self, x_mean: float) -> int:
        from math import floor
        return int(floor((x_mean / self.delta) + 0.5))

    def _clip(self, b: float, g: float, c: float) -> Dict[str, float]:
        T = self.tuning
        b = min(max(b, T.beta_min), T.beta_max)
        g = min(max(g, T.gamma_min), T.gamma_max)
        c = min(max(c, T.clamp_min), T.clamp_max)
        return {"beta": b, "gamma": g, "clamp": c}

    def _blend(self, base: Dict[str, float], target: Dict[str, float], w: float) -> Dict[str, float]:
        """Linear blend toward target with clipping."""
        w = float(min(max(w, 0.0), 1.0))
        b0, g0, c0 = float(base.get("beta", 1.0)), float(base.get("gamma", 0.5)), float(base.get("clamp", 5.0))
        bt, gt, ct = float(target.get("beta", 1.2)), float(target.get("gamma", 0.5)), float(target.get("clamp", 5.0))
        return self._clip(b0 + w * (bt - b0), g0 + w * (gt - g0), c0 + w * (ct - c0))

    # -------------------------------
    # Target policies (by intent/FSM)
    # -------------------------------
    def _target_stabilize(self, p_half: Optional[float]) -> Dict[str, float]:
        T = self.tuning
        if p_half is None:
            gamma = 0.5 * (T.gamma_damp_lo + T.gamma_damp_hi)
        else:
            # near barrier → stronger damping
            d = abs(0.5 - max(0.0, min(1.0, p_half)))
            nearness = 1.0 - min(max(d / 0.5, 0.0), 1.0)
            gamma = T.gamma_damp_lo + nearness * (T.gamma_damp_hi - T.gamma_damp_lo)
        return {"beta": T.beta_hold, "gamma": gamma, "clamp": T.clamp_safe}

    def _target_hold(self, p_half: Optional[float]) -> Dict[str, float]:
        T = self.tuning
        gamma = T.gamma_damp_hi if (p_half is None or p_half < 0.25) \
                else 0.5 * (T.gamma_damp_lo + T.gamma_damp_hi)
        return {"beta": T.beta_hold, "gamma": gamma, "clamp": T.clamp_safe}

    def _target_step(self,
                     direction: int,
                     p_half: Optional[float],
                     r: Optional[float],
                     kappa: float) -> tuple[Dict[str, float], bool]:
        """
        Finite‑state crossing logic.
        Returns (target_controls, done_flag).
        """
        T = self.tuning
        d = self.delta

        # Robust predicates (None‑safe)
        at_mid = ((p_half is not None and p_half >= 0.48) or
                  (r is not None and abs(r) >= 0.45 * d))
        well_locked = ((p_half is not None and p_half <= T.p_half_lock) or
                       (kappa >= 0.20))

        if self.state.phase == "LOCKED":
            if direction == 0:
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, True
            # Start moving; if already mid, advance, else stay LOCKED until mid sensed
            self.state.phase = "MID" if at_mid else "LOCKED"
            return {"beta": T.beta_boost, "gamma": T.gamma_damp_lo, "clamp": T.clamp_safe, "_blend": T.blend_cross}, False

        if self.state.phase == "MID":
            self.state.dwell += 1
            if self.state.dwell >= T.dwell_mid:
                self.state.phase, self.state.dwell = "CROSSING", 0
            return {"beta": T.beta_boost, "gamma": T.gamma_damp_lo, "clamp": T.clamp_safe, "_blend": T.blend_cross}, False

        if self.state.phase == "CROSSING":
            # Crossed if sufficiently near/over the barrier by p½ or residual sign aligns with direction
            crossed = ((p_half is not None and p_half >= T.p_half_target) or
                       (r is not None and (1 if direction > 0 else -1) * r > 0.0))
            if crossed:
                self.state.phase, self.state.dwell = "CAPTURE", 0
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_capture}, False
            return {"beta": T.beta_boost, "gamma": T.gamma_damp_lo, "clamp": T.clamp_safe, "_blend": T.blend_cross}, False

        if self.state.phase == "CAPTURE":
            self.state.dwell += 1
            if self.state.dwell >= T.dwell_lock and well_locked:
                self.state.phase, self.state.dwell = "LOCKED", 0
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, True
            # Keep damping high while capturing
            return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_capture}, False

        # Unknown phase → reset
        self.state.phase, self.state.dwell = "LOCKED", 0
        return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, False

    # -------------------------------
    # Relaxation (calmness) processing
    # -------------------------------
    def _relaxation_calm(self, tele: Dict[str, Any]) -> float:
        """
        Distill optional diffusion/decay signals into calmness ∈ [0,1].
          • Larger (ℱ, λ, D) ⇒ calmer (↑)
          • Larger η         ⇒ more active (↓)
        """
        T = self.tuning

        def _get(keys: tuple[str, ...]) -> Optional[float]:
            for k in keys:
                if k in tele and tele[k] is not None:
                    try: return float(tele[k])
                    except Exception: pass
            return None

        def _squash_pos(x: Optional[float], scale: float) -> Optional[float]:
            if x is None or scale <= 0: return None
            v = abs(float(x)); return v / (v + scale)

        F   = _get(("F", "ℱ", "folds"))
        eta = _get(("eta", "η"))
        lam = _get(("lambda", "λ"))
        Dv  = _get(("D", "smooth", "smoothing"))

        vals = []
        for s in (_squash_pos(F, T.relax_F_scale),
                  _squash_pos(lam, T.relax_lambda_scale),
                  _squash_pos(Dv, T.relax_D_scale)):
            if s is not None: vals.append(s)
        se = _squash_pos(eta, T.relax_eta_scale)
        if se is not None: vals.append(1.0 - se)  # inverse effect

        if not vals: return 0.0
        calm = sum(vals) / len(vals)
        return float(max(0.0, min(1.0, calm)))

    def _apply_relaxation(self, target: Dict[str, float],
                          calm: float, *, phase: ModePhase) -> Dict[str, float]:
        """Adjust β, γ, and ⛔ as calmness increases."""
        if calm <= 0.0: return target
        T = self.tuning
        beta  = float(target.get("beta", T.beta_hold))
        gamma = float(target.get("gamma", 0.5))
        clamp = float(target.get("clamp", T.clamp_safe))

        # Soft, phase‑aware modulation
        beta *= (1.0 - T.relax_beta_soften * calm *
                 (1.0 if phase in {"LOCKED", "CAPTURE"} else 0.5))
        gamma += T.relax_gamma_boost * calm
        clamp *= (1.0 - T.relax_clamp_soften * calm)

        out = dict(target)
        out.update({"beta": beta, "gamma": gamma, "clamp": clamp})
        return out

    def _relax_blend_weight(self, w: float, calm: float, *, phase: ModePhase | str) -> float:
        """Phase‑aware blend modulation with calmness."""
        w = float(min(max(w, 0.0), 1.0))
        if calm <= 0.0: return w
        T = self.tuning
        if phase in {"MID", "CROSSING"}:
            w *= (1.0 - T.relax_blend_cross_suppress * calm)
        else:
            w += T.relax_blend_hold_boost * calm
        return float(min(max(w, 0.0), 1.0))


# ============================================================
# 5) Null controller (no‑op)
# ============================================================

class NullRungController(RungController):
    """Pass‑through version of RungController that applies no overrides."""
    def __init__(self, delta: float, **_: Any) -> None:
        super().__init__(delta=delta)
        self.intent = RungIntent.STABILIZE

    def update(self, tele: Dict[str, float],
               ctrl_from_supervisor: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        return dict(ctrl_from_supervisor or {"beta": 1.0, "gamma": 0.5, "clamp": 5.0})
