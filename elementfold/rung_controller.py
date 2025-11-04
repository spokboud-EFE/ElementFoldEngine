# elementfold/rung_controller.py
# ──────────────────────────────────────────────────────────────────────────────
# RungController — on‑demand click (rung) control for ElementFold
#     • default: passthrough (does nothing; stable with existing code)
#     • 'hold'     → keep the system centered on the nearest rung
#     • 'step_up'  → deliberately reach the mid‑step, cross, and capture the next rung
#     • 'step_down'→ same but toward the previous rung
#
# Where it plugs in (one line change in your training loop):
#     ctrl_sup = sup.update(tele)
#     ctrl_out = rung.update(tele, ctrl_sup)      # ← add this
#     model.apply_control(**ctrl_out)
#
# Non‑expert picture:
#   Think of a washboard with bumps every δ⋆ in the hidden log variable X.
#   The Supervisor keeps the ride smooth; this controller, when told,
#   gently lowers damping (γ), raises exposure (β), and walks you over the
#   half‑bump, then restores damping to lock the next groove.
#
# Inputs it reads (best effort — all optional, safe fallbacks):
#   tele["delta"] or tele["δ⋆"] or passed-in delta → the click size
#   tele["kappa"] or tele["κ"]                   → coherence score
#   tele["p_half"] or tele["p½"]                 → “how near the half‑step” fraction
#   tele["x_mean"] (preferred if present)        → average X to compute residuals
#
# Outputs it sets:
#   A tiny override of beta / gamma / clamp (⛔) via {beta:..., gamma:..., clamp:...}
#
# Safety rails:
#   • Hard clamps on β, γ, ⛔ (see _clip()) so you can’t push unstable commands.
#   • Internal state machine with dwell times; once a click completes, it re‑locks.
#
# MIT‑style tiny utility; zero dependencies outside standard library.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Literal

Mode = Literal["off", "passthrough", "hold", "step_up", "step_down"]

# ──────────────────────────────────────────────────────────────────────────────
# User intent (what you want done)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RungIntent:
    """
    What should the controller try to do?

    mode
      "off" / "passthrough": do nothing (just return the Supervisor’s control)
      "hold":                center and keep the nearest rung
      "step_up":             cross the barrier and capture the next higher rung
      "step_down":           same toward the next lower rung

    steps
      How many clicks to execute (for step_*). After each captured rung,
      the internal counter decreases; reaching 0 returns to "hold".
    """
    mode: Mode = "passthrough"
    steps: int = 0

# ──────────────────────────────────────────────────────────────────────────────
# Tuning knobs (gentle defaults; all values dimensionless)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RungTuning:
    """
    Gentle defaults that behave well on CPU/GPU smoke tests.
    """
    # Acceptance band half‑width; ~= δ⋆/6 by convention
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

# ──────────────────────────────────────────────────────────────────────────────
# Internal state (simple finite‑state machine)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _RungState:
    phase: Literal["IDLE", "LOCKED", "MID", "CROSSING", "CAPTURE"] = "IDLE"
    dwell: int = 0
    done_steps: int = 0
    # Book‑keeping for residuals (if x_mean available)
    last_k: Optional[int] = None
    last_x: Optional[float] = None

# ──────────────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────────────

class RungController:
    """
    RungController — non‑intrusive, on‑demand rung control.

    Usage (one line in your loop):
        ctrl_sup = sup.update(tele)
        ctrl_out = rung.update(tele, ctrl_sup)
        model.apply_control(**ctrl_out)

    Notes for non‑experts:
      • You don’t have to pass every telemetry key; it will work with κ (kappa)
        and p½ (p_half) alone. If you also provide x_mean (average ledger X),
        the rung index/residuals are computed more precisely.

      • “Hold” gently increases damping (γ) and centers you on the rung.
        “Step” lowers damping, increases exposure (β), waits until p½ shows
        you’re over the ridge, then increases damping to lock the next rung.

      • If you never set an intent (default), it behaves like a wire: it
        simply returns the Supervisor’s control unchanged.
    """

    def __init__(self,
                 delta: float,
                 intent: Optional[RungIntent] = None,
                 tuning: Optional[RungTuning] = None):
        self.delta = float(delta)
        self.intent = intent or RungIntent("passthrough", 0)
        self.tuning = tuning or RungTuning()
        self.state = _RungState()

    # ——— public API ————————————————————————————————————————————————

    def set_intent(self, mode: Mode, steps: int = 0) -> None:
        """Change the goal on the fly (safe during training or inference)."""
        self.intent = RungIntent(mode=mode, steps=max(0, int(steps)))
        self._reset_fsm()

    def clear(self) -> None:
        """Return to passthrough; state machine resets."""
        self.set_intent("passthrough", steps=0)

    def status(self) -> Dict[str, object]:
        """Tiny JSON‑like snapshot (nice to print in Studio)."""
        return {
            "mode": self.intent.mode,
            "remaining_steps": self.intent.steps,
            "phase": self.state.phase,
            "dwell": self.state.dwell,
            "delta": self.delta
        }

    # ——— main hook ————————————————————————————————————————————————

    def update(self,
               tele: Dict[str, float],
               ctrl_from_supervisor: Optional[Dict[str, float]] = None
               ) -> Dict[str, float]:
        """
        Called once per training/inference step.

        tele  — coherence metrics; best‑effort keys used:
                'delta'|'δ⋆', 'kappa'|'κ', 'p_half'|'p½', 'x_mean'
        ctrl_from_supervisor — existing {beta,gamma,clamp} suggestion;
                we blend toward our targets; if None, we start from {1.0,0.5,5.0}.
        """
        # 0) Safe defaults
        ctrl = dict(ctrl_from_supervisor or {"beta": 1.0, "gamma": 0.5, "clamp": 5.0})

        # Early exit when passthrough/off
        if self.intent.mode in ("off", "passthrough"):
            return ctrl

        # 1) Read telemetry (best‑effort, tolerant to missing keys)
        delta = float(tele.get("delta", tele.get("δ⋆", self.delta)))
        if delta != self.delta:
            self.delta = delta  # follow runtime’s δ⋆ if it changes

        kappa = float(tele.get("kappa", tele.get("κ", 0.0)))
        p_half = tele.get("p_half", tele.get("p½", None))
        x_mean = tele.get("x_mean", None)  # optional but helpful

        # 2) Decide targets from mode + state
        if self.intent.mode == "hold":
            target = self._target_hold(kappa, p_half)
            return self._blend(ctrl, target, self.tuning.blend_hold)

        if self.intent.mode in ("step_up", "step_down"):
            target, completed = self._target_step(direction=+1 if self.intent.mode == "step_up" else -1,
                                                  kappa=kappa, p_half=p_half, x_mean=x_mean)
            out = self._blend(ctrl, target, target.pop("_blend", self.tuning.blend_cross))
            if completed:
                # Count the click and either continue or fall back to "hold"
                self.intent.steps = max(0, self.intent.steps - 1)
                if self.intent.steps == 0:
                    self.intent.mode = "hold"
                    self._reset_fsm(to="LOCKED")
            return out

        # Unknown mode → passthrough
        return ctrl

    # ——— internal helpers ———————————————————————————————————————————

    def _reset_fsm(self, to: _RungState["phase"].__args__ = "IDLE") -> None:
        self.state = _RungState(phase=to, dwell=0, done_steps=0, last_k=None, last_x=None)

    def _epsilon(self) -> float:
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

    # ——— HOLD (keep nearest rung centered) —————————————————————————

    def _target_hold(self, kappa: float, p_half: Optional[float]) -> Dict[str, float]:
        """
        Simple, gentle centering:
          • slightly higher damping (γ) to absorb chatter,
          • modest β to keep responsiveness,
          • clamp at a safe value.
        """
        T = self.tuning
        # If p_half says we’re far from the barrier, use a bit more damping; else, stay nimble
        gamma = T.gamma_damp_hi if (p_half is None or p_half < 0.25) else 0.5 * (T.gamma_damp_lo + T.gamma_damp_hi)
        return {"beta": T.beta_hold, "gamma": gamma, "clamp": T.clamp_safe}

    # ——— STEP (cross barrier then capture) ——————————————————————————

    def _target_step(self,
                     direction: int,
                     kappa: float,
                     p_half: Optional[float],
                     x_mean: Optional[float]) -> tuple[Dict[str, float], bool]:
        """
        State machine:
            LOCKED   → (lower γ, raise β) → MID (approach barrier)
            MID      → (β↑, γ↓) hold mid dwell → CROSSING
            CROSSING → once p_half high or residual shows crossing, go CAPTURE
            CAPTURE  → γ↑ (damping), clamp safe; after dwell, count +1 step
        Returns: (target_control, completed: bool)
        """
        T = self.tuning
        eps = self._epsilon()

        # Heuristics when x_mean is available (preferred)
        k_now, r = None, None
        if x_mean is not None:
            k_now = self._nearest_k(x_mean)
            r = x_mean - k_now * self.delta  # absolute residual (±)

        # Phase transitions (no hard dependence on x_mean)
        if self.state.phase in ("IDLE",):
            self.state.phase = "LOCKED"
            self.state.dwell = 0

        # Decide with p_half if available; otherwise rely on dwell + kappa
        at_mid = (p_half is not None and p_half >= 0.48) or (r is not None and abs(r) >= 0.45 * self.delta)
        well_locked = (p_half is not None and p_half <= T.p_half_lock) or (kappa >= 0.20)

        if self.state.phase == "LOCKED":
            # Start leaning toward the barrier in the chosen direction
            self.state.dwell = 0
            if at_mid:
                self.state.phase = "MID"
            # Target: cross‑friendly set (nimble)
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
            # We consider the new rung "locked" once p_half is small (or κ recovered)
            if self.state.dwell >= T.dwell_lock and well_locked:
                # completed one click
                self.state.phase = "LOCKED"
                self.state.dwell = 0
                self.state.done_steps += 1
                return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, True
            # Keep damping up during capture
            return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_capture}, False

        # Fallback
        return {"beta": T.beta_hold, "gamma": T.gamma_damp_hi, "clamp": T.clamp_safe, "_blend": T.blend_hold}, False
