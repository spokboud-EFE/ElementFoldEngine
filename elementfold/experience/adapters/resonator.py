# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ElementFold · experience/adapters/resonator.py                               ║
# ║──────────────────────────────────────────────────────────────────────────────║
# ║  Resonator Adapter — the physical heart of the Studio’s control surface.     ║
# ║                                                                              ║
# ║  Purpose:                                                                    ║
# ║   • Bridge the relaxation core (runtime + control + telemetry) with the UI.  ║
# ║   • Respect shaping / forcing modes.                                         ║
# ║   • Produce real-time narrative feedback (β, γ, κ, ⛔, λ, D, ∇Φ, ℱ, δ★).     ║
# ║                                                                              ║
# ║  Public contract (through AdapterRegistry):                                  ║
# ║    @AdapterRegistry.register_fn("resonator")                                 ║
# ║    def make_resonator_adapter() → Adapter                                    ║
# ║                                                                              ║
# ║  The adapter is NumPy-based, self-contained, and readable for hardware or    ║
# ║  physics engineers wanting to inject real sensors or actuators later.       ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations
import math, time, numpy as np
from typing import Dict, Any, Tuple

from elementfold.core import control, telemetry, runtime
from elementfold.experience.adapters.base import (
    Adapter, AdapterRegistry, AdapterSpec, AdapterMeta,
    with_spec, with_meta,
)

# ────────────────────────────────────────────────────────────────────────────────
# Adapter specification (used by Studio and registry)
# ────────────────────────────────────────────────────────────────────────────────
_resonator_spec = AdapterSpec(
    name="resonator",
    description="Resonant adapter coupling the relaxation field with the Studio.",
    expects={"ΔΦ": None},  # minimal placeholder (no fixed input)
    predicts={
        "folds": "ℱ cumulative relaxation",
        "z": "redshift equivalent (e^ℱ − 1)",
        "A": "brightness attenuation (≈ e^{−2ℱ})",
        "narrative": "short Unicode telemetry summary",
    },
    wait="simulate_only",
)

_resonator_meta = AdapterMeta(
    kind="physics",
    what="Resonator adapter — drives and reads field oscillations.",
    why="Serves as the central physical control bridge between shaping and forcing.",
    actions=("tick", "pulse", "status", "reset"),
    params={"δ★": "click size", "λ": "relax rate", "γ": "damping"},
)

# ────────────────────────────────────────────────────────────────────────────────
# The Adapter itself
# ────────────────────────────────────────────────────────────────────────────────
@AdapterRegistry.register_fn("resonator")
@with_spec(_resonator_spec)
@with_meta(_resonator_meta)
def make_resonator_adapter() -> Adapter:
    class ResonatorAdapter(Adapter):
        """Unified NumPy resonator; mode-aware and narrative."""

        def __init__(self):
            super().__init__("Resonator")
            # field state variables
            self.phase = 0.0
            self.velocity = 0.0
            self.fold_clock = 0.0
            self.last_tick = time.time()
            self.last_report: Dict[str, float] = {}
            self.coherence = 1.0

        # ── Core behavior ───────────────────────────────────────────────────────
        def infer(self, model=None, data=None, **kw) -> Dict[str, Any]:
            """
            Perform one relaxation step depending on mode:
              • shaping → smooth relaxation
              • forcing → impulsive update
            """
            mode = control.get_mode()
            params = telemetry.snapshot()
            dt = 0.05

            # interpret λ and γ from telemetry if available
            lam = float(params.get("lambda_relax", 0.3))
            gam = float(params.get("gamma_damping", 0.5))
            delta = float(params.get("delta_star", 0.31))

            # simple oscillator dynamics
            if mode == "forcing":
                impulse = np.random.uniform(-1, 1) * 0.8
                self.velocity += impulse
            else:  # shaping
                self.velocity += -lam * self.phase - gam * self.velocity

            self.phase += self.velocity * dt
            self.phase = max(-delta, min(delta, self.phase))
            # compute fold accumulation ℱ
            self.fold_clock += abs(self.velocity) * dt
            self.coherence = max(0.0, 1.0 - 0.5 * abs(self.phase / delta))
            z = math.exp(self.fold_clock) - 1.0
            atten = math.exp(-2.0 * self.fold_clock)

            self.last_report = {
                "δ★": delta,
                "λ": lam,
                "γ": gam,
                "ℱ": self.fold_clock,
                "κ": self.coherence,
                "z": z,
                "A": atten,
            }
            return self.last_report

        # ── Passive observation hook ────────────────────────────────────────────
        def observe(self, tele: Dict[str, Any]) -> None:
            """Update from external telemetry (used by Studio sync)."""
            self.last_state = dict(tele)
            self.coherence = float(tele.get("κ", tele.get("kappa", self.coherence)))

        # ── Reset ───────────────────────────────────────────────────────────────
        def reset(self) -> None:
            self.phase = 0.0
            self.velocity = 0.0
            self.fold_clock = 0.0
            self.coherence = 1.0
            self.last_report.clear()

        # ── Simulation / diagnostic ─────────────────────────────────────────────
        def simulate(self, ticks: int = 100) -> Dict[str, np.ndarray]:
            """Generate synthetic oscillation traces for testing."""
            t = np.linspace(0, 2 * math.pi, ticks)
            phase = np.sin(t) * 0.3
            folds = np.linspace(0, 1.0, ticks)
            κ = np.cos(t) * 0.5 + 0.5
            return {"t": t, "phase": phase, "folds": folds, "κ": κ}

        # ── Human narrative ─────────────────────────────────────────────────────
        def narrate(self, state: Dict[str, Any] | None = None) -> str:
            st = state or self.last_report
            if not st:
                return "🌀 Resonator idle — awaiting first tick."
            κ = st.get("κ", self.coherence)
            ℱ = st.get("ℱ", self.fold_clock)
            mode = control.get_mode()
            if mode == "forcing":
                tone = "⚡ field pulsed"
            else:
                tone = "🎛️ shaping field"
            return (
                f"{tone} — κ={κ:.3f} (coherence), ℱ={ℱ:.3f} (folds) "
                f"→ {'stable' if κ>0.9 else 'relaxing' if κ>0.5 else 'unstable'}."
            )

    return ResonatorAdapter()
