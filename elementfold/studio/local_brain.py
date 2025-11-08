"""
studio/local_brain.py — The Local Brain 🧠

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The Local Brain is the Studio’s intuition.
• It watches the Factory’s telemetry, reads the ledgers,
  and murmurs comments about coherence, stability, and rhythm.
• It never overrides control — it advises, narrates, and learns
  from how the system behaves.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from elementfold.core.control.factory import Factory
from elementfold.core.control.ledger import Ledger
from elementfold.core.physics.safety_guard import SafetyGuard
from elementfold.core.telemetry import TelemetryBus


# ====================================================================== #
# 🧩 LocalBrain — advisory and narrative engine
# ====================================================================== #
@dataclass
class LocalBrain:
    """
    Observes Factory and TelemetryBus, provides commentary and suggestions.

    It’s rule-based, transparent, and human-readable.
    """

    factory: Factory
    telemetry: TelemetryBus
    guard: SafetyGuard
    verbose: bool = True
    mood: str = "neutral"
    last_comment: str = ""
    _last_eval_time: float = 0.0
    _interval: float = 2.0  # seconds between comment updates

    # ------------------------------------------------------------------ #
    # 🎧 Main loop interface
    # ------------------------------------------------------------------ #
    def tick(self) -> Optional[str]:
        """Evaluate system and possibly emit a comment."""
        now = time.perf_counter()
        if now - self._last_eval_time < self._interval:
            return None
        self._last_eval_time = now

        comment = self._evaluate_state()
        self.last_comment = comment
        if comment and self.verbose:
            print(comment)
        return comment

    # ------------------------------------------------------------------ #
    # 🧮 Evaluation logic
    # ------------------------------------------------------------------ #
    def _evaluate_state(self) -> str:
        """Inspect factory state and generate advisory text."""
        try:
            snap = self.factory.snapshot()
            if not snap:
                return "🤔 waiting for cores..."
            comments: List[str] = []

            for name, state in snap.items():
                κ = state.get("kappa", 1.0)
                λ = state["params"].get("lambda", 0.1) if "params" in state else 0.1
                D = state["params"].get("D", 0.05) if "params" in state else 0.05
                dt = state.get("dt", 0.01)
                mode = state.get("mode", "shaping")

                stable = self.guard.check_stability({"lambda": λ, "D": D, "dt": dt})

                # Choose comment tone
                if κ > 0.97 and stable:
                    tone = "🌤️ harmony stable"
                elif κ < 0.7:
                    tone = "🌧️ coherence weak"
                elif not stable:
                    tone = "⚠️ nearing instability"
                else:
                    tone = "🌫️ breathing normally"

                # Small randomized phrasing to avoid monotony
                phr = random.choice(
                    [
                        f"Core {name} {tone} (κ={κ:.2f}, λΔt={λ*dt:.3f}, DΔt={D*dt:.3f})",
                        f"{name}: κ={κ:.2f} — {tone}",
                        f"{tone} at {name} (Δt={dt:.3g})",
                    ]
                )
                comments.append(phr)

                # Suggest parameter adjustment if marginal
                if 0.9 < κ < 0.95:
                    comments.append(f"🪶 {name}: consider increasing λ slightly for tighter relaxation.")
                if κ < 0.5:
                    comments.append(f"🪶 {name}: coherence low, reduce D or shorten dt.")
                if not stable:
                    comments.append(f"🪶 {name}: try λ={λ*0.8:.3g}, D={D*0.8:.3g} to regain stability.")

            # Combine and set mood
            joined = "\n".join(comments)
            self._update_mood(joined)
            return joined
        except Exception as exc:
            return f"[local_brain] error evaluating state: {exc}"

    # ------------------------------------------------------------------ #
    # 🎭 Mood tracking
    # ------------------------------------------------------------------ #
    def _update_mood(self, text: str) -> None:
        """Adjust internal mood based on content."""
        if "🌧️" in text or "⚠️" in text:
            self.mood = "concerned"
        elif "🌤️" in text:
            self.mood = "calm"
        else:
            self.mood = "neutral"

    # ------------------------------------------------------------------ #
    # 🧘 Manual invocation
    # ------------------------------------------------------------------ #
    def narrate(self) -> None:
        """Speak the latest comment again, or evaluate if none."""
        if not self.last_comment:
            self.tick()
        else:
            print(self.last_comment)

    # ------------------------------------------------------------------ #
    # 💡 Suggestion API
    # ------------------------------------------------------------------ #
    def suggest(self) -> Dict[str, Any]:
        """
        Return programmatic suggestions based on current mood.
        Example:
            {'adjust_lambda': 0.9, 'recommendation': 'reduce diffusion slightly'}
        """
        sug: Dict[str, Any] = {}
        if self.mood == "calm":
            sug["recommendation"] = "system stable; maintain parameters"
        elif self.mood == "concerned":
            sug["recommendation"] = "reduce step size or lower diffusion"
            sug["adjust_lambda"] = 0.9
            sug["adjust_D"] = 0.9
        elif self.mood == "neutral":
            sug["recommendation"] = "monitor coherence; no change yet"
        return sug
