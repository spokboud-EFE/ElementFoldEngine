# ElementFold · experience/local_brain.py
# ============================================================
# Runtime-Aware Brain Interface (Physics Runtime Edition)
# ------------------------------------------------------------
# Purpose:
#   • Expose a JSON reasoning loop between an LLM and the ElementFold engine.
#   • The brain observes telemetry and config, proposes parameter changes,
#     and receives feedback from the relaxation runtime.
#
# Public API:
#   - LocalBrain(engine)
#   - step()      → runs one observe → reason → act cycle
#   - teach()     → returns canonical prompt examples
#
# Notes:
#   • Uses RelaxationModel as the runtime engine.
#   • Works with any BackgroundModel implementing .chat(messages=[...]).
# ============================================================

from __future__ import annotations
import json, time
from typing import Any, Dict, Optional

# Import the current physics runtime classes
from ..core.model import RelaxationModel as Engine
from ..utils.display import info, warn, success, debug
from ..core.telemetry import summary as measure_telemetry

# Optional model layer (local or remote reasoning LLM)
try:
    from .background_model import BackgroundModel
except Exception:  # pragma: no cover
    BackgroundModel = None


# ============================================================
# Default prompt template
# ============================================================

_BASE_PROMPT = """
You are a runtime controller for the ElementFold relaxation engine.

You will receive:
  1. The current telemetry of the field (variance, energy, coherence, etc.).
  2. The current configuration (lambda, D, phi_inf, dt).
Your task is to respond with one JSON object describing what to do next.

Valid actions:
  {"action":"adjust","params":{"lambda":0.3,"D":0.2},"comment":"Reduce lambda to slow relaxation."}
  {"action":"hold","comment":"Stable; no change."}

Do not output text before or after the JSON.
"""


# ============================================================
# LocalBrain class
# ============================================================

class LocalBrain:
    """Bridge between the physics runtime (Engine) and a text reasoning model."""

    def __init__(self, engine: Engine, *, model: Optional[Any] = None, verbose: bool = True):
        self.engine = engine
        self.model = model or (BackgroundModel() if BackgroundModel else None)
        self.verbose = verbose
        if self.model is None:
            warn("⚠ No brain model available — LocalBrain will use echo mode.")
        if verbose:
            info("🧠 LocalBrain initialized.")

    # --------------------------------------------------------
    # Observation: collect telemetry snapshot
    # --------------------------------------------------------

    def _observe(self) -> Dict[str, Any]:
        try:
            # Ask runtime for last state telemetry
            tele = self.engine.telemetry(self.engine.background) if hasattr(self.engine, "telemetry") else {}
        except Exception as e:
            warn(f"Telemetry read failed: {e}")
            tele = {}
        cfg = getattr(self.engine, "background", None)
        config = {
            "lambda": getattr(cfg, "lambda_", None),
            "D": getattr(cfg, "D", None),
            "phi_inf": getattr(cfg, "phi_inf", None),
            "dt": getattr(self.engine, "dt", None),
        }
        return {"telemetry": tele, "config": config}

    # --------------------------------------------------------
    # Reasoning: ask the model
    # --------------------------------------------------------

    def _query_brain(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        msg = _BASE_PROMPT + "\nObservation:\n" + json.dumps(observation, indent=2)
        if self.model is None:
            debug("Echo mode (no brain).")
            return {"action": "hold", "comment": "no brain available"}

        try:
            reply = self.model.chat([{"role": "user", "content": msg}])
            txt = reply.text if hasattr(reply, "text") else str(reply)
            start = txt.find("{")
            end = txt.rfind("}")
            if start != -1 and end != -1:
                return json.loads(txt[start:end + 1])
            warn("Brain reply not JSON-parsable; ignoring.")
        except Exception as e:
            warn(f"Brain query failed: {e}")
        return {"action": "hold", "comment": "error or invalid reply"}

    # --------------------------------------------------------
    # Act: apply control
    # --------------------------------------------------------

    def _act(self, decision: Dict[str, Any]) -> None:
        act = (decision.get("action") or "").lower()
        params = decision.get("params") or {}
        comment = decision.get("comment", "")
        if act == "adjust" and isinstance(params, dict):
            lam = params.get("lambda") or params.get("lambda_")
            D = params.get("D")
            phi_inf = params.get("phi_inf")
            if self.verbose:
                info(f"Applying λ={lam}  D={D}  Φ∞={phi_inf}  — {comment}")
            if hasattr(self.engine, "apply_control"):
                self.engine.apply_control(beta=lam, gamma=D, clamp=phi_inf)
        elif act == "hold":
            if self.verbose:
                debug(f"Holding parameters — {comment}")
        else:
            warn(f"Unknown action '{act}'; ignoring.")

    # --------------------------------------------------------
    # Step loop
    # --------------------------------------------------------

    def step(self) -> Dict[str, Any]:
        """Run one full reasoning step (observe → reason → act)."""
        obs = self._observe()
        decision = self._query_brain(obs)
        self._act(decision)
        return decision

    # --------------------------------------------------------
    # Teaching examples
    # --------------------------------------------------------

    @staticmethod
    def teach() -> Dict[str, Any]:
        return {
            "examples": [
                {
                    "telemetry": {"variance": 0.01, "coherence": 0.98},
                    "config": {"lambda": 0.3, "D": 0.2},
                    "expected": {"action": "hold", "comment": "stable state"},
                },
                {
                    "telemetry": {"variance": 0.1, "coherence": 0.7},
                    "config": {"lambda": 0.2, "D": 0.1},
                    "expected": {
                        "action": "adjust",
                        "params": {"D": 0.18},
                        "comment": "increase diffusion",
                    },
                },
            ]
        }


# ============================================================
# CLI Demo
# ============================================================

if __name__ == "__main__":
    from ..utils.config import Config
    cfg = Config()
    eng = Engine(cfg.background, cfg.device if hasattr(cfg, "device") else None)
    brain = LocalBrain(eng)
    for i in range(3):
        print(f"\n— Step {i+1} —")
        decision = brain.step()
        print("Brain decision:", json.dumps(decision, indent=2))
        time.sleep(1.5)
