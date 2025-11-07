# ElementFold · experience/local_brain.py
# ============================================================
# Runtime-Aware Brain Interface
# ------------------------------------------------------------
# Purpose:
#   • Expose a JSON reasoning loop between an LLM and the ElementFold engine.
#   • The model observes telemetry and config, proposes parameter changes,
#     and receives feedback from the runtime.
#
# Public API:
#   - LocalBrain(engine)
#   - step()      → runs one observe→reason→act cycle
#   - teach()     → loads a few canonical examples (prompt scaffolds)
#
# Notes:
#   • This module is runtime-safe: no writes outside working dir.
#   • Works with any background_model implementing .chat(messages=[...]).
# ============================================================

from __future__ import annotations
import json, time
from typing import Any, Dict, Optional

try:
    from ..core.runtime import Engine
    from ..utils.display import info, warn, success, debug
    from ..core.telemetry import measure as measure_telemetry
except Exception:
    raise RuntimeError("LocalBrain requires core.runtime and utils.display")

# Optional model layer (can be local or remote)
try:
    from .background_model import BackgroundModel
except Exception:  # pragma: no cover
    BackgroundModel = None


# ============================================================
# Default prompt template
# ============================================================

_BASE_PROMPT = """
You are a runtime controller for the ElementFold simulation engine.

You will receive:
  1. The current telemetry of the field (variance, coherence, etc.).
  2. The current configuration (lambda, D, phi_inf, dt).
Your task is to reason and respond with *one JSON object* indicating
what to do next. Valid actions:

  {"action":"adjust","params":{"lambda":0.3,"D":0.2},"comment":"Reduce lambda to slow relaxation."}
  {"action":"hold","comment":"Stable; no change."}

Do not output text before or after the JSON.
"""

# ============================================================
# LocalBrain class
# ============================================================

class LocalBrain:
    """
    Bridge between Engine (simulation) and a text model (brain).

    Steps:
      1. Collect telemetry from engine.model (variance, etc.).
      2. Query the background model with telemetry JSON.
      3. Parse the JSON reply.
      4. Apply parameter updates via engine.apply_control().
    """

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
            _, X = self.engine.infer(x=None, strategy="greedy")
            tele = measure_telemetry(X, self.engine.cfg.delta)
        except Exception:
            tele = {"variance": None, "coherence": None}
        cfg = {
            "lambda": getattr(self.engine.cfg, "lambda_", None),
            "D": getattr(self.engine.cfg, "D", None),
            "phi_inf": getattr(self.engine.cfg, "phi_inf", None),
            "dt": getattr(self.engine.cfg, "dt", None),
        }
        return {"telemetry": tele, "config": cfg}

    # --------------------------------------------------------
    # Reasoning: ask the model
    # --------------------------------------------------------

    def _query_brain(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        msg = _BASE_PROMPT + "\nObservation:\n" + json.dumps(observation, indent=2)
        if self.model is None:
            debug("Echo mode (no model).")
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
                info(f"Applying λ={lam} D={D} Φ∞={phi_inf}  — {comment}")
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
        """
        Run one full reasoning step (observe → reason → act).
        Returns the brain's decision for logging.
        """
        obs = self._observe()
        decision = self._query_brain(obs)
        self._act(decision)
        return decision

    # --------------------------------------------------------
    # Teaching examples (optional)
    # --------------------------------------------------------

    @staticmethod
    def teach() -> Dict[str, Any]:
        """Return example observation→action pairs for prompting."""
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
                    "expected": {"action": "adjust", "params": {"D": 0.18}, "comment": "increase diffusion"},
                },
            ]
        }


# ============================================================
# CLI Demo
# ============================================================

if __name__ == "__main__":
    from ..core.runtime import Engine
    from ..utils.config import Config
    cfg = Config()
    eng = Engine(cfg)
    brain = LocalBrain(eng)
    for i in range(3):
        print(f"\n— Step {i+1} —")
        decision = brain.step()
        print("Brain decision:", json.dumps(decision, indent=2))
        time.sleep(1.5)
