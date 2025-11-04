# ElementFold · experience/adapters/resonator.py
# Minimal "resonant hardware" adapters that expose a unified telemetry → control loop
# through the Studio. They register under two names:
#   • "resonator"      — generic resonant mode (electrical / mechanical) simulator/driver
#   • "interferometer" — photonic flavor alias with identical contract
#
# Contract with Studio:
#   adapter_factory() -> runner
#   runner(model, prompt, style) -> str
# The runner keeps internal state (driver + RungController) across calls so /mod ... works interactively.
#
# Commands (typed as the "text" in Studio when /mod resonator is active):
#   help                — show a short cheat‑sheet
#   status              — print controller + driver state
#   init δ=0.5          — reset session and set δ⋆ (click size)
#   hold                — lock the nearest rung
#   step up [N]         — cross upward N clicks (default 1)
#   step down [N]       — cross downward N clicks (default 1)
#   delta <value>       — change δ⋆ on the fly
#   tick [N]            — run N control ticks (default 1) using live/sim telemetry
#   driver sim|null     — switch between a simulator and a no‑op sink
#
# Telemetry keys expected from drivers:
#    {"delta"| "δ⋆", "kappa"| "κ", "p_half"| "p½", "x_mean"}
#
# Control dictionary returned to drivers:
#    {"beta": float, "gamma": float, "clamp": float}
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Callable, Tuple
import re
import math
import time

from elementfold.rung_controller import RungController, RungIntent
from elementfold.control import Supervisor
from .base import AdapterRegistry


# ————————————————————————————————————————————————————————————————
# Lightweight drivers
# ————————————————————————————————————————————————————————————————

class _BaseDriver:
    """Minimal driver API used by the adapters."""

    def read_telemetry(self, delta: float) -> Dict[str, float]:
        raise NotImplementedError

    def apply_control(self, beta: float, gamma: float, clamp: float) -> None:
        """Hardware would act here. Simulation just updates its internal state."""
        raise NotImplementedError

    def snapshot(self) -> Dict[str, float]:
        """Optional debug snapshot for status printing."""
        return {}


class SimResonatorDriver(_BaseDriver):
    """
    Toy resonator:
      • x_mean evolves like a damped integrator of the 'drive' u = (β - 1) - 0.8*(γ - 0.5).
      • p½ peaks near half‑steps; κ is high near rungs.
    This is intentionally simple — it exercises the control loop without hardware.
    """
    def __init__(self):
        self.x = 0.0
        self.v = 0.0
        self.last_ctrl = {"beta": 1.0, "gamma": 0.5, "clamp": 5.0}

    def _fold(self, x: float, delta: float) -> Tuple[int, float]:
        # nearest rung index and residual
        k = int(math.floor((x / delta) + 0.5))
        r = x - k * delta
        return k, r

    def read_telemetry(self, delta: float) -> Dict[str, float]:
        # Compute residual normalized to half‑step
        k, r = self._fold(self.x, delta)
        phalf = min(1.0, abs(r) / (0.5 * delta)) if delta > 0 else 0.0
        # Coherence proxy: high near rung, low near barrier
        kappa = max(0.0, 1.0 - phalf)
        return {"δ⋆": float(delta), "κ": float(kappa), "kappa": float(kappa), "p½": float(phalf), "p_half": float(phalf), "x_mean": float(self.x)}

    def apply_control(self, beta: float, gamma: float, clamp: float) -> None:
        # Simple damped update; clamp acts as a max "kick" scale
        u = (beta - 1.0) - 0.8 * (gamma - 0.5)
        # Integrate with light damping toward nearest rung to simulate capture
        kick = max(-1.0, min(1.0, u)) * (0.06 + 0.01 * clamp)
        self.v = 0.85 * self.v + kick
        self.x += self.v
        self.last_ctrl = {"beta": beta, "gamma": gamma, "clamp": clamp}

    def snapshot(self) -> Dict[str, float]:
        out = dict(x=self.x, v=self.v)
        out.update({f"ctrl_{k}": v for k, v in self.last_ctrl.items()})
        return out


class NullDriver(_BaseDriver):
    """No‑op sink; useful to sanity‑check the adapter without a simulator."""
    def read_telemetry(self, delta: float) -> Dict[str, float]:
        return {"δ⋆": float(delta), "κ": 0.0, "p½": 0.0, "x_mean": 0.0}

    def apply_control(self, beta: float, gamma: float, clamp: float) -> None:
        pass


# ————————————————————————————————————————————————————————————————
# Helpers
# ————————————————————————————————————————————————————————————————

_CMD_RE = re.compile(r'^\s*(?P<cmd>help|status|init|hold|step(?:\s+(?:up|down))?|delta|tick|driver)\b(?P<rest>.*)$', re.I)

def _parse_int(s: str, default: int = 1) -> int:
    try:
        return int(s.strip())
    except Exception:
        return default

def _parse_float_after_eq(s: str, key: str, default: Optional[float]) -> Optional[float]:
    m = re.search(rf'{re.escape(key)}\s*=\s*([-+]?(\d+(\.\d*)?|\.\d+))', s)
    return float(m.group(1)) if m else default


# ————————————————————————————————————————————————————————————————
# Adapter factories
# ————————————————————————————————————————————————————————————————

def _make_runner(kind: str) -> Callable:
    """
    Build a stateful runner. 'kind' is a flavor tag (e.g., 'resonator', 'interferometer').
    """
    driver: _BaseDriver = SimResonatorDriver()  # default simulator
    sup = Supervisor()
    delta = 0.5
    rung = RungController(delta=delta)

    def _tick(n: int = 1) -> str:
        lines = []
        nonlocal delta, driver, sup, rung
        for i in range(max(1, n)):
            tele = driver.read_telemetry(delta)
            # Prefer runtime δ⋆ if provided by telemetry
            delta = float(tele.get("δ⋆", tele.get("delta", delta)))
            # Supervisor suggests a base control; RungController blends toward its target
            ctrl_sup = sup.update(tele)
            ctrl_out = rung.update(tele, ctrl_sup)
            driver.apply_control(**ctrl_out)
            st = rung.status()
            lines.append(f"tick {i+1:02d} • phase={st['phase']:<9} κ={tele.get('κ',0):.3f} p½={tele.get('p½',0):.3f} "
                         f"ctrl={{β:{ctrl_out['beta']:.2f}, γ:{ctrl_out['gamma']:.2f}, ⛔:{ctrl_out['clamp']:.1f}}}")
        return "\n".join(lines)

    def run(_model, prompt: str, _style=None) -> str:
        nonlocal delta, driver, sup, rung
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = "help"

        m = _CMD_RE.match(prompt)
        if not m:
            # Default action: run one tick
            return _tick(1)

        cmd = m.group("cmd").lower()
        rest = (m.group("rest") or "").strip()

        if cmd == "help":
            return (
                f"[{kind}] commands:\n"
                "  help                — this help\n"
                "  status              — controller + driver snapshot\n"
                "  init δ=<value>      — reset session and set δ⋆\n"
                "  hold                — keep the nearest rung centered\n"
                "  step up [N]         — cross upward N clicks\n"
                "  step down [N]       — cross downward N clicks\n"
                "  delta <value>       — change δ⋆ on the fly\n"
                "  tick [N]            — run N control ticks\n"
                "  driver sim|null     — switch driver\n"
            )

        if cmd == "status":
            st = rung.status()
            snap = driver.snapshot()
            return ("status:"
                    f" mode={st.get('mode', '?')}"
                    f" phase={st.get('phase', '?')}"
                    f" dwell={st.get('dwell', 0)}"
                    f" δ⋆={st.get('delta', delta)}"
                    f" done_steps={st.get('remaining_steps', 'n/a')}\n"
                    f" driver={driver.__class__.__name__} • {snap}")

        if cmd == "init":
            new_delta = _parse_float_after_eq(rest, "δ", None)
            if new_delta is None:
                new_delta = _parse_float_after_eq(rest, "delta", None)
            delta = float(new_delta or delta)
            sup = Supervisor()
            rung = RungController(delta=delta)
            driver = SimResonatorDriver()
            return f"initialized: δ⋆={delta}"

        if cmd == "hold":
            rung.set_intent("hold", 0)
            return "intent: hold\n" + _tick(1)

        if cmd.startswith("step"):
            up = "down" not in rest.lower()
            n = _parse_int(rest.split()[-1] if rest else "", 1)
            rung.set_intent("step_up" if up else "step_down", steps=n)
            return f"intent: step_{'up' if up else 'down'} ×{n}\n" + _tick(max(3, n*2))

        if cmd == "delta":
            try:
                delta = float(rest.strip())
                rung.delta = delta
                return f"δ⋆ set to {delta}"
            except Exception:
                return "usage: delta <float>"

        if cmd == "tick":
            n = _parse_int(rest, 1)
            return _tick(n)

        if cmd == "driver":
            choice = (rest or "").strip().lower()
            if choice == "sim":
                driver = SimResonatorDriver()
                return "driver → SimResonatorDriver"
            if choice == "null":
                driver = NullDriver()
                return "driver → NullDriver"
            return "driver choices: sim|null"

        # Fallback: one tick
        return _tick(1)

    return run


@AdapterRegistry.register_fn("resonator")
def make_resonator_adapter():
    return _make_runner("resonator")


@AdapterRegistry.register_fn("interferometer")
def make_interferometer_adapter():
    # Alias flavor — identical contract; different name helps discoverability
    return _make_runner("interferometer")