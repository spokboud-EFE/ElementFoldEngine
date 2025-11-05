# ElementFold · experience/adapters/resonator.py
# ──────────────────────────────────────────────────────────────────────────────
# Minimal “resonant hardware” adapter that exposes a unified telemetry → control
# loop through the Studio. Registers under two names:
#   • "resonator"      — generic resonant (electrical / mechanical) simulator/driver
#   • "interferometer" — photonic flavor alias with identical contract
#
# Contract with Studio:
#   adapter_factory() -> runner
#   runner(model, prompt, style) -> str
# The runner keeps internal state (driver + RungController) across calls so
# `/mod resonator` stays interactive.
#
# Commands (typed as the “text” in Studio when /mod resonator is active):
#   help                    — show this cheat‑sheet
#   status                  — print controller + driver snapshot
#   init δ=0.5              — reset session and set δ⋆ (click size)
#   hold                    — lock the nearest rung (intent=hold)
#   step up [N]             — cross upward N clicks (default 1)
#   step down [N]           — cross downward N clicks (default 1)
#   seek k=<K>              — seek to absolute rung index K
#   delta <value>           — change δ⋆ on the fly
#   tick [N]                — run N control ticks (default 1) with live telemetry
#   driver sim|null         — switch between a simulator and a no‑op sink
#
# Telemetry keys expected from drivers:
#   {"delta"|"δ⋆", "kappa"|"κ", "p_half"|"p½", "x_mean"}
#
# Control dictionary returned to drivers:
#   {"beta": float, "gamma": float, "clamp": float}
#
# Design aims: simple, readable, forgiving. Hardware folks can drop in a real
# driver as long as it honors read_telemetry/apply_control contracts.
from __future__ import annotations

from typing import Dict, Optional, Callable, Tuple
import re
import math

from elementfold.rung_controller import RungController, RungIntent
from elementfold.control import Supervisor
from .base import AdapterRegistry


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight drivers
# ──────────────────────────────────────────────────────────────────────────────

class _BaseDriver:
    """Minimal driver API used by the adapter."""

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
    Tiny toy resonator:
      • x_mean evolves like a damped integrator of the 'drive'
            u = (β − 1) − 0.8·(γ − 0.5)
      • p½ (pronounced “p‑half”) peaks near half‑steps; κ is high near rungs.
    Intentionally simple — just enough to exercise the control loop.
    """
    def __init__(self):
        self.x = 0.0  # position in the hidden coordinate X
        self.v = 0.0  # velocity-ish internal state
        self.last_ctrl = {"beta": 1.0, "gamma": 0.5, "clamp": 5.0}

    def _fold(self, x: float, delta: float) -> Tuple[int, float]:
        # Nearest rung index and residual
        k = int(math.floor((x / delta) + 0.5))
        r = x - k * delta
        return k, r

    def read_telemetry(self, delta: float) -> Dict[str, float]:
        k, r = self._fold(self.x, delta)
        # Barrier nearness proxy: 0 at rung center, 1 near half‑step
        phalf = min(1.0, abs(r) / (0.5 * delta)) if delta > 0 else 0.0
        # Coherence proxy: high near rung, low near barrier
        kappa = max(0.0, 1.0 - phalf)
        return {
            "δ⋆": float(delta),
            "delta": float(delta),
            "κ": float(kappa),
            "kappa": float(kappa),
            "p½": float(phalf),
            "p_half": float(phalf),
            "x_mean": float(self.x),
        }

    def apply_control(self, beta: float, gamma: float, clamp: float) -> None:
        # Simple damped update; clamp acts as a max “kick” scale
        u = (beta - 1.0) - 0.8 * (gamma - 0.5)
        # Integrate with light damping toward nearest rung to simulate capture
        kick = max(-1.0, min(1.0, u)) * (0.06 + 0.01 * clamp)
        self.v = 0.85 * self.v + kick
        self.x += self.v
        self.last_ctrl = {"beta": beta, "gamma": gamma, "clamp": clamp}

    def snapshot(self) -> Dict[str, float]:
        out = {"x": self.x, "v": self.v}
        out.update({f"ctrl_{k}": v for k, v in self.last_ctrl.items()})
        return out


class NullDriver(_BaseDriver):
    """No‑op sink; useful to sanity‑check the adapter without a simulator."""
    def read_telemetry(self, delta: float) -> Dict[str, float]:
        return {"δ⋆": float(delta), "delta": float(delta), "κ": 0.0, "p½": 0.0, "x_mean": 0.0}

    def apply_control(self, beta: float, gamma: float, clamp: float) -> None:
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Accepts: help | status | init | hold | step [up|down] [N] | seek ... | delta ... | tick ... | driver ...
_CMD_RE = re.compile(
    r'^\s*(?P<cmd>help|status|init|hold|step(?:\s+(?:up|down))?|seek|delta|tick|driver)\b(?P<rest>.*)$',
    re.IGNORECASE,
)

def _parse_int(s: str, default: int = 1) -> int:
    try:
        return int(s.strip())
    except Exception:
        return default

def _parse_float_after_eq(s: str, key: str, default: Optional[float]) -> Optional[float]:
    # matches key=<float>
    m = re.search(rf'{re.escape(key)}\s*=\s*([-+]?(\d+(\.\d*)?|\.\d+))', s)
    return float(m.group(1)) if m else default

def _parse_first_float(s: str) -> Optional[float]:
    m = re.search(r'([-+]?(\d+(\.\d*)?|\.\d+))', s)
    return float(m.group(1)) if m else None

def _parse_seek_target(rest: str) -> Optional[int]:
    """
    Accept a few friendly forms:
      seek k=7
      seek to 7
      seek 7
    """
    m = re.search(r'k\s*=\s*(-?\d+)', rest, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'\bto\s+(-?\d+)\b', rest, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    m = re.search(r'\b(-?\d+)\b', rest)
    return int(m.group(1)) if m else None


# ──────────────────────────────────────────────────────────────────────────────
# Adapter factories
# ──────────────────────────────────────────────────────────────────────────────

def _make_runner(kind: str) -> Callable:
    """
    Build a stateful runner. 'kind' is a flavor tag (e.g., 'resonator', 'interferometer').
    Keeps driver + supervisor + rung controller alive across invocations.
    """
    driver: _BaseDriver = SimResonatorDriver()  # default simulator
    sup = Supervisor()
    delta = 0.5
    rung = RungController(delta=delta)          # starts in intent=STABILIZE

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
            lines.append(
                f"tick {i+1:02d} • phase={st.get('phase','?'):<9} "
                f"κ={tele.get('κ', tele.get('kappa', 0.0)):.3f} "
                f"p½={tele.get('p½', tele.get('p_half', 0.0)):.3f} "
                f"ctrl={{β:{ctrl_out['beta']:.2f}, γ:{ctrl_out['gamma']:.2f}, ⛔:{ctrl_out['clamp']:.1f}}}"
            )
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
                "  help                    — this help\n"
                "  status                  — controller + driver snapshot\n"
                "  init δ=<value>          — reset session and set δ⋆\n"
                "  hold                    — keep the nearest rung centered\n"
                "  step up [N]             — cross upward N clicks\n"
                "  step down [N]           — cross downward N clicks\n"
                "  seek k=<K>              — seek to absolute rung index K\n"
                "  delta <value>           — change δ⋆ on the fly\n"
                "  tick [N]                — run N control ticks\n"
                "  driver sim|null         — switch driver\n"
            )

        if cmd == "status":
            st = rung.status()                  # controller snapshot
            tele = driver.read_telemetry(delta) # non‑destructive read for display
            snap = driver.snapshot()
            # Prefer the new controller keys; fall back gracefully
            intent = st.get("intent", "?")
            k_tgt = st.get("k_target", st.get("target_k", None))
            band = st.get("band", "?")
            phase = st.get("phase", "?")
            dwell = st.get("dwell", 0)
            plan = st.get("plan")  # may be None
            return (
                "status:\n"
                f"  intent={intent}  k_target={k_tgt}  phase={phase}  dwell={dwell}\n"
                f"  δ⋆={st.get('delta', delta)}  band={band}\n"
                f"  plan={plan}\n"
                f"  tele={{κ:{tele.get('κ',0):.3f}, p½:{tele.get('p½',0):.3f}, x:{tele.get('x_mean',0):.3f}}}\n"
                f"  driver={driver.__class__.__name__} • {snap}"
            )

        if cmd == "init":
            # Accept init δ=... or init delta=...
            new_delta = _parse_float_after_eq(rest, "δ", None)
            if new_delta is None:
                new_delta = _parse_float_after_eq(rest, "delta", None)
            delta = float(new_delta or delta)
            sup = Supervisor()
            rung = RungController(delta=delta)
            driver = SimResonatorDriver()
            return f"initialized: δ⋆={delta}"

        if cmd == "hold":
            rung.hold()
            return "intent: hold\n" + _tick(1)

        if cmd == "seek":
            k = _parse_seek_target(rest)
            if k is None:
                return "usage: seek k=<int>   (examples: 'seek k=7', 'seek to 3')"
            rung.seek_to(k)
            return f"intent: seek → k={k}\n" + _tick(4)

        if cmd.startswith("step"):
            up = "down" not in rest.lower()
            # Try to parse trailing number in “step up 3”; default to 1
            parts = rest.split()
            n = 1
            if parts:
                try:
                    n = int(parts[-1])
                except Exception:
                    n = 1
            if up:
                rung.set_intent("step_up", steps=max(1, n))
            else:
                rung.set_intent("step_down", steps=max(1, n))
            return f"intent: step_{'up' if up else 'down'} ×{n}\n" + _tick(max(3, n * 2))

        if cmd == "delta":
            # Accept “delta 0.4” or “delta δ=0.4”
            val = _parse_first_float(rest)
            if val is None:
                val = _parse_float_after_eq(rest, "δ", None)
            if val is None:
                return "usage: delta <float>"
            delta = float(val)
            rung.delta = delta  # controller follows runtime δ⋆
            return f"δ⋆ set to {delta}"

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
