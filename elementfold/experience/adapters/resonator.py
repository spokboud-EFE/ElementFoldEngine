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
#   spec                    — show the adapter contract (expected tensors, outputs)
#   status                  — print controller + driver snapshot (+ feed readiness)
#   init δ=0.5              — reset session and set δ⋆ (click size)
#   hold                    — lock the nearest rung (intent=hold)
#   step up [N]             — cross upward N clicks (default 1)
#   step down [N]           — cross downward N clicks (default 1)
#   seek k=<K>              — seek to absolute rung index K
#   delta <value>           — change δ⋆ on the fly
#   tick [N]                — run N control ticks (default 1) with live telemetry
#   pred on|off|reset       — show/hide/reset relaxation predictions (ℱ, z, e^{-2ℱ})
#   driver sim|null|live    — switch between simulator, no‑op, or live‑feed driver
#   feed clear              — (live) drop any attached feed (runner.set_feed(...) exists)
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

from typing import Dict, Optional, Callable, Tuple, Any
import re
import math

from elementfold.rung_controller import RungController
from elementfold.core.control import Supervisor

# New: adapter spec / readiness plumbing (compatible additions in base.py)
from .base import (
    AdapterRegistry,
    AdapterSpec,
    TensorSpec,
    with_spec,
)

# ──────────────────────────────────────────────────────────────────────────────
# Adapter contract (plain words)
# ──────────────────────────────────────────────────────────────────────────────
#
# We declare what a “live” resonator expects if a device is attached. We keep
# it *minimal* on purpose: a single 1‑D tensor “X” of ledger positions (δ⋆
# clicks) is enough to compute κ, p½, and x_mean. Studio (or your driver code)
# can call `runner.set_feed({"X": torch.tensor([...])})`. If no feed is present,
# the adapter happily runs in simulator mode.

try:
    import torch  # optional (only used for dtype name in pretty docs)
    _float32 = torch.float32
except Exception:  # pragma: no cover
    torch = None   # type: ignore
    _float32 = "float32"  # human-friendly fallback

_resonator_spec = AdapterSpec(
    name="resonator",
    description="Rung‑aware resonator control. Accepts live ledger values X (δ⋆ clicks) or simulates.",
    expects={
        # Rank‑1 time series of ledger seats. If you need (B,T), stream B traces one at a time.
        "X": TensorSpec(shape=(-1,), dtype=_float32, device="any",
                        doc="ledger positions (δ⋆ units); shape (T,)"),
        # (Optional) You can still change δ⋆ with the 'delta' command; not enforced here.
    },
    predicts={
        "ctrl": "β (responsiveness), γ (damping), ⛔ (clamp) per tick",
        "folds": "ℱ — cumulative relaxation folds; z = e^ℱ − 1; dim tilt A ≈ e^{−2ℱ}",
    },
    wait="allow_sim",  # if feed missing, offer simulation instead of blocking
)

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
        _, r = self._fold(self.x, delta)
        half = 0.5 * delta if delta > 0 else 1.0
        # Barrier nearness proxy: 0 at rung center, 1 near half‑step
        phalf = min(1.0, abs(r) / max(1e-9, half))
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

class LiveFeedDriver(_BaseDriver):
    """
    Live driver that *reads* telemetry from an injected feed:
        runner.set_feed({"X": torch.tensor([...])})

    We consume one sample per tick and synthesize the same telemetry keys the
    controller expects: κ, p½, x_mean. This is deliberately simple and stable.
    """
    def __init__(self, feed_ref: Dict[str, Any]):
        self._feed_ref = feed_ref
        self._i = 0
        self._last_ctrl = {"beta": 1.0, "gamma": 0.5, "clamp": 5.0}

    def _current_x(self) -> Optional[float]:
        X = self._feed_ref.get("X", None)
        if X is None:
            return None
        try:
            # numpy / torch / list — all duck‑typed via __len__/__getitem__
            if len(X) == 0:
                return None
            i = min(self._i, len(X) - 1)
            v = X[i]
            # Torch / numpy scalar → Python float
            return float(v.item()) if hasattr(v, "item") else float(v)
        except Exception:
            return None

    def read_telemetry(self, delta: float) -> Dict[str, float]:
        x = self._current_x()
        if x is None:
            # Not ready; return a neutral telemetry snapshot
            return {"δ⋆": float(delta), "delta": float(delta), "κ": 0.0, "p½": 0.0, "x_mean": 0.0}
        # Compute residual and simple κ/p½ proxies (same as simulator)
        k = int(math.floor((x / delta) + 0.5)) if delta > 0 else 0
        r = x - k * delta
        half = 0.5 * delta if delta > 0 else 1.0
        phalf = min(1.0, abs(r) / max(1e-9, half))
        kappa = max(0.0, 1.0 - phalf)
        return {
            "δ⋆": float(delta),
            "delta": float(delta),
            "κ": float(kappa),
            "kappa": float(kappa),
            "p½": float(phalf),
            "p_half": float(phalf),
            "x_mean": float(x),
        }

    def apply_control(self, beta: float, gamma: float, clamp: float) -> None:
        # Live mode does not “move” hardware; we just advance the cursor along X.
        self._last_ctrl = {"beta": beta, "gamma": gamma, "clamp": clamp}
        self._i += 1  # consume one sample per tick

    def snapshot(self) -> Dict[str, float]:
        out = {"i": self._i}
        out.update({f"ctrl_{k}": v for k, v in self._last_ctrl.items()})
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

# Accepts: help | spec | status | init | hold | step | seek | delta | tick | driver | feed
_CMD_RE = re.compile(
    r'^\s*(?P<cmd>help|spec|status|init|hold|seek|delta|tick|driver|step|feed)\b(?P<rest>.*)$',
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
# Adapter factory and runner
# ──────────────────────────────────────────────────────────────────────────────

def _make_runner(kind: str) -> Callable:
    """
    Build a stateful runner. 'kind' is a flavor tag (e.g., 'resonator', 'interferometer').
    Keeps driver + supervisor + rung controller alive across invocations.
    """

    # Runtime state
    driver: _BaseDriver = SimResonatorDriver()  # default simulator
    sup = Supervisor()
    delta = 0.5
    rung = RungController(delta=delta)          # starts in intent=STABILIZE

    # Live‑feed store (for driver=live); Studio can set via runner.set_feed(...)
    feed: Dict[str, Any] = {}

    # Relaxation predictions (ℱ clock). We keep it deliberately simple:
    #   η ≈ η0 + a·p½ + b·(1−κ), then ℱ ← ℱ + η  (per tick)
    # Use gentle coefficients; Studio can reset via 'pred reset'.
    F = 0.0
    show_pred = True
    eta0, a_p, b_k = 0.005, 0.040, 0.020

    def _update_relaxation(tele: Dict[str, float]) -> Tuple[float, float, float]:
        nonlocal F
        kappa = float(tele.get("κ", tele.get("kappa", 0.0)))
        phalf = float(tele.get("p½", tele.get("p_half", 0.0)))
        eta = max(0.0, eta0 + a_p * phalf + b_k * (1.0 - kappa))
        F = max(0.0, F + eta)  # monotone clock
        z = math.exp(F) - 1.0
        atten = math.exp(-2.0 * F)
        return (F, z, atten)

    def _pred_suffix(tele: Dict[str, float]) -> str:
        if not show_pred:
            return ""
        Fv, zv, av = _update_relaxation(tele)
        return f"  ℱ={Fv:.3f}  z={zv:.3f}  A≈e^(-2ℱ)={av:.3f}"

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
                + _pred_suffix(tele)
            )
        return "\n".join(lines)

    def _pretty_spec() -> str:
        spec = AdapterRegistry.spec("resonator") or _resonator_spec
        return "contract:\n" + spec.pretty()

    def _readiness_summary() -> str:
        # Validate current feed (if any) against the spec
        rep = AdapterRegistry.validate("resonator", feed)
        # Short UX string
        if rep.ok:
            return "feed: ✓ ready"
        msg = "feed: wait — " + rep.summary
        return msg

    def run(_model, prompt: str, _style=None) -> str:
        nonlocal delta, driver, sup, rung, F, show_pred
        if not isinstance(prompt, str) or not prompt.strip():
            prompt = "help"

        m = _CMD_RE.match(prompt)
        if not m:
            # Default action: run one tick
            return _tick(1)

        cmd = (m.group("cmd") or "").lower()
        rest = (m.group("rest") or "").strip()

        if cmd == "help":
            return (
                f"[{kind}] commands:\n"
                "  help                    — this help\n"
                "  spec                    — show expected inputs / predicted outputs\n"
                "  status                  — controller + driver snapshot\n"
                "  init δ=<value>          — reset session and set δ⋆\n"
                "  hold                    — keep the nearest rung centered\n"
                "  step up [N]             — cross upward N clicks\n"
                "  step down [N]           — cross downward N clicks\n"
                "  seek k=<K>              — seek to absolute rung index K\n"
                "  delta <value>           — change δ⋆ on the fly\n"
                "  tick [N]                — run N control ticks\n"
                "  pred on|off|reset       — show/hide/reset relaxation predictions\n"
                "  driver sim|null|live    — switch driver\n"
                "  feed clear              — clear any attached live feed\n"
            )

        if cmd == "spec":
            return _pretty_spec()

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
            feed_line = _readiness_summary()
            return (
                "status:\n"
                f"  intent={intent}  k_target={k_tgt}  phase={phase}  dwell={dwell}\n"
                f"  δ⋆={st.get('delta', delta)}  band={band}\n"
                f"  plan={plan}\n"
                f"  {feed_line}\n"
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
            F = 0.0  # reset predictions clock
            # Keep current driver choice, but reset its internal state if simulation
            if isinstance(driver, SimResonatorDriver):
                driver = SimResonatorDriver()
            elif isinstance(driver, LiveFeedDriver):
                driver = LiveFeedDriver(feed)
            else:
                driver = NullDriver()
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

        if cmd == "step":
            # Accept "step up [N]" or "step down [N]"
            up = "down" not in rest.lower()
            # Extract trailing integer if present; default 1
            n = 1
            mnum = re.search(r'(-?\d+)\s*$', rest)
            if mnum:
                try:
                    n = max(1, int(mnum.group(1)))
                except Exception:
                    n = 1
            if up:
                rung.set_intent("step_up", steps=n)
            else:
                rung.set_intent("step_down", steps=n)
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
            F = 0.0             # restart predictions for new scale
            return f"δ⋆ set to {delta}"

        if cmd == "tick":
            n = _parse_int(rest, 1)
            return _tick(n)

        if cmd == "pred":
            s = rest.lower()
            if "reset" in s:
                F = 0.0
                return "predictions: ℱ reset to 0"
            if "off" in s:
                show_pred = False
                return "predictions: off"
            if "on" in s or s == "":
                show_pred = True
                return "predictions: on"
            return "usage: pred on|off|reset"

        if cmd == "driver":
            choice = (rest or "").strip().lower()
            if choice == "sim":
                driver = SimResonatorDriver()
                return "driver → SimResonatorDriver"
            if choice == "null":
                driver = NullDriver()
                return "driver → NullDriver"
            if choice == "live":
                driver = LiveFeedDriver(feed)
                return "driver → LiveFeedDriver (waiting on runner.set_feed({...}))"
            return "driver choices: sim|null|live"

        if cmd == "feed":
            if "clear" in rest.lower():
                feed.clear()
                return "feed: cleared"
            return "feed: this command only supports 'feed clear' here; inject data via runner.set_feed({...})"

        # Fallback: one tick
        return _tick(1)

    # Attach handy hooks for Studio / scripts (no change to runner signature)
    def _set_feed(data: Dict[str, Any]) -> str:
        """Programmatic injection point for live driver feeds (e.g., {'X': tensor})."""
        nonlocal feed
        if not isinstance(data, dict):
            return "set_feed: expected a dict like {'X': tensor}"
        feed = dict(data)  # shallow copy
        rep = AdapterRegistry.validate("resonator", feed)
        return ("feed: ✓ ready" if rep.ok else "feed: wait — " + rep.summary)

    def _get_feed() -> Dict[str, Any]:
        return dict(feed)

    def _simulate(T: int = 512, noise: float = 0.02, drift: float = 0.1) -> Dict[str, Any]:
        """
        Produce a simple synthetic ledger series X for experiments.
        X wanders across rungs with small noise and drift to exercise crossings.
        """
        try:
            import numpy as np
        except Exception:
            np = None
        if np is None:
            # minimal fallback without numpy
            Xs = []
            x = 0.0
            for t in range(int(T)):
                x += drift + noise * (0.5 - (t % 2))
                Xs.append(x)
            return {"X": Xs}
        rng = np.random.default_rng(0xE1F0LD)
        X = np.zeros(int(T), dtype=float)
        v = 0.0
        for t in range(int(T)):
            # light random walk with occasional pushes through half‑steps
            a = drift + 0.03 * rng.standard_normal()
            v = 0.95 * v + a
            X[t] = (X[t - 1] if t else 0.0) + v + noise * rng.standard_normal()
        return {"X": X}

    # Decorate runner object with helpful attributes/methods
    run.__adapter_spec__ = _resonator_spec  # type: ignore[attr-defined]
    run.set_feed = _set_feed                # type: ignore[attr-defined]
    run.get_feed = _get_feed                # type: ignore[attr-defined]
    run.simulate = _simulate                # type: ignore[attr-defined]
    return run


@AdapterRegistry.register_fn("resonator")
@with_spec(_resonator_spec)
def make_resonator_adapter():
    return _make_runner("resonator")


@AdapterRegistry.register_fn("interferometer")
@with_spec(AdapterSpec(
    name="interferometer",
    description="Photonic alias of the resonator contract.",
    expects=_resonator_spec.expects,
    predicts=_resonator_spec.predicts,
    wait=_resonator_spec.wait,
))
def make_interferometer_adapter():
    # Alias flavor — identical contract; different name helps discoverability
    return _make_runner("interferometer")
