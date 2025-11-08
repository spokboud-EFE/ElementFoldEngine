"""
core/control/runtime.py — Heartbeat of ElementFold cores 🫀

This runtime is the live clock of a core: it owns time t, phase φ, and
controls how each relaxation tick (δ★) advances the system.

It does not know physics; it only calls a step function that applies
the active physical law (λ relax-rate, D smoothing, etc.).  The runtime
measures rhythm and tells the Factory when to breathe.

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• Every core has a pulse (t, dt, phase).
• Each step is a click of δ★ — the universal increment.
• The runtime keeps that pulse steady, even under forcing.
• Telemetry tells the Studio how the core is breathing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Union

# ---------------------------------------------------------------------- #
# Typing aliases
# ---------------------------------------------------------------------- #
StepFn = Callable[["RuntimeState", float], Optional["RuntimeState"]]
HookFn = Callable[["RuntimeState"], None]


# ====================================================================== #
# 📦 RuntimeState — container for all evolving quantities
# ====================================================================== #
@dataclass
class RuntimeState:
    """
    Mutable snapshot of a single ElementFold core.

    Attributes
    ----------
    t : float
        Current time in relaxation units.
    dt : float
        Step size (Δt) for the integrator.
    mode : str
        'shaping' → equilibrium-seeking updates,
        'forcing' → deliberate impulses / depins.
    phase : float
        Coherence phase, used by Synchronizer.
    fields : Dict[str, Any]
        Named tensors or scalars (Φ, ∇Φ, ℱ, etc.).
    params : Dict[str, Any]
        Model/control parameters (λ, D, σₓ, ...).
    kappa : float
        Coherence factor (for Coupler entanglement).
    """

    t: float = 0.0
    dt: float = 1e-3
    mode: str = "shaping"
    phase: float = 0.0
    fields: Dict[str, Any] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    kappa: float = 1.0

    def shallow_copy(self) -> "RuntimeState":
        """Return a shallow structural copy (safe for inspectors)."""
        return RuntimeState(
            t=self.t,
            dt=self.dt,
            mode=self.mode,
            phase=self.phase,
            fields=dict(self.fields),
            params=dict(self.params),
            kappa=self.kappa,
        )


# ====================================================================== #
# ⚙️  Runtime — the heartbeat and conductor for one core
# ====================================================================== #
class Runtime:
    """
    Drives the state of a single core through time.

    It calls `step_fn(state, dt)` once per tick and updates the
    coherence phase.  Telemetry is best-effort: no blocking, no noise.
    """

    def __init__(
        self,
        step_fn: Optional[StepFn] = None,
        *,
        init_state: Optional[RuntimeState] = None,
        epoch_interval: int = 100,
        telemetry: Optional[Any] = None,
    ) -> None:
        self._step_fn: StepFn = step_fn or self._noop_step
        self.state: RuntimeState = init_state or RuntimeState()
        self.epoch_interval: int = max(1, int(epoch_interval))
        self.telemetry = telemetry

        # External synchronization handle (set by Synchronizer)
        self.phase: float = self.state.phase

        # Hook table
        self._hooks: Dict[str, HookFn] = {}

        # Control flags
        self._running: bool = False

    # ------------------------------------------------------------------ #
    # 🪝 Hook management
    # ------------------------------------------------------------------ #
    def register_hook(self, name: str, fn: HookFn) -> None:
        """Attach a lifecycle hook: on_step | on_epoch | on_finish."""
        if name not in ("on_step", "on_epoch", "on_finish"):
            raise ValueError("Hook name must be one of: on_step, on_epoch, on_finish")
        self._hooks[name] = fn

    # ------------------------------------------------------------------ #
    # 🎚 Mode and parameter control
    # ------------------------------------------------------------------ #
    @property
    def mode(self) -> str:
        return self.state.mode

    def set_mode(self, mode: str) -> None:
        """Switch between shaping ↔ forcing."""
        if mode not in ("shaping", "forcing"):
            raise ValueError("mode must be 'shaping' or 'forcing'")
        self.state.mode = mode
        self._emit("⚙️  mode.changed", mode=mode)

    def set_params(self, **kwargs: Any) -> None:
        """Update physical/control parameters in place."""
        self.state.params.update(kwargs)
        self._emit("🧮 params.updated", params=list(kwargs.keys()))

    def set_param_map(self, params: Mapping[str, Any]) -> None:
        self.state.params.update(dict(params))
        self._emit("🧮 params.updated", params=list(params.keys()))

    def add_field(self, name: str, value: Any) -> None:
        self.state.fields[name] = value
        self._emit("🌐 field.added", field=name)

    def update_field(self, name: str, value: Any) -> None:
        if name not in self.state.fields:
            raise KeyError(f"Unknown field '{name}'")
        self.state.fields[name] = value
        self._emit("🌐 field.updated", field=name)

    def get_field(self, name: str) -> Any:
        return self.state.fields[name]

    # ------------------------------------------------------------------ #
    # ▶️ Lifecycle control
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Begin runtime loop."""
        self._running = True
        self._emit("▶️ runtime.start")

    def stop(self) -> None:
        """Stop execution and run finish hooks."""
        was_running = self._running
        self._running = False
        if was_running:
            self._emit("⏹ runtime.stop")
            self._call_hook("on_finish")

    # ------------------------------------------------------------------ #
    # ⏱ Step execution
    # ------------------------------------------------------------------ #
    def step(self, dt: Optional[float] = None) -> None:
        """
        Execute a single heartbeat step.
        - Applies the provided step function.
        - Updates t, dt, and phase.
        - Fires on_step / on_epoch hooks.
        """
        if not self._running:
            # allow single manual ticks
            self._emit("⚠️  autostep")

        step_dt = float(dt if dt is not None else self.state.dt)
        try:
            new_state = self._step_fn(self.state, step_dt)
        except (ArithmeticError, ValueError) as exc:
            self._emit("💥 step.error", error=str(exc))
            raise
        except Exception as exc:
            # unknown exception — report and stop gracefully
            self._emit("💥 step.exception", error=type(exc).__name__, msg=str(exc))
            self.stop()
            raise

        if isinstance(new_state, RuntimeState):
            self.state = new_state

        # advance internal clocks
        self.state.t += step_dt
        self.state.dt = step_dt
        self.state.phase = self.phase

        # rhythmic telemetry (every click)
        self._emit("🔂 step", t=self.state.t, dt=step_dt, phase=self.phase)

        # hooks
        self._call_hook("on_step")
        # check epoch rhythm
        epoch_index = int(round(self.state.t / step_dt))
        if epoch_index % self.epoch_interval == 0:
            self._call_hook("on_epoch")

    def run(
        self,
        *,
        steps: Optional[int] = None,
        t_end: Optional[float] = None,
        dt: Optional[float] = None,
        max_wall_seconds: Optional[float] = None,
    ) -> None:
        """
        Continuous run loop — the long breath of the core.
        Stops when step or time limits are reached.
        """
        if steps is None and t_end is None:
            raise ValueError("Provide either steps or t_end")
        if dt is not None:
            self.state.dt = float(dt)

        self.start()
        start_wall = time.perf_counter()
        n = 0
        while self._running:
            if steps is not None and n >= steps:
                break
            if t_end is not None and self.state.t >= t_end:
                break
            if max_wall_seconds is not None and (time.perf_counter() - start_wall) >= max_wall_seconds:
                break
            self.step(self.state.dt)
            n += 1
        self.stop()

    # ------------------------------------------------------------------ #
    # 📸 Snapshot & narrative diagnostics
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """
        Return a portable description of current state
        (used by Factory snapshots and telemetry panels).
        """
        return {
            "t": round(self.state.t, 9),
            "dt": self.state.dt,
            "mode": self.state.mode,
            "phase": self.state.phase,
            "params": dict(self.state.params),
            "fields": {k: _shape_of(v) for k, v in self.state.fields.items()},
            "kappa": self.state.kappa,
        }

    # ------------------------------------------------------------------ #
    # 📨 Internals — emission & hooks
    # ------------------------------------------------------------------ #
    def _emit(self, event: str, **payload: Any) -> None:
        """Non-blocking emission to telemetry."""
        if self.telemetry is None:
            return
        try:
            self.telemetry.emit(event, **payload)
        except AttributeError:
            # telemetry object missing emit()
            pass
        except Exception as exc:
            # only log internal errors; never raise
            print(f"[telemetry-error] {type(exc).__name__}: {exc}")

    def _call_hook(self, name: str) -> None:
        fn = self._hooks.get(name)
        if fn is None:
            return
        try:
            fn(self.state)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            self._emit("🪝 hook.error", name=name, error=str(exc))

    # ------------------------------------------------------------------ #
    # 🚫 Default no-op step
    # ------------------------------------------------------------------ #
    @staticmethod
    def _noop_step(state: RuntimeState, dt: float) -> Optional[RuntimeState]:
        """Fallback step when no physics is defined."""
        # Nothing moves; we still honour the heartbeat
        return None


# ====================================================================== #
# 🧩 Helpers
# ====================================================================== #
def _shape_of(x: Any) -> Union[str, Dict[str, int]]:
    """Best-effort structural summary for telemetry snapshots."""
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            return {"shape": tuple(int(d) for d in x.shape), "ndim": int(x.ndim)}
    except ImportError:
        pass
    except Exception as exc:
        print(f"[shape-info] numpy check failed: {exc}")

    try:
        import torch
        if isinstance(x, torch.Tensor):
            return {"shape": tuple(int(d) for d in x.shape), "ndim": int(x.dim())}
    except ImportError:
        pass
    except Exception as exc:
        print(f"[shape-info] torch check failed: {exc}")

    if hasattr(x, "shape"):
        shape = getattr(x, "shape")
        try:
            return {"shape": tuple(int(d) for d in shape)}
        except Exception:
            return {"shape": str(shape)}

    return type(x).__name__
