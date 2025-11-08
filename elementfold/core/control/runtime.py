"""
core/control/runtime.py — Runtime Controller ⚙️
──────────────────────────────────────────────────────────────────────
Purpose
  • Manage timing, parameters, and relaxation mode per core.
  • Stay idle until a device is attached.
  • Never simulate or tick automatically without explicit start().
  • Provide clean start/stop, step(), and safe exception handling.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Any

from elementfold.core.physics.safety_guard import SafetyGuard
from elementfold.core.telemetry.bus import TelemetryBus


# ==================================================================== #
# 🧩 Runtime
# ==================================================================== #
@dataclass
class Runtime:
    """
    Controls one core’s timing and state.

    • Idle until activated (no device → mode='idle')
    • step(dt) updates internal clock and publishes telemetry
    • Thread-safe start/stop
    """

    name: str = "unnamed"
    guard: SafetyGuard = field(default_factory=SafetyGuard)
    telemetry: TelemetryBus = field(default_factory=TelemetryBus)

    # internal state
    t: float = 0.0                        # accumulated time
    mode: str = "idle"                    # 'idle' | 'shaping' | 'forcing'
    params: Dict[str, float] = field(default_factory=dict)
    state: Any = field(default_factory=lambda: type("State", (), {"kappa": 1.0})())
    _running: bool = False
    _thread: Optional[threading.Thread] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _stop_flag: threading.Event = field(default_factory=threading.Event)
    _dt: float = 0.05                     # default integration step [s]

    # -------------------------------------------------------------- #
    # ▶️  Lifecycle
    # -------------------------------------------------------------- #
    def start(self, dt: float = 0.05) -> None:
        """
        Begin the runtime loop only if mode != 'idle'.
        """
        with self._lock:
            if self._running:
                print(f"[runtime:{self.name}] already running.")
                return
            if self.mode == "idle":
                print(f"[runtime:{self.name}] idle — no device attached.")
                return

            self._running = True
            self._dt = dt
            self._stop_flag.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            print(f"[runtime:{self.name}] started in {self.mode} mode (Δt={dt}).")
            self.telemetry.publish(
                "▶️ runtime.start", {"core": self.name, "mode": self.mode}
            )

    def stop(self) -> None:
        """Stop the runtime loop gracefully."""
        with self._lock:
            if not self._running:
                print(f"[runtime:{self.name}] stop() — already idle.")
                return
            self._stop_flag.set()
            self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.telemetry.publish("⏹ runtime.stop", {"core": self.name})
        print(f"[runtime:{self.name}] stopped and idle.")

    # -------------------------------------------------------------- #
    # 🔁  Loop (only when running)
    # -------------------------------------------------------------- #
    def _loop(self) -> None:
        """Internal timing loop."""
        try:
            while not self._stop_flag.is_set():
                self.step(self._dt)
                time.sleep(self._dt)
        except Exception as exc:
            self.telemetry.publish(
                "❗ runtime.error", {"core": self.name, "error": str(exc)}
            )
            print(f"[runtime:{self.name}] error: {exc}")
        finally:
            self._running = False
            self.mode = "idle"

    # -------------------------------------------------------------- #
    # 🕒  Step
    # -------------------------------------------------------------- #
    def step(self, dt: Optional[float] = None) -> None:
        """Advance the runtime clock one step and emit telemetry."""
        dt = dt or self._dt
        self.t += dt
        self.telemetry.publish(
            "🩺 runtime.step",
            {"core": self.name, "t": round(self.t, 3), "mode": self.mode},
        )

    # -------------------------------------------------------------- #
    # ⚙️  Parameter and mode control
    # -------------------------------------------------------------- #
    def set_param_map(self, params: Dict[str, float]) -> None:
        """Update parameters through SafetyGuard clamping."""
        with self._lock:
            safe_params = self.guard.clamp_params(params)
            self.params.update(safe_params)
            self.telemetry.publish(
                "⚙️ runtime.params", {"core": self.name, "params": safe_params}
            )
            print(f"[runtime:{self.name}] parameters updated: {safe_params}")

    def set_mode(self, mode: str) -> None:
        """Switch operational mode."""
        allowed = {"idle", "shaping", "forcing"}
        if mode not in allowed:
            raise ValueError(f"[runtime:{self.name}] invalid mode '{mode}'")
        self.mode = mode
        self.telemetry.publish(
            "🎚 mode.change", {"core": self.name, "mode": self.mode}
        )
        print(f"[runtime:{self.name}] mode → {mode}")

    # -------------------------------------------------------------- #
    # 🧠  Status helpers
    # -------------------------------------------------------------- #
    def is_running(self) -> bool:
        return self._running

    def is_idle(self) -> bool:
        return self.mode == "idle" or not self._running

    def summary(self) -> Dict[str, Any]:
        """Return a concise state snapshot."""
        return {
            "name": self.name,
            "t": self.t,
            "mode": self.mode,
            "running": self._running,
            "params": self.params,
            "kappa": getattr(self.state, "kappa", 1.0),
        }

    def __repr__(self) -> str:
        run = "running" if self._running else "idle"
        return f"<Runtime {self.name} mode={self.mode} {run} t={self.t:.3f}>"
