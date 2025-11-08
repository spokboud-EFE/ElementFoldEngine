"""
studio/studio.py — The Studio Conductor 🎛️

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The Studio sits above the Factory and speaks plain words.
• It watches telemetry, adjusts parameters, and lets humans
  explore relaxation in real time.
• Think of it as an intelligent command surface:
  one prompt that can start, stop, sync, and narrate.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from elementfold.core.control.factory import Factory
from elementfold.core.control.runtime import Runtime
from elementfold.core.control.ledger import Ledger
from elementfold.core.physics.safety_guard import SafetyGuard
from elementfold.core.telemetry import TelemetryBus
from elementfold.studio.factory_monitor import FactoryMonitor


# ====================================================================== #
# 🎛️  Studio — top-level controller for Factory and Telemetry
# ====================================================================== #
@dataclass
class Studio:
    """Interactive orchestration surface connecting Factory + SafetyGuard."""

    factory: Factory = field(default_factory=Factory)
    guard: SafetyGuard = field(default_factory=SafetyGuard)
    monitor: Optional[FactoryMonitor] = None
    _thread: Optional[threading.Thread] = None
    _running: bool = False

    # ------------------------------------------------------------------ #
    # 🧱 Setup and teardown
    # ------------------------------------------------------------------ #
    # elementfold/studio/studio.py
    def add_core(self, name: str, runtime: Optional[Runtime] = None, ledger: Optional[Any] = None) -> None:
        """
        Create and register a new core.

        NOTE: 'ledger' is ignored in the new Factory API; the Factory creates its own Ledger.
        This keeps backward compatibility with old calls that passed a ledger.
        """
        self.factory.register_core(name, runtime or Runtime())
        print(f"[studio] 🎬 Core '{name}' added to factory.")


    def start(self) -> None:
        """Start factory + optional monitor."""
        if self._running:
            print("[studio] already running.")
            return
        self._running = True
        self.factory.start()
        if not self.monitor:
            self.monitor = FactoryMonitor(self.factory.telemetry)
            self.monitor.start()
        print("[studio] 🚀 started.")

    def stop(self) -> None:
        """Stop factory and monitor."""
        self.factory.stop()
        if self.monitor:
            self.monitor.stop()
            self.monitor = None
        self._running = False
        print("[studio] 🛑 stopped.")

    # ------------------------------------------------------------------ #
    # ⚙️  Parameter management
    # ------------------------------------------------------------------ #
    def update_params(self, core_name: str, **params: Any) -> None:
        """Safely update parameters for a specific core via SafetyGuard."""
        if core_name not in self.factory.cores:
            print(f"[studio] ❌ core '{core_name}' not found.")
            return
        safe_params = self.guard.clamp_params(params)
        core = self.factory.cores[core_name]
        core.runtime.set_param_map(safe_params)
        print(f"[studio] ⚙️ parameters updated for '{core_name}': {safe_params}")

    def show_limits(self) -> None:
        """Display current safety limits."""
        print(self.guard.describe())

    # ------------------------------------------------------------------ #
    # 📡  Telemetry summaries
    # ------------------------------------------------------------------ #
    def snapshot(self) -> None:
        """Print a snapshot of all core states."""
        snap = self.factory.snapshot()
        print(json.dumps(snap, indent=2, ensure_ascii=False))

    def summary(self) -> None:
        """Print a short textual summary (for quick monitoring)."""
        print(self.factory.summary())

    # ------------------------------------------------------------------ #
    # 🎚️ Run helpers
    # ------------------------------------------------------------------ #
    def run_steps(self, steps: int, dt: float) -> None:
        """Run a finite number of steps on all cores."""
        print(f"[studio] 🧩 running {steps} steps (dt={dt}) ...")
        self.factory.run(steps=steps, dt=dt)
        print("[studio] ✅ run complete.")

    def run_async(self, steps: int, dt: float) -> None:
        """Run in a background thread so user can still interact."""
        if self._thread and self._thread.is_alive():
            print("[studio] background run already in progress.")
            return

        def _target():
            self.run_steps(steps, dt)

        self._thread = threading.Thread(target=_target, daemon=True)
        self._thread.start()
        print("[studio] 🧵 background run started.")

    # ------------------------------------------------------------------ #
    # 🗣️ Narrative commands
    # ------------------------------------------------------------------ #
    def narrate(self) -> None:
        """Emit a human-readable summary via Factory telemetry."""
        snap = self.factory.snapshot()
        for name, state in snap.items():
            κ = state.get("kappa", 0.0)
            t = state.get("t", 0.0)
            mode = state.get("mode", "?")
            print(f"🎵 Core {name}: t={t:.3f} κ={κ:.3f} mode={mode}")

    # ------------------------------------------------------------------ #
    # 🔚 Cleanup
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Ensure everything stops gracefully."""
        self.stop()
        self.factory.clear_ledgers()
        print("[studio] 🌙 shutdown complete.")
