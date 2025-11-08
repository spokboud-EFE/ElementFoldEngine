"""
core/control/factory.py — The Conductor 🏭 of ElementFold

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The Factory builds, synchronizes, and entangles multiple cores.
• Each core is a Runtime wrapped with its own Ledger and coupled
  through shared Synchronizer (δ★) and Coupler (ρ, κ).
• The TelemetryBus carries their voices; the Studio listens in.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .runtime import Runtime
from .ledger import Ledger
from .synchronizer import Synchronizer
from .coupler import Coupler
from ..telemetry import TelemetryBus


# ====================================================================== #
# 🧩 CoreInstance — a single living core built by the Factory
# ====================================================================== #
@dataclass
class CoreInstance:
    """Encapsulates a Runtime, its Ledger, and connection metadata."""

    name: str
    runtime: Runtime
    ledger: Ledger
    synchronizer: Synchronizer
    mode: str = "shaping"

    def tick(self, dt: float) -> None:
        """Advance one δ★ tick with safety guards."""
        try:
            self.runtime.step(dt)
            self.synchronizer.align(self.runtime)
            self.ledger.record_step(self.runtime, dt)
        except (ArithmeticError, ValueError) as exc:
            print(f"[{self.name}] ⚠️ numerical instability: {exc}")
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[{self.name}] 💥 runtime exception: {exc}")
            self.runtime.stop()


# ====================================================================== #
# 🏭 Factory — orchestrates, synchronizes, and couples cores
# ====================================================================== #
@dataclass
class Factory:
    """Central conductor managing all active cores."""

    telemetry: TelemetryBus = field(default_factory=TelemetryBus)
    synchronizer: Synchronizer = field(default_factory=Synchronizer)
    coupler: Coupler = field(default_factory=Coupler)
    cores: Dict[str, CoreInstance] = field(default_factory=dict)
    tick_interval: float = 0.01  # default Δt between steps
    running: bool = False

    # ------------------------------------------------------------------ #
    # 🧱 Construction
    # ------------------------------------------------------------------ #
    def register_core(
        self,
        name: str,
        runtime: Runtime,
        ledger: Optional[Ledger] = None,
    ) -> None:
        """Attach a Runtime to the Factory with its own Ledger."""
        if name in self.cores:
            raise ValueError(f"Core '{name}' already registered.")

        ledger = ledger or Ledger(name=name, telemetry=self.telemetry)
        self.cores[name] = CoreInstance(
            name=name,
            runtime=runtime,
            ledger=ledger,
            synchronizer=self.synchronizer,
        )
        self.telemetry.emit("🏗️ core.registered", core=name)
        self.telemetry.emit("📖 ledger.attached", core=name, entries=len(ledger))

    # ------------------------------------------------------------------ #
    # ▶️ Lifecycle control
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start global loop — begins ticking all cores."""
        if self.running:
            return
        self.running = True
        self.telemetry.emit("🏭 factory.start", cores=len(self.cores))

    def stop(self) -> None:
        """Stop all runtimes and mark factory as halted."""
        if not self.running:
            return
        self.running = False
        self.telemetry.emit("⛔ factory.stop")
        for c in self.cores.values():
            c.runtime.stop()

    # ------------------------------------------------------------------ #
    # 🔁 Synchronization & Entanglement
    # ------------------------------------------------------------------ #
    def synchronize(self) -> None:
        """Align δ★ and phase across all cores."""
        self.synchronizer.synchronize(list(self.cores.values()))
        self.telemetry.emit("🕰️ delta.star.sync", cores=len(self.cores))

    def entangle(self) -> None:
        """Apply coupling (ρ, κ) across all cores."""
        self.coupler.couple(list(self.cores.values()))
        self.telemetry.emit("🕸️ entanglement.updated", rho=self.coupler.rho, kappa=self.coupler.kappa)

    # ------------------------------------------------------------------ #
    # 🫀 Main step loop
    # ------------------------------------------------------------------ #
    def step_all(self, dt: Optional[float] = None) -> None:
        """Advance all registered cores by one δ★ tick."""
        step_dt = dt if dt is not None else self.tick_interval
        for core in self.cores.values():
            core.tick(step_dt)
        self.synchronize()
        self.entangle()

    def run(
        self,
        *,
        steps: Optional[int] = None,
        duration: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> None:
        """
        Continuous orchestration loop.
        - steps : number of δ★ ticks to perform
        - duration : wall-clock seconds to run
        - dt : step size override
        """
        if not self.cores:
            self.telemetry.emit("⚠️ factory.empty")
            return

        self.start()
        start = time.perf_counter()
        n = 0
        while self.running:
            if steps is not None and n >= steps:
                break
            if duration is not None and (time.perf_counter() - start) >= duration:
                break
            self.step_all(dt)
            n += 1
        self.stop()
        self.telemetry.emit("🏁 factory.run.complete", steps=n)

    # ------------------------------------------------------------------ #
    # 📸 Diagnostics
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Dict]:
        """Collect current state of all cores (for Studio display)."""
        snap = {name: c.runtime.snapshot() for name, c in self.cores.items()}
        self.telemetry.emit("📸 factory.snapshot", cores=len(snap))
        return snap

    def summary(self) -> str:
        """Return short textual summary for debugging."""
        lines: List[str] = []
        for name, c in self.cores.items():
            entry = c.ledger.latest()
            if entry:
                lines.append(f"{name}: t={entry.t:.3f} κ={entry.kappa:.2f} {entry.notes}")
            else:
                lines.append(f"{name}: (no ledger yet)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # 🧹 Maintenance
    # ------------------------------------------------------------------ #
    def clear_ledgers(self) -> None:
        """Erase all core ledgers."""
        for c in self.cores.values():
            c.ledger.clear()
        self.telemetry.emit("🧹 ledgers.cleared", cores=len(self.cores))

    def __repr__(self) -> str:
        return f"<Factory cores={len(self.cores)} running={self.running}>"
