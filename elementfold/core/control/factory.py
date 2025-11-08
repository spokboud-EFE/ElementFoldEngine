"""
core/control/factory.py — Device-Aware Factory 🏭
──────────────────────────────────────────────────────────────────────
Purpose
  • Manage cores and devices in a single Studio session.
  • Never run unless a device is attached.
  • Clean, idempotent start/stop, safe for headless sessions.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from elementfold.core.control.runtime import Runtime
from elementfold.core.control.ledger import Ledger
from elementfold.core.control.synchronizer import Synchronizer
from elementfold.core.control.coupler import Coupler
from elementfold.core.telemetry.bus import TelemetryBus
from elementfold.core.physics.safety_guard import SafetyGuard


@dataclass
class CoreWrapper:
    """A minimal container tying together runtime, ledger, and optional driver."""
    name: str
    runtime: Runtime
    ledger: Ledger
    driver: Optional[Any] = None


# ====================================================================== #
# 🏭 Factory
# ====================================================================== #
@dataclass
class Factory:
    """Central orchestrator for all cores and devices."""

    telemetry: TelemetryBus = field(default_factory=TelemetryBus)
    guard: SafetyGuard = field(default_factory=SafetyGuard)
    synchronizer: Synchronizer = field(default_factory=Synchronizer)
    coupler: Coupler = field(default_factory=Coupler)

    cores: Dict[str, CoreWrapper] = field(default_factory=dict)
    devices: Dict[str, Any] = field(default_factory=dict)
    _running: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ #
    # ⚙️  Core management
    # ------------------------------------------------------------------ #
    def register_core(self, name: str, runtime: Optional[Runtime] = None) -> None:
        """Register a new core (without auto-starting it)."""
        if name in self.cores:
            print(f"[factory] core '{name}' already exists.")
            return
        self.cores[name] = CoreWrapper(name=name, runtime=runtime or Runtime(), ledger=Ledger())
        self.telemetry.publish("🏗️ core.registered", {"core": name})
        print(f"[factory] core '{name}' registered.")

    def attach_device(self, core_name: str, driver: Any) -> None:
        """Attach a driver or dataset to a core."""
        if core_name not in self.cores:
            raise ValueError(f"[factory] unknown core '{core_name}'")
        core = self.cores[core_name]
        core.driver = driver
        self.devices[core_name] = driver
        print(f"[factory] device attached to core '{core_name}'.")
        self.telemetry.publish("🔌 device.attached", {"core": core_name})

    def detach_device(self, core_name: str) -> None:
        """Detach driver and stop the core."""
        core = self.cores.get(core_name)
        if not core:
            return
        core.driver = None
        self.devices.pop(core_name, None)
        self.telemetry.publish("🧲 device.detached", {"core": core_name})
        print(f"[factory] device detached from '{core_name}'.")

    # ------------------------------------------------------------------ #
    # 🕓  Device awareness
    # ------------------------------------------------------------------ #
    def has_device(self) -> bool:
        """Return True if at least one device is attached."""
        return bool(self.devices)

    # ------------------------------------------------------------------ #
    # ▶️  Start / Stop
    # ------------------------------------------------------------------ #
    def start(self, device: Optional[Any] = None) -> None:
        """
        Start the factory only if a device is attached or provided.
        Idempotent and thread-safe.
        """
        with self._lock:
            if self._running:
                print("[factory] already running.")
                return
            if not (device or self.has_device()):
                print("[factory] no device attached — idle mode.")
                self._running = False
                return

            # attach provided device if core empty
            if device and isinstance(device, dict):
                for name, drv in device.items():
                    self.attach_device(name, drv)

            self._running = True
            self.telemetry.publish("🏭 factory.start", {"cores": len(self.cores)})
            print(f"[factory] started with {len(self.devices)} active device(s).")

    def stop(self) -> None:
        """Stop all active cores, close ledgers, clear telemetry."""
        with self._lock:
            if not self._running:
                print("[factory] stop() called — already idle.")
                return
            for name, core in self.cores.items():
                try:
                    core.ledger.close()
                except Exception:
                    pass
            self.telemetry.publish("⛔ factory.stop", {})
            self.telemetry.close()
            self.devices.clear()
            self._running = False
            print("[factory] stopped and cleared all devices.")

    # ------------------------------------------------------------------ #
    # 📸  Snapshot for panels
    # ------------------------------------------------------------------ #
    def snapshot(self) -> Dict[str, Any]:
        """Return current runtime summaries for all cores."""
        data: Dict[str, Any] = {}
        for name, core in self.cores.items():
            rt = core.runtime
            data[name] = {
                "t": getattr(rt, "t", 0.0),
                "mode": getattr(rt, "mode", "idle"),
                "kappa": getattr(rt.state, "kappa", 1.0) if hasattr(rt, "state") else 1.0,
                "params": getattr(rt, "params", {}),
            }
        self.telemetry.publish("📸 factory.snapshot", {"cores": len(data)})
        return data

    # ------------------------------------------------------------------ #
    # 🧹  Maintenance
    # ------------------------------------------------------------------ #
    def clear_ledgers(self) -> None:
        """Clear all ledger entries from disk (non-blocking)."""
        for name, core in self.cores.items():
            try:
                core.ledger.clear()
            except Exception:
                pass
        self.telemetry.publish("🧹 ledgers.cleared", {"cores": len(self.cores)})
        print("[factory] ledgers cleared.")

    # ------------------------------------------------------------------ #
    # 🧠  Diagnostics
    # ------------------------------------------------------------------ #
    def status(self) -> Dict[str, Any]:
        """Return concise status summary."""
        return {
            "running": self._running,
            "cores": len(self.cores),
            "devices": len(self.devices),
            "has_device": self.has_device(),
        }

    def __repr__(self) -> str:
        state = "running" if self._running else "idle"
        return f"<Factory cores={len(self.cores)} devices={len(self.devices)} state={state}>"
