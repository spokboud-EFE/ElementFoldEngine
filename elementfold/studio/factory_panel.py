"""
studio/factory_panel.py — Factory Overview Panel 🏭
──────────────────────────────────────────────────────────────────────
Purpose
  • Display factory-level information: cores, devices, modes.
  • If no devices: display calm waiting banner.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys
import time
import threading
from typing import Optional

from elementfold.core.control.factory import Factory
from elementfold.core.physics.safety_guard import SafetyGuard


class _C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    MAGENTA = "\033[35m"
    GREY = "\033[90m"


def _color(msg: str, color: str) -> str:
    return f"{color}{msg}{_C.RESET}"


class FactoryPanel:
    """High-level factory overview, passive when idle."""

    def __init__(
        self,
        factory: Factory,
        guard: Optional[SafetyGuard] = None,
        refresh_interval: float = 3.0,
    ) -> None:
        self.factory = factory
        self.guard = guard or SafetyGuard(verbose=False)
        self.refresh_interval = refresh_interval
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(_color("🏭 FactoryPanel started (waiting for device...)", _C.DIM))

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print(_color("🛑 FactoryPanel stopped.", _C.DIM))

    def _loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                self._render()
            except Exception as exc:
                print(_color(f"[factory-panel] render error: {exc}", _C.MAGENTA))
            time.sleep(self.refresh_interval)

    def _render(self) -> None:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()

        print(_color("ElementFold · Factory Overview", _C.BOLD))
        print(_C.DIM + time.strftime("%Y-%m-%d %H:%M:%S") + _C.RESET)
        print("=" * 70)

        devices = getattr(self.factory, "devices", None)
        if not devices:
            print(_color("🕓 Waiting for device to attach...", _C.DIM))
            print(_C.DIM + "(no cores active)\n" + _C.RESET)
            return

        # render each core
        for name, core in self.factory.cores.items():
            dev = getattr(core, "driver", None)
            driver_name = dev.__class__.__name__ if dev else "—"
            mode = getattr(core.runtime, "mode", "?")
            print(f" {name:<10} driver={driver_name:<20} mode={mode}")

        print("\n" + _color("Safety Limits:", _C.BOLD))
        for k, (lo, hi) in self.guard.limits.items():
            print(f"  {_color(k,_C.GREY)}: {lo:.2g} – {hi:.2g}")

        sys.stdout.flush()
