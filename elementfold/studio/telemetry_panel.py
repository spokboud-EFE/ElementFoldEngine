"""
studio/telemetry_panel.py — Passive Telemetry Panel 📊
──────────────────────────────────────────────────────────────────────
Purpose
  • Display live Factory telemetry when devices exist.
  • Stay calm and idle when no devices are attached.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import sys
import time
import threading
from typing import Any, Dict, Optional

from elementfold.core.telemetry.bus import TelemetryBus
from elementfold.core.control.factory import Factory
from elementfold.core.physics.safety_guard import SafetyGuard


class _C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    GREY = "\033[90m"


def _color(msg: str, color: str) -> str:
    return f"{color}{msg}{_C.RESET}"


class TelemetryPanel:
    """
    Live telemetry dashboard (read-only).

    • If no device: prints a calm waiting message every few seconds.
    • If device attached: displays summary of Factory telemetry.
    """

    def __init__(
        self,
        factory: Factory,
        guard: Optional[SafetyGuard] = None,
        refresh_interval: float = 2.0,
    ) -> None:
        self.factory = factory
        self.guard = guard or SafetyGuard(verbose=False)
        self.refresh_interval = refresh_interval
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ---------------------------------------------------------------- #
    # ▶️ Control
    # ---------------------------------------------------------------- #
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(_color("📊 TelemetryPanel waiting for device...", _C.DIM))

    def stop(self) -> None:
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print(_color("🛑 TelemetryPanel stopped.", _C.DIM))

    # ---------------------------------------------------------------- #
    # 🔁 Main loop
    # ---------------------------------------------------------------- #
    def _loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                self._render()
            except Exception as exc:
                print(_color(f"[panel] render error: {exc}", _C.YELLOW))
            time.sleep(self.refresh_interval)

    # ---------------------------------------------------------------- #
    # 🖼️ Render
    # ---------------------------------------------------------------- #
    def _render(self) -> None:
        """Render snapshot or waiting message."""
        sys.stdout.write("\033[2J\033[H")  # clear
        sys.stdout.flush()

        print(_color("ElementFold · Telemetry Panel", _C.BOLD))
        print(_C.DIM + time.strftime("%Y-%m-%d %H:%M:%S") + _C.RESET)
        print("=" * 70)

        # No devices yet?
        if not getattr(self.factory, "devices", None):
            print(_color("🕓 Waiting for device to attach...", _C.DIM))
            print(_C.DIM + "(no telemetry updates)\n" + _C.RESET)
            return

        # Otherwise render telemetry snapshot
        snap = self.factory.snapshot()
        for name, state in snap.items():
            t = state.get("t", 0.0)
            mode = state.get("mode", "?")
            kappa = state.get("kappa", 0.0)
            color = _C.CYAN if mode == "shaping" else _C.MAGENTA
            print(
                f" {name:<10} t={t:>6.3f}  κ={kappa:>5.3f}  mode={_color(mode,color)}"
            )

        print("\n" + _color("Safety Limits:", _C.BOLD))
        for k, (lo, hi) in self.guard.limits.items():
            print(f"  {_color(k,_C.GREY)}: {lo:.2g} – {hi:.2g}")

        sys.stdout.flush()
