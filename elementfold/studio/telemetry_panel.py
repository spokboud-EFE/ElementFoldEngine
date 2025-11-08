"""
studio/telemetry_panel.py — Live Telemetry Panel 📊

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The Panel is the face of the Studio — it refreshes continuously,
  showing every core’s vital signs and parameters.
• It never breaks; it simply repaints, narrating in color.
• This runs in any ANSI terminal; no curses or external libs needed.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import sys
import threading
import time
from typing import Any, Dict, Optional

from elementfold.core.telemetry import TelemetryBus
from elementfold.core.control.factory import Factory
from elementfold.core.physics.safety_guard import SafetyGuard


# ---------------------------------------------------------------------- #
# 🎨  Color codes
# ---------------------------------------------------------------------- #
class _C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"
    BLUE = "\033[34m"
    GREY = "\033[90m"


def _color(val: Any, color: str) -> str:
    return f"{color}{val}{_C.RESET}"


# ====================================================================== #
# 📊 TelemetryPanel — live refreshing terminal dashboard
# ====================================================================== #
class TelemetryPanel:
    """
    Continuously refreshes the live view of all cores, parameters, and modes.

    Usage:
        panel = TelemetryPanel(factory)
        panel.start()     # run in background
        panel.stop()      # when done
    """

    def __init__(
        self,
        factory: Factory,
        *,
        guard: Optional[SafetyGuard] = None,
        refresh_interval: float = 1.0,
        show_power: bool = True,
    ) -> None:
        self.factory = factory
        self.guard = guard or SafetyGuard(verbose=False)
        self.refresh_interval = refresh_interval
        self.show_power = show_power
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    # ▶️ Control
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Launch background refresh loop."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(_color("📊 TelemetryPanel started — live view active.", _C.BOLD))

    def stop(self) -> None:
        """Stop refreshing."""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print(_color("🛑 TelemetryPanel stopped.", _C.DIM))

    # ------------------------------------------------------------------ #
    # 🔁 Main loop
    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        while not self._stop_flag.is_set():
            self._render()
            time.sleep(self.refresh_interval)

    # ------------------------------------------------------------------ #
    # 🖼️ Render logic
    # ------------------------------------------------------------------ #
    def _render(self) -> None:
        """Redraw the terminal with current factory and core data."""
        try:
            # Clear terminal for full refresh
            sys.stdout.write("\033[2J\033[H")
            sys.stdout.flush()

            print(_color("ElementFold · Live Telemetry Panel", _C.BOLD))
            print(_C.DIM + time.strftime("%Y-%m-%d %H:%M:%S") + _C.RESET)
            print("=" * 70)

            # Factory state summary
            summary = self.factory.summary().splitlines()
            for line in summary:
                print(_color(line, _C.GREEN))

            # Per-core details
            print("\n" + _color("Per-Core Parameters:", _C.BOLD))
            snap = self.factory.snapshot()
            for name, state in snap.items():
                t = state.get("t", 0.0)
                mode = state.get("mode", "?")
                kappa = state.get("kappa", 0.0)
                phase = state.get("phase", 0.0)
                color = _C.CYAN if mode == "shaping" else _C.MAGENTA
                print(
                    f" {name:<10} t={t:>6.3f}  κ={kappa:>5.3f}  "
                    f"phase={phase:>5.3f}  mode={_color(mode, color)}"
                )

            # Optional: safety limits overview
            print("\n" + _color("Safety Limits:", _C.BOLD))
            for k, (lo, hi) in self.guard.limits.items():
                print(f"  {_color(k, _C.GREY)}: {lo:.2g} – {hi:.2g}")

            # Optional: performance / backend power indicator
            if self.show_power:
                from elementfold.core.physics.field import BACKEND
                icon = "⚡" if BACKEND.current() == "torch" else "🧮"
                print(f"\nBackend: {icon} {BACKEND.current()}")

            print(_C.DIM + "\n(Press Ctrl+C or use panel.stop() to exit)" + _C.RESET)

            sys.stdout.flush()
        except Exception as exc:
            print(_color(f"[panel] render error: {exc}", _C.RED))
