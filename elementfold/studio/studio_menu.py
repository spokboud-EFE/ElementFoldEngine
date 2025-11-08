"""
studio/studio_menu.py — Persistent Control Shell 🎚️

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The Studio Menu is the human interface to ElementFold.
• It never hides what’s happening; the telemetry panel stays alive
  while you type commands.
• The menu is hierarchical but fluid:
    Factory ▸ Core ▸ Local Brain ▸ Utilities ▸ Exit
• Commands are short, words are clear, and every action narrates itself.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from elementfold.studio.studio import Studio
from elementfold.studio.telemetry_panel import TelemetryPanel
from elementfold.core.physics.safety_guard import SafetyGuard


# ---------------------------------------------------------------------- #
# 🎨 Simple color helpers
# ---------------------------------------------------------------------- #
class _C:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    YELLOW = "\033[33m"
    RED = "\033[31m"


def _color(msg: str, color: str) -> str:
    return f"{color}{msg}{_C.RESET}"


# ====================================================================== #
# 🧠 StudioMenu — persistent interactive shell
# ====================================================================== #
class StudioMenu:
    """
    Persistent hierarchical command shell for Studio.
    The telemetry panel refreshes in parallel; input remains non-blocking.
    """

    def __init__(self, studio: Optional[Studio] = None) -> None:
        self.studio = studio or Studio()
        self.panel = TelemetryPanel(self.studio.factory, guard=self.studio.guard, refresh_interval=1.0)
        self._input_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()

    # ------------------------------------------------------------------ #
    # 🎬 Entry point
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start panel and input loop."""
        self.studio.add_core("alpha")  # ensure one core exists
        self.studio.start()
        self.panel.start()
        print(_color("🎚️ StudioMenu started. Type 'help' for options.", _C.BOLD))
        self._run_input_loop()

    # ------------------------------------------------------------------ #
    # 🧾 Command interpreter
    # ------------------------------------------------------------------ #
    def _run_input_loop(self) -> None:
        """Simple REPL loop; safe against interrupts."""
        try:
            while not self._stop_flag.is_set():
                sys.stdout.write(_C.CYAN + "\n> " + _C.RESET)
                sys.stdout.flush()
                cmd = sys.stdin.readline().strip()
                if not cmd:
                    continue
                self._dispatch(cmd)
        except (KeyboardInterrupt, EOFError):
            print(_color("\n[menu] Interrupted — shutting down...", _C.YELLOW))
            self.stop()

    # ------------------------------------------------------------------ #
    # 🎯 Command dispatch
    # ------------------------------------------------------------------ #
    def _dispatch(self, cmd: str) -> None:
        """Interpret and execute a single command."""
        args = cmd.split()
        if not args:
            return
        main = args[0].lower()

        # Factory controls
        if main in ("start", "run"):
            steps = int(args[1]) if len(args) > 1 else 10
            dt = float(args[2]) if len(args) > 2 else 0.05
            self.studio.run_async(steps, dt)

        elif main == "stop":
            self.studio.stop()

        elif main == "add":
            name = args[1] if len(args) > 1 else f"core_{len(self.studio.factory.cores)+1}"
            self.studio.add_core(name)

        elif main == "sync":
            self.studio.factory.synchronize()

        elif main == "entangle":
            self.studio.factory.entangle()

        # Core / parameter commands
        elif main == "set":
            if len(args) < 3:
                print("Usage: set <core> <param>=<value> ...")
            else:
                core = args[1]
                params = {}
                for pair in args[2:]:
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        try:
                            params[k] = float(v)
                        except ValueError:
                            params[k] = v
                self.studio.update_params(core, **params)

        elif main == "summary":
            self.studio.summary()

        elif main == "snapshot":
            self.studio.snapshot()

        # Local Brain / commentary
        elif main == "narrate":
            self.studio.narrate()

        elif main == "limits":
            self.studio.show_limits()

        elif main == "help":
            self._show_help()

        elif main in ("exit", "quit"):
            self.stop()

        else:
            print(_color(f"[menu] Unknown command: {cmd}", _C.RED))

    # ------------------------------------------------------------------ #
    # 📖 Help text
    # ------------------------------------------------------------------ #
    def _show_help(self) -> None:
        """Display available commands."""
        print(_color("\nStudioMenu Commands:", _C.BOLD))
        print(" start [steps dt]     — Run background relaxation loop")
        print(" stop                 — Stop factory and panel")
        print(" add [name]           — Add a new core")
        print(" set <core> k=v ...   — Update parameters safely")
        print(" sync / entangle      — Synchronize or couple cores")
        print(" summary / snapshot   — Display current state")
        print(" narrate              — Local brain commentary")
        print(" limits               — Show safety limits")
        print(" help                 — Show this help text")
        print(" exit / quit          — Shutdown everything\n")

    # ------------------------------------------------------------------ #
    # 🔚 Shutdown
    # ------------------------------------------------------------------ #
    def stop(self) -> None:
        """Stop all background threads and exit cleanly."""
        self._stop_flag.set()
        self.panel.stop()
        self.studio.shutdown()
        print(_color("👋 StudioMenu exited gracefully.", _C.DIM))
        sys.exit(0)
