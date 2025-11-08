"""
studio/studio_manager.py — Studio Session Manager 🪄

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The StudioManager replaces shell scripts and tmux.
• It spawns and supervises the Menu, Panels, and Brain in separate
  Python processes, so you can attach/detach or run headless safely.
• It prints a dim, calm farewell when everything stops.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Optional

from elementfold.studio.studio import Studio
from elementfold.studio.studio_menu import StudioMenu
from elementfold.studio.telemetry_panel import TelemetryPanel
from elementfold.studio.studio_brain_panel import StudioBrainPanel


# ====================================================================== #
# 🧩 StudioManager
# ====================================================================== #
@dataclass
class StudioManager:
    """
    Manages Studio subsystems as separate processes.
    Provides start, attach, detach, and shutdown control.
    """

    refresh_interval: float = 1.0
    brain_refresh: float = 2.0
    _studio: Optional[Studio] = None
    _menu_proc: Optional[mp.Process] = None
    _telemetry_proc: Optional[mp.Process] = None
    _brain_proc: Optional[mp.Process] = None
    _stop_event: mp.Event = mp.Event()

    # ------------------------------------------------------------------ #
    # 🎬 Launch
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Spawn Studio and its components in separate processes."""
        if self._studio:
            print("[manager] Studio already running.")
            return

        self._studio = Studio()
        self._studio.add_core("alpha")

        # Start background processes
        self._telemetry_proc = mp.Process(target=self._run_panel, name="TelemetryPanel")
        self._brain_proc = mp.Process(target=self._run_brain_panel, name="BrainPanel")
        self._menu_proc = mp.Process(target=self._run_menu, name="StudioMenu")

        print("🪄 Starting Studio session...")
        self._telemetry_proc.start()
        time.sleep(0.2)
        self._brain_proc.start()
        time.sleep(0.2)
        self._menu_proc.start()
        print("[manager] All subsystems launched. Type Ctrl+C or use menu 'exit' to quit.")

    # ------------------------------------------------------------------ #
    # 🧭 Attach/Detach
    # ------------------------------------------------------------------ #
    def attach(self) -> None:
        """Attach to existing session (reprint PIDs and status)."""
        print(f"[manager] Attached to session:")
        for name, proc in self._procs().items():
            print(f"  {name:<15} PID={proc.pid if proc else '—'}  alive={proc.is_alive() if proc else False}")

    def detach(self) -> None:
        """Detach without killing processes (panels keep running)."""
        print("[manager] Detached from Studio session. Panels continue in background.")

    # ------------------------------------------------------------------ #
    # 🧹 Shutdown
    # ------------------------------------------------------------------ #
    def shutdown(self) -> None:
        """Gracefully terminate all sub-processes."""
        print("[manager] Shutting down Studio...")
        self._stop_event.set()
        for name, proc in self._procs().items():
            if proc and proc.is_alive():
                proc.terminate()
                print(f"  🔚 {name} terminated (PID {proc.pid})")
        self._studio = None
        self._menu_proc = None
        self._telemetry_proc = None
        self._brain_proc = None
        print("\033[2m[studio] ✨ system cooled and exited\033[0m")

    # ------------------------------------------------------------------ #
    # 🧠 Process targets
    # ------------------------------------------------------------------ #
    def _run_panel(self) -> None:
        try:
            studio = Studio()
            studio.add_core("alpha")
            studio.start()
            panel = TelemetryPanel(studio.factory, guard=studio.guard, refresh_interval=self.refresh_interval)
            panel.start()
            while not self._stop_event.is_set():
                time.sleep(0.1)
            panel.stop()
            studio.shutdown()
        except Exception as exc:
            print(f"[TelemetryPanel] crashed: {exc}")
            traceback.print_exc()

    def _run_brain_panel(self) -> None:
        try:
            studio = Studio()
            studio.add_core("alpha")
            studio.start()
            brain = StudioBrainPanel(studio, refresh_interval=self.brain_refresh)
            brain.start()
            while not self._stop_event.is_set():
                time.sleep(0.1)
            brain.stop()
            studio.shutdown()
        except Exception as exc:
            print(f"[BrainPanel] crashed: {exc}")
            traceback.print_exc()

    def _run_menu(self) -> None:
        try:
            menu = StudioMenu()
            menu.start()
        except KeyboardInterrupt:
            print("\n[manager] Menu interrupted by user.")
            self.shutdown()
        except Exception as exc:
            print(f"[StudioMenu] crashed: {exc}")
            traceback.print_exc()

    # ------------------------------------------------------------------ #
    # 📦 Utilities
    # ------------------------------------------------------------------ #
    def _procs(self) -> dict[str, Optional[mp.Process]]:
        return {
            "menu": self._menu_proc,
            "telemetry": self._telemetry_proc,
            "brain": self._brain_proc,
        }
