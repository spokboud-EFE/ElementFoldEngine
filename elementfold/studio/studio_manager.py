"""
studio/studio_manager.py — Stable Studio Session Manager 🪄
──────────────────────────────────────────────────────────────────────
Purpose
  • Manage one Studio session, no automatic fake runs.
  • Panels start only when explicitly attached.
  • Each process has a PID file under ~/.elementfold/pids.
  • Clean shutdown guarantees: no zombie processes, no repainting.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations
import multiprocessing as mp
import os
import signal
import sys
import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict

from elementfold.studio.studio import Studio
from elementfold.studio.telemetry_panel import TelemetryPanel
from elementfold.studio.studio_brain_panel import StudioBrainPanel
from elementfold.studio.studio_menu import StudioMenu


# ------------------------------------------------------------------ #
# Paths & constants
# ------------------------------------------------------------------ #
ROOT_DIR = Path.home() / ".elementfold"
PID_DIR = ROOT_DIR / "pids"
LOG_DIR = ROOT_DIR / "logs"
SESSION_FILE = ROOT_DIR / "session.json"

for d in (PID_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# Helper: PID file management
# ------------------------------------------------------------------ #
def write_pid(name: str, pid: int) -> None:
    (PID_DIR / f"{name}.pid").write_text(str(pid))


def clear_pids() -> None:
    for p in PID_DIR.glob("*.pid"):
        try:
            p.unlink()
        except OSError:
            pass


# ------------------------------------------------------------------ #
# Helper: graceful signal handling
# ------------------------------------------------------------------ #
def _graceful_exit(signum, frame):
    print(f"\033[2m[studio] caught signal {signum}, shutting down...\033[0m")
    sys.stdout.flush()
    StudioManager().shutdown()
    sys.exit(0)


signal.signal(signal.SIGINT, _graceful_exit)
signal.signal(signal.SIGTERM, _graceful_exit)


# ================================================================== #
# 🧩 StudioManager
# ================================================================== #
@dataclass
class StudioManager:
    """
    Single-session manager.  No fake runs; Studio remains idle until
    a device is attached.  Panels can be started/stopped explicitly.
    """

    refresh_interval: float = 1.0
    brain_refresh: float = 2.0
    _studio: Optional[Studio] = None
    _telemetry_proc: Optional[mp.Process] = None
    _brain_proc: Optional[mp.Process] = None
    _menu_proc: Optional[mp.Process] = None
    _stop_event: mp.Event = mp.Event()
    _session_id: float = time.time()

    # -------------------------------------------------------------- #
    # 🎬 Session control
    # -------------------------------------------------------------- #
    def start(self) -> None:
        """Initialize a headless Studio session."""
        if self._studio:
            print("[manager] Studio already running.")
            return

        self._studio = Studio()
        self._studio.add_core("alpha")  # core exists but idle
        print("[manager] Studio initialized — idle, no device attached.")
        self._write_session_file(status="idle")

    # -------------------------------------------------------------- #
    # 🔗 Attach / Detach panels
    # -------------------------------------------------------------- #
    def attach_panels(self) -> None:
        """Attach telemetry & brain panels as subprocesses."""
        if not self._studio:
            self.start()

        print("[manager] Attaching panels...")
        self._telemetry_proc = self._spawn(
            "telemetry", self._run_panel, LOG_DIR / "telemetry.log"
        )
        self._brain_proc = self._spawn(
            "brain", self._run_brain, LOG_DIR / "brain.log"
        )
        self._menu_proc = self._spawn(
            "menu", self._run_menu, LOG_DIR / "menu.log"
        )
        self._write_session_file(status="attached")
        print(f"[manager] Panels attached. Logs → {LOG_DIR}")

    def detach_panels(self) -> None:
        """Leave panels running in background."""
        print("[manager] Detached; panels continue quietly.")
        self._write_session_file(status="detached")

    # -------------------------------------------------------------- #
    # 🧹 Shutdown
    # -------------------------------------------------------------- #
    def shutdown(self) -> None:
        """Terminate panels and Studio gracefully."""
        print("[manager] shutting down Studio...")
        self._stop_event.set()
        for name, proc in self._procs().items():
            if proc and proc.is_alive():
                try:
                    proc.terminate()
                    print(f"  🔚 {name} terminated (PID {proc.pid})")
                except Exception:
                    pass
        if self._studio:
            try:
                self._studio.shutdown()
            except Exception:
                pass
        clear_pids()
        self._write_session_file(status="stopped")
        print("\033[2m[studio] ✨ system cooled and exited\033[0m")

    # -------------------------------------------------------------- #
    # 🧠 Internal process targets
    # -------------------------------------------------------------- #
    def _run_panel(self) -> None:
        studio = self._studio or Studio()
        panel = TelemetryPanel(
            studio.factory, guard=studio.guard, refresh_interval=self.refresh_interval
        )
        panel.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.2)
        finally:
            panel.stop()
            studio.shutdown()

    def _run_brain(self) -> None:
        studio = self._studio or Studio()
        brain = StudioBrainPanel(studio, refresh_interval=self.brain_refresh)
        brain.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.2)
        finally:
            brain.stop()
            studio.shutdown()

    def _run_menu(self) -> None:
        try:
            menu = StudioMenu(self._studio)
            menu.start()
        except KeyboardInterrupt:
            print("\n[manager] Menu interrupted by user.")
            self.shutdown()

    # -------------------------------------------------------------- #
    # 🧰 Utilities
    # -------------------------------------------------------------- #
    def _spawn(self, name: str, target, logfile: Path) -> mp.Process:
        """Start a subprocess with redirected output."""
        def _wrapped():
            sys.stdout = open(logfile, "w", buffering=1)
            sys.stderr = sys.stdout
            write_pid(name, os.getpid())
            try:
                target()
            except Exception as e:
                print(f"[{name}] crashed: {e}")
                import traceback
                traceback.print_exc(file=sys.stdout)

        p = mp.get_context("spawn").Process(target=_wrapped, name=name)
        p.start()
        return p

    def _procs(self) -> Dict[str, Optional[mp.Process]]:
        return {
            "telemetry": self._telemetry_proc,
            "brain": self._brain_proc,
            "menu": self._menu_proc,
        }

    def _write_session_file(self, status: str) -> None:
        """Record session metadata for reattachment."""
        data = {
            "session_id": self._session_id,
            "status": status,
            "pids": {
                k: (v.pid if v and v.is_alive() else None)
                for k, v in self._procs().items()
            },
            "timestamp": time.time(),
        }
        SESSION_FILE.write_text(json.dumps(data, indent=2))
