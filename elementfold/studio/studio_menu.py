"""
studio/studio_menu.py — Persistent Control Shell 🎚️

This is the human-facing shell for ElementFold.
It keeps the live telemetry & brain panels running while you steer the Factory.
Commands are short and safe; everything narrates itself.

Hierarchy (conceptual)
- Factory:  start/stop/run/sync/entangle
- Core:     add/remove/list, set params, set mode
- Brain:    toggle brain/panel, narrative, advice
- Utility:  snapshot/summary/limits/save/clear/backends/exit
"""

from __future__ import annotations

import shlex
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from elementfold.studio.studio import Studio
from elementfold.studio.telemetry_panel import TelemetryPanel
from elementfold.studio.studio_brain_panel import StudioBrainPanel
from elementfold.core.physics.safety_guard import SafetyGuard
from elementfold.core.physics.field import BACKEND as GLOBAL_BACKEND


# ---------- Colors & formatting -------------------------------------------------
class _C:
    RESET = "\n\033[0m"
    R = "\033[31m"
    G = "\033[32m"
    Y = "\033[33m"
    B = "\033[34m"
    C = "\033[36m"
    M = "\033[35m"
    DIM = "\033[2m"
    BOLD = "\033[1m"


def _ok(msg: str) -> str: return f"{_C.G}✔ {msg}{_C.RESET}"
def _warn(msg: str) -> str: return f"{_C.Y}⚠ {msg}{_C.RESET}"
def _err(msg: str) -> str: return f"{_C.R}✖ {msg}{_C.RESET}"
def _info(msg: str) -> str: return f"{_C.C}{msg}{_C.RESET}"
def _bold(msg: str) -> str: return f"{_C.BOLD}{msg}{_C.RESET}"


# ---------- StudioMenu ----------------------------------------------------------
class StudioMenu:
    """
    Persistent interactive shell that controls the Studio.
    It keeps TelemetryPanel + BrainPanel alive while accepting commands.

    Usage:
        menu = StudioMenu()
        menu.start()
    """

    def __init__(self, studio: Optional[Studio] = None) -> None:
        self.studio = studio or Studio()
        self.guard: SafetyGuard = self.studio.guard
        self.panel: Optional[TelemetryPanel] = None
        self.brain_panel: Optional[StudioBrainPanel] = None
        self._stop_flag = threading.Event()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start the Studio, panel, brain, and enter the REPL."""
        # Ensure at least one core exists for a friendly start.
        if not self.studio.factory.cores:
            self.studio.add_core("alpha")

        # Start Studio (spawns Factory + monitor)
        self.studio.start()

        # Launch live panels
        self._ensure_panel()
        self._ensure_brain_panel()

        print(_bold("🎚  StudioMenu started. Type 'help' for commands."))
        self._repl()

    def stop(self) -> None:
        """Stop panels and shutdown studio cleanly."""
        self._stop_panels()
        self.studio.shutdown()
        import sys
        print("\033[2m[studio] ✨ system cooled and exited\033[0m")
        sys.stdout.flush()
        sys.exit(0)


    # ------------------------------------------------------------------ #
    # Panels
    # ------------------------------------------------------------------ #
    def _ensure_panel(self) -> None:
        if self.panel and self.panel._thread and self.panel._thread.is_alive():
            return
        self.panel = TelemetryPanel(self.studio.factory, guard=self.guard, refresh_interval=1.0)
        self.panel.start()

    def _ensure_brain_panel(self) -> None:
        if self.brain_panel and self.brain_panel._thread and self.brain_panel._thread.is_alive():
            return
        self.brain_panel = StudioBrainPanel(self.studio, refresh_interval=2.0)
        self.brain_panel.start()

    def _stop_panels(self) -> None:
        try:
            if self.panel:
                self.panel.stop()
                self.panel = None
        except Exception as exc:
            print(_warn(f"panel stop: {exc}"))
        try:
            if self.brain_panel:
                self.brain_panel.stop()
                self.brain_panel = None
        except Exception as exc:
            print(_warn(f"brain panel stop: {exc}"))

    # ------------------------------------------------------------------ #
    # REPL
    # ------------------------------------------------------------------ #
    def _repl(self) -> None:
        while not self._stop_flag.is_set():
            try:
                sys.stdout.write("\n" + _C.C + "> " + _C.RESET)
                sys.stdout.flush()
                raw = sys.stdin.readline()
                if not raw:
                    # EOF (e.g. ctrl-d)
                    self.stop()
                    return
                line = raw.strip()
                if not line:
                    continue
                self._dispatch(line)
            except KeyboardInterrupt:
                print(_warn("\nInterrupted — type 'exit' to quit safely."))
            except Exception as exc:
                print(_err(f"unhandled error: {exc}"))

    # ------------------------------------------------------------------ #
    # Command dispatch
    # ------------------------------------------------------------------ #
    def _dispatch(self, line: str) -> None:
        args = shlex.split(line)
        cmd = args[0].lower()

        # ---------- Factory ----------
        if cmd in ("start", "run"):
            steps = int(args[1]) if len(args) > 1 and args[1].isdigit() else 20
            dt = float(args[2]) if len(args) > 2 else 0.05
            self.studio.run_async(steps=steps, dt=dt)

        elif cmd in ("runsec", "runfor"):
            duration = float(args[1]) if len(args) > 1 else 5.0
            dt = float(args[2]) if len(args) > 2 else 0.05
            print(_info(f"⏱ running for {duration}s (dt={dt})…"))
            self.studio.factory.run(duration=duration, dt=dt)

        elif cmd == "stop":
            self.studio.stop()

        elif cmd == "sync":
            self.studio.factory.synchronize()

        elif cmd == "entangle":
            self.studio.factory.entangle()

        elif cmd == "list":
            cores = ", ".join(self.studio.factory.cores.keys()) or "(none)"
            print(_info(f"cores: {cores}"))

        elif cmd == "add":
            name = args[1] if len(args) > 1 else f"core_{len(self.studio.factory.cores)+1}"
            self.studio.add_core(name)

        elif cmd in ("rm", "remove", "del", "delete"):
            if len(args) < 2:
                print("usage: remove <core>")
                return
            name = args[1]
            core = self.studio.factory.cores.pop(name, None)
            if core:
                print(_ok(f"removed core '{name}'"))
            else:
                print(_warn(f"core '{name}' not found"))

        # ---------- Core params & mode ----------
        elif cmd == "set":
            if len(args) < 3:
                print("usage: set <core> k=v [k2=v2 …]")
                return
            core = args[1]
            kv = self._parse_kv(args[2:])
            self.studio.update_params(core, **kv)

        elif cmd == "mode":
            if len(args) != 3 or args[2] not in ("shaping", "forcing"):
                print("usage: mode <core> shaping|forcing")
                return
            core = args[1]
            try:
                self.studio.factory.cores[core].runtime.set_mode(args[2])
                print(_ok(f"{core}: mode → {args[2]}"))
            except KeyError:
                print(_warn(f"core '{core}' not found"))

        # ---------- Panels & brain ----------
        elif cmd == "panel":
            if len(args) == 1:
                print("usage: panel on|off")
            elif args[1] == "on":
                self._ensure_panel()
                print(_ok("telemetry panel on"))
            elif args[1] == "off":
                if self.panel:
                    self.panel.stop()
                    self.panel = None
                    print(_ok("telemetry panel off"))
            else:
                print("usage: panel on|off")

        elif cmd == "brain":
            if len(args) == 1:
                print("usage: brain on|off|narrate|advise")
            elif args[1] == "on":
                self._ensure_brain_panel()
                print(_ok("brain panel on"))
            elif args[1] == "off":
                if self.brain_panel:
                    self.brain_panel.stop()
                    self.brain_panel = None
                    print(_ok("brain panel off"))
            elif args[1] in ("say", "narrate"):
                try:
                    assert self.brain_panel is not None
                    self.brain_panel.brain.narrate()
                except AssertionError:
                    print(_warn("brain panel is not running (use 'brain on')"))
            elif args[1] in ("advise", "advice"):
                try:
                    assert self.brain_panel is not None
                    print(self.brain_panel.brain.suggest())
                except AssertionError:
                    print(_warn("brain panel is not running (use 'brain on')"))
            else:
                print("usage: brain on|off|narrate|advise")

        # ---------- Utility / info ----------
        elif cmd in ("summary", "sum"):
            self.studio.summary()

        elif cmd in ("snapshot", "snap"):
            self.studio.snapshot()

        elif cmd == "limits":
            self.studio.show_limits()

        elif cmd == "save":
            # save ledgers to a directory
            out = Path(args[1]) if len(args) > 1 else Path("ledgers")
            out.mkdir(parents=True, exist_ok=True)
            for name, core in self.studio.factory.cores.items():
                dst = out / f"{name}_ledger.jsonl"
                core.ledger.save(dst)
                print(_ok(f"ledger saved: {dst}"))

        elif cmd == "clear":
            self.studio.factory.clear_ledgers()
            print(_ok("all ledgers cleared"))

        elif cmd == "backend":
            if len(args) == 1:
                icon = "⚡" if GLOBAL_BACKEND.current() == "torch" else "🧮"
                print(_info(f"backend: {icon} {GLOBAL_BACKEND.current()}"))
            else:
                desired = args[1].lower()
                if desired in ("numpy", "torch"):
                    if desired == "torch":
                        # Promote only if torch is installed; otherwise stick to numpy
                        try:
                            import torch  # noqa
                            GLOBAL_BACKEND.backend = "torch"
                            print(_ok("backend → torch"))
                        except Exception:
                            print(_warn("torch not available; staying on numpy"))
                            GLOBAL_BACKEND.backend = "numpy"
                    else:
                        GLOBAL_BACKEND.backend = "numpy"
                        print(_ok("backend → numpy"))
                else:
                    print("usage: backend [numpy|torch]")

        elif cmd in ("help", "?"):
            self._print_help()

        elif cmd in ("exit", "quit"):
            self.stop()
            return

        else:
            print(_warn(f"unknown command: {cmd} (type 'help')"))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _parse_kv(self, pairs: list[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for p in pairs:
            if "=" not in p:
                print(_warn(f"ignored '{p}' (expected k=v)"))
                continue
            k, v = p.split("=", 1)
            k = k.strip()
            val: Any = v.strip()
            # try numeric cast
            try:
                if "." in val or "e" in val.lower():
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                # keep as string
                pass
            out[k] = val
        return out

    def _print_help(self) -> None:
        print(_C.BOLD + "\nStudioMenu — Commands" + _C.RESET)
        print(_C.DIM + "Factory" + _C.RESET)
        print("  start [steps dt]         run [steps dt]      — run N steps in background")
        print("  runsec <seconds> [dt]                        — run for wall-clock seconds")
        print("  stop    sync    entangle                     — control orchestration")
        print("  add [name]   remove <name>   list            — manage cores")

        print(_C.DIM + "\nCore" + _C.RESET)
        print("  set <core> k=v [k2=v2 …]  — safely update params via SafetyGuard")
        print("  mode <core> shaping|forcing")

        print(_C.DIM + "\nBrain & Panels" + _C.RESET)
        print("  panel on|off              — toggle telemetry panel")
        print("  brain on|off|narrate|advise")

        print(_C.DIM + "\nUtility" + _C.RESET)
        print("  summary   snapshot        — inspect state")
        print("  limits                    — show safety limits")
        print("  backend [numpy|torch]     — view or force backend")
        print("  save [dir]                — save all ledgers as JSONL")
        print("  clear                     — clear all ledgers")
        print("  help                      — this help")
        print("  exit                      — shutdown cleanly")
