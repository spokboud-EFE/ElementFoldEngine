"""
studio/studio_brain_panel.py — The Thinking Side of the Studio 🧠

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• The Brain Panel is the Studio’s inner voice, printed in real time.
• It listens to the Local Brain and shows its mood, comments, and advice.
• The panel never blocks; it refreshes quietly beside the telemetry view.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Optional

from elementfold.studio.local_brain import LocalBrain
from elementfold.studio.studio import Studio

# ---------------------------------------------------------------------- #
# 🎨 Colors
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
    GREY = "\033[90m"


def _color(msg: str, color: str) -> str:
    return f"{color}{msg}{_C.RESET}"


# ====================================================================== #
# 🧠 StudioBrainPanel — live reflection of LocalBrain state
# ====================================================================== #
class StudioBrainPanel:
    """
    Continuously prints LocalBrain mood, latest comments, and suggestions.

    Usage:
        panel = StudioBrainPanel(studio)
        panel.start()
        panel.stop()
    """

    def __init__(
        self,
        studio: Studio,
        *,
        refresh_interval: float = 2.0,
    ) -> None:
        self.studio = studio
        self.brain = LocalBrain(
            studio.factory,
            studio.factory.telemetry,
            studio.guard,
            verbose=False,
        )
        self.refresh_interval = refresh_interval
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._history: list[str] = []

    # ------------------------------------------------------------------ #
    # ▶️ Control
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Start background refresh."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print(_color("🧠 BrainPanel started — listening to thoughts...", _C.BOLD))

    def stop(self) -> None:
        """Stop background refresh."""
        self._stop_flag.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        print(_color("🧘 BrainPanel stopped.", _C.DIM))

    # ------------------------------------------------------------------ #
    # 🔁 Loop
    # ------------------------------------------------------------------ #
    def _loop(self) -> None:
        while not self._stop_flag.is_set():
            self._render()
            time.sleep(self.refresh_interval)

    # ------------------------------------------------------------------ #
    # 🖼️ Render
    # ------------------------------------------------------------------ #
    def _render(self) -> None:
        try:
            comment = self.brain.tick()
            mood = self.brain.mood
            if comment:
                self._history.append(comment)
                if len(self._history) > 3:
                    self._history.pop(0)

            mood_color = {
                "calm": _C.GREEN,
                "neutral": _C.CYAN,
                "concerned": _C.YELLOW,
            }.get(mood, _C.MAGENTA)

            sys.stdout.write("\033[2K\r")  # clear current line
            sys.stdout.write(_color("🧠  Local Brain ", _C.BOLD) + _color(f"({mood})", mood_color) + "\n")

            for line in self._history[-3:]:
                wrapped = line.replace("\n", " ")
                sys.stdout.write(_color(f"   {wrapped}\n", _C.GREY))

            sugg = self.brain.suggest()
            if sugg:
                sys.stdout.write(_color(f"   💡 {sugg.get('recommendation','')}\n", _C.CYAN))
            sys.stdout.flush()

        except Exception as exc:
            sys.stdout.write(_color(f"[brain_panel] render error: {exc}\n", _C.RED))
            sys.stdout.flush()
