# ElementFold · utils/display.py
# ============================================================
# Unified console + telemetry display for ElementFold
# ------------------------------------------------------------
# • Handles colored logging with safe fallbacks (ASCII/TTY).
# • Maintains a recent buffer mirrored to the web dashboard.
# • Provides banner(), progress(), param(), etc.
# ============================================================

from __future__ import annotations
import os, sys, time, math, threading
from collections import deque
from typing import Any, Dict, List, Optional

# ============================================================
# Shared glyph set (used by console + Studio UI)
# ============================================================

GLYPHS: Dict[str, str] = {
    "bar_fill": "█",
    "bar_empty": "░",
    "prog_fill": "█",
    "prog_empty": "░",
    "sep": "  ",
}

# ============================================================
# Public interface
# ============================================================

__all__ = [
    "GLYPHS",
    "banner", "progress", "format_seconds",
    "info", "success", "warn", "error", "debug",
    "section", "param",
    "recent", "recent_json", "clear_recent",
]

# ============================================================
# Terminal capability detection
# ============================================================

def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR", "") not in {"", "0"}:
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False

_COL = _supports_color()

class _C:
    RESET = "\033[0m" if _COL else ""
    BOLD = "\033[1m" if _COL else ""
    FAINT = "\033[2m" if _COL else ""
    RED = "\033[31m" if _COL else ""
    GREEN = "\033[32m" if _COL else ""
    YELLOW = "\033[33m" if _COL else ""
    BLUE = "\033[34m" if _COL else ""
    CYAN = "\033[36m" if _COL else ""
    GRAY = "\033[90m" if _COL else ""

def _paint(txt: str, *styles: str) -> str:
    if not _COL or not styles:
        return txt
    return "".join(styles) + txt + _C.RESET

# ============================================================
# Recent buffer (for Studio mirror)
# ============================================================

_RECENT: deque[Dict[str, Any]] = deque(maxlen=600)
_LOCK = threading.Lock()

def _push(level: str, text: str) -> None:
    with _LOCK:
        _RECENT.append({
            "t": time.time(),
            "level": level,
            "text": text,
        })

def recent(n: int = 200) -> List[str]:
    with _LOCK:
        items = list(_RECENT)[-int(max(1, n)):]
    return [f"{r['level']}: {r['text']}" for r in items]

def recent_json(n: int = 200) -> List[Dict[str, Any]]:
    with _LOCK:
        return list(_RECENT)[-int(max(1, n)):]

def clear_recent() -> None:
    with _LOCK:
        _RECENT.clear()

# ============================================================
# Banners, progress, and timers
# ============================================================

def banner(lambda_: float, D: float, phi_inf: float) -> str:
    """Return one-line simulation banner."""
    return (
        f"⟲ ElementFold Relaxation Engine ⟲{GLYPHS['sep']}"
        f"λ={lambda_:.3f}{GLYPHS['sep']}D={D:.3f}{GLYPHS['sep']}Φ∞={phi_inf:.3f}"
    )

def format_seconds(secs: float) -> str:
    s = int(max(0, round(float(secs))))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

def progress(step: int, total: int, width: int = 30,
             prefix: str = "", start_time: Optional[float] = None) -> str:
    """Progress bar for long-running simulations."""
    total = max(1, total)
    frac = min(max(step / total, 0.0), 1.0)
    w = max(1, width)
    k = int(round(w * frac))
    bar = GLYPHS["prog_fill"] * k + GLYPHS["prog_empty"] * (w - k)
    pct = f"{100.0 * frac:5.1f}%"
    eta = ""
    if start_time and step > 0:
        elapsed = time.time() - start_time
        rate = elapsed / step
        remain = max(0.0, rate * (total - step))
        eta = f"  ETA {format_seconds(remain)}"
    return f"{prefix} [{bar}] {pct}{eta}"

# ============================================================
# Logging utilities
# ============================================================

def _emit(level: str, msg: str, color: str = "") -> None:
    color_map = {
        "INFO": _C.BLUE,
        "OK": _C.GREEN,
        "WARN": _C.YELLOW,
        "ERR": _C.RED,
        "DBG": _C.GRAY,
    }
    style = color_map.get(level, "")
    prefix = _paint(f"[{level}]", style, _C.BOLD)
    text = _paint(msg, style)
    print(f"{prefix} {text}", flush=True)
    _push(level, msg)

def info(msg: str) -> None: _emit("INFO", msg)
def success(msg: str) -> None: _emit("OK", msg)
def warn(msg: str) -> None: _emit("WARN", msg)
def error(msg: str) -> None: _emit("ERR", msg)
def debug(msg: str) -> None: _emit("DBG", msg)

# ============================================================
# Section headers and parameter visualization
# ============================================================

def section(title: str) -> None:
    line = "─" * max(10, len(title) + 10)
    print(f"\n{_paint(title, _C.BOLD)}\n{_paint(line, _C.GRAY)}", flush=True)
    _push("SEC", title)

def param(name: str,
          value: float,
          meaning: str = "",
          quality: str = "neutral",
          unit: Optional[str] = None,
          maxv: Optional[float] = None) -> None:
    """Render a single telemetry or control parameter."""
    color = {
        "good": _C.GREEN, "warn": _C.YELLOW, "bad": _C.RED,
        "neutral": _C.BLUE
    }.get(quality, _C.BLUE)
    try:
        v = float(value)
        txt = f"{name} {v:.5g}"
    except Exception:
        txt = f"{name} n/a"
    if unit:
        txt += f" {unit}"
    if maxv is not None and isinstance(value, (int, float)):
        m = float(maxv)
        v = min(max(value, 0.0), m)
        filled = int(round(10 * v / m))
        bar = GLYPHS["bar_fill"] * filled + GLYPHS["bar_empty"] * (10 - filled)
        txt += f" [{bar}]"
    if meaning:
        txt += f" — {meaning}"
    print(_paint(txt, color))
    _push("PARAM", txt)

# ============================================================
# End of file
# ============================================================
