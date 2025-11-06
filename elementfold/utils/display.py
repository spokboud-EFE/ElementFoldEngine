# ElementFold · utils/display.py
# ──────────────────────────────────────────────────────────────────────────────
# A small, narrative-aware display layer for the console (and a mirror buffer
# for the Web UI). Keeps the legacy helpers (banner, gauge, progress,
# format_seconds) while adding color log lines and parameter narration.
#
# Design:
#   • Zero hard deps: uses ANSI; if colorama is installed, it will be enabled on
#     Windows automatically. If the terminal lacks unicode/colors, we degrade.
#   • Ring buffer of recent lines so the UI can mirror console output.
#   • Friendly one-liners for parameters with short interpretations.
#
# Public API (stable):
#   - banner(delta, beta, gamma) -> str
#   - gauge(name, val, maxv, width=10) -> str
#   - progress(step, total, width=30, prefix="", start_time=None) -> str
#   - format_seconds(secs) -> str
#   - info(msg), success(msg), warn(msg), error(msg), debug(msg)
#   - section(title: str)
#   - kv(label, value, *, color=None, unit=None)  -> prints "label: value unit"
#   - param(name, value, *, meaning="", quality="neutral", unit=None, maxv=None, width=10)
#   - recent(n=200) -> List[str]
#   - recent_json(n=200) -> List[dict]   (timestamped records)
#   - clear_recent()
#
# MIT-style tiny utility. © 2025 ElementFold authors.

from __future__ import annotations

import math
import os
import sys
import time
from collections import deque
from typing import Any, Dict, List, Optional

__all__ = [
    "banner", "gauge", "progress", "format_seconds",
    "info", "success", "warn", "error", "debug",
    "section", "kv", "param",
    "recent", "recent_json", "clear_recent",
]

# ──────────────────────────────────────────────────────────────────────────────
# Capability detection
# ──────────────────────────────────────────────────────────────────────────────

def _supports_unicode() -> bool:
    enc = getattr(sys.stdout, "encoding", "") or ""
    return "UTF" in enc.upper()

def _supports_color() -> bool:
    """
    Very conservative color support:
      • honor NO_COLOR if present,
      • require a TTY,
      • otherwise disable color.
    """
    if "NO_COLOR" in os.environ:
        return False
    try:
        return bool(sys.stdout.isatty())
    except Exception:
        return False

# Best-effort Windows ANSI init (no hard dependency).
try:
    import colorama  # type: ignore
    colorama.just_fix_windows_console()
except Exception:
    pass

# ──────────────────────────────────────────────────────────────────────────────
# Glyphs + ANSI palette
# ──────────────────────────────────────────────────────────────────────────────

_UNI = _supports_unicode()
_COL = _supports_color()

GLYPHS = {
    "spin": "⟲" if _UNI else "*",
    "star": "⋆" if _UNI else "*",
    "delta": "δ⋆" if _UNI else "delta*",
    "beta": "β" if _UNI else "beta",
    "gamma": "γ" if _UNI else "gamma",
    "clamp": "⛔" if _UNI else "CLAMP",
    "bar_fill": "▮" if _UNI else "#",
    "bar_empty": "▯" if _UNI else "-",
    "prog_fill": "█" if _UNI else "=",
    "prog_empty": "░" if _UNI else "-",
    "sep": "  ",
}

class _C:
    # ANSI codes (kept tiny; bold + faint + basic 8 colors)
    RESET = "\033[0m" if _COL else ""
    BOLD = "\033[1m" if _COL else ""
    FAINT = "\033[2m" if _COL else ""
    DIM = FAINT
    # Colors
    RED = "\033[31m" if _COL else ""
    GREEN = "\033[32m" if _COL else ""
    YELLOW = "\033[33m" if _COL else ""
    BLUE = "\033[34m" if _COL else ""
    MAGENTA = "\033[35m" if _COL else ""
    CYAN = "\033[36m" if _COL else ""
    GRAY = "\033[90m" if _COL else ""

def _paint(txt: str, *styles: str) -> str:
    if not _COL or not styles:
        return txt
    return "".join(styles) + txt + _C.RESET

def _ts() -> str:
    # Short timestamp (mm:ss)
    t = time.localtime()
    return f"{t.tm_min:02d}:{t.tm_sec:02d}"

# ──────────────────────────────────────────────────────────────────────────────
# Recent buffer (for UI mirroring)
# ──────────────────────────────────────────────────────────────────────────────

_RECENT: deque[Dict[str, Any]] = deque(maxlen=600)

def _push_recent(level: str, text: str) -> None:
    _RECENT.append({
        "t": time.time(),
        "ts": _ts(),
        "level": level,
        "text": text,
    })

def recent(n: int = 200) -> List[str]:
    """Most-recent lines as plain strings (for quick dumps)."""
    items = list(_RECENT)[-int(max(1, n)):]
    return [f"[{r['ts']}] {r['level']}: {r['text']}" for r in items]

def recent_json(n: int = 200) -> List[Dict[str, Any]]:
    """Most-recent lines as JSON records (for web UI)."""
    return list(_RECENT)[-int(max(1, n)):]

def clear_recent() -> None:
    _RECENT.clear()

# ──────────────────────────────────────────────────────────────────────────────
# Legacy helpers (unchanged signatures)
# ──────────────────────────────────────────────────────────────────────────────

def banner(delta: float, beta: float, gamma: float) -> str:
    head = f"{GLYPHS['spin']} ElementFold {GLYPHS['spin']}"
    d = float(delta); b = float(beta); g = float(gamma)
    return f"{head}{GLYPHS['sep']}{GLYPHS['delta']}={d:.18f}{GLYPHS['sep']}{GLYPHS['beta']}={b:.2f}{GLYPHS['sep']}{GLYPHS['gamma']}={g:.2f}"

def gauge(name: str, val: float, maxv: float, width: int = 10) -> str:
    m = max(1e-12, float(maxv))
    v = float(val)
    if not math.isfinite(v): v = 0.0
    v = min(max(v, 0.0), m)
    w = max(1, int(width))
    k = int(round(w * (v / m)))
    filled = GLYPHS["bar_fill"] * k
    empty  = GLYPHS["bar_empty"]  * (w - k)
    label = name
    if not _UNI:
        # Map common greek labels to ASCII words in ASCII mode
        label_map = {"β": "beta", "γ": "gamma", "⛔": "CLAMP"}
        label = label_map.get(name.strip(), name)
    return f"{label}{filled}{empty} {v:.2f}/{m:.2f}"

def progress(step: int, total: int, width: int = 30, prefix: str = "", start_time: float | None = None) -> str:
    t = max(1, int(total))
    s = min(max(0, int(step)), t)
    frac = s / t
    w = max(1, int(width))
    k = int(round(w * frac))
    bar = GLYPHS["prog_fill"] * k + GLYPHS["prog_empty"] * (w - k)
    pct = f"{100.0 * frac:5.1f}%"
    eta_txt = ""
    if start_time is not None and s > 0:
        elapsed = max(0.0, time.time() - float(start_time))
        rate = elapsed / s
        remain = max(0.0, rate * (t - s))
        eta_txt = f"  ETA {format_seconds(remain)}"
    head = (prefix + " ") if prefix else ""
    return f"{head}[{bar}] {pct}{eta_txt}"

def format_seconds(secs: float) -> str:
    s = int(max(0, round(float(secs))))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"

# ──────────────────────────────────────────────────────────────────────────────
# Friendly line loggers
# ──────────────────────────────────────────────────────────────────────────────

def _emit(level: str, text: str, color: Optional[str] = None, icon: Optional[str] = None) -> None:
    if icon is None:
        icon = {"INFO": "ℹ", "OK": "✓", "WARN": "⚠", "ERR": "✖", "DBG": "·"}.get(level, "·") if _UNI else "*"
    left = f"[{_ts()}] {level}:"
    if color == "green":
        left = _paint(left, _C.GREEN, _C.BOLD)
        text = _paint(text, _C.GREEN)
    elif color == "yellow":
        left = _paint(left, _C.YELLOW, _C.BOLD)
        text = _paint(text, _C.YELLOW)
    elif color == "red":
        left = _paint(left, _C.RED, _C.BOLD)
        text = _paint(text, _C.RED)
    elif color == "blue":
        left = _paint(left, _C.CYAN, _C.BOLD)
        text = _paint(text, _C.CYAN)
    elif color == "magenta":
        left = _paint(left, _C.MAGENTA, _C.BOLD)
        text = _paint(text, _C.MAGENTA)
    elif color == "gray":
        left = _paint(left, _C.GRAY)
        text = _paint(text, _C.GRAY)
    line = f"{left} {icon} {text}"
    print(line)
    _push_recent(level, text)

def info(msg: str) -> None:
    _emit("INFO", msg, color="blue")

def success(msg: str) -> None:
    _emit("OK", msg, color="green")

def warn(msg: str) -> None:
    _emit("WARN", msg, color="yellow")

def error(msg: str) -> None:
    _emit("ERR", msg, color="red")

def debug(msg: str) -> None:
    _emit("DBG", msg, color="gray")

def section(title: str) -> None:
    bar = "─" * max(6, min(60, len(title) + 10)) if _UNI else "-" * max(6, min(60, len(title) + 10))
    hdr = _paint(title, _C.BOLD)
    print(f"\n{hdr}\n{_paint(bar, _C.GRAY)}")
    _push_recent("SEC", title)

def kv(label: str, value: Any, *, color: Optional[str] = None, unit: Optional[str] = None) -> None:
    s = f"{label}: {value}"
    if unit:
        s += f" {unit}"
    _emit("INFO", s, color=color or "blue")

# ──────────────────────────────────────────────────────────────────────────────
# Parameter narration (one row per number)
# ──────────────────────────────────────────────────────────────────────────────

def _quality_color(quality: str) -> str:
    q = (quality or "neutral").lower()
    if q in ("good", "ok", "healthy", "safe", "strong"): return "green"
    if q in ("warn", "risky", "edge", "uncertain"): return "yellow"
    if q in ("bad", "danger", "unstable", "fail"): return "red"
    if q in ("neutral", "info"): return "blue"
    return "blue"

def param(
    name: str,
    value: float,
    *,
    meaning: str = "",
    quality: str = "neutral",     # "good" | "warn" | "bad" | "neutral"
    unit: Optional[str] = None,
    maxv: Optional[float] = None,
    width: int = 10,
    advice: Optional[str] = None,
) -> None:
    """
    Render a single narrative line for a numeric parameter, with an optional bar
    when maxv is provided. Example:
      κ 0.91  [██████░░]  — coherence high near rung; next: hold / tick 3
    """
    # Left label + value
    try:
        is_finite = math.isfinite(float(value))
    except Exception:
        is_finite = False
    val_txt = f"{float(value):.3f}" if is_finite else "n/a"
    left = f"{name} {val_txt}"
    if unit:
        left += f" {unit}"

    # Optional bar
    bar_txt = ""
    if (maxv is not None) and is_finite:
        bar_txt = "  " + gauge("", float(value), float(maxv), width=width)[1:]  # drop first char (label)

    # Meaning + advice
    meaning = (meaning or "").strip()
    hint = f" — {meaning}" if meaning else ""
    if advice:
        hint += f"; next: {advice}"

    color = _quality_color(quality)
    _emit("INFO", f"{left}{bar_txt}{hint}", color=color)

# ──────────────────────────────────────────────────────────────────────────────
# Tiny demo main (optional)
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    section("Display demo")
    print(banner(0.030908106561043047, 1.0, 0.5))
    info("Training started")
    print(progress(10, 100, prefix="train", start_time=time.time() - 3.2))
    param("κ", 0.91, meaning="coherence high near rung", quality="good", maxv=1.0, advice="hold | tick 3")
    param("p½", 0.07, meaning="barrier far", quality="good", maxv=1.0)
    param("β", 1.28, meaning="exposure mild", quality="neutral", maxv=2.0)
    param("γ", 0.46, meaning="damping calm", quality="good", maxv=1.0)
    success("Model ready")
