# ElementFold · utils/logging.py
# ──────────────────────────────────────────────────────────────────────────────
# Small, dependency‑free helpers for pretty terminal readouts.
# Unicode is lovely, but terminals differ; we auto‑detect support and fall back
# to ASCII so logs remain readable everywhere.
#
# Public API (unchanged signatures for drop‑in use):
#   • banner(delta, beta, gamma) -> str
#   • gauge(name, val, maxv, width=10) -> str
#   • progress(step, total, width=30, prefix="", start_time=None) -> str
#   • format_seconds(secs) -> str

from __future__ import annotations

import math      # numeric guards (isfinite), floor/ceil
import time      # wall‑clock stamps for ETA
import sys       # detect terminal encoding


# ──────────────────────────────────────────────────────────────────────────────
# Unicode detection + glyph palette
# ──────────────────────────────────────────────────────────────────────────────

def _supports_unicode() -> bool:
    """Best‑effort: true when stdout encoding looks UTF‑ish."""
    enc = getattr(sys.stdout, "encoding", "") or ""
    return "UTF" in enc.upper()


def _glyphs(use_unicode: bool) -> dict[str, str]:
    """Return a tiny palette of glyphs + label aliases."""
    if use_unicode:
        return {
            # banners / separators
            "spin": "⟲",
            "star": "⋆",
            # units / labels
            "delta": "δ⋆",
            "beta": "β",
            "gamma": "γ",
            "clamp": "⛔",
            # bars
            "g_filled": "▮",
            "g_empty":  "▯",
            "p_filled": "█",
            "p_empty":  "░",
            # ascii fallbacks for labels (unused in unicode mode)
            "beta_txt": "β",
            "gamma_txt": "γ",
            "clamp_txt": "⛔",
            "delta_txt": "δ⋆",
            # misc
            "sep": "  ",
        }
    else:
        return {
            "spin": "*",
            "star": "*",
            "delta": "delta*",
            "beta": "beta",
            "gamma": "gamma",
            "clamp": "CLAMP",
            "g_filled": "#",
            "g_empty":  "-",
            "p_filled": "=",
            "p_empty":  "-",
            "beta_txt": "beta",
            "gamma_txt": "gamma",
            "clamp_txt": "CLAMP",
            "delta_txt": "delta*",
            "sep": "  ",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────────

def banner(delta: float, beta: float, gamma: float) -> str:
    """
    One‑line identity string for runs, showing the coherence click and nominal controls.

    Example (Unicode):
      ⟲ ElementFold ⟲  δ⋆=0.030908106561043047  β=1.00  γ=0.50

    Example (ASCII fallback):
      * ElementFold *  delta*=0.030908106561043047  beta=1.00  gamma=0.50
    """
    g = _glyphs(_supports_unicode())
    d = float(delta)
    b = float(beta)
    gm = float(gamma)
    # Compose a simple decorative head depending on glyph set.
    head = f"{g['spin']} ElementFold {g['spin']}"
    return f"{head}{g['sep']}{g['delta']}={d:.18f}{g['sep']}{g['beta']}={b:.2f}{g['sep']}{g['gamma']}={gm:.2f}"


def gauge(name: str, val: float, maxv: float, width: int = 10) -> str:
    """
    Compact bar meter for a value in [0, maxv]. We clamp input and render a bar.

    Args:
      name:  short label (e.g., 'β', 'γ', '⛔'); ASCII fallback maps to 'beta', 'gamma', 'CLAMP'.
      val:   current value (float).
      maxv:  maximum scale for the bar (float).
      width: number of bar cells (int), default 10.

    Returns:
      e.g.  β▮▮▮▮▯▯▯▯▯ 1.04/2.00    or    beta####------ 1.04/2.00
    """
    g = _glyphs(_supports_unicode())

    # Normalize and clamp numeric input.
    m = max(1e-12, float(maxv))
    v = float(val)
    if not math.isfinite(v):
        v = 0.0
    v = min(max(v, 0.0), m)

    w = max(1, int(width))
    k = int(round(w * (v / m)))
    filled = g["g_filled"] * k
    empty  = g["g_empty"]  * (w - k)

    # Label: respect caller string, but in ASCII mode swap Greek for readable words when it matches.
    label = name
    if not _supports_unicode():
        if name.strip() in {"β", "beta"}:
            label = g["beta_txt"]
        elif name.strip() in {"γ", "gamma"}:
            label = g["gamma_txt"]
        elif name.strip() in {"⛔", "clamp", "CLAMP"}:
            label = g["clamp_txt"]

    return f"{label}{filled}{empty} {v:.2f}/{m:.2f}"


def progress(
    step: int,
    total: int,
    width: int = 30,
    prefix: str = "",
    start_time: float | None = None,
) -> str:
    """
    Single‑line progress bar with optional ETA.

    Args:
      step:       current step index (0‑ or 1‑based; we clamp).
      total:      total number of steps (>0).
      width:      bar cells to draw.
      prefix:     optional text to prepend (e.g., 'train').
      start_time: epoch seconds from time.time() for ETA; if None, ETA omitted.

    Returns:
      '[████░░░░░░░░░░░░░░░░░░]  33.3%  ETA 0:42'  (Unicode)
      '[======----------------]  33.3%  ETA 0:42'  (ASCII)
    """
    g = _glyphs(_supports_unicode())
    t = max(1, int(total))
    s = min(max(0, int(step)), t)
    frac = s / t

    w = max(1, int(width))
    k = int(round(w * frac))
    bar = g["p_filled"] * k + g["p_empty"] * (w - k)
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
    """
    Convert seconds → 'H:MM:SS' (hours omitted if zero).

    Examples:
      5.4   → '0:05'
      75.0  → '1:15'
      3671  → '1:01:11'
    """
    s = int(max(0, round(float(secs))))
    h, r = divmod(s, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h > 0 else f"{m:d}:{s:02d}"
