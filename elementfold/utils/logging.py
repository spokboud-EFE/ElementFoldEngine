# ElementFold · utils/logging.py
# Small, dependency‑free helpers for pretty terminal readouts.
# We keep everything simple and deterministic so logs are portable.

import math   # ✴ numeric guards (isfinite), floor/ceil
import time   # ✴ wall‑clock stamps for progress ETA


def banner(delta: float, beta: float, gamma: float) -> str:
    """
    One‑line identity string for runs, showing the coherence click δ⋆ and
    nominal control knobs (β exposure, γ damping). Purely cosmetic.

    Example:
      ⟲ ElementFold ⟲  δ⋆=0.030908106561043047  β=1.00  γ=0.50
    """
    d = float(delta)                     # ✴ ensure plain float
    b = float(beta)                      # ✴ format as fixed‑width
    g = float(gamma)                     # ✴ format as fixed‑width
    return f"⟲ ElementFold ⟲  δ⋆={d:.18f}  β={b:.2f}  γ={g:.2f}"  # 🄱 Unicode banner


def gauge(name: str, val: float, maxv: float, width: int = 10) -> str:
    """
    Compact bar meter for a value in [0, maxv].
    We clamp input, then render a bar with filled ▮ and empty ▯ cells.

    Args:
      name:  short label (e.g., 'β', 'γ', '⛔')
      val:   current value (float)
      maxv:  maximum scale for the bar (float)
      width: number of bar cells (int), default 10
    """
    v = float(val)                       # ✴ numeric normalize
    m = max(1e-12, float(maxv))          # ✴ avoid division by zero
    w = max(1, int(width))               # ✴ at least 1 cell
    if not math.isfinite(v):             # ✴ NaN/Inf guard
        v = 0.0
    v = min(max(v, 0.0), m)              # ✴ clamp to [0, m]
    k = int(round(w * (v / m)))          # ✴ filled cell count
    filled = "▮" * k                     # ✴ filled glyphs
    empty  = "▯" * (w - k)               # ✴ empty glyphs
    return f"{name}{filled}{empty} {v:.2f}/{m:.2f}"  # ✴ e.g., β▮▮▮▮▯▯▯▯▯ 0.80/2.00


def progress(step: int, total: int, width: int = 30, prefix: str = "", start_time: float | None = None) -> str:
    """
    Single‑line progress bar with optional ETA, suitable for periodic prints.

    Args:
      step:       current step index (0‑based or 1‑based; we normalize)
      total:      total number of steps expected (>0)
      width:      number of bar cells to draw
      prefix:     optional text to prepend (e.g., 'train')
      start_time: wall‑clock timestamp from time.time() for ETA; if None, ETA is omitted
    """
    t = max(1, int(total))               # ✴ guard total
    s = min(max(0, int(step)), t)        # ✴ clamp step into [0,t]
    frac = s / t                         # ✴ completion fraction
    w = max(1, int(width))               # ✴ bar width
    k = int(round(w * frac))             # ✴ filled cells
    bar = "█" * k + "░" * (w - k)        # ✴ solid + light
    pct = f"{100.0 * frac:5.1f}%"        # ✴ percent fixed width

    eta_txt = ""                         # ✴ default: no ETA
    if start_time is not None and s > 0: # ✴ compute ETA only with progress
        elapsed = max(0.0, time.time() - float(start_time))  # ✴ seconds since start
        rate = elapsed / s                                   # ✴ sec/step
        remain = max(0.0, rate * (t - s))                    # ✴ seconds remaining
        eta_txt = f"  ETA {format_seconds(remain)}"          # ✴ pretty ETA

    head = (prefix + " ") if prefix else ""  # ✴ prefix spacing
    return f"{head}[{bar}] {pct}{eta_txt}"   # ✴ final line


def format_seconds(secs: float) -> str:
    """
    Convert seconds → 'H:MM:SS' with hours omitted if zero.

    Examples:
      5.4   → '0:05'
      75.0  → '1:15'
      3671  → '1:01:11'
    """
    s = int(max(0, round(float(secs))))   # ✴ clamp and round
    h, r = divmod(s, 3600)                # ✴ hours, remainder
    m, s = divmod(r, 60)                  # ✴ minutes, seconds
    if h > 0:                             # ✴ show hours when nonzero
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"               # ✴ mm:ss
