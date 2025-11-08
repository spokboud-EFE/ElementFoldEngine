"""
core/physics/foldclock.py — The Fold Clock ⏱️

──────────────────────────────────────────────────────────────────────
Human narrative
──────────────────────────────────────────────────────────────────────
• Every core ticks by a fixed logarithmic increment δ★.
• Each tick adds to the universal fold counter ℱ.
• ℱ measures how much relaxation a system has undergone.
• This module defines how to compute, accumulate, and narrate ℱ.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
from elementfold.core.physics.field import Field, BACKEND

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ====================================================================== #
# ⚙️ FoldClock — accumulates δ★ ticks into ℱ folds
# ====================================================================== #
@dataclass
class FoldClock:
    """
    Maintains the relation between time, δ★, and cumulative folds ℱ.
    """

    delta_star: float = 0.32000062   # canonical logarithmic increment
    folds: float = 0.0               # accumulated folds ℱ
    last_tick_time: float = field(default_factory=time.perf_counter)
    auto_sync: bool = True           # if True, updates every runtime step

    # ------------------------------------------------------------------ #
    # 🕰️ Core methods
    # ------------------------------------------------------------------ #
    def tick(self, dt: float) -> float:
        """
        Advance the fold counter by one runtime step.

        ℱ ← ℱ + (dt / δ★)
        Returns the updated ℱ value.
        """
        start = time.perf_counter()
        self.folds += dt / self.delta_star
        BACKEND.record(time.perf_counter() - start)
        return self.folds

    def reset(self) -> None:
        """Reset the fold counter."""
        self.folds = 0.0
        self.last_tick_time = time.perf_counter()

    # ------------------------------------------------------------------ #
    # 🧩 Derived quantities
    # ------------------------------------------------------------------ #
    def click_count(self) -> int:
        """Return number of full δ★ clicks elapsed."""
        return int(math.floor(self.folds))

    def remainder(self) -> float:
        """Return fraction of the next click."""
        return self.folds - math.floor(self.folds)

    def phase_angle(self) -> float:
        """
        Map fold fraction to an angular phase in radians (0 → 2π).
        Useful for periodic narration or coupling.
        """
        return 2 * math.pi * self.remainder()

    # ------------------------------------------------------------------ #
    # 🌐 Utilities
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the current state."""
        return {
            "delta_star": self.delta_star,
            "folds": self.folds,
            "clicks": self.click_count(),
            "remainder": self.remainder(),
            "phase_angle": self.phase_angle(),
        }

    def __repr__(self) -> str:
        frac = self.remainder()
        return (
            f"<FoldClock ℱ={self.folds:.3f} (δ★={self.delta_star:.6f}) "
            f"clicks={self.click_count()} frac={frac:.3f}>"
        )


# ====================================================================== #
# 🔗 Utility: compute fold field from η (share-rate) and path
# ====================================================================== #
def accumulate_folds(eta_field: Field, path_length: float) -> float:
    """
    Integrate share rate η along a path to get total folds ℱ.

        ℱ = ∫ η ds  ≈  mean(η) * path_length
    """
    start = time.perf_counter()
    backend = eta_field.backend
    try:
        if backend == "torch" and _TORCH_AVAILABLE:
            folds_val = float(eta_field.data.mean() * path_length)
        else:
            folds_val = float(np.mean(np.asarray(eta_field.data)) * path_length)
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[foldclock] accumulate_folds error: {exc}")
        folds_val = float(np.mean(np.asarray(eta_field.data)) * path_length)
    finally:
        BACKEND.record(time.perf_counter() - start)
    return folds_val


# ====================================================================== #
# 🪶 Narrative helper
# ====================================================================== #
def narrate(clock: FoldClock) -> str:
    """
    Return a human-readable sentence describing the current fold state.
    Example: "ℱ 5.02 — five clicks complete, sixth in progress."
    """
    k = clock.click_count()
    frac = clock.remainder()
    if frac < 0.1:
        mood = "steady"
    elif frac < 0.5:
        mood = "rising"
    else:
        mood = "approaching shift"
    return f"ℱ {clock.folds:.2f} — {k} clicks complete, next {mood}."
