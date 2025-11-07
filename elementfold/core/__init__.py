# ElementFold · core/__init__.py
# ============================================================
# Core package — the coherent engine of ElementFold.
#
# Contains:
#   • fgn.py         → Fold–Gate–Norm kernel (FGNBlock, Gate, Norm)
#   • model.py       → Model backbone (FGN stack + rotary)
#   • variational.py → Variational ledger (convex spacing law)
#   • ledger.py      → δ⋆-circle geometry & kernels
#   • control.py     → Supervisor (β, γ, ⛔ controller)
#   • telemetry.py   → Coherence metrics (κ, p½, etc.)
#   • runtime.py     → Engine orchestration spine
#
# This package can run entirely stand-alone with PyTorch + stdlib.
# ============================================================

from __future__ import annotations

# Core engine primitives
from .fgn import FGNBlock, Gate, Norm, FoldGrid
from .model import Model
from .variational import VariationalLedger
from .ledger import (
    phase,
    rung_residual,
    half_click_margin,
    snap_to_rung,
    seat_index,
    seat_index_int,
    wrapped_distance,
    periodic_mean,
    periodic_lerp,
    char_kernel,
    vm_kernel,
    invariants,
    check_identities,
    kappa,
    p_half,
)
from .control import Supervisor
from .telemetry import measure, normalize, pretty
from .runtime import Engine

__all__ = [
    # Core compute blocks
    "FoldGrid", "Gate", "Norm", "FGNBlock",
    # Models
    "Model", "VariationalLedger",
    # Ledger geometry
    "phase", "rung_residual", "half_click_margin", "snap_to_rung",
    "seat_index", "seat_index_int", "wrapped_distance",
    "periodic_mean", "periodic_lerp",
    "char_kernel", "vm_kernel",
    "invariants", "check_identities",
    "kappa", "p_half",
    # Control / telemetry / runtime
    "Supervisor", "measure", "normalize", "pretty", "Engine",
]
