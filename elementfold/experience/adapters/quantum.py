# ElementFold · quantum.py
# Click‑geometry & coherence algebra — high‑level helpers for experiments.
#
# Why this module exists:
#   We already have the atomic pieces:
#     • ledger.phase(x, δ⋆), rung_residual(x, δ⋆), kernels, invariants…
#     • verify.py for checks/diagnostics.
#   This file composes them into *experiment-friendly* utilities with clear
#   docstrings and narrative comments so you can prototype quickly and safely.
#
# Contents:
#   • theta_star(δ⋆), click_angles(T, δ⋆)              — canonical angles.
#   • compile_multiples(m, δ⋆), palindrome(seq)        — small sequence combinators.
#   • roots_of_unity(C), project_to_root(z, C)         — algebraic helpers on S¹.
#   • harmonic_response(x, δ⋆, C)                      — response at the C‑th harmonic.
#   • coherence_spectrum(X, δ⋆, maxC=16)               — quick spectrum over capacities.
#   • nearest_click(x, δ⋆)                              — (k, r) wrapper (sugary rung).
#
# All ops are torch‑based and device‑agnostic.

from __future__ import annotations
import math
from typing import Iterable, Tuple, Dict
import torch

from elementfold.ledger import (  # pragma: no cover
    phase as _phase,
    rung_residual as _rung_residual,
    char_kernel as _char_kernel,
)
# ———————————————————————————————————————————————————————————
# Angles & click sequences
# ———————————————————————————————————————————————————————————

def theta_star(delta: float) -> float:
    """
    θ⋆ = 2π · δ⋆
    Why this number?
      • RotaryClick uses a fixed rotation per time step: θ_t = t·θ⋆.
      • δ⋆ therefore controls the "tempo" on the circle; θ⋆ is its angle.
    """
    return 2.0 * math.pi * float(delta)

def click_angles(T: int, delta: float, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """
    Build angles θ_t = t · θ⋆ for t = 0..T−1 as a tensor (T,).
    We keep this in one place so downstream code uses a single convention.
    """
    th = theta_star(delta)
    dev = device or torch.device("cpu")
    return (torch.arange(int(T), device=dev, dtype=dtype or torch.float32) * float(th))

def compile_multiples(m: Iterable[int], delta: float) -> list[float]:
    """
    Turn a list of integer multiples into the corresponding angle set:
      m = [0,1,2] → [0·θ⋆, 1·θ⋆, 2·θ⋆].
    """
    th = theta_star(delta)
    return [int(k) * th for k in m]

def palindrome(seq: Iterable[int]) -> list[int]:
    """
    Simple symmetric sequence builder:
      seq + reversed(seq)
    Useful for symmetric probing schedules around a center.
    """
    s = list(seq)
    return s + s[::-1]

# ———————————————————————————————————————————————————————————
# Roots of unity helpers
# ———————————————————————————————————————————————————————————

def roots_of_unity(C: int, device: torch.device | None = None, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    """
    ζ^a for a=0..C−1 on the unit circle (complex tensor, shape (C,)).
    """
    C = int(max(1, C))
    dev = device or torch.device("cpu")
    a = torch.arange(C, device=dev, dtype=torch.float32)
    z = torch.exp(2j * math.pi * a / float(C))
    return z.to(dtype)

def project_to_root(z: torch.Tensor, C: int) -> torch.Tensor:
    """
    Project complex phases z (any shape) to the nearest C‑th root of unity.
    We do this by snapping the angle to the nearest multiple of 2π/C.
    """
    # Get angle in (−π, π], build indices, and quantize to nearest multiple.
    angle = torch.angle(z.to(torch.complex64))
    step = 2.0 * math.pi / float(max(1, C))
    k = torch.round(angle / step)
    snapped = k * step
    return torch.exp(1j * snapped)

# ———————————————————————————————————————————————————————————
# Harmonic / spectral readouts on the ledger
# ———————————————————————————————————————————————————————————

def harmonic_response(x: torch.Tensor, delta: float, C: int) -> torch.Tensor:
    """
    Response at the C‑th harmonic:
      r_C = |⟨cos(2π·C·x/δ⋆)⟩|  (magnitude on the circle)
    Accepts x with shape (B,), (T,), or (B,T); we always average over available dims.
    """
    # Build a character at C‑fold frequency by scaling x/δ⋆.
    # cos(2π·C·x/δ⋆) = cos(2π·(x / (δ⋆/C)))
    delta_C = float(delta) / float(max(1, C))
    if x.dim() == 2:
        xc = x.mean(dim=1)
    else:
        xc = x
    # Use the project’s character kernel (cosine on circle) for consistency.
    # char_kernel(a,b,δ) = cos(2π(a−b)/δ); here we compare to 0 baseline ⇒ cos(2π·x/δ)
    val = _char_kernel(xc, torch.zeros_like(xc), delta_C)
    # Return absolute mean (magnitude). Shape collapses to scalar per batch row or all.
    return val.mean().abs()

def coherence_spectrum(X: torch.Tensor, delta: float, maxC: int = 16) -> Dict[int, float]:
    """
    Compute a quick spectrum across harmonics C=1..maxC:
      spec[C] = harmonic_response(X, δ⋆, C)
    Good for sanity: low C should dominate if seats/blocks are coherent.
    """
    out: Dict[int, float] = {}
    for C in range(1, int(maxC) + 1):
        r = float(harmonic_response(X, delta, C).item())
        out[C] = r
    return out

# ———————————————————————————————————————————————————————————
# Sugary wrappers around rung/phase (single import for experiments)
# ———————————————————————————————————————————————————————————

def nearest_click(x: torch.Tensor, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (k, r) where k is the integer rung index and r ∈ (−δ⋆/2, δ⋆/2] is the residual.
    Exactly mirrors ledger.rung_residual but kept here for locality in notebooks.
    """
    return _rung_residual(x, delta)

def phase(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    e^{i·2πx/δ⋆} complex phase on the circle (thin wrapper to ledger.phase).
    """
    return _phase(x, delta)

__all__ = [
    "theta_star", "click_angles", "compile_multiples", "palindrome",
    "roots_of_unity", "project_to_root",
    "harmonic_response", "coherence_spectrum",
    "nearest_click", "phase",
]
