# ElementFold · quantum.py
# Click‑geometry & coherence algebra — high‑level helpers for experiments.
#
# Import policy: STRICT absolute import only (per project policy).
#   from elementfold.ledger import phase, rung_residual, char_kernel
#
# Contents:
#   • theta_star(δ⋆), click_angles(T, δ⋆)               — canonical angles
#   • compile_multiples(m, δ⋆), palindrome(seq)         — tiny sequence combinators
#   • roots_of_unity(C), project_to_root(z, C)          — algebra on S¹
#   • harmonic_response / _tensor                        — C‑th harmonic response (scalar / vectorized)
#   • coherence_spectrum / _tensor                       — quick spectrum across harmonics
#   • nearest_click(x, δ⋆), phase(x, δ⋆)                 — sugary wrappers
#   • kappa / p_half / kappa_p_half                      — convenience coherence readouts
#
# All ops are torch‑native, device/dtype aware, and @no_grad where appropriate.

from __future__ import annotations

import math
from typing import Iterable, Tuple, Dict

import torch

# — Strict absolute import (no fallbacks) —
from elementfold.core.ledger import (  # pragma: no cover
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
      • RotaryClick uses θ_t = t·θ⋆ (fixed rotation per step).
      • δ⋆ sets the circular “tempo”; θ⋆ is its angle.
    """
    return 2.0 * math.pi * float(delta)


def click_angles(
    T: int,
    delta: float,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Build angles θ_t = t · θ⋆ for t = 0..T−1 as a tensor of shape (T,).
    """
    th = theta_star(delta)
    dev = device or torch.device("cpu")
    dt = dtype or torch.float32
    return torch.arange(int(T), device=dev, dtype=dt) * float(th)


def compile_multiples(m: Iterable[int], delta: float) -> list[float]:
    """m = [0,1,2] → [0·θ⋆, 1·θ⋆, 2·θ⋆]."""
    th = theta_star(delta)
    return [int(k) * th for k in m]


def palindrome(seq: Iterable[int]) -> list[int]:
    """Symmetric probe builder: seq + reversed(seq)."""
    s = list(seq)
    return s + s[::-1]


# ———————————————————————————————————————————————————————————
# Roots of unity on S¹
# ———————————————————————————————————————————————————————————

def roots_of_unity(
    C: int,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """
    ζ^a for a=0..C−1 on the unit circle (complex tensor, shape (C,)).
    Implemented with torch.polar so it honors device/dtype.
    """
    C = int(max(1, C))
    dev = device or torch.device("cpu")
    a = torch.arange(C, device=dev, dtype=torch.float32)        # (C,)
    angles = (2.0 * math.pi / float(C)) * a                     # radians
    z = torch.polar(torch.ones_like(angles), angles)            # complex64 default
    return z.to(dtype=dtype)


def project_to_root(z: torch.Tensor, C: int) -> torch.Tensor:
    """
    Snap complex phases `z` to the nearest C‑th root of unity by quantizing angles
    to multiples of 2π/C, then reconstruct via torch.polar.
    """
    step = 2.0 * math.pi / float(max(1, C))
    ang = torch.angle(z.to(torch.complex64))
    k = torch.round(ang / step)                                 # nearest multiple index
    snapped = k * step                                          # snapped angle
    return torch.polar(torch.ones_like(snapped), snapped)       # e^{i·snapped}


# ———————————————————————————————————————————————————————————
# Harmonic / spectral readouts (scalar + vectorized tensor forms)
# ———————————————————————————————————————————————————————————

@torch.no_grad()
def harmonic_response(x: torch.Tensor, delta: float, C: int) -> torch.Tensor:
    """
    Response at the C‑th harmonic (scalar):
        r_C = |⟨cos(2π·C·x/δ⋆)⟩|

    Accepts x with shape (B,), (T,), or (B,T); we average over available dims.
    """
    delta_C = float(delta) / float(max(1, C))
    xc = x.mean(dim=1) if x.dim() == 2 else x
    val = _char_kernel(xc, torch.zeros_like(xc), delta_C)       # cos(2π·xc / δ⋆_C)
    return val.mean().abs()                                     # scalar magnitude


@torch.no_grad()
def harmonic_response_tensor(
    X: torch.Tensor,
    delta: float,
    Cs: torch.Tensor | None = None,
    maxC: int = 16,
    reduce_time: bool = True,
    reduce_batch: bool = False,
) -> torch.Tensor:
    """
    Vectorized harmonic response across multiple C at once.

    Args:
        X:            (B,T) or (T,) or (B,) real tensor.
        delta:        δ⋆.
        Cs:           optional 1‑D int tensor of harmonics; if None uses 1..maxC.
        maxC:         used when Cs is None.
        reduce_time:  if True and X has a time dim, average across T first.
        reduce_batch: if True and X has a batch dim, average across B at the end.

    Returns:
        R:
          • (len(Cs),)         if no batch and reduce_time=True
          • (B, len(Cs))       if batch exists and reduce_time=True and reduce_batch=False
          • (len(Cs),)         if batch exists and reduce_time=True and reduce_batch=True
    """
    x = X
    if x.dim() == 2:                 # (B,T)
        pass
    elif x.dim() == 1:               # treat as (1,T) by default
        x = x.unsqueeze(0)
    else:
        raise ValueError("X must be 1‑D or 2‑D tensor")

    if reduce_time:
        x = x.mean(dim=-1, keepdim=False)   # (B,)

    # harmonics
    if Cs is None:
        Cs = torch.arange(1, int(maxC) + 1, device=x.device, dtype=torch.float32)
    else:
        Cs = torch.as_tensor(Cs, device=x.device, dtype=torch.float32)

    # cos(2π·C·x/δ⋆) with broadcasting:
    xb = x.view(-1, 1).to(dtype=torch.float32)                   # (B,1)
    Cb = Cs.view(1, -1)                                          # (1,C)
    angles = (2.0 * math.pi / float(delta)) * (xb * Cb)          # (B,C)
    vals = torch.cos(angles).abs()                               # (B,C)
    if not reduce_batch:
        return vals                                              # (B,C)
    return vals.mean(dim=0)                                      # (C,)


@torch.no_grad()
def coherence_spectrum(X: torch.Tensor, delta: float, maxC: int = 16) -> Dict[int, float]:
    """
    Quick dictionary spectrum across harmonics C=1..maxC:
        spec[C] = |⟨cos(2π·C·X/δ⋆)⟩|
    """
    out: Dict[int, float] = {}
    for C in range(1, int(maxC) + 1):
        r = float(harmonic_response(X, delta, C).item())
        out[C] = r
    return out


@torch.no_grad()
def coherence_spectrum_tensor(
    X: torch.Tensor,
    delta: float,
    maxC: int = 16,
    per_sample: bool = False,
) -> torch.Tensor:
    """
    Tensor spectrum across C=1..maxC (vectorized).

    Args:
        X:           (B,T) or (T,) or (B,)
        delta:       δ⋆
        maxC:        highest harmonic to include
        per_sample:  return (B, C) if True (averaged over time only), else (C,) averaged over batch
    """
    if per_sample:
        return harmonic_response_tensor(X, delta, Cs=None, maxC=maxC, reduce_time=True, reduce_batch=False)
    return harmonic_response_tensor(X, delta, Cs=None, maxC=maxC, reduce_time=True, reduce_batch=True)


# ———————————————————————————————————————————————————————————
# Sugary wrappers around rung/phase + coherence readouts
# ———————————————————————————————————————————————————————————

def nearest_click(x: torch.Tensor, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    (k, r) where k is the integer rung index and r ∈ (−δ⋆/2, δ⋆/2] is the residual.
    Thin alias to ledger.rung_residual for locality in notebooks.
    """
    return _rung_residual(x, delta)


def phase(x: torch.Tensor, delta: float) -> torch.Tensor:
    """
    e^{i·2πx/δ⋆} complex phase on the circle (alias to ledger.phase).
    """
    return _phase(x, delta)


@torch.no_grad()
def kappa(X: torch.Tensor, delta: float) -> torch.Tensor:
    """
    κ = |⟨e^{i·2πX/δ⋆}⟩|  (phase concentration).
    If X is (B,T), we reduce over T and then report a scalar magnitude.
    """
    x = X.mean(dim=1) if X.dim() == 2 else X
    ph = phase(x, delta)
    return torch.abs(ph.mean())


@torch.no_grad()
def p_half(X: torch.Tensor, delta: float, eps: float = 1e-9) -> torch.Tensor:
    """
    p½ = P(|r| ≥ δ⋆/2 − ε) — fraction touching the half‑click boundary.
    If X is (B,T), we reduce over T and then report a scalar fraction.
    """
    x = X.mean(dim=1) if X.dim() == 2 else X
    _, r = _rung_residual(x, delta)
    return (r.abs() >= (float(delta) / 2.0 - float(eps))).float().mean()


@torch.no_grad()
def kappa_p_half(X: torch.Tensor, delta: float, eps: float = 1e-9) -> Dict[str, float]:
    """One‑stop: {'kappa': κ, 'p_half': p½} as Python floats."""
    return {"kappa": float(kappa(X, delta).item()), "p_half": float(p_half(X, delta, eps=eps).item())}


__all__ = [
    # angles & sequences
    "theta_star", "click_angles", "compile_multiples", "palindrome",
    # S¹ algebra
    "roots_of_unity", "project_to_root",
    # harmonic/spectral
    "harmonic_response", "harmonic_response_tensor",
    "coherence_spectrum", "coherence_spectrum_tensor",
    # rung/phase & coherence readouts
    "nearest_click", "phase", "kappa", "p_half", "kappa_p_half",
]
