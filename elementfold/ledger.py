# ElementFold · ledger.py
# ──────────────────────────────────────────────────────────────────────────────
# Geometry of coherence — the δ⋆ circle
#
# This module is the “small math” behind ElementFold:
#   • phase(x, δ⋆)          : map any real x to the unit circle e^{i·2πx/δ⋆}
#   • rung_residual(x, δ⋆)  : split x = k·δ⋆ + r with r ∈ (−½δ⋆, ½δ⋆]
#   • wrapped_distance      : shortest arc length on the circle (always ≤ ½δ⋆)
#   • periodic_mean / lerp  : circular mean and interpolation along the short arc
#   • seat_index[_int]      : continuous / integer seat indices within a click
#   • char_kernel / vm_kernel: exact cosine kernel and a smooth von‑Mises kernel
#   • invariants + checks   : tiny gauge‑free identities used across the codebase
#   • kappa / p_half        : coherence and half‑click contact rate (telemetry staples)
#
# All functions are vectorized (broadcasting like PyTorch ops), device‑agnostic,
# and carefully documented for non‑experts. No heavy dependencies.

from __future__ import annotations

import math
import torch

__all__ = [
    "phase", "rung_residual", "half_click_margin", "snap_to_rung",
    "seat_index", "seat_index_int",
    "wrapped_distance", "periodic_mean", "periodic_lerp",
    "char_kernel", "vm_kernel",
    "invariants", "check_identities",
    "kappa", "p_half",
]


# ──────────────────────────────────────────────────────────────────────────────
# Core circular mapping
# ──────────────────────────────────────────────────────────────────────────────

def phase(x: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Map a real position x onto the unit circle with period δ⋆ (delta).

        ϕ(x) = e^{i·2πx/δ⋆}

    Args:
        x:     real scalar or tensor (any broadcastable shape).
        delta: positive float — the fundamental “click” size δ⋆.

    Returns:
        Complex tensor with |ϕ| = 1 (dtype: complex64/complex128 based on x).
    """
    a = (2.0 * math.pi / float(delta)) * torch.as_tensor(x, dtype=torch.get_default_dtype())
    one = torch.ones_like(a)
    return torch.polar(one, a)  # radius=1, angle=a


def rung_residual(x: torch.Tensor | float, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Decompose x into an integer rung k and a signed residual r within the current click:

        x = k·δ⋆ + r,   with   r ∈ (−½δ⋆, ½δ⋆]   (open on the left, closed on the right).

    Why this interval? Having +½δ⋆ included and −½δ⋆ excluded removes a double‑count at the boundary.

    Implementation details:
      • We first do “round to nearest” with a symmetric trick k = ⌊x/δ⋆ + 0.5⌋.
      • Then we normalize the rare exact left‑edge case r = −½δ⋆ → (+½δ⋆, k−1)
        so r always obeys (−½δ⋆, ½δ⋆].

    Args:
        x:     real scalar or tensor.
        delta: click size δ⋆.

    Returns:
        (k, r): integer tensor k (int64) and residual tensor r (same dtype as x).
    """
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype())
    d = float(delta)

    # Round to nearest integer rung index (works for negative x as well).
    k = torch.floor((x_t / d) + 0.5).to(torch.int64)
    r = x_t - k.to(x_t.dtype) * d

    # Normalize exact left edge to the right edge so r ∈ (−½δ⋆, ½δ⋆]
    half = d / 2.0
    cond = r <= -half  # extremely rare unless x sits exactly at the boundary
    if cond.any():
        r = torch.where(cond, r + d, r)
        k = torch.where(cond, k - 1, k)

    return k, r


def half_click_margin(x: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Safety gap to the seat boundary (½δ⋆ from the center of a click):

        margin = ½δ⋆ − |r|

    Positive margin ⇒ “safe” (small perturbations won’t flip the rung).
    Negative margin ⇒ already beyond the boundary.

    Returns:
        Tensor, same shape as x, real.
    """
    _, r = rung_residual(x, delta)
    return (float(delta) / 2.0) - r.abs()


def snap_to_rung(x: torch.Tensor | float, delta: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Snap x to its nearest click center k·δ⋆.

    Returns:
        (x_snap, k, r) where:
          • x_snap = k·δ⋆ (exact center),
          • k      = integer rung index,
          • r      = residual within the click, r ∈ (−½δ⋆, ½δ⋆].
    """
    k, r = rung_residual(x, delta)
    x_snap = k.to(r.dtype) * float(delta)
    return x_snap, k, r


# ──────────────────────────────────────────────────────────────────────────────
# Seats inside a click (C “seats” partition one click)
# ──────────────────────────────────────────────────────────────────────────────

def seat_index(x: torch.Tensor | float, delta: float, C: int) -> torch.Tensor:
    """
    Continuous seat coordinate within a click with capacity C (e.g., C=6 → hexagonal),
    reported in [0, C).

    Implementation:
        s = (C/δ⋆)·x   wrapped into [0, C) with remainder.
    """
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype())
    C = int(C)
    s = (C / float(delta)) * x_t
    return torch.remainder(s, C)


def seat_index_int(x: torch.Tensor | float, delta: float, C: int) -> torch.Tensor:
    """
    Integer seat index in {0, 1, …, C−1} by *nearest* rounding in seat units.

    We round with the same symmetric trick as rungs and then wrap into the valid range.
    """
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype())
    C = int(C)
    s = (C / float(delta)) * x_t
    idx = torch.floor(s + 0.5).to(torch.int64)
    return torch.remainder(idx, C)


# ──────────────────────────────────────────────────────────────────────────────
# Distances and averages on the circle
# ──────────────────────────────────────────────────────────────────────────────

def wrapped_distance(x: torch.Tensor | float, y: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Shortest arc length between x and y on the δ⋆ circle (always in [0, ½δ⋆]).

    We compute the difference, wrap into (−½δ⋆, ½δ⋆], then take |·|.
    """
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype())
    y_t = torch.as_tensor(y, dtype=torch.get_default_dtype())
    d = (x_t - y_t).remainder(float(delta))              # [0, δ⋆)
    half = float(delta) / 2.0
    d = torch.where(d > half, d - float(delta), d)       # → (−½δ⋆, ½δ⋆]
    return d.abs()


def periodic_mean(x: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Circular mean of points on the δ⋆ circle:
      1) map to unit circle,
      2) average complex numbers,
      3) convert angle back to [0, δ⋆).

    Returns:
        Real scalar tensor in [0, δ⋆).
    """
    ph = phase(x, delta)
    ph_mean = ph.mean()
    ang = torch.angle(ph_mean)                           # (−π, π]
    pos = (ang * float(delta)) / (2.0 * math.pi)
    return pos.remainder(float(delta))


def periodic_lerp(x: torch.Tensor | float, y: torch.Tensor | float, w: float, delta: float) -> torch.Tensor:
    """
    Interpolate from x to y along the *shortest* circular arc with weight w ∈ [0,1].

    Implementation: move (y−x) into (−½δ⋆, ½δ⋆], then take x + w·that, wrapped back to [0, δ⋆).
    """
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype())
    y_t = torch.as_tensor(y, dtype=torch.get_default_dtype())
    d = (y_t - x_t)                                      # raw difference
    d = (d + float(delta) / 2.0).remainder(float(delta)) - (float(delta) / 2.0)  # (−½δ⋆, ½δ⋆]
    z = x_t + float(w) * d
    return torch.remainder(z, float(delta))


# ──────────────────────────────────────────────────────────────────────────────
# Circular kernels (how “in tune” two positions are)
# ──────────────────────────────────────────────────────────────────────────────

def char_kernel(x: torch.Tensor | float, y: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Exact circular similarity (character of the 1‑D representation):

        K(x, y) = cos(2π(x−y)/δ⋆)

    Equals +1 when x and y coincide and −1 when they are half a click apart.
    """
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype())
    y_t = torch.as_tensor(y, dtype=torch.get_default_dtype())
    return torch.cos(2.0 * math.pi * (x_t - y_t) / float(delta))


def vm_kernel(x: torch.Tensor | float, y: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Smooth, positive kernel on the circle (von Mises–type):

        K(x, y) = exp( cos(2π(x−y)/δ⋆) )

    Peaks at 1 when x=y and decays smoothly as the wrapped distance grows.
    """
    x_t = torch.as_tensor(x, dtype=torch.get_default_dtype())
    y_t = torch.as_tensor(y, dtype=torch.get_default_dtype())
    return torch.exp(torch.cos(2.0 * math.pi * (x_t - y_t) / float(delta)))


# ──────────────────────────────────────────────────────────────────────────────
# Invariants and simple checks (tiny, handy utilities)
# ──────────────────────────────────────────────────────────────────────────────

def invariants(U: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gauge‑free channels built from a potential U (all pointwise):
        Γ = e^{+U}      (clock)
        n = e^{−2U}     (index)
        I = e^{+2U}     (intensity)

    Returns:
        (Γ, n, I) as tensors broadcasting with U.
    """
    U_t = torch.as_tensor(U, dtype=torch.get_default_dtype())
    Gam = torch.exp(+U_t)
    nidx = torch.exp(-2.0 * U_t)
    I = torch.exp(+2.0 * U_t)
    return Gam, nidx, I


def check_identities(U: torch.Tensor | float, eps: float = 1e-6) -> dict:
    """
    Verify the simple multiplicative identities:
        Γ · n^{1/2} ≈ 1   and   I · n ≈ 1.

    Returns:
        {'err_Gam_nhalf': max|Γ√n − 1|,
         'err_I_n':      max|In − 1|,
         'ok':           both ≤ eps}
    """
    Gam, n, I = invariants(U)
    lhs1 = Gam * torch.sqrt(n)
    lhs2 = I * n
    err1 = float((lhs1 - 1.0).abs().max().item())
    err2 = float((lhs2 - 1.0).abs().max().item())
    return {"err_Gam_nhalf": err1, "err_I_n": err2, "ok": bool(err1 <= eps and err2 <= eps)}


# ──────────────────────────────────────────────────────────────────────────────
# Telemetry primitives (coherence and boundary contact)
# ──────────────────────────────────────────────────────────────────────────────

def kappa(x: torch.Tensor | float, delta: float) -> float:
    """
    Phase concentration κ ∈ [0,1]: magnitude of the mean unit‑phase vector.
    κ≈1 means phases are tightly aligned; κ≈0 means they are spread out.
    """
    ph = phase(x, delta)
    return float(torch.abs(ph.mean()).item())


def p_half(x: torch.Tensor | float, delta: float, eps: float = 1e-9) -> float:
    """
    Fraction of samples on or beyond the half‑click boundary (i.e., |r| ≥ ½δ⋆ − ε).

    A larger value means you are skimming the boundary and more likely to flip rungs.
    """
    _, r = rung_residual(x, delta)
    return float((r.abs() >= (float(delta) / 2.0 - float(eps))).float().mean().item())
