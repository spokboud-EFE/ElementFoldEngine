# ElementFold · ledger.py
# ──────────────────────────────────────────────────────────────────────────────
# Geometry of coherence — the δ⋆ circle
#
# This module is the “small math” behind ElementFold:
#   • phase(x, δ⋆)           : map any real x to the unit circle e^{i·2πx/δ⋆}
#   • rung_residual(x, δ⋆)   : split x = k·δ⋆ + r with r ∈ (−½δ⋆, ½δ⋆]
#   • half_click_margin      : safety gap to the seat boundary (½δ⋆)
#   • snap_to_rung           : project x to its nearest click center k·δ⋆
#   • wrapped_distance       : shortest arc length on the circle (≤ ½δ⋆)
#   • periodic_mean / lerp   : circular mean and interpolation along the short arc
#   • seat_index[_int]       : continuous / integer seat indices within a click
#   • char_kernel / vm_kernel: cosine and von‑Mises circular kernels
#   • invariants + checks    : tiny gauge‑free identities (telemetry helpers)
#   • kappa / p_half         : phase concentration and half‑click contact rate
#
# All functions are vectorized, device‑agnostic, and torch‑native.

from __future__ import annotations

import math
import torch

__all__ = [
    "phase",
    "rung_residual",
    "half_click_margin",
    "snap_to_rung",
    "seat_index",
    "seat_index_int",
    "wrapped_distance",
    "periodic_mean",
    "periodic_lerp",
    "char_kernel",
    "vm_kernel",
    "invariants",
    "check_identities",
    "kappa",
    "p_half",
]

# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_tensor(x: torch.Tensor | float, *, like: torch.Tensor | None = None,
                   dtype: torch.dtype | None = None) -> torch.Tensor:
    """Convert x to a tensor on the same device as `like` (if given)."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype) if dtype is not None else x
    device = like.device if isinstance(like, torch.Tensor) else None
    return torch.tensor(x, dtype=dtype or torch.get_default_dtype(), device=device)

def _require_positive_delta(delta: float) -> float:
    d = float(delta)
    if not (d > 0.0):
        raise ValueError("delta (δ⋆) must be positive.")
    return d

def _real_dtype(x: torch.Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32

# ──────────────────────────────────────────────────────────────────────────────
# Core circular mapping
# ──────────────────────────────────────────────────────────────────────────────

def phase(x: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Map a real position x onto the unit circle with period δ⋆:

        ϕ(x) = e^{i·2πx/δ⋆}

    Returns a complex tensor (complex64/complex128).
    """
    d = _require_positive_delta(delta)
    x_t = _ensure_tensor(x)
    # Use float32/float64 trig for stability (even if input is bf16/fp16).
    rdtype = _real_dtype(x_t)
    a = (2.0 * math.pi / d) * x_t.to(rdtype)
    one = torch.ones_like(a, dtype=rdtype)
    return torch.polar(one, a)  # → complex dtype matched to rdtype

def rung_residual(x: torch.Tensor | float, delta: float) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Decompose x into an integer rung k and a signed residual r within the current click:

        x = k·δ⋆ + r,   with   r ∈ (−½δ⋆, ½δ⋆]  (left‑open, right‑closed).

    Ties are rounded with k = ⌊x/δ⋆ + 1/2⌋, then the rare left‑edge r = −½δ⋆
    is normalized to (+½δ⋆, k−1).
    """
    d = _require_positive_delta(delta)
    x_t = _ensure_tensor(x)
    k = torch.floor((x_t / d) + 0.5).to(torch.int64)
    r = x_t - k.to(x_t.dtype) * d

    half = d / 2.0
    # Normalize exact left edge to right edge: (−½δ⋆) → (+½δ⋆, k−1)
    cond = r <= -half
    if torch.any(cond):
        r = torch.where(cond, r + d, r)
        k = torch.where(cond, k - 1, k)
    return k, r

def half_click_margin(x: torch.Tensor | float, delta: float) -> torch.Tensor:
    """Safety gap to the boundary: margin = ½δ⋆ − |r|."""
    d = _require_positive_delta(delta)
    _, r = rung_residual(x, d)
    return (d / 2.0) - r.abs()

def snap_to_rung(x: torch.Tensor | float, delta: float
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Project x to its nearest click center k·δ⋆.

    Returns:
      x_snap (center), k (int64 rung), r (residual in (−½δ⋆, ½δ⋆]).
    """
    d = _require_positive_delta(delta)
    k, r = rung_residual(x, d)
    x_snap = k.to(r.dtype) * d
    return x_snap, k, r

# ──────────────────────────────────────────────────────────────────────────────
# Seats inside a click (C seats partition one click)
# ──────────────────────────────────────────────────────────────────────────────

def seat_index(x: torch.Tensor | float, delta: float, C: int) -> torch.Tensor:
    """
    Continuous seat coordinate within a click with capacity C, reported in [0, C).

    s = (C/δ⋆)·x  wrapped into [0, C) with remainder.
    """
    if C <= 0:
        raise ValueError("C (seat capacity) must be a positive integer.")
    d = _require_positive_delta(delta)
    x_t = _ensure_tensor(x)
    s = (float(C) / d) * x_t.to(_real_dtype(x_t))
    return torch.remainder(s, float(C)).to(x_t.dtype)

def seat_index_int(x: torch.Tensor | float, delta: float, C: int) -> torch.Tensor:
    """
    Integer seat index in {0,…,C−1} via nearest rounding in seat units.
    """
    if C <= 0:
        raise ValueError("C (seat capacity) must be a positive integer.")
    d = _require_positive_delta(delta)
    x_t = _ensure_tensor(x)
    s = (float(C) / d) * x_t.to(_real_dtype(x_t))
    idx = torch.floor(s + 0.5).to(torch.int64)
    return torch.remainder(idx, C)

# ──────────────────────────────────────────────────────────────────────────────
# Distances and averages on the circle
# ──────────────────────────────────────────────────────────────────────────────

def wrapped_distance(x: torch.Tensor | float, y: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Shortest arc length between x and y on the δ⋆ circle (in [0, ½δ⋆]).

    Implementation: wrap (x−y) to (−½δ⋆, ½δ⋆], then take |·|.
    """
    d = _require_positive_delta(delta)
    x_t = _ensure_tensor(x)
    y_t = _ensure_tensor(y, like=x_t, dtype=x_t.dtype)
    diff = torch.remainder(x_t - y_t, d)                  # [0, δ⋆)
    half = d / 2.0
    diff = torch.where(diff > half, diff - d, diff)       # (−½δ⋆, ½δ⋆]
    return diff.abs()

def periodic_mean(x: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Circular mean on the δ⋆ circle:
      map to unit phases → average complex → angle → [0, δ⋆).

    If phases cancel (mean magnitude≈0), returns 0 (conventional choice).
    """
    d = _require_positive_delta(delta)
    ph = phase(x, d)
    m = ph.mean()
    mag = torch.abs(m)
    rdtype = torch.float64 if m.dtype == torch.complex128 else torch.float32
    if torch.lt(mag, torch.tensor(1e-12, dtype=rdtype, device=mag.device)):
        return torch.zeros((), dtype=rdtype, device=mag.device)
    ang = torch.angle(m)                                   # (−π, π]
    pos = (ang * d) / (2.0 * math.pi)
    return torch.remainder(pos, d)

def periodic_lerp(x: torch.Tensor | float, y: torch.Tensor | float,
                  w: float, delta: float) -> torch.Tensor:
    """
    Interpolate from x to y along the shortest circular arc with weight w∈[0,1].
    """
    d = _require_positive_delta(delta)
    x_t = _ensure_tensor(x)
    y_t = _ensure_tensor(y, like=x_t, dtype=x_t.dtype)
    rdtype = _real_dtype(x_t)
    diff = (y_t.to(rdtype) - x_t.to(rdtype))
    diff = (diff + d / 2.0).remainder(d) - (d / 2.0)      # (−½δ⋆, ½δ⋆]
    z = x_t.to(rdtype) + float(w) * diff
    return torch.remainder(z, d).to(x_t.dtype)

# ──────────────────────────────────────────────────────────────────────────────
# Circular kernels
# ──────────────────────────────────────────────────────────────────────────────

def char_kernel(x: torch.Tensor | float, y: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Exact circular similarity:

        K(x, y) = cos(2π(x−y)/δ⋆)
    """
    d = _require_positive_delta(delta)
    x_t = _ensure_tensor(x)
    y_t = _ensure_tensor(y, like=x_t, dtype=x_t.dtype)
    rdtype = _real_dtype(x_t)
    return torch.cos(2.0 * math.pi * (x_t.to(rdtype) - y_t.to(rdtype)) / d).to(x_t.dtype)

def vm_kernel(x: torch.Tensor | float, y: torch.Tensor | float, delta: float) -> torch.Tensor:
    """
    Smooth positive kernel on the circle (von‑Mises‑type):

        K(x, y) = exp( cos(2π(x−y)/δ⋆) )
    """
    d = _require_positive_delta(delta)
    x_t = _ensure_tensor(x)
    y_t = _ensure_tensor(y, like=x_t, dtype=x_t.dtype)
    rdtype = _real_dtype(x_t)
    z = torch.cos(2.0 * math.pi * (x_t.to(rdtype) - y_t.to(rdtype)) / d)
    return torch.exp(z).to(x_t.dtype)

# ──────────────────────────────────────────────────────────────────────────────
# Invariants and simple checks
# ──────────────────────────────────────────────────────────────────────────────

def invariants(U: torch.Tensor | float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gauge‑free channels built from a potential U (pointwise):
        Γ = e^{+U}      (clock)
        n = e^{−2U}     (index)
        I = e^{+2U}     (intensity)
    """
    U_t = _ensure_tensor(U)
    rdtype = _real_dtype(U_t)
    U_r = U_t.to(rdtype)
    Gam = torch.exp(+U_r)
    nidx = torch.exp(-2.0 * U_r)
    I = torch.exp(+2.0 * U_r)
    # Return in the input's real dtype
    return Gam.to(U_t.dtype), nidx.to(U_t.dtype), I.to(U_t.dtype)

def check_identities(U: torch.Tensor | float, eps: float = 1e-6) -> dict:
    """
    Verify:
        Γ · n^{1/2} ≈ 1   and   I · n ≈ 1.
    """
    Gam, n, I = invariants(U)
    lhs1 = Gam * torch.sqrt(n)
    lhs2 = I * n
    err1 = float((lhs1 - 1.0).abs().amax().item())
    err2 = float((lhs2 - 1.0).abs().amax().item())
    return {"err_Gam_nhalf": err1, "err_I_n": err2, "ok": bool(err1 <= eps and err2 <= eps)}

# ──────────────────────────────────────────────────────────────────────────────
# Telemetry primitives
# ──────────────────────────────────────────────────────────────────────────────

def kappa(x: torch.Tensor | float, delta: float) -> float:
    """
    Phase concentration κ ∈ [0,1]: |mean unit‑phase|.
    κ≈1 ⇒ tightly aligned phases; κ≈0 ⇒ spread out.
    """
    d = _require_positive_delta(delta)
    ph = phase(x, d)
    return float(torch.abs(ph.mean()).item())

def p_half(x: torch.Tensor | float, delta: float, eps: float = 1e-9) -> float:
    """
    Fraction of samples on or beyond the half‑click boundary:
        |r| ≥ ½δ⋆ − ε
    """
    d = _require_positive_delta(delta)
    _, r = rung_residual(x, d)
    thr = (d / 2.0) - float(eps)
    return float((r.abs() >= thr).to(torch.float32).mean().item())
