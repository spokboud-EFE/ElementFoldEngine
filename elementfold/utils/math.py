# ElementFold · utils/math.py
# Small numerical helpers that keep training/inference stable.
#
# Additions in this version:
#   • Relaxation clock (diffusion/decay) primitives:
#       folds_from_path(eta, ds) → ℱ = ∑ η·ds
#       redshift_from_folds(ℱ)   → z where 1+z = e^ℱ
#       attenuation_from_folds(ℱ, power) → e^{−power·ℱ}  (power=2 → surface brightness tilt)
#       compose_folds(*ℱ_i) / accumulate_folds(prev, inc)
#   • Lightweight Laplacian (periodic) and a discrete diffuse+decay step:
#       laplacian(x, dims, spacing, boundary='periodic')
#       relax_diffuse_step(φ, dt, λ, D, ...)
#
# Plain words:
#   The “relaxation clock” counts small shares along a path: each little η·ds adds
#   one more tiny fold. Sum them to get ℱ; exponentiate to read color stretch
#   (1+z=e^ℱ) or brightness tilt (∝ e^{−2ℱ}). The Laplacian/decay helpers let you
#   run a gentle smoothing + letting‑go step when φ is defined on a grid.

from __future__ import annotations

import math
from typing import Tuple, Iterable, Sequence

import torch


# ———————————————————————————————————————————————————————————
# Wrapping / modular arithmetic
# ———————————————————————————————————————————————————————————

def mod(x: torch.Tensor, m: float | int) -> torch.Tensor:
    """x mod m, preserving dtype/device."""
    return torch.remainder(x, float(m))


def wrap_centered(x: torch.Tensor, period: float) -> torch.Tensor:
    """
    Wrap real values into the centered interval (−period/2, +period/2].
    """
    p = float(period)
    x = torch.remainder(x + 0.5 * p, p)
    x = torch.where(x <= 0, x + p, x)
    return x - 0.5 * p


def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap radians into (−π, π]."""
    return wrap_centered(x, 2.0 * math.pi)


# ———————————————————————————————————————————————————————————
# Stable norms / normalization
# ———————————————————————————————————————————————————————————

def safe_norm(x: torch.Tensor, p: float = 2.0, dim: int = -1, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    ‖x‖_p stabilized with ε.
    """
    if p == 2.0:
        return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=keepdim), min=eps))
    if p == 1.0:
        return torch.clamp(torch.sum(torch.abs(x), dim=dim, keepdim=keepdim), min=eps)
    # generic
    return torch.clamp(torch.sum(torch.abs(x) ** p, dim=dim, keepdim=keepdim), min=eps) ** (1.0 / p)


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """x / max(‖x‖₂, ε)"""
    n = safe_norm(x, p=2.0, dim=dim, keepdim=True, eps=eps)
    return x / n


def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity between tensors along `dim`.
    """
    a_hat = l2_normalize(a, dim=dim, eps=eps)
    b_hat = l2_normalize(b, dim=dim, eps=eps)
    return torch.sum(a_hat * b_hat, dim=dim)


def rms(x: torch.Tensor, dim: int | None = None, keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Root-mean-square with ε for stability.
    """
    if dim is None:
        return torch.sqrt(torch.clamp(torch.mean(x * x), min=eps))
    return torch.sqrt(torch.clamp(torch.mean(x * x, dim=dim, keepdim=keepdim), min=eps))


# ———————————————————————————————————————————————————————————
# Log-sum-exp / soft utilities
# ———————————————————————————————————————————————————————————

def logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Stable log ∑ exp(x)."""
    m = torch.max(x, dim=dim, keepdim=True).values
    z = torch.log(torch.clamp(torch.sum(torch.exp(x - m), dim=dim, keepdim=True), min=1e-20)) + m
    return z if keepdim else z.squeeze(dim)


def softclip(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """Smoothly clamp x into [lo, hi] using tanh."""
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + half * torch.tanh((x - mid) / max(1e-6, half))


# ———————————————————————————————————————————————————————————
# EMA / parameter utilities
# ———————————————————————————————————————————————————————————

@torch.no_grad()
def ema_(target: torch.nn.Module | torch.Tensor, source: torch.nn.Module | torch.Tensor, decay: float = 0.99) -> None:
    """
    In-place exponential moving average:
        target ← decay * target + (1 - decay) * source
    Works for nn.Modules (matches parameters by name) or plain tensors.
    """
    d = float(decay)
    if isinstance(target, torch.nn.Module) and isinstance(source, torch.nn.Module):
        t_params = dict(target.named_parameters())
        s_params = dict(source.named_parameters())
        for k, tp in t_params.items():
            sp = s_params.get(k, None)
            if (sp is None) or (sp.data.shape != tp.data.shape):
                continue
            tp.data.mul_(d).add_(sp.data, alpha=(1.0 - d))
        return
    if isinstance(target, torch.Tensor) and isinstance(source, torch.Tensor):
        target.mul_(d).add_(source, alpha=(1.0 - d))
        return
    raise TypeError("ema_ expects both args to be nn.Module or both to be torch.Tensor")


# ———————————————————————————————————————————————————————————
# Relaxation clock (η, ℱ) and gentle decay helpers
# ———————————————————————————————————————————————————————————
# Plain words:
#   • η is a small, per‑segment “share rate”.
#   • ds is how long that segment is.
#   • ℱ is the count of “folds” (little shares that compounded): ℱ = ∑ η·ds.
#   • 1+z = e^ℱ (color stretch). Surface brightness picks up an extra e^{−2ℱ}.
#
# Shapes:
#   • eta, ds can be scalars, vectors, or tensors; they must broadcast together.
#   • Use `dim` to choose which axis to sum over (default: last axis).
#   • All routines are torch-native and keep the device of the inputs.

def _device_of(*xs: object) -> torch.device:
    for x in xs:
        if isinstance(x, torch.Tensor):
            return x.device
    return torch.device("cpu")


def _to_tensor(x: float | torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def folds_from_path(eta: float | torch.Tensor,
                    ds: float | torch.Tensor,
                    dim: int = -1) -> torch.Tensor:
    """
    Compute folds ℱ from a per‑segment share rate η and segment lengths ds.

    ℱ = ∑ η · ds   (sum along `dim`)

    Args
    ----
    eta : scalar or Tensor
        Share rate per unit path for each segment.
    ds  : scalar or Tensor
        Segment length(s), same shape/broadcast as `eta`.
    dim : int
        Dimension to sum across (default: last).

    Returns
    -------
    Tensor of folds ℱ with `eta`/`ds` broadcast shape reduced over `dim`.
    """
    dev = _device_of(eta, ds)
    e = _to_tensor(eta, device=dev)
    d = _to_tensor(ds, device=dev)
    prod = e * d
    return prod.sum(dim=dim) if prod.ndim > 0 else prod


def compose_folds(*folds: float | torch.Tensor) -> torch.Tensor:
    """
    Combine multiple fold counts by addition (ℱ_total = ∑ ℱ_i).
    """
    if len(folds) == 0:
        return _to_tensor(0.0)
    dev = _device_of(*folds)
    stack = [ _to_tensor(f, device=dev) for f in folds ]
    return torch.stack(stack, dim=0).sum(dim=0)


def accumulate_folds(prev: float | torch.Tensor, inc: float | torch.Tensor) -> torch.Tensor:
    """
    Incremental update: ℱ_new = ℱ_prev + inc
    """
    dev = _device_of(prev, inc)
    return _to_tensor(prev, device=dev) + _to_tensor(inc, device=dev)


def redshift_from_folds(folds: float | torch.Tensor) -> torch.Tensor:
    """
    z from folds ℱ, via 1+z = exp(ℱ).
    """
    f = _to_tensor(folds, device=_device_of(folds))
    # Use torch.expm1 for small-fold stability
    return torch.expm1(f)


def attenuation_from_folds(folds: float | torch.Tensor, power: float = 1.0) -> torch.Tensor:
    """
    Generic attenuation factor from ℱ: A = exp(−power · ℱ).
      • power=1 → amplitude-like
      • power=2 → surface brightness tilt (two powers for energy+rate)
    """
    f = _to_tensor(folds, device=_device_of(folds))
    return torch.exp(-float(power) * f)


def share_decay_update(y: float | torch.Tensor,
                       eta: float | torch.Tensor,
                       ds: float | torch.Tensor,
                       power: float = 1.0) -> torch.Tensor:
    """
    Apply one segment of “sharing” to a signal y:
        y_out = y * exp(−power · η · ds)

    This is a tiny convenience when stepping through segments.
    """
    dev = _device_of(y, eta, ds)
    y0 = _to_tensor(y, device=dev)
    A = attenuation_from_folds(_to_tensor(eta, device=dev) * _to_tensor(ds, device=dev), power=power)
    return y0 * A


# ———————————————————————————————————————————————————————————
# Laplacian (periodic) and a discrete diffuse+decay step
# ———————————————————————————————————————————————————————————
# Plain words:
#   φ wants to calm: local letting‑go pulls it toward φ∞, and smoothing shares
#   with neighbors. One explicit Euler step does:
#       φ ← φ + dt * ( −λ (φ − φ∞) + D ∇² φ )
#
# Notes:
#   • Default dims = (−1,) to avoid mixing batches/channels by accident.
#   • Boundary handling: we implement a safe, fast **periodic** Laplacian using
#     torch.roll. Other boundaries can be added later if needed.

def _as_tuple(x: int | Sequence[int]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    return tuple(int(d) for d in x)


def laplacian(x: torch.Tensor,
              dims: int | Sequence[int] = (-1,),
              spacing: float | Sequence[float] = 1.0,
              boundary: str = "periodic") -> torch.Tensor:
    """
    N‑D Laplacian with **periodic** boundaries via second differences.

    Args
    ----
    x        : Tensor
    dims     : int or sequence of ints (which axes to treat as spatial; default: last axis)
    spacing  : scalar or per‑dim grid spacing Δ (meters, pixels, arbitrary units)
    boundary : currently only 'periodic' is supported

    Returns
    -------
    Tensor with same shape as x: ∑_d (x_{+1} + x_{−1} − 2x) / Δ_d²
    """
    if boundary != "periodic":
        raise ValueError("laplacian: only boundary='periodic' is supported currently")

    dims = _as_tuple(dims)
    if isinstance(spacing, (tuple, list)):
        if len(spacing) != len(dims):
            raise ValueError("laplacian: spacing length must match dims length")
        spacings = [float(s) for s in spacing]
    else:
        spacings = [float(spacing)] * len(dims)

    out = torch.zeros_like(x)
    for d, dx in zip(dims, spacings):
        out = out + (torch.roll(x, shifts=+1, dims=d) + torch.roll(x, shifts=-1, dims=d) - 2.0 * x) / (dx * dx)
    return out


@torch.no_grad()
def relax_diffuse_step(phi: torch.Tensor,
                       dt: float,
                       lam: float = 0.0,
                       D: float = 0.0,
                       *,
                       spacing: float | Sequence[float] = 1.0,
                       dims: int | Sequence[int] = (-1,),
                       boundary: str = "periodic",
                       phi_inf: float | torch.Tensor = 0.0) -> torch.Tensor:
    """
    One explicit Euler step of:  dφ/dt = −λ(φ − φ∞) + D ∇²φ

    Args
    ----
    phi     : field tensor (any shape)
    dt      : time step
    lam     : local letting‑go rate λ (≥0)
    D       : smoothing strength D (≥0)
    spacing : grid spacing(s) for Laplacian
    dims    : spatial dims to smooth (default: last axis only)
    boundary: 'periodic' (others can be added later)
    phi_inf : calm baseline φ∞ (scalar or broadcastable tensor)

    Returns
    -------
    Updated φ with the same shape/device as input.

    Stability note
    --------------
    For explicit schemes, use conservative steps (e.g., in 1D, D·dt/Δ² ≲ 0.5).
    """
    out = phi

    if lam != 0.0:
        out = out + float(dt) * (-float(lam)) * (out - _to_tensor(phi_inf, device=out.device))

    if D != 0.0:
        lap = laplacian(out, dims=dims, spacing=spacing, boundary=boundary)
        out = out + float(dt) * float(D) * lap

    return out
