# ElementFold · utils/math.py
# ============================================================
# Small numerical helpers that keep training and inference stable.
#
# This module collects all the "gentle" math the coherence engine uses:
#   • Safe wrapping and modular arithmetic for angles or periodic signals.
#   • Stable norms and normalization for vectors and tensors.
#   • Exponential-moving-average utilities for slow parameter relaxation.
#   • The "relaxation clock" — a tiny bookkeeping system for how small shares
#     of change (η·ds) accumulate into a total number of folds ℱ.
#   • Simple diffusion/decay updates for fields defined on grids.
#
# Each function carries two layers of comments:
#   — Technical: what the code does exactly.
#   — Plain words: what that means physically inside the ElementFold model.
#
# Every function is torch-native, device-aware, and written to stay stable even
# when values are near zero or extremely large.
# ============================================================

from __future__ import annotations
import math
from typing import Sequence, Tuple
import torch

# ============================================================
# 1. Wrapping and modular arithmetic
# ============================================================
# In a coherence engine, many signals are cyclic: phases, angles, resonances.
# These helpers keep numbers within a consistent range so they never drift
# outside their natural period.

def mod(x: torch.Tensor, m: float | int) -> torch.Tensor:
    """Compute x modulo m (torch-safe)."""
    return torch.remainder(x, float(m))

def wrap_centered(x: torch.Tensor, period: float) -> torch.Tensor:
    """
    Wrap any real value into the centered interval (−period/2, +period/2].
    This is useful for phase differences: it ensures continuity around zero.
    """
    p = float(period)
    # Shift half-period forward so remainder is positive, then shift back.
    x = torch.remainder(x + 0.5 * p, p)
    # Remainder can be zero or negative, push negatives one period forward.
    x = torch.where(x <= 0, x + p, x)
    # Center back around zero.
    return x - 0.5 * p

def wrap_angle(x: torch.Tensor) -> torch.Tensor:
    """Wrap angles (radians) into the canonical interval (−π, π]."""
    return wrap_centered(x, 2.0 * math.pi)


# ============================================================
# 2. Norms and normalization
# ============================================================
# These keep magnitudes well-behaved even when tensors are nearly zero.

def safe_norm(x: torch.Tensor, p: float = 2.0, dim: int = -1,
              keepdim: bool = False, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute ‖x‖ₚ safely with a small ε floor to avoid divide-by-zero.
    """
    if p == 2.0:
        # Fast path for the common Euclidean norm.
        return torch.sqrt(torch.clamp(torch.sum(x * x, dim=dim, keepdim=keepdim), min=eps))
    if p == 1.0:
        return torch.clamp(torch.sum(torch.abs(x), dim=dim, keepdim=keepdim), min=eps)
    # Generic p-norm.
    return torch.clamp(torch.sum(torch.abs(x) ** p, dim=dim, keepdim=keepdim), min=eps) ** (1.0 / p)

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize x so that its L2 norm along the given dimension equals 1.
    In ElementFold this is used to keep vectors on the unit sphere in latent space.
    """
    n = safe_norm(x, p=2.0, dim=dim, keepdim=True, eps=eps)
    return x / n

def cosine_sim(a: torch.Tensor, b: torch.Tensor, dim: int = -1,
               eps: float = 1e-8) -> torch.Tensor:
    """
    Cosine similarity between two tensors along `dim`.
    Measures alignment of directions, ignoring magnitude.
    """
    a_hat = l2_normalize(a, dim=dim, eps=eps)
    b_hat = l2_normalize(b, dim=dim, eps=eps)
    return torch.sum(a_hat * b_hat, dim=dim)

def rms(x: torch.Tensor, dim: int | None = None, keepdim: bool = False,
        eps: float = 1e-8) -> torch.Tensor:
    """
    Root-mean-square (energy measure) with ε for numerical safety.
    """
    if dim is None:
        return torch.sqrt(torch.clamp(torch.mean(x * x), min=eps))
    return torch.sqrt(torch.clamp(torch.mean(x * x, dim=dim, keepdim=keepdim), min=eps))


# ============================================================
# 3. Soft operations: log-sum-exp and soft clipping
# ============================================================
# Softmax-style operations appear in phase alignment and energy pooling.

def logsumexp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """Stable computation of log(∑exp(x)) avoiding overflow."""
    m = torch.max(x, dim=dim, keepdim=True).values
    z = torch.log(torch.clamp(torch.sum(torch.exp(x - m), dim=dim, keepdim=True), min=1e-20)) + m
    return z if keepdim else z.squeeze(dim)

def softclip(x: torch.Tensor, lo: float = -1.0, hi: float = 1.0) -> torch.Tensor:
    """
    Smoothly clamp x into [lo, hi] using tanh.
    Unlike hard clipping, this keeps gradients continuous.
    """
    mid = 0.5 * (hi + lo)
    half = 0.5 * (hi - lo)
    return mid + half * torch.tanh((x - mid) / max(1e-6, half))


# ============================================================
# 4. Exponential-moving-average utilities
# ============================================================
# These implement a slow relaxation between two states.
# target ← decay·target + (1−decay)·source

@torch.no_grad()
def ema_(target: torch.nn.Module | torch.Tensor,
         source: torch.nn.Module | torch.Tensor,
         decay: float = 0.99) -> None:
    """
    In-place exponential moving average.

    Works for:
      • nn.Module → updates parameters of one model from another.
      • torch.Tensor → updates values directly.
    """
    d = float(decay)
    if isinstance(target, torch.nn.Module) and isinstance(source, torch.nn.Module):
        # Match parameters by name.
        t_params = dict(target.named_parameters())
        s_params = dict(source.named_parameters())
        for k, tp in t_params.items():
            sp = s_params.get(k)
            if sp is None or sp.shape != tp.shape:
                continue
            tp.data.mul_(d).add_(sp.data, alpha=(1.0 - d))
    elif isinstance(target, torch.Tensor) and isinstance(source, torch.Tensor):
        target.mul_(d).add_(source, alpha=(1.0 - d))
    else:
        raise TypeError("ema_ expects both args to be nn.Module or both Tensors")


# ============================================================
# 5. The Relaxation Clock (η, ℱ)
# ============================================================
# This section formalizes the "counting of folds" idea:
# each segment of a path contributes η·ds to a cumulative total ℱ.
# From that, one can derive color stretch (redshift) or attenuation.

def _device_of(*xs) -> torch.device:
    """Return the first tensor’s device, or CPU if none found."""
    for x in xs:
        if isinstance(x, torch.Tensor):
            return x.device
    return torch.device("cpu")

def _to_tensor(x: float | torch.Tensor, *, device: torch.device | None = None) -> torch.Tensor:
    """Convert scalars to float tensors on the given device."""
    return torch.as_tensor(x, dtype=torch.float32, device=device)

def folds_from_path(eta: float | torch.Tensor,
                    ds: float | torch.Tensor,
                    dim: int = -1) -> torch.Tensor:
    """
    Compute total folds ℱ = ∑ η·ds along the specified dimension.
    Physically: how much "sharing" (relaxation) a wave has experienced.
    """
    dev = _device_of(eta, ds)
    e, d = _to_tensor(eta, device=dev), _to_tensor(ds, device=dev)
    prod = e * d
    return prod.sum(dim=dim) if prod.ndim > 0 else prod

def compose_folds(*folds: float | torch.Tensor) -> torch.Tensor:
    """Combine multiple fold counts by addition (ℱ_total = ∑ℱᵢ)."""
    if not folds:
        return _to_tensor(0.0)
    dev = _device_of(*folds)
    return torch.stack([_to_tensor(f, device=dev) for f in folds]).sum(dim=0)

def accumulate_folds(prev: float | torch.Tensor,
                     inc: float | torch.Tensor) -> torch.Tensor:
    """Incrementally update fold count: ℱ ← ℱ + inc."""
    dev = _device_of(prev, inc)
    return _to_tensor(prev, device=dev) + _to_tensor(inc, device=dev)

def redshift_from_folds(folds: float | torch.Tensor) -> torch.Tensor:
    """
    Convert fold count ℱ into color stretch z using 1+z = e^ℱ.
    Uses expm1 for accuracy when ℱ is small.
    """
    f = _to_tensor(folds, device=_device_of(folds))
    return torch.expm1(f)

def attenuation_from_folds(folds: float | torch.Tensor,
                           power: float = 1.0) -> torch.Tensor:
    """
    Compute attenuation factor A = e^{−power·ℱ}.
      • power=1 → amplitude attenuation
      • power=2 → surface-brightness tilt (energy+rate)
    """
    f = _to_tensor(folds, device=_device_of(folds))
    return torch.exp(-float(power) * f)

def share_decay_update(y: float | torch.Tensor,
                       eta: float | torch.Tensor,
                       ds: float | torch.Tensor,
                       power: float = 1.0) -> torch.Tensor:
    """
    Apply one small sharing step to a signal y:
        y_out = y * exp(−power·η·ds)
    Conceptually: one more tiny click of relaxation along the path.
    """
    dev = _device_of(y, eta, ds)
    y0 = _to_tensor(y, device=dev)
    A = attenuation_from_folds(_to_tensor(eta, device=dev) * _to_tensor(ds, device=dev),
                               power=power)
    return y0 * A


# ============================================================
# 6. Laplacian and Relaxation Step for Fields
# ============================================================
# This section implements a discrete version of:
#     ∂φ/∂t = −λ(φ − φ∞) + D∇²φ
# Used to smooth and let go simultaneously.

def _as_tuple(x: int | Sequence[int]) -> Tuple[int, ...]:
    """Ensure an integer or sequence becomes a tuple."""
    return (x,) if isinstance(x, int) else tuple(int(d) for d in x)

def laplacian(x: torch.Tensor,
              dims: int | Sequence[int] = (-1,),
              spacing: float | Sequence[float] = 1.0,
              boundary: str = "periodic") -> torch.Tensor:
    """
    Compute the N-D Laplacian of x with periodic boundaries.

    Plain words:
        Each point looks at its neighbors along `dims` and measures how
        different it is from their average. Positive values mean "peaks"
        that will relax down; negative values mean "valleys" that will rise.
    """
    if boundary != "periodic":
        raise ValueError("laplacian: only boundary='periodic' is supported")
    dims = _as_tuple(dims)
    # Accept either one spacing or one per dimension.
    spacings = [float(spacing)] * len(dims) if isinstance(spacing, (int, float)) \
                else [float(s) for s in spacing]
    out = torch.zeros_like(x)
    for d, dx in zip(dims, spacings):
        # Periodic neighbor difference: forward + backward − 2×center.
        out += (torch.roll(x, +1, d) + torch.roll(x, -1, d) - 2.0 * x) / (dx * dx)
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
    One explicit Euler step for the relaxation equation:
        dφ/dt = −λ(φ − φ∞) + D∇²φ

    Plain words:
        • λ controls how quickly each point gives up its excess tension.
        • D controls how strongly neighbors share their differences.
        • φ∞ is the calm baseline the field moves toward.
    """
    out = phi
    if lam:
        # Local letting-go toward calm baseline.
        out = out + float(dt) * (-float(lam)) * (out - _to_tensor(phi_inf, device=out.device))
    if D:
        # Spatial smoothing (diffusion).
        out = out + float(dt) * float(D) * laplacian(out, dims=dims, spacing=spacing, boundary=boundary)
    return out
