"""
core/physics/smoothing.py — Gentle Filters and Diffusion 🌬️

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• Relaxation calms differences, but smoothing decides how
  that calm spreads across space.
• Here live the filters: gradient, Laplacian, Gaussian.
• NumPy paints softly by default; Torch takes the brush only
  when scenes grow too large or slow.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
from elementfold.core.physics.field import Field, BACKEND

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]


# ====================================================================== #
# 🎚️ Basic finite differences
# ====================================================================== #
def gradient(field: Field) -> Tuple[Field, ...]:
    """
    Compute spatial gradients ∇Φ using central differences.
    Returns a tuple of Field components.
    """
    arr = field.data
    backend = field.backend
    start = time.perf_counter()

    try:
        if backend == "torch" and _TORCH_AVAILABLE:
            grads = torch.gradient(arr)
            comps = [Field(name=f"{field.name}_grad{i}", data=g, backend="torch") for i, g in enumerate(grads)]
        else:
            grads_np = np.gradient(np.asarray(arr))
            comps = [Field(name=f"{field.name}_grad{i}", data=g, backend="numpy") for i, g in enumerate(grads_np)]
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[smoothing] gradient fallback: {exc}")
        grads_np = np.gradient(np.asarray(arr))
        comps = [Field(name=f"{field.name}_grad{i}", data=g, backend="numpy") for i, g in enumerate(grads_np)]
    finally:
        BACKEND.record(time.perf_counter() - start)

    return tuple(comps)


def laplacian(field: Field) -> Field:
    """
    Compute Laplacian ∇²Φ = div(∇Φ) using second-order finite differences.
    This is a thin alias that defers to relaxation.laplacian when needed.
    """
    arr = field.data
    backend = field.backend
    start = time.perf_counter()

    try:
        if backend == "torch" and _TORCH_AVAILABLE:
            ndim = arr.ndim
            lap = torch.zeros_like(arr)
            if ndim == 1:
                lap[1:-1] = arr[:-2] - 2 * arr[1:-1] + arr[2:]
            elif ndim == 2:
                lap[1:-1, 1:-1] = (
                    arr[:-2, 1:-1]
                    + arr[2:, 1:-1]
                    + arr[1:-1, :-2]
                    + arr[1:-1, 2:]
                    - 4 * arr[1:-1, 1:-1]
                )
            elif ndim == 3:
                lap[1:-1, 1:-1, 1:-1] = (
                    arr[:-2, 1:-1, 1:-1]
                    + arr[2:, 1:-1, 1:-1]
                    + arr[1:-1, :-2, 1:-1]
                    + arr[1:-1, 2:, 1:-1]
                    + arr[1:-1, 1:-1, :-2]
                    + arr[1:-1, 1:-1, 2:]
                    - 6 * arr[1:-1, 1:-1, 1:-1]
                )
            else:
                raise ValueError(f"Unsupported dimension {ndim}")
            result = lap
        else:
            arr_np = np.asarray(arr)
            ndim = arr_np.ndim
            lap = np.zeros_like(arr_np)
            if ndim == 1:
                lap[1:-1] = arr_np[:-2] - 2 * arr_np[1:-1] + arr_np[2:]
            elif ndim == 2:
                lap[1:-1, 1:-1] = (
                    arr_np[:-2, 1:-1]
                    + arr_np[2:, 1:-1]
                    + arr_np[1:-1, :-2]
                    + arr_np[1:-1, 2:]
                    - 4 * arr_np[1:-1, 1:-1]
                )
            elif ndim == 3:
                lap[1:-1, 1:-1, 1:-1] = (
                    arr_np[:-2, 1:-1, 1:-1]
                    + arr_np[2:, 1:-1, 1:-1]
                    + arr_np[1:-1, :-2, 1:-1]
                    + arr_np[1:-1, 2:, 1:-1]
                    + arr_np[1:-1, 1:-1, :-2]
                    + arr_np[1:-1, 1:-1, 2:]
                    - 6 * arr_np[1:-1, 1:-1, 1:-1]
                )
            else:
                raise ValueError(f"Unsupported dimension {ndim}")
            result = lap
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[smoothing] laplacian error: {exc}")
        arr_np = np.asarray(arr)
        lap = np.zeros_like(arr_np)
        if arr_np.ndim == 1:
            lap[1:-1] = arr_np[:-2] - 2 * arr_np[1:-1] + arr_np[2:]
        result = lap
    finally:
        BACKEND.record(time.perf_counter() - start)

    return Field(name=f"{field.name}_lap", data=result, backend=backend)


# ====================================================================== #
# 🌫️ Gaussian smoothing
# ====================================================================== #
@dataclass
class GaussianSmoother:
    """
    Apply isotropic Gaussian smoothing to a Field.

    σ controls blur; larger σ → smoother result.
    """

    sigma: float = 1.0

    def smooth(self, field: Field) -> Field:
        """Apply Gaussian blur in-place and return new Field."""
        backend = field.backend
        data = field.data
        start = time.perf_counter()

        try:
            if backend == "torch" and _TORCH_AVAILABLE:
                ndim = data.ndim
                # Construct 1D Gaussian kernel
                radius = int(3 * self.sigma)
                x = torch.arange(-radius, radius + 1, dtype=torch.float64)
                kernel1d = torch.exp(-0.5 * (x / self.sigma) ** 2)
                kernel1d /= kernel1d.sum()
                # Apply separable convolution
                weight = kernel1d
                out = data.clone()
                for dim in range(ndim):
                    # Reshape kernel for dimension
                    shape = [1] * ndim
                    shape[dim] = -1
                    w = weight.view(*shape)
                    out = F.conv1d(
                        out.unsqueeze(0).unsqueeze(0),
                        w.unsqueeze(0).unsqueeze(0),
                        padding=radius,
                    )[0, 0]
                result = out
            else:
                from scipy.ndimage import gaussian_filter

                arr_np = np.asarray(data)
                result = gaussian_filter(arr_np, sigma=self.sigma)
        except ImportError:
            print("[smoothing] scipy not available; fallback to simple averaging.")
            arr_np = np.asarray(data)
            kernel = np.ones((3,) * arr_np.ndim) / (3 ** arr_np.ndim)
            from scipy.signal import convolve  # type: ignore
            result = convolve(arr_np, kernel, mode="same")
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[smoothing] gaussian error: {exc}")
            arr_np = np.asarray(data)
            result = arr_np  # no smoothing fallback
        finally:
            BACKEND.record(time.perf_counter() - start)

        return Field(name=f"{field.name}_smooth", data=result, backend=backend)
