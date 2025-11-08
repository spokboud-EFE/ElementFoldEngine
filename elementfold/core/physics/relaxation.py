"""
core/physics/relaxation.py — The Relaxation Law λ–D 🌊

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• This is the heart of ElementFold’s shaping mode.
• λ (lambda) — how fast the field lets go of excess tension.
• D (diffusion) — how fast the field shares its tension with neighbors.
• Together they write the universal rule:
      ∂Φ/∂t = -λ(Φ - Φ∞) + D ∇²Φ
• Torch may join if NumPy starts to slow, but NumPy remains the
  reference implementation.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from elementfold.core.physics.field import Field, BACKEND

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ====================================================================== #
# 🧮 Helper: Laplacian (finite difference)
# ====================================================================== #
def laplacian(field: Field) -> Field:
    """Compute ∇²Φ with simple central differences (NumPy or Torch)."""
    data = field.data
    backend = field.backend

    start = time.perf_counter()
    try:
        if backend == "torch" and _TORCH_AVAILABLE:
            # For 1D, 2D, 3D torch tensors
            ndim = data.ndim
            lap = torch.zeros_like(data)
            if ndim == 1:
                lap[1:-1] = data[:-2] - 2 * data[1:-1] + data[2:]
            elif ndim == 2:
                lap[1:-1, 1:-1] = (
                    data[:-2, 1:-1]
                    + data[2:, 1:-1]
                    + data[1:-1, :-2]
                    + data[1:-1, 2:]
                    - 4 * data[1:-1, 1:-1]
                )
            elif ndim == 3:
                lap[1:-1, 1:-1, 1:-1] = (
                    data[:-2, 1:-1, 1:-1]
                    + data[2:, 1:-1, 1:-1]
                    + data[1:-1, :-2, 1:-1]
                    + data[1:-1, 2:, 1:-1]
                    + data[1:-1, 1:-1, :-2]
                    + data[1:-1, 1:-1, 2:]
                    - 6 * data[1:-1, 1:-1, 1:-1]
                )
            else:
                raise ValueError(f"Unsupported dimension {ndim}")
            result = lap
        else:
            arr = np.asarray(data)
            ndim = arr.ndim
            lap = np.zeros_like(arr)
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
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[relaxation] laplacian fallback ({type(exc).__name__}: {exc})")
        arr = np.asarray(field.data)
        lap = np.zeros_like(arr)
        if arr.ndim == 1:
            lap[1:-1] = arr[:-2] - 2 * arr[1:-1] + arr[2:]
        result = lap
    finally:
        BACKEND.record(time.perf_counter() - start)

    return Field(name=f"{field.name}_lap", data=result, backend=field.backend)


# ====================================================================== #
# ⚙️ Relaxation update
# ====================================================================== #
@dataclass
class RelaxationLaw:
    """
    Implements the λ–D relaxation update:
        Φ(t+Δt) = Φ + Δt * (-λ(Φ - Φ∞) + D ∇²Φ)
    """

    lambda_: float = 0.1  # relaxation rate λ
    D: float = 0.05       # diffusion coefficient D
    phi_inf: float = 0.0  # calm baseline Φ∞
    safety_clamp: float = 1e6  # to prevent runaway

    def step(self, field: Field, dt: float) -> Field:
        """Return the next field value after one relaxation tick."""
        start = time.perf_counter()
        backend = field.backend

        try:
            lap = laplacian(field)
            if backend == "torch" and _TORCH_AVAILABLE:
                phi_inf_t = torch.as_tensor(self.phi_inf, dtype=field.data.dtype)
                dphi = -self.lambda_ * (field.data - phi_inf_t) + self.D * lap.data
                new_data = field.data + dt * dphi
            else:
                arr = np.asarray(field.data)
                dphi = -self.lambda_ * (arr - self.phi_inf) + self.D * np.asarray(lap.data)
                new_data = arr + dt * dphi

            # clamp for safety
            np_clip = np.clip if backend == "numpy" else (torch.clamp if _TORCH_AVAILABLE else np.clip)
            new_data = np_clip(new_data, -self.safety_clamp, self.safety_clamp)
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[relaxation] step error ({type(exc).__name__}: {exc}); fallback to numpy.")
            arr = np.asarray(field.data)
            lap = np.asarray(laplacian(field).data)
            dphi = -self.lambda_ * (arr - self.phi_inf) + self.D * lap
            new_data = np.clip(arr + dt * dphi, -self.safety_clamp, self.safety_clamp)
            backend = "numpy"
        finally:
            BACKEND.record(time.perf_counter() - start)

        return Field(name=field.name, data=new_data, backend=backend)

    # ------------------------------------------------------------------ #
    # 🧩 Narrative helper
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """Plain-language description for Studio panels."""
        return (
            f"λ={self.lambda_:.3f}, D={self.D:.3f}, Φ∞={self.phi_inf:.3f} — "
            "relaxing toward calm."
        )


# ====================================================================== #
# 🧪 Standalone RHS function (for Runtime integration)
# ====================================================================== #
def relaxation_rhs(state: Dict[str, Any], dt: float, params: Optional[Dict[str, Any]] = None) -> Dict[str, Field]:
    """
    Standalone right-hand-side for Runtime.step_fn interface.

    Expects `state` with key 'Phi' (Field) and params containing λ, D, Φ∞.
    Returns updated state dict.
    """
    p = params or {}
    lam = float(p.get("lambda", 0.1))
    D = float(p.get("D", 0.05))
    phi_inf = float(p.get("phi_inf", 0.0))
    law = RelaxationLaw(lambda_=lam, D=D, phi_inf=phi_inf)
    phi: Field = state["Phi"]
    next_phi = law.step(phi, dt)
    return {"Phi": next_phi}
