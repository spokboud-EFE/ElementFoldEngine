"""
core/physics/motion.py — Coherence and Coupling in Motion 🌌

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• Motion here means the quiet drift of coherence, not travel in space.
• κ (kappa) measures harmony — how well phases align.
• ρ (rho) measures coupling — how strongly neighboring cores feel one another.
• ∇Φ guides both: motion follows the slope of tension.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
from elementfold.core.physics.field import Field, BACKEND
from elementfold.core.physics.smoothing import gradient

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ====================================================================== #
# 🧭 Drift along ∇Φ
# ====================================================================== #
def drift_velocity(phi_field: Field, kappa: float = 1.0) -> Tuple[Field, ...]:
    """
    Compute drift velocity components proportional to -κ ∇Φ.
    """
    start = time.perf_counter()
    backend = phi_field.backend
    try:
        grads = gradient(phi_field)
        if backend == "torch" and _TORCH_AVAILABLE:
            comps = [
                Field(name=f"v{i}", data=-kappa * g.data, backend="torch") for i, g in enumerate(grads)
            ]
        else:
            comps = [
                Field(name=f"v{i}", data=-kappa * np.asarray(g.data), backend="numpy")
                for i, g in enumerate(grads)
            ]
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[motion] drift_velocity fallback: {exc}")
        arr = np.asarray(phi_field.data)
        grads_np = np.gradient(arr)
        comps = [
            Field(name=f"v{i}", data=-kappa * g, backend="numpy") for i, g in enumerate(grads_np)
        ]
        backend = "numpy"
    finally:
        BACKEND.record(time.perf_counter() - start)
    return tuple(comps)


# ====================================================================== #
# 🕸️ Coupling model
# ====================================================================== #
@dataclass
class CouplingLaw:
    """
    Represents ρ-coupling between cores or regions.

    ΔΦ_coupled = ρ (Φ_neighbor − Φ_local)
    Δκ_coupled = ρ (κ_neighbor − κ_local)
    """

    rho: float = 0.01  # coupling strength
    max_change: float = 0.1

    def apply_field_coupling(self, phi_local: Field, phi_neighbor: Field) -> Field:
        """Adjust Φ by coupling difference with a neighbor."""
        start = time.perf_counter()
        backend = phi_local.backend
        try:
            if backend == "torch" and _TORCH_AVAILABLE:
                delta = self.rho * (phi_neighbor.data - phi_local.data)
                delta = torch.clamp(delta, -self.max_change, self.max_change)
                new_data = phi_local.data + delta
            else:
                arr_local = np.asarray(phi_local.data)
                arr_neigh = np.asarray(phi_neighbor.data)
                delta = self.rho * (arr_neigh - arr_local)
                np.clip(delta, -self.max_change, self.max_change, out=delta)
                new_data = arr_local + delta
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[motion] apply_field_coupling error: {exc}")
            arr_local = np.asarray(phi_local.data)
            arr_neigh = np.asarray(phi_neighbor.data)
            delta = self.rho * (arr_neigh - arr_local)
            np.clip(delta, -self.max_change, self.max_change, out=delta)
            new_data = arr_local + delta
            backend = "numpy"
        finally:
            BACKEND.record(time.perf_counter() - start)
        return Field(name=f"{phi_local.name}_coupled", data=new_data, backend=backend)

    def update_kappa(self, kappa_local: float, kappa_neighbor: float) -> float:
        """
        Update scalar κ by coupling to neighbor.
        Returns new κ_local.
        """
        delta = self.rho * (kappa_neighbor - kappa_local)
        new_val = kappa_local + max(min(delta, self.max_change), -self.max_change)
        return new_val


# ====================================================================== #
# 🧘 Coherence law (κ evolution)
# ====================================================================== #
@dataclass
class CoherenceLaw:
    """
    Models how coherence κ relaxes toward unity and responds to drift.
    dκ/dt = α(1−κ) − β|∇Φ|²
    """

    alpha: float = 0.05  # relaxation toward unity
    beta: float = 0.01   # loss term due to gradient energy

    def step(self, phi_field: Field, kappa_field: Field, dt: float) -> Field:
        """Advance κ field one step."""
        start = time.perf_counter()
        backend = phi_field.backend
        try:
            grads = gradient(phi_field)
            grad_energy = sum(g * g for g in grads)
            if backend == "torch" and _TORCH_AVAILABLE:
                d_kappa = self.alpha * (1.0 - kappa_field.data) - self.beta * grad_energy.data
                new_data = kappa_field.data + dt * d_kappa
            else:
                arr_k = np.asarray(kappa_field.data)
                arr_g = sum(np.asarray(g.data) ** 2 for g in grads)
                d_kappa = self.alpha * (1.0 - arr_k) - self.beta * arr_g
                new_data = arr_k + dt * d_kappa
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[motion] coherence.step fallback: {exc}")
            arr_k = np.asarray(kappa_field.data)
            arr_g = sum(np.asarray(g.data) ** 2 for g in gradient(phi_field))
            d_kappa = self.alpha * (1.0 - arr_k) - self.beta * arr_g
            new_data = arr_k + dt * d_kappa
            backend = "numpy"
        finally:
            BACKEND.record(time.perf_counter() - start)
        return Field(name="kappa", data=new_data, backend=backend)
