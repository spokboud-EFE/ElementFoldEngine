# ElementFold · core/data.py
# ============================================================
# Unified data structures for the Relaxation model
# ------------------------------------------------------------
# This module replaces the former torch dataset with lightweight,
# pure-Python dataclasses that describe:
#     • Field states (Φ, t, grid spacing, BC)
#     • Background and optical parameters
#     • Path segments used in fold/redshift integration
#
# All are serializable (to/from dict) and independent of any framework.
# ============================================================

from __future__ import annotations

import dataclasses
import numpy as np
from typing import Callable, Literal, Any, List, Dict, Tuple


# ============================================================
# Basic dataclasses
# ============================================================

@dataclasses.dataclass
class BackgroundParams:
    """Physical background coefficients for the relaxation PDE."""
    lambda_: float          # local letting-go rate λ
    D: float                # spatial smoothing coefficient D
    phi_inf: float = 0.0    # asymptotic calm baseline Φ∞

    def as_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class OpticalParams:
    """Functions defining σ(Φ), n(Φ,ν), η(Φ,ν)."""
    sigma: Callable[[float], float]
    n: Callable[[float, float], float]
    eta: Callable[[float, float], float]
    nu0: float = 1.0        # reference frequency (for normalization)

    def as_dict(self) -> Dict[str, str]:
        # only stores function names; actual callables registered elsewhere
        return {"sigma": getattr(self.sigma, "__name__", "lambda"),
                "n": getattr(self.n, "__name__", "lambda"),
                "eta": getattr(self.eta, "__name__", "lambda"),
                "nu0": self.nu0}


@dataclasses.dataclass
class FieldState:
    """Grid snapshot of the resonance potential Φ."""
    phi: np.ndarray                     # array (1D/2D/3D)
    t: float                            # current simulation time
    spacing: Tuple[float, ...]          # Δx, Δy, (Δz)
    bc: Literal["neumann", "periodic"]  # boundary condition

    def copy(self) -> "FieldState":
        return FieldState(self.phi.copy(), self.t, tuple(self.spacing), self.bc)

    def shape(self) -> Tuple[int, ...]:
        return self.phi.shape

    def ndim(self) -> int:
        return self.phi.ndim

    def as_dict(self) -> Dict[str, Any]:
        return {
            "phi": self.phi.tolist(),
            "t": self.t,
            "spacing": list(self.spacing),
            "bc": self.bc,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FieldState":
        phi = np.array(d["phi"], dtype=float)
        return cls(phi=phi, t=float(d["t"]),
                   spacing=tuple(d["spacing"]), bc=str(d["bc"]))


# ============================================================
# Path representation
# ============================================================

@dataclasses.dataclass
class PathSegment:
    """A small element of a light or signal path."""
    ds: float       # physical path length
    phi: float      # field sample (Φ) along the segment
    nu: float       # frequency of the signal there

    def as_dict(self) -> Dict[str, float]:
        return {"ds": self.ds, "phi": self.phi, "nu": self.nu}


Path = List[PathSegment]


# ============================================================
# Helpers
# ============================================================

def to_dict(obj: Any) -> Dict[str, Any]:
    """Recursively convert dataclasses to plain dicts."""
    if dataclasses.is_dataclass(obj):
        return {k: to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def from_dict(cls, data: Dict[str, Any]) -> Any:
    """Recreate dataclass instances from dicts."""
    if cls is FieldState:
        return FieldState.from_dict(data)
    return cls(**data)


# ============================================================
# Convenience constructors
# ============================================================

def default_background(lambda_: float = 0.33,
                       D: float = 0.15,
                       phi_inf: float = 0.0) -> BackgroundParams:
    """Return a typical background configuration."""
    return BackgroundParams(lambda_=lambda_, D=D, phi_inf=phi_inf)


def default_optics() -> OpticalParams:
    """Return a simple, safe default optical parameterization."""

    def sigma(phi: float) -> float:
        return 1.0 + 0.01 * phi

    def n(phi: float, nu: float) -> float:
        return 1.0 + 0.001 * phi + 0.0001 * np.log(max(nu, 1e-12))

    def eta(phi: float, nu: float) -> float:
        return 0.02 + 0.005 * phi + 0.0005 * np.log(max(nu, 1e-12))

    return OpticalParams(sigma=sigma, n=n, eta=eta, nu0=1.0)


def empty_state(shape: Tuple[int, ...] = (64, 64),
                spacing: Tuple[float, ...] = (1.0, 1.0),
                bc: Literal["neumann", "periodic"] = "neumann",
                t0: float = 0.0) -> FieldState:
    """Create a zero-initialized field state."""
    phi = np.zeros(shape, dtype=float)
    return FieldState(phi=phi, t=t0, spacing=spacing, bc=bc)
