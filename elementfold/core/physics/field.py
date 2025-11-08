"""
core/physics/field.py — The Fabric of Φ 🌌

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• Every measurable quantity in ElementFold is a Field.
• NumPy is the primary craftsman — deterministic and precise.
• Torch is the reserve engine — activated only when NumPy lags.
• The BackendManager watches wall time and promotes once, safely.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]
Backend = Literal["numpy", "torch"]


# ====================================================================== #
# 🧭 BackendManager — watches performance and promotes if needed
# ====================================================================== #
class BackendManager:
    """
    Watches cumulative operation time and promotes backend to Torch
    if NumPy becomes slow (walltime > threshold_s). Thread-safe.
    """

    def __init__(self, threshold_s: float = 0.5):
        self.backend: Backend = "numpy"
        self.threshold_s = threshold_s
        self._accum = 0.0
        self._promoted = False
        self._lock = threading.Lock()

    def record(self, duration: float) -> None:
        """Accumulate duration and promote if cumulative exceeds threshold."""
        if not _TORCH_AVAILABLE or self._promoted or self.backend != "numpy":
            return
        with self._lock:
            self._accum += duration
            if self._accum > self.threshold_s:
                self._promoted = True
                self.backend = "torch"
                print("⚡ backend.promotion → torch (NumPy too slow)")

    def current(self) -> Backend:
        return self.backend


# global manager
BACKEND = BackendManager()


# ====================================================================== #
# 🎛 Backend helpers
# ====================================================================== #
def _ensure_backend(x: ArrayLike, backend: Backend) -> ArrayLike:
    """Convert array to target backend if needed."""
    if backend == "numpy":
        if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            try:
                return x.detach().cpu().numpy()
            except (RuntimeError, AttributeError) as exc:
                print(f"[field] torch→numpy detach failed: {exc}")
                return np.asarray(x)
        return np.asarray(x)

    if backend == "torch" and _TORCH_AVAILABLE:
        if isinstance(x, np.ndarray):
            try:
                return torch.from_numpy(x)
            except (ValueError, TypeError) as exc:
                print(f"[field] numpy→torch conversion failed: {exc}")
                return x
        return x
    return x


def _asarray(data: Any, backend: Backend) -> ArrayLike:
    """Create array respecting backend."""
    if backend == "torch" and _TORCH_AVAILABLE:
        try:
            return torch.as_tensor(data)
        except (TypeError, ValueError) as exc:
            print(f"[field] torch.as_tensor failed: {exc}")
    return np.asarray(data)


# ====================================================================== #
# 🧩 Field — container for spatial or scalar quantities
# ====================================================================== #
@dataclass
class Field:
    """Numeric field with metadata and backend awareness."""

    name: str
    data: ArrayLike
    units: str = "a.u."
    backend: Backend = field(default="numpy")
    description: str = ""

    # ------------------------------------------------------------------ #
    # 🏗️ Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def zeros(cls, shape: Tuple[int, ...], *, name: str = "Φ") -> Field:
        """Create a zero field following global backend choice."""
        backend = BACKEND.current()
        try:
            if backend == "torch" and _TORCH_AVAILABLE:
                data = torch.zeros(shape, dtype=torch.float64)
            else:
                data = np.zeros(shape, dtype=float)
                backend = "numpy"
        except (MemoryError, ValueError) as exc:
            print(f"[field] zeros allocation failed ({exc}); fallback to numpy.")
            data = np.zeros(shape, dtype=float)
            backend = "numpy"
        return cls(name=name, data=data, backend=backend)

    @classmethod
    def from_array(
        cls, arr: ArrayLike, *, name: str = "Φ", backend: Optional[Backend] = None
    ) -> Field:
        """Wrap an existing array into a Field."""
        backend = backend or BACKEND.current()
        data = _ensure_backend(arr, backend)
        return cls(name=name, data=data, backend=backend)

    # ------------------------------------------------------------------ #
    # 🧮 Arithmetic operations
    # ------------------------------------------------------------------ #
    def _binary_op(self, other: Any, op) -> Field:
        """Internal helper for arithmetic with timing & fallbacks."""
        left = _ensure_backend(self.data, self.backend)
        right = _ensure_backend(
            other.data if isinstance(other, Field) else other, self.backend
        )
        start = time.perf_counter()
        try:
            result = op(left, right)
        except (ValueError, TypeError) as exc:
            # fallback to numpy
            result = op(_ensure_backend(left, "numpy"), _ensure_backend(right, "numpy"))
            self.backend = "numpy"
            print(f"🧮 backend.fallback — {type(exc).__name__}: {exc}")
        finally:
            duration = time.perf_counter() - start
            BACKEND.record(duration)

        return Field(name=self.name, data=result, units=self.units, backend=self.backend)

    def __add__(self, other: Any) -> Field:
        return self._binary_op(other, lambda a, b: a + b)

    def __sub__(self, other: Any) -> Field:
        return self._binary_op(other, lambda a, b: a - b)

    def __mul__(self, other: Any) -> Field:
        return self._binary_op(other, lambda a, b: a * b)

    def __truediv__(self, other: Any) -> Field:
        return self._binary_op(other, lambda a, b: a / b)

    def __neg__(self) -> Field:
        start = time.perf_counter()
        try:
            result = -self.data
        except (RuntimeError, TypeError) as exc:
            print(f"[field] negation failed: {exc}")
            result = -_ensure_backend(self.data, "numpy")
            self.backend = "numpy"
        BACKEND.record(time.perf_counter() - start)
        return Field(self.name, result, self.units, self.backend)

    # ------------------------------------------------------------------ #
    # 🌡️ Metrics and utilities
    # ------------------------------------------------------------------ #
    def norm(self) -> float:
        """Compute L2 norm."""
        start = time.perf_counter()
        try:
            if self.backend == "torch" and _TORCH_AVAILABLE:
                val = float(torch.linalg.norm(self.data))
            else:
                val = float(np.linalg.norm(np.asarray(self.data)))
        except (ValueError, TypeError) as exc:
            print(f"[field] norm failed: {exc}")
            val = float(np.linalg.norm(np.asarray(self.data)))
            self.backend = "numpy"
        finally:
            BACKEND.record(time.perf_counter() - start)
        return val

    def mean(self) -> float:
        """Compute mean value."""
        start = time.perf_counter()
        try:
            if self.backend == "torch" and _TORCH_AVAILABLE:
                val = float(self.data.mean())
            else:
                val = float(np.mean(np.asarray(self.data)))
        except (ValueError, TypeError) as exc:
            print(f"[field] mean failed: {exc}")
            val = float(np.mean(np.asarray(self.data)))
            self.backend = "numpy"
        finally:
            BACKEND.record(time.perf_counter() - start)
        return val

    def clamp(self, min_value: float, max_value: float) -> None:
        """In-place clamp for safety (does not create new Field)."""
        start = time.perf_counter()
        try:
            if self.backend == "torch" and _TORCH_AVAILABLE:
                self.data.clamp_(min_value, max_value)
            else:
                np.clip(self.data, min_value, max_value, out=self.data)
        except (ValueError, TypeError) as exc:
            print(f"[field] clamp failed: {exc}")
            self.data = np.clip(_ensure_backend(self.data, "numpy"), min_value, max_value)
            self.backend = "numpy"
        finally:
            BACKEND.record(time.perf_counter() - start)

    def copy(self, name: Optional[str] = None) -> Field:
        """Return a shallow copy."""
        try:
            if self.backend == "torch" and _TORCH_AVAILABLE:
                data = self.data.clone()
            else:
                data = self.data.copy()
        except (AttributeError, RuntimeError) as exc:
            print(f"[field] copy failed: {exc}")
            data = np.copy(_ensure_backend(self.data, "numpy"))
            self.backend = "numpy"
        return Field(name or self.name, data, self.units, self.backend)

    # ------------------------------------------------------------------ #
    # 🔍 Representation
    # ------------------------------------------------------------------ #
    def __repr__(self) -> str:
        icon = "🧮" if self.backend == "numpy" else "⚡"
        shape = tuple(self.data.shape) if hasattr(self.data, "shape") else ()
        return f"<Field {self.name!r} {icon} shape={shape} units={self.units}>"
