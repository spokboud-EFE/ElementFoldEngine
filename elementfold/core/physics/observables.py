"""
core/physics/observables.py — What the Universe Shows Us 🌈

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• These are the observables — what the calm field looks like to an eye.
• Redshift:  how much color stretches across ℱ folds.
• Brightness: how much light softens as it shares with the medium.
• Time dilation: how much rhythm slows as the field settles.
• Bending: how the light path curves through gradients of n(Φ, ν).
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from elementfold.core.physics.field import Field, BACKEND
from elementfold.core.physics.optics import OpticalLaw
from elementfold.core.physics.foldclock import FoldClock

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ====================================================================== #
# 🔭  Redshift
# ====================================================================== #
def redshift_from_folds(folds: float) -> float:
    """
    Convert cumulative folds ℱ into redshift:
        1 + z = exp(ℱ)
    """
    try:
        return math.exp(folds) - 1.0
    except OverflowError as exc:
        print(f"[observables] redshift overflow: {exc}")
        return float("inf")


def redshift_field(fold_field: Field) -> Field:
    """Elementwise version of redshift_from_folds."""
    start = time.perf_counter()
    backend = fold_field.backend
    try:
        if backend == "torch" and _TORCH_AVAILABLE:
            z = torch.exp(fold_field.data) - 1.0
        else:
            z = np.exp(np.asarray(fold_field.data)) - 1.0
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[observables] redshift_field fallback: {exc}")
        z = np.exp(np.asarray(fold_field.data)) - 1.0
        backend = "numpy"
    finally:
        BACKEND.record(time.perf_counter() - start)
    return Field(name="redshift", data=z, backend=backend)


# ====================================================================== #
# 💡  Surface Brightness Tilt
# ====================================================================== #
def brightness_from_folds(folds: float) -> float:
    """
    Surface brightness tilt from fold count:
        I_obs / I_emit = exp(-2ℱ)
    """
    try:
        return math.exp(-2.0 * folds)
    except OverflowError as exc:
        print(f"[observables] brightness overflow: {exc}")
        return 0.0


def brightness_field(fold_field: Field) -> Field:
    """Elementwise version of brightness_from_folds."""
    start = time.perf_counter()
    backend = fold_field.backend
    try:
        if backend == "torch" and _TORCH_AVAILABLE:
            tilt = torch.exp(-2.0 * fold_field.data)
        else:
            tilt = np.exp(-2.0 * np.asarray(fold_field.data))
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[observables] brightness_field fallback: {exc}")
        tilt = np.exp(-2.0 * np.asarray(fold_field.data))
        backend = "numpy"
    finally:
        BACKEND.record(time.perf_counter() - start)
    return Field(name="brightness_tilt", data=tilt, backend=backend)


# ====================================================================== #
# ⏱️  Apparent Time Dilation
# ====================================================================== #
def time_dilation(phi_obs: float, phi_emit: float, folds: float) -> float:
    """
    Apparent time dilation between emitter and observer:
        τ_obs / τ_emit ≈ (1 + ΔΦ/c²) * exp(ℱ)
    For normalized units (c=1), we take:
        τ_obs / τ_emit ≈ exp(ℱ) * (1 + (Φ_obs - Φ_emit))
    """
    try:
        return math.exp(folds) * (1.0 + (phi_obs - phi_emit))
    except OverflowError as exc:
        print(f"[observables] time_dilation overflow: {exc}")
        return float("inf")


def time_dilation_field(phi_obs: Field, phi_emit: Field, fold_field: Field) -> Field:
    """Elementwise dilation field."""
    start = time.perf_counter()
    backend = fold_field.backend
    try:
        if backend == "torch" and _TORCH_AVAILABLE:
            ratio = torch.exp(fold_field.data) * (1.0 + (phi_obs.data - phi_emit.data))
        else:
            ratio = np.exp(np.asarray(fold_field.data)) * (
                1.0 + (np.asarray(phi_obs.data) - np.asarray(phi_emit.data))
            )
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[observables] time_dilation_field fallback: {exc}")
        ratio = np.exp(np.asarray(fold_field.data)) * (
            1.0 + (np.asarray(phi_obs.data) - np.asarray(phi_emit.data))
        )
        backend = "numpy"
    finally:
        BACKEND.record(time.perf_counter() - start)
    return Field(name="time_dilation", data=ratio, backend=backend)


# ====================================================================== #
# 🌈  Chromatic Bending
# ====================================================================== #
def chromatic_bending(optical_field: Field, nu1: float, nu2: float) -> float:
    """
    Estimate chromatic bending angle difference between two frequencies.
    Δθ ≈ ∫ |∇n(ν1) - ∇n(ν2)| ds  (simplified average form)
    For simplicity we return average gradient magnitude difference.
    """
    start = time.perf_counter()
    backend = optical_field.backend
    try:
        arr = optical_field.data
        if backend == "torch" and _TORCH_AVAILABLE:
            g1 = torch.gradient(arr * (nu1 / 1e14))
            g2 = torch.gradient(arr * (nu2 / 1e14))
            diffs = [torch.abs(a - b).mean() for a, b in zip(g1, g2)]
            delta = float(sum(diffs))
        else:
            arr_np = np.asarray(arr)
            g1 = np.gradient(arr_np * (nu1 / 1e14))
            g2 = np.gradient(arr_np * (nu2 / 1e14))
            delta = float(sum(np.mean(np.abs(a - b)) for a, b in zip(g1, g2)))
    except (ValueError, TypeError, RuntimeError) as exc:
        print(f"[observables] chromatic_bending error: {exc}")
        arr_np = np.asarray(optical_field.data)
        g1 = np.gradient(arr_np * (nu1 / 1e14))
        g2 = np.gradient(arr_np * (nu2 / 1e14))
        delta = float(sum(np.mean(np.abs(a - b)) for a, b in zip(g1, g2)))
    finally:
        BACKEND.record(time.perf_counter() - start)
    return delta


# ====================================================================== #
# 🗣️  Narrative summary
# ====================================================================== #
def summarize(clock: FoldClock, phi_obs: float, phi_emit: float) -> str:
    """Return a short plain-language description of all observables."""
    z = redshift_from_folds(clock.folds)
    bright = brightness_from_folds(clock.folds)
    dil = time_dilation(phi_obs, phi_emit, clock.folds)
    return (
        f"ℱ {clock.folds:.2f} → z={z:.3f}, "
        f"I/I₀={bright:.3e}, τ_ratio={dil:.3f}"
    )
