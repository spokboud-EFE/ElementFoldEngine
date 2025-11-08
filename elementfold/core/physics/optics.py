"""
core/physics/optics.py — Light Through the Relaxing Field 💡

──────────────────────────────────────────────────────────────────────
──────────────────────────────────────────────────────────────────────
• Φ sets how the background “feels” to light.
• The optical index n(Φ, ν) tells how much the field slows phase speed.
• The share-rate η(Φ, ν) tells how much sharpness each wave crest gives back.
• Together they explain redshift, dimming, and bending across folds.
──────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from elementfold.core.physics.field import Field, BACKEND

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


# ====================================================================== #
# 📐 Base relations
# ====================================================================== #
@dataclass
class OpticalLaw:
    """
    Defines mappings between resonance potential Φ and
    observable optical quantities n and η.

    n(Φ, ν) = 1 − α Φ + β Φ² + γ (ν/ν₀)⁻²
    η(Φ, ν) = κ |∂n/∂Φ| · |Φ|
    """

    alpha: float = 2.0e-2     # linear sensitivity
    beta: float = 1.0e-4      # quadratic correction
    gamma: float = 0.0        # chromatic term coefficient
    nu0: float = 5.0e14       # reference frequency (Hz)
    kappa: float = 1.0        # proportionality for share-rate η

    # ------------------------------------------------------------------ #
    # 💡 Optical index n(Φ, ν)
    # ------------------------------------------------------------------ #
    def index(self, phi_field: Field, nu: float) -> Field:
        """Compute refractive index n(Φ, ν)."""
        start = time.perf_counter()
        backend = phi_field.backend
        data = phi_field.data
        try:
            if backend == "torch" and _TORCH_AVAILABLE:
                x = data
                chrom = self.gamma * (nu / self.nu0) ** (-2.0)
                n = 1.0 - self.alpha * x + self.beta * x ** 2 + chrom
            else:
                x = np.asarray(data)
                chrom = self.gamma * (nu / self.nu0) ** (-2.0)
                n = 1.0 - self.alpha * x + self.beta * x ** 2 + chrom
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[optics] index fallback: {exc}")
            x = np.asarray(data)
            chrom = self.gamma * (nu / self.nu0) ** (-2.0)
            n = 1.0 - self.alpha * x + self.beta * x ** 2 + chrom
            backend = "numpy"
        finally:
            BACKEND.record(time.perf_counter() - start)
        return Field(name="n", data=n, backend=backend)

    # ------------------------------------------------------------------ #
    # 🌊 Share-rate η(Φ, ν)
    # ------------------------------------------------------------------ #
    def share_rate(self, phi_field: Field, nu: float) -> Field:
        """
        Compute share-rate η(Φ, ν) = κ |∂n/∂Φ| · |Φ|.
        Measures how fast sharpness (contrast) is shared with background.
        """
        start = time.perf_counter()
        backend = phi_field.backend
        data = phi_field.data
        try:
            if backend == "torch" and _TORCH_AVAILABLE:
                dn_dphi = -self.alpha + 2.0 * self.beta * data
                eta = self.kappa * torch.abs(dn_dphi) * torch.abs(data)
            else:
                arr = np.asarray(data)
                dn_dphi = -self.alpha + 2.0 * self.beta * arr
                eta = self.kappa * np.abs(dn_dphi) * np.abs(arr)
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[optics] share_rate fallback: {exc}")
            arr = np.asarray(data)
            dn_dphi = -self.alpha + 2.0 * self.beta * arr
            eta = self.kappa * np.abs(dn_dphi) * np.abs(arr)
            backend = "numpy"
        finally:
            BACKEND.record(time.perf_counter() - start)
        return Field(name="eta", data=eta, backend=backend)

    # ------------------------------------------------------------------ #
    # 🔦 Combined observable helpers
    # ------------------------------------------------------------------ #
    def refractivity(self, phi_field: Field, nu: float) -> Field:
        """Return (n-1): the deviation from vacuum."""
        n_field = self.index(phi_field, nu)
        if n_field.backend == "torch" and _TORCH_AVAILABLE:
            dev = n_field.data - 1.0
        else:
            dev = np.asarray(n_field.data) - 1.0
        return Field(name="refractivity", data=dev, backend=n_field.backend)

    def attenuation(self, phi_field: Field, nu: float) -> Field:
        """
        Empirical dimming factor A ≈ exp(-2 η ℱ)
        (here we just compute local coefficient e^(-2η)).
        """
        eta_field = self.share_rate(phi_field, nu)
        data = eta_field.data
        backend = eta_field.backend
        try:
            if backend == "torch" and _TORCH_AVAILABLE:
                att = torch.exp(-2.0 * data)
            else:
                att = np.exp(-2.0 * np.asarray(data))
        except (ValueError, TypeError, RuntimeError) as exc:
            print(f"[optics] attenuation error: {exc}")
            att = np.exp(-2.0 * np.asarray(data))
            backend = "numpy"
        return Field(name="attenuation", data=att, backend=backend)

    # ------------------------------------------------------------------ #
    # 🗣️ Narrative helper
    # ------------------------------------------------------------------ #
    def describe(self) -> str:
        """Short description for Studio panels."""
        return (
            f"n(Φ, ν)=1-{self.alpha}Φ+{self.beta}Φ²+γ(ν/ν₀)⁻²; "
            f"η=κ|∂n/∂Φ||Φ|; α={self.alpha}, β={self.beta}, κ={self.kappa}"
        )
