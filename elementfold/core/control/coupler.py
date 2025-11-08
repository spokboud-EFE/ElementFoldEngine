"""
elementfold.core.control.coupler
--------------------------------
Coupler — manages inter-core coupling and coherence feedback.

It is responsible for sharing parameters such as ρ (coupling strength)
and κ (coherence factor) between cores.  Factory calls it after each
synchronization step to harmonize collective behavior.
"""

from __future__ import annotations
import threading
from typing import List


class Coupler:
    """Handles inter-core coupling and coherence sharing."""

    def __init__(self, rho: float = 0.05, kappa: float = 1.0) -> None:
        self.rho: float = rho       # coupling coefficient (strength of connection)
        self.kappa: float = kappa   # global coherence baseline
        self._lock = threading.Lock()

    # ---------------------------------------------------------
    # --- Core coupling methods
    # ---------------------------------------------------------
    def couple(self, cores: List) -> None:
        """
        Apply mutual coherence influence among all cores.
        Each core’s κ moves slightly toward the mean of all others.
        """
        if not cores:
            return
        with self._lock:
            try:
                mean_kappa = sum(c.runtime.state.kappa for c in cores) / len(cores)
                for c in cores:
                    current = c.runtime.state.kappa
                    # weighted blend toward global coherence
                    new_kappa = (1 - self.rho) * current + self.rho * mean_kappa
                    c.runtime.state.kappa = max(0.0, min(2.0, new_kappa))
                self.kappa = mean_kappa
            except AttributeError:
                raise ValueError("[Coupler] Core runtime state missing kappa attribute")

    # ---------------------------------------------------------
    # --- Diagnostics
    # ---------------------------------------------------------
    def report(self) -> str:
        """Return short text summary of coupling state."""
        return f"[Coupler] ρ={self.rho:.3f}, κ̄={self.kappa:.3f}"
