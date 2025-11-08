"""
elementfold.core.control.synchronizer
-------------------------------------
The Synchronizer keeps all active cores in phase coherence.
It aligns runtime clocks, manages shared δ★ (delta-star), and
smooths discrepancies between local runtime states.

Used by Factory to ensure synchronized operations across
multiple simulation cores or hardware loops.
"""

from __future__ import annotations
import time
import threading
import math
from typing import List, Optional


class Synchronizer:
    """
    Synchronizer — maintains temporal and phase alignment
    across all cores in the system.

    Each core is expected to expose a `.runtime` object with
    at least: `.phase`, `.tick_rate`, `.last_sync`.
    """

    def __init__(self) -> None:
        self.delta_star: float = 0.32000062   # universal step (baseline)
        self.phase_reference: float = 0.0     # reference phase value
        self.last_sync: float = time.time()
        self.lock = threading.Lock()
        self.active: bool = False

    # ---------------------------------------------------------
    # --- Core Synchronization Methods
    # ---------------------------------------------------------
    def align_core(self, core) -> None:
        """Align a single core’s runtime phase to the reference."""
        try:
            with self.lock:
                core.runtime.phase = self.phase_reference
                core.runtime.last_sync = time.time()
        except AttributeError:
            raise ValueError("[Synchronizer] Core runtime missing phase attributes")

    def synchronize(self, cores: List) -> None:
        """
        Bring all given cores into alignment.
        Computes mean phase and adjusts all runtimes toward it.
        """
        if not cores:
            return
        try:
            with self.lock:
                mean_phase = sum(c.runtime.phase for c in cores) / len(cores)
                for c in cores:
                    c.runtime.phase = mean_phase
                    c.runtime.last_sync = time.time()
                self.phase_reference = mean_phase
                self.last_sync = time.time()
        except Exception as exc:
            raise RuntimeError(f"[Synchronizer] Synchronization failed: {exc}") from exc

    # ---------------------------------------------------------
    # --- Continuous Background Sync Loop
    # ---------------------------------------------------------
    def continuous_sync(self, cores: List, interval: float = 1.0) -> None:
        """Run a continuous background synchronization loop."""
        if self.active:
            return
        self.active = True

        def _loop() -> None:
            while self.active:
                self.synchronize(cores)
                time.sleep(interval)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()

    def stop(self) -> None:
        """Stop continuous background synchronization."""
        self.active = False

    # ---------------------------------------------------------
    # --- Diagnostics and Utilities
    # ---------------------------------------------------------
    def measure_phase_drift(self, cores: List) -> float:
        """
        Compute RMS phase deviation across cores.
        Returns 0 if perfectly aligned.
        """
        if not cores:
            return 0.0
        phases = [c.runtime.phase for c in cores]
        mean_phase = sum(phases) / len(phases)
        rms = math.sqrt(sum((p - mean_phase) ** 2 for p in phases) / len(phases))
        return rms

    def report_status(self, cores: Optional[List] = None) -> str:
        """Return a compact string summary of the synchronization state."""
        drift = self.measure_phase_drift(cores) if cores else 0.0
        delta = time.time() - self.last_sync
        return (
            f"[Synchronizer] Δ★={self.delta_star:.6f}, "
            f"phase_ref={self.phase_reference:.4f}, "
            f"drift={drift:.6e}, "
            f"last_sync={delta:.2f}s ago"
        )
