# ElementFold · core/runtime.py
# ============================================================
# Runtime — orchestration spine for relaxation simulations
# ------------------------------------------------------------
# Responsibilities
#   • run_loop()      – evolve Φ with hooks and logging
#   • simulate_once() – single run returning state + metrics
#   • hooks: on_step, on_epoch, on_finish (optional)
#
# Works with RelaxationModel (core/model.py) and Telemetry.
# ============================================================

from __future__ import annotations
import time
import numpy as np
from typing import Callable, Dict, Optional, Any

from .data import FieldState
from .model import RelaxationModel
from .telemetry import summary as telemetry_summary
from ..helpers import verify as verify  # only for runtime invariants (optional)


# ============================================================
# Main run loop
# ============================================================

def run_loop(model: RelaxationModel,
             state: FieldState,
             steps: int = 100,
             dt: Optional[float] = None,
             hooks: Optional[Dict[str, Callable]] = None,
             log_interval: int = 10,
             verbose: bool = True) -> FieldState:
    """
    Generic simulation driver.
    Parameters
    ----------
    model : RelaxationModel
        The physical model to integrate.
    state : FieldState
        Initial condition Φ, t, spacing, bc.
    steps : int
        Number of integration steps to run.
    dt : float | None
        Time step; computed safely if None.
    hooks : dict[str, callable]
        Optional callbacks: {'on_step', 'on_epoch', 'on_finish'}.
    log_interval : int
        Print or emit telemetry every N steps.
    verbose : bool
        Whether to print progress to stdout.

    Returns
    -------
    FieldState (final).
    """
    on_step = None
    on_epoch = None
    on_finish = None
    if hooks:
        on_step = hooks.get("on_step")
        on_epoch = hooks.get("on_epoch")
        on_finish = hooks.get("on_finish")

    start_time = time.time()
    current = state
    for i in range(1, steps + 1):
        current = model.evolve(current, steps=1, dt=dt)
        metrics = model.telemetry(current)

        # Optional runtime invariant check
        try:
            if hasattr(verify, "check_monotone_calming"):
                verify.check_monotone_calming(current.phi)
        except Exception:
            pass

        if on_step:
            on_step(current, metrics)

        if verbose and (i % log_interval == 0 or i == steps):
            elapsed = time.time() - start_time
            print(f"[step {i:05d}]  t={current.t:0.5f}  "
                  f"var={metrics['variance']:.3e}  grad²={metrics['grad_L2']:.3e}  "
                  f"energy={metrics['energy']:.3e}  (dt={dt or 'auto'})  "
                  f"{elapsed:0.2f}s")

        if on_epoch and (i % log_interval == 0):
            on_epoch(current, metrics)

    if on_finish:
        on_finish(current)

    return current


# ============================================================
# One-shot convenience wrapper
# ============================================================

def simulate_once(model: RelaxationModel,
                  state: FieldState,
                  steps: int = 100,
                  dt: Optional[float] = None,
                  hooks: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """
    Single-run convenience wrapper that executes run_loop and
    returns {'state': FieldState, 'metrics': Dict[str,float]}.
    """
    final_state = run_loop(model, state, steps, dt, hooks, verbose=False)
    metrics = telemetry_summary(final_state.phi,
                                model.background.lambda_,
                                model.background.D,
                                model.background.phi_inf,
                                final_state.spacing,
                                final_state.bc)
    return {"state": final_state, "metrics": metrics}


# ============================================================
# Minimal logging hook
# ============================================================

def print_step_hook(state: FieldState, metrics: Dict[str, float]) -> None:
    """Simple default hook printing variance and energy."""
    print(f"t={state.t:.4f}  variance={metrics['variance']:.4e}  energy={metrics['energy']:.4e}")


# ============================================================
# Exports
# ============================================================

__all__ = ["run_loop", "simulate_once", "print_step_hook"]
