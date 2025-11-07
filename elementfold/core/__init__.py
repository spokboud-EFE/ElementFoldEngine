# elementfold/core/__init__.py  (modern physics API)

from . import data, control, fgn, telemetry, quantize, runtime

# control (PDE)
from .control import safe_dt, step, step_torch, step_semi_implicit, Controller

# folds / observables
from .fgn import folds, redshift_from_F, brightness_tilt, time_dilation, bend

# telemetry
from .telemetry import summary as telemetry_summary, variance, grad_L2, total_energy

__all__ = [
    "data", "control", "fgn", "telemetry", "quantize", "runtime",
    "safe_dt", "step", "step_torch", "step_semi_implicit", "Controller",
    "folds", "redshift_from_F", "brightness_tilt", "time_dilation", "bend",
    "telemetry_summary", "variance", "grad_L2", "total_energy",
]
