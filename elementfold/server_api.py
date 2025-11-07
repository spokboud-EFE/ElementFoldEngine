# ElementFold · server_api.py
# ============================================================
# Minimal REST schema for relaxation physics API
# ------------------------------------------------------------
# Endpoints:
#   /simulate   → evolve Φ field for N steps
#   /folds      → integrate cumulative folds ℱ
#   /redshift   → compute (1+z) = e^ℱ − 1
#   /brightness → brightness tilt and geometric dimming
#   /bend       → color-dependent angular deflection
# ============================================================

from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union
from .data import PathSegment


# ============================================================
# Data schemas
# ============================================================

@dataclass
class SimulateRequest:
    shape: List[int] = None            # grid shape (e.g. [64,64])
    spacing: List[float] = None        # Δx, Δy, (Δz)
    bc: str = "neumann"                # boundary condition
    lambda_: float = 0.33              # letting-go rate
    D: float = 0.15                    # smoothing coefficient
    phi_inf: float = 0.0               # baseline
    steps: int = 10                    # integration steps
    dt: Optional[float] = None         # explicit time step (auto if None)

@dataclass
class SimulateResponse:
    phi: List[Any]                     # final Φ array (nested lists)
    t: float                           # final time
    metrics: Dict[str, float]          # variance, grad², energy

@dataclass
class PathEntry:
    ds: float
    phi: float
    nu: float

@dataclass
class PathRequest:
    path: List[PathEntry]
    params: Optional[Dict[str, Any]] = None

@dataclass
class FoldsResponse:
    F: float

@dataclass
class RedshiftResponse:
    z: float

@dataclass
class BrightnessResponse:
    I_obs: float

@dataclass
class BendResponse:
    dtheta: float

@dataclass
class ErrorResponse:
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


# ============================================================
# JSON helpers
# ============================================================

def parse_json(body: Union[str, bytes]) -> Dict[str, Any]:
    if not body:
        return {}
    if isinstance(body, bytes):
        body = body.decode("utf-8")
    data = json.loads(body)
    if not isinstance(data, dict):
        raise ValueError("top-level JSON must be an object")
    return data


def _json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return 0.0
    return x


def to_json(obj: Any) -> bytes:
    if hasattr(obj, "__dataclass_fields__"):
        obj = asdict(obj)
    obj = _json_sanitize(obj)
    return json.dumps(obj, ensure_ascii=False, allow_nan=False).encode("utf-8")


# ============================================================
# Request coercion
# ============================================================

def coerce_simulate_request(payload: Dict[str, Any]) -> SimulateRequest:
    shape = [int(x) for x in payload.get("shape", [64, 64])]
    spacing = [float(x) for x in payload.get("spacing", [1.0, 1.0])]
    bc = str(payload.get("bc", "neumann"))
    lam = float(payload.get("lambda", payload.get("lambda_", 0.33)))
    D = float(payload.get("D", 0.15))
    phi_inf = float(payload.get("phi_inf", 0.0))
    steps = int(payload.get("steps", 10))
    dt = payload.get("dt", None)
    try:
        dt = float(dt) if dt is not None else None
    except Exception:
        dt = None
    return SimulateRequest(shape=shape, spacing=spacing, bc=bc,
                           lambda_=lam, D=D, phi_inf=phi_inf,
                           steps=steps, dt=dt)


def coerce_path_request(payload: Dict[str, Any]) -> PathRequest:
    raw = payload.get("path", [])
    path = []
    for seg in raw:
        try:
            ds = float(seg.get("ds", 1.0))
            phi = float(seg.get("phi", 0.0))
            nu = float(seg.get("nu", 1.0))
            path.append(PathEntry(ds=ds, phi=phi, nu=nu))
        except Exception:
            continue
    return PathRequest(path=path, params=payload.get("params", None))


# ============================================================
# Response packers
# ============================================================

def pack_simulate_response(phi: Any,
                           t: float,
                           metrics: Dict[str, float]) -> SimulateResponse:
    if hasattr(phi, "tolist"):
        phi = phi.tolist()
    return SimulateResponse(phi=phi, t=float(t), metrics=_json_sanitize(metrics))


def pack_error(code: str, message: str, details: Optional[Dict[str, Any]] = None) -> ErrorResponse:
    return ErrorResponse(code=code, message=message, details=details)
