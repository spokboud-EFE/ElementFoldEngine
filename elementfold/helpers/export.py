# ElementFold · helpers/export.py
# ============================================================
# Portable state and parameter exporters for relaxation runs
# ------------------------------------------------------------
# Responsibilities
#   • dump_state / load_state        – Φ arrays + metadata to .json / .npy
#   • dump_params / load_params      – background + optics to .json
#   • save_bundle / load_bundle      – combined archive (.npz)
#
# All functions are NumPy / JSON only (no torch dependency).
# ============================================================

from __future__ import annotations
import os, json, time
import numpy as np
from typing import Any, Dict

from ..core.data import FieldState, BackgroundParams, OpticalParams


# ============================================================
# Helpers
# ============================================================

def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


def _timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ============================================================
# Field state I/O
# ============================================================

def dump_state(state: FieldState, path: str) -> str:
    """
    Save Φ, spacing, bc, and t to a portable JSON + NPY pair.
    """
    _ensure_dir(path)
    base, _ = os.path.splitext(path)
    npy_path = base + ".npy"
    meta_path = base + ".json"

    np.save(npy_path, state.phi.astype(np.float32))

    meta = {
        "format": "elementfold.state.v1",
        "time_utc": _timestamp(),
        "shape": list(state.phi.shape),
        "spacing": list(state.spacing),
        "bc": state.bc,
        "t": float(state.t),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return base


def load_state(base_path: str) -> FieldState:
    """
    Load a FieldState from a JSON + NPY pair written by dump_state().
    """
    base, _ = os.path.splitext(base_path)
    npy_path = base + ".npy"
    meta_path = base + ".json"
    phi = np.load(npy_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return FieldState(
        phi=np.array(phi, dtype=float),
        t=float(meta.get("t", 0.0)),
        spacing=tuple(meta.get("spacing", [1.0] * phi.ndim)),
        bc=meta.get("bc", "neumann"),
    )


# ============================================================
# Parameter I/O
# ============================================================

def dump_params(background: BackgroundParams,
                optics: OpticalParams,
                path: str) -> str:
    """
    Save background and optical parameter sets as JSON.
    """
    _ensure_dir(path)
    payload = {
        "format": "elementfold.params.v1",
        "time_utc": _timestamp(),
        "background": background.as_dict(),
        "optics": optics.as_dict(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_params(path: str) -> Dict[str, Any]:
    """
    Load background/optics parameters from JSON.
    Returns {'background': BackgroundParams, 'optics': OpticalParams}.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    bg = BackgroundParams(**payload["background"])
    op = OpticalParams(**payload["optics"])
    return {"background": bg, "optics": op}


# ============================================================
# Combined bundle (for quick checkpoints)
# ============================================================

def save_bundle(state: FieldState,
                background: BackgroundParams,
                optics: OpticalParams,
                path: str) -> str:
    """
    Store Φ and parameters together in one .npz archive.
    """
    _ensure_dir(path)
    np.savez_compressed(
        path,
        phi=state.phi.astype(np.float32),
        t=state.t,
        spacing=np.array(state.spacing, dtype=np.float32),
        bc=np.string_(state.bc),
        lambda_=background.lambda_,
        D=background.D,
        phi_inf=background.phi_inf,
        nu0=optics.nu0,
    )
    return path


def load_bundle(path: str) -> Dict[str, Any]:
    """
    Read a .npz bundle back into state + params.
    """
    data = np.load(path, allow_pickle=False)
    phi = data["phi"]
    t = float(data["t"])
    spacing = tuple(data["spacing"].tolist())
    bc = str(data["bc"].tolist().decode("utf-8")) if isinstance(data["bc"], np.ndarray) else "neumann"
    bg = BackgroundParams(float(data["lambda_"]), float(data["D"]), float(data["phi_inf"]))
    # Optics reconstructed minimally (functional forms supplied elsewhere)
    from ..core.data import default_optics
    op = default_optics()
    op.nu0 = float(data.get("nu0", 1.0))
    return {"state": FieldState(phi, t, spacing, bc), "background": bg, "optics": op}
