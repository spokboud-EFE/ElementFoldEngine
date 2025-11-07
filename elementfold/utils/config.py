# ElementFold · utils/config.py
# ============================================================
# Configuration Carrier — typed, self-validating, and minimal.
# ------------------------------------------------------------
# Purpose:
#   • Store all simulation parameters (λ, D, Φ∞, grid, dt, etc.).
#   • Load/save as JSON.
#   • Normalize paths and devices (CPU/GPU optional).
#   • Stay forward-compatible with future schema updates.
# ============================================================

from __future__ import annotations
import os, json
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


# ============================================================
# Helpers
# ============================================================

def _normalize_device(label: Optional[str]) -> Optional[str]:
    s = (label or "").strip().lower()
    if s in {"", "auto", "default", "none"}:
        return None
    if s == "cpu":
        return "cpu"
    if s.startswith("cuda") or s == "gpu":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return None


def _atomic_write(path: str, text: str, *, encoding: str = "utf-8") -> None:
    folder = os.path.dirname(path) or "."
    os.makedirs(folder, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp, path)


# ============================================================
# Config dataclass
# ============================================================

@dataclass
class Config:
    # --- numerical simulation parameters ---
    lambda_: float = 0.33        # letting-go rate
    D: float = 0.15              # diffusion strength
    phi_inf: float = 0.0         # calm baseline
    dt: float = 0.05             # timestep
    steps: int = 100             # number of updates per run
    grid: Tuple[int, int] = (64, 64)  # simulation grid
    spacing: Tuple[float, float] = (1.0, 1.0)
    bc: str = "neumann"          # boundary condition
    seed: int = 1234

    # --- output / runtime ---
    save_every: int = 50
    out_dir: str = "runs"
    device: Optional[str] = None
    ui: str = "auto"
    schema_version: int = 1

    def __post_init__(self) -> None:
        self._validate()

    # ========================================================
    # Conversion helpers
    # ========================================================

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Config:
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        filt = {k: d[k] for k in d if k in valid}
        return cls(**filt)

    @classmethod
    def from_json(cls, s: str) -> Config:
        try:
            return cls.from_dict(json.loads(s))
        except Exception as e:
            raise ValueError(f"invalid JSON ({e})")

    @classmethod
    def load(cls, path: str, encoding: str = "utf-8") -> Config:
        with open(path, "r", encoding=encoding) as f:
            return cls.from_json(f.read())

    def save(self, path: str, indent: int = 2) -> None:
        payload = self.to_json(indent)
        try:
            _atomic_write(path, payload)
        except Exception:
            with open(path, "w", encoding="utf-8") as f:
                f.write(payload)

    # ========================================================
    # Validation
    # ========================================================

    def _validate(self) -> None:
        self.lambda_ = float(max(0.0, self.lambda_))
        self.D = float(max(0.0, self.D))
        self.dt = float(max(1e-9, self.dt))
        self.steps = max(1, int(self.steps))
        self.grid = tuple(max(2, int(x)) for x in self.grid)
        self.spacing = tuple(float(x) for x in self.spacing)
        if self.bc not in {"neumann", "periodic"}:
            self.bc = "neumann"
        self.save_every = max(1, int(self.save_every))
        self.device = _normalize_device(self.device)
        self.seed = int(self.seed)

    # ========================================================
    # Convenience: kwargs for runtime
    # ========================================================

    def to_kwargs(self) -> Dict[str, Any]:
        """Return args suitable for core.runtime.simulate_once."""
        return dict(
            lambda_=self.lambda_,
            D=self.D,
            phi_inf=self.phi_inf,
            dt=self.dt,
            steps=self.steps,
            spacing=self.spacing,
            bc=self.bc,
        )
