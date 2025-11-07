# ElementFold · config.py
# ============================================================
# Configuration Carrier — small, typed, self-validating.
#
# Purpose
# -------
# • Store all training-time knobs in one safe dataclass.
# • Load/save as JSON (file or string).
# • Normalize devices, UI, and rung intent names.
# • Stay compatible across schema versions.
# • No external helpers (self‑contained; no core.utils dependency).
# ============================================================

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple

# ============================================================
# Normalization helpers
# ============================================================

def _normalize_device(label: Optional[str]) -> Optional[str]:
    """Map arbitrary user labels → {'cuda','cpu', None (auto choose later)}."""
    s = (label or "").strip().lower()
    if s in ("", "auto", "autodetect", "default", "none", "null"):
        return None
    if s == "cpu":
        return "cpu"
    if s in ("cuda", "gpu", "cuda:0", "cuda0"):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else None
        except Exception:
            return None
    return None  # unknown → defer to runtime auto-pick

def _normalize_ui(label: Optional[str]) -> str:
    s = (label or "auto").strip().lower()
    return s if s in {"auto", "unicode", "ascii"} else "auto"

def _normalize_intent(label: Optional[str]) -> str:
    """Normalize rung intent → {'stabilize','hold','seek'}."""
    s = (label or "stabilize").strip().lower()
    aliases = {
        "stabilise": "stabilize",
        "stabilizer": "stabilize",
        "stabilise/hold": "hold",
        "lock": "hold",
        "center": "hold",
        "step": "seek",
        "seek": "seek",
        "hold": "hold",
        "stabilize": "stabilize",
    }
    return aliases.get(s, "stabilize")

# ============================================================
# Tiny, local atomic writer (no external utils)
# ============================================================

def _atomic_write(path: str, text: str, *, encoding: str = "utf-8") -> None:
    """Best-effort atomic file write using a temp file + replace."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding=encoding) as f:
        f.write(text)
    os.replace(tmp_path, path)

# ============================================================
# Config dataclass
# ============================================================

@dataclass
class Config:
    # --- core geometry / runtime ---
    device: Optional[str] = "cuda"      # 'cuda'→uses if available, else auto; 'cpu' or None also ok
    steps: int = 200
    vocab: int = 256
    d: int = 128
    layers: int = 4
    heads: int = 4
    seq_len: int = 128
    fold: str = "grid"
    delta: float = 0.030908106561043047
    capacities: Tuple[int, ...] = (2, 6, 10, 14)
    batch: int = 32
    use_data: bool = True

    # --- optimizer / schedule ---
    lr: float = 2e-4
    wd: float = 0.01
    warmup_frac: float = 0.10
    clip_norm: float = 1.0
    tv_weight: float = 0.0

    # --- UI / output ---
    print_every: Optional[int] = None
    ui: str = "auto"
    out: Optional[str] = None

    # --- rung control ---
    rung_intent: str = "stabilize"
    rung_target_k: Optional[int] = None
    rung_band: Optional[float] = None
    rung_loss_weight: float = 0.0

    # --- optional modality checkpoints ---
    lang_ckpt: Optional[str] = None
    vision_ckpt: Optional[str] = None
    audio_ckpt: Optional[str] = None

    # --- meta ---
    schema_version: int = 1

    def __post_init__(self) -> None:
        self._validate()

    # ========================================================
    # Train-loop argument subset
    # ========================================================
    def to_kwargs(self) -> Dict[str, Any]:
        """Subset of fields the training loop understands."""
        return dict(
            device=self.device,
            steps=self.steps,
            vocab=self.vocab,
            d=self.d,
            layers=self.layers,
            heads=self.heads,
            seq_len=self.seq_len,
            fold=self.fold,
            delta=self.delta,
            capacities=self.capacities,
            batch=self.batch,
            use_data=self.use_data,
            lr=self.lr,
            wd=self.wd,
            warmup_frac=self.warmup_frac,
            clip_norm=self.clip_norm,
            tv_weight=self.tv_weight,
            out=self.out,
            print_every=self.print_every,
            ui=self.ui,
            rung_intent=self.rung_intent,
            rung_target_k=self.rung_target_k,
            rung_band=self.rung_band,
            rung_loss_weight=self.rung_loss_weight,
        )

    # ========================================================
    # JSON I/O
    # ========================================================
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Config:
        """Ignore unknown keys for forward/backward compatibility."""
        if not isinstance(d, dict):
            raise TypeError(f"from_dict expects dict, got {type(d).__name__}")
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
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

    def save(self, path: str, indent: int = 2, encoding: str = "utf-8") -> None:
        """Write pretty JSON (atomic best-effort), with fallback to direct write."""
        payload = self.to_json(indent)
        try:
            _atomic_write(path, payload, encoding=encoding)
        except Exception:
            with open(path, "w", encoding=encoding) as f:
                f.write(payload)

    # ========================================================
    # Validation & normalization
    # ========================================================
    def _validate(self) -> None:
        """Clamp numeric ranges, normalize strings, ensure non-empty lists."""
        self.steps = max(1, int(self.steps))
        self.vocab = max(2, int(self.vocab))
        self.d = max(4, int(self.d))
        self.layers = max(1, int(self.layers))
        self.heads = max(1, int(self.heads))
        self.seq_len = max(1, int(self.seq_len))
        self.batch = max(1, int(self.batch))

        if not isinstance(self.capacities, (tuple, list)) or not self.capacities:
            self.capacities = (2, 6, 10, 14)
        self.capacities = tuple(max(1, int(c)) for c in self.capacities)

        self.lr = float(max(1e-7, min(1.0, float(self.lr))))
        self.wd = float(max(0.0, float(self.wd)))
        self.warmup_frac = float(max(0.0, min(0.99, float(self.warmup_frac))))
        self.clip_norm = float(max(0.0, float(self.clip_norm)))
        self.tv_weight = float(max(0.0, float(self.tv_weight)))
        self.delta = float(self.delta)

        self.ui = _normalize_ui(self.ui)
        self.device = _normalize_device(self.device)
        self.rung_intent = _normalize_intent(self.rung_intent)
        self.rung_target_k = None if self.rung_target_k is None else int(self.rung_target_k)
        self.rung_band = None if self.rung_band is None else float(self.rung_band)
        self.rung_loss_weight = float(max(0.0, float(self.rung_loss_weight)))

# ============================================================
# Legacy globals (kept for backward-compat scripts)
# ============================================================
lang_ckpt = None
vision_ckpt = None
audio_ckpt = None
