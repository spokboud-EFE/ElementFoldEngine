# ElementFold · config.py
# ──────────────────────────────────────────────────────────────────────────────
# A tiny, typed configuration carrier with JSON I/O.
# Read this like a checklist:
#   • It stores the handful of knobs the training loop understands.
#   • It can be serialized to / from JSON (files or strings).
#   • It validates gently so bad values don’t slip through.
#
# Public surface (unchanged spirit, slightly richer set):
#   Config(...)                          — dataclass with safe defaults
#   cfg.to_kwargs()                      — only args the training loop accepts
#   cfg.to_dict() / cfg.to_json()        — full snapshot (pretty JSON)
#   Config.from_dict(d) / from_json(s)   — tolerant loaders (ignore unknown keys)
#   Config.load(path) / cfg.save(path)   — small file helpers
#
# Notes:
#   • We include a few extra training knobs already supported by train_loop:
#       lr, wd, warmup_frac, clip_norm, tv_weight, ui, print_every, out,
#       and rung_* (intent/target/band/loss_weight).
#   • Device: 'cuda' | 'cpu' | None ('auto'); we normalize politely.

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Tuple
import json
import sys


def _normalize_device(label: Optional[str]) -> Optional[str]:
    """
    Map user labels → {'cuda','cpu',None}. If CUDA was requested but appears
    unavailable at import time, fall back to None (auto) rather than error.
    """
    s = (label or "").strip().lower()
    if s in ("", "auto", "autodetect", "default", "none", "null"):
        return None
    if s in ("cpu",):
        return "cpu"
    if s in ("cuda", "gpu", "cuda:0", "cuda0"):
        try:
            import torch  # local import to avoid hard dependency if unused
            return "cuda" if torch.cuda.is_available() else None
        except Exception:
            return None
    # Unknown → auto
    return None


def _normalize_ui(label: str) -> str:
    s = (label or "auto").strip().lower()
    if s not in {"auto", "unicode", "ascii"}:
        return "auto"
    return s


def _normalize_intent(label: Optional[str]) -> str:
    """
    Keep intent as a *string* to avoid import cycles with RungIntent.
    Accept a few spelling variants; return one of: 'stabilize' | 'hold' | 'seek'.
    """
    s = (label or "stabilize").strip().lower()
    aliases = {
        "stabilise": "stabilize",
        "stabilizer": "stabilize",
        "stabilise/hold": "hold",
        "lock": "hold",
        "center": "hold",
        "step": "seek",  # legacy UI often uses 'step_*' — SEEK does the walking FSM
        "seek": "seek",
        "hold": "hold",
        "stabilize": "stabilize",
    }
    return aliases.get(s, "stabilize")


@dataclass
class Config:
    # — core training/runtime knobs —
    device: Optional[str] = "cuda"                 # 'cuda' | 'cpu' | None (auto) — normalized in __post_init__
    steps: int = 200                               # total optimization steps
    vocab: int = 256                               # tokenizer/model vocabulary size
    d: int = 128                                   # feature width
    layers: int = 4                                # number of FGN blocks
    heads: int = 4                                 # kept for parity with attention configs
    seq_len: int = 128                             # max sequence length
    fold: str = "grid"                             # fold kind ('grid' FGN currently)
    delta: float = 0.030908106561043047            # δ⋆ coherence click
    capacities: Tuple[int, ...] = (2, 6, 10, 14)   # seat capacities per block
    batch: int = 32                                # batch size
    use_data: bool = True                          # use DataLoader vs. synthetic tokens

    # — optimizer / schedule / regularization (already supported by train_loop) —
    lr: float = 2e-4                               # AdamW learning rate
    wd: float = 0.01                               # AdamW weight decay
    warmup_frac: float = 0.10                      # fraction of steps for warmup
    clip_norm: float = 1.0                         # gradient‑norm clip
    tv_weight: float = 0.0                         # total‑variation weight on ledger (0 = off)

    # — UI / logging / output —
    print_every: Optional[int] = None              # e.g. 50 → progress line every 50 steps; None = silent
    ui: str = "auto"                               # 'unicode' | 'ascii' | 'auto'
    out: Optional[str] = None                      # optional path/dir for checkpoint

    # — rung control (flows into RungController via train_loop) —
    rung_intent: str = "stabilize"                 # 'stabilize' | 'hold' | 'seek'
    rung_target_k: Optional[int] = None            # target rung index (for SEEK/HOLD); None = heuristic / nearest
    rung_band: Optional[float] = None              # acceptance half‑band; None → δ⋆/6
    rung_loss_weight: float = 0.0                  # add small rung penalty to total loss (0 = off)

    # — optional modality checkpoints (kept for UX parity; not used by train_loop) —
    lang_ckpt: Optional[str] = None
    vision_ckpt: Optional[str] = None
    audio_ckpt: Optional[str] = None

    # — meta —
    schema_version: int = 1                        # future‑proofing: bump if structure changes

    # Ensure sensible values even when built from sparse/older dicts.
    def __post_init__(self) -> None:
        self._validate()

    # ——————————————————————————————————————————————————————————————
    # Train‑loop kwargs (strict subset that train_loop accepts)
    # ——————————————————————————————————————————————————————————————
    def to_kwargs(self) -> Dict[str, Any]:
        """
        Only the arguments the training loop expects. We deliberately exclude
        any UI‑only or checkpoint‑only fields the loop does not accept.
        """
        return dict(
            # core
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
            # opt/schedule/regularization
            lr=self.lr,
            wd=self.wd,
            warmup_frac=self.warmup_frac,
            clip_norm=self.clip_norm,
            tv_weight=self.tv_weight,
            # output/logging
            out=self.out,
            print_every=self.print_every,
            ui=self.ui,
            # rung control
            rung_intent=self.rung_intent,
            rung_target_k=self.rung_target_k,
            rung_band=self.rung_band,
            rung_loss_weight=self.rung_loss_weight,
        )

    # — snapshots / JSON I/O —
    def to_dict(self) -> Dict[str, Any]:
        """Full dataclass → dict (safe to JSON‑dump)."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize the full config (including non‑training fields) to JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """
        Construct from a plain dict. Unknown keys are ignored so older/newer configs
        stay compatible across versions.
        """
        if not isinstance(d, dict):
            raise TypeError(f"Config.from_dict expects dict, got {type(d).__name__}")
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: d[k] for k in d.keys() if k in valid}
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> "Config":
        """Parse JSON produced by to_json()/manual edits and build a Config."""
        try:
            payload = json.loads(s)
        except Exception as e:
            raise ValueError(f"Config.from_json: invalid JSON ({e})")
        return cls.from_dict(payload)

    @classmethod
    def load(cls, path: str, encoding: str = "utf-8") -> "Config":
        """Read JSON config from a file path."""
        with open(path, "r", encoding=encoding) as f:
            return cls.from_json(f.read())

    def save(self, path: str, indent: int = 2, encoding: str = "utf-8") -> None:
        """Write JSON config to a file path (pretty)."""
        try:
            # Optional convenience if your tree provides an atomic writer.
            from .utils.io import write_text  # type: ignore[attr-defined]
            write_text(path, self.to_json(indent=indent))
        except Exception:
            with open(path, "w", encoding=encoding) as f:
                f.write(self.to_json(indent=indent))

    # — gentle sanity checks / normalization —
    def _validate(self) -> None:
        """Clamp and normalize a few fields to keep the engine sensible."""
        # shapes / counts
        self.steps = max(1, int(self.steps))
        self.vocab = max(2, int(self.vocab))
        self.d = max(4, int(self.d))
        self.layers = max(1, int(self.layers))
        self.heads = max(1, int(self.heads))
        self.seq_len = max(1, int(self.seq_len))
        self.batch = max(1, int(self.batch))

        # capacities: non‑empty, positive ints
        if not isinstance(self.capacities, (tuple, list)) or len(self.capacities) == 0:
            self.capacities = (2, 6, 10, 14)
        self.capacities = tuple(max(1, int(c)) for c in self.capacities)

        # numeric knobs
        self.lr = float(max(1e-7, min(1.0, float(self.lr))))
        self.wd = float(max(0.0, float(self.wd)))
        self.warmup_frac = float(max(0.0, min(0.99, float(self.warmup_frac))))
        self.clip_norm = float(max(0.0, float(self.clip_norm)))
        self.tv_weight = float(max(0.0, float(self.tv_weight)))
        self.delta = float(self.delta)

        # UI & device
        self.ui = _normalize_ui(self.ui)
        self.device = _normalize_device(self.device)

        # rung control
        self.rung_intent = _normalize_intent(self.rung_intent)
        self.rung_target_k = (None if self.rung_target_k is None else int(self.rung_target_k))
        self.rung_band = (None if self.rung_band is None else float(self.rung_band))
        self.rung_loss_weight = float(max(0.0, float(self.rung_loss_weight)))


# Legacy module‑level names (kept for backward compatibility with earlier snippets)
lang_ckpt = None
vision_ckpt = None
audio_ckpt = None
