# ElementFold · config.py
# A tiny, typed configuration carrier with friendly JSON I/O.
# It holds just the knobs the training loop needs (device, steps, d, …) and
# provides helpers so callers can serialize/deserialize configs cleanly.

from __future__ import annotations                           # Forward annotations (older Python)
from dataclasses import dataclass, asdict, field             # Typed, lightweight config container
from typing import Tuple, Optional, Dict, Any                # Minimal typing for clarity
import json                                                  # JSON (text) ↔ dict conversion


@dataclass
class Config:
    # — core training/runtime knobs —
    device: str = "cuda"                                     # 'cuda' | 'cpu' (training loop also accepts None → auto)
    steps: int = 200                                         # total optimization steps
    vocab: int = 256                                         # tokenizer/model vocabulary size
    d: int = 128                                             # feature width
    layers: int = 4                                          # number of FGN blocks
    heads: int = 4                                           # kept for parity with attention configs
    seq_len: int = 128                                       # max sequence length
    fold: str = "grid"                                       # fold kind ('grid' FGN currently)
    delta: float = 0.030908106561043047                      # δ⋆ coherence click
    capacities: Tuple[int, ...] = (2, 6, 10, 14)             # seat capacities per block
    batch: int = 32                                          # batch size
    use_data: bool = True                                    # use DataLoader vs. synthetic tokens

    # — optional modality checkpoints (UX layer can use these) —
    lang_ckpt: Optional[str] = None                          # path to language steering/model ckpt
    vision_ckpt: Optional[str] = None                        # path to vision steering/model ckpt
    audio_ckpt: Optional[str] = None                         # path to audio steering/model ckpt

    # — helpers: dictionary → kwargs for train_loop —
    def to_kwargs(self) -> Dict[str, Any]:
        """
        Only the arguments the training loop expects. We deliberately exclude
        modality checkpoint paths so callers don’t accidentally pass them.
        """
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
        )

    # — convenience: full dict snapshot (includes ckpt fields) —
    def to_dict(self) -> Dict[str, Any]:
        """Full dataclass → dict (safe to JSON‑dump)."""
        return asdict(self)

    # — pretty JSON text for saving to disk or logs —
    def to_json(self, indent: int = 2) -> str:
        """Serialize the full config (including ckpt fields) to JSON."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    # — load from python dict (ignore unknown keys gracefully) —
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        """
        Construct a Config from a plain dict. Unknown keys are ignored so
        older/newer configs stay compatible across versions.
        """
        if not isinstance(d, dict):
            raise TypeError(f"Config.from_dict expects dict, got {type(d).__name__}")
        # Filter only fields declared on the dataclass
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        filtered = {k: d[k] for k in d.keys() if k in valid}
        cfg = cls(**filtered)
        cfg._validate()  # ensure values are sensible
        return cfg

    # — load from JSON text (pairs with to_json) —
    @classmethod
    def from_json(cls, s: str) -> "Config":
        """Parse a JSON string produced by to_json()/manual edits and build a Config."""
        try:
            payload = json.loads(s)
        except Exception as e:
            raise ValueError(f"Config.from_json: invalid JSON ({e})")
        return cls.from_dict(payload)

    # — internal sanity checks (kept gentle) —
    def _validate(self) -> None:
        """
        Clamp/normalize a few fields to keep the engine in a sensible range.
        We keep this light‑touch: enforce positivity and basic structure.
        """
        self.steps = max(1, int(self.steps))
        self.vocab = max(2, int(self.vocab))
        self.d = max(4, int(self.d))
        self.layers = max(1, int(self.layers))
        self.heads = max(1, int(self.heads))
        self.seq_len = max(1, int(self.seq_len))
        self.batch = max(1, int(self.batch))
        # capacities must be a non‑empty tuple of positive ints
        if not isinstance(self.capacities, (tuple, list)) or len(self.capacities) == 0:
            self.capacities = (2, 6, 10, 14)
        self.capacities = tuple(max(1, int(c)) for c in self.capacities)
        # device: normalize common variants
        dev = (self.device or "").strip().lower()
        if dev in ("auto", "autodetect", "default"):
            self.device = None  # let the training loop pick
        elif dev in ("cuda", "gpu", "cuda:0"):
            self.device = "cuda"
        elif dev in ("cpu",):
            self.device = "cpu"
        else:
            # Unknown label → defer to loop auto‑select by setting None
            self.device = None

# Legacy names kept for backward compatibility with earlier snippets
lang_ckpt = None
vision_ckpt = None
audio_ckpt = None
