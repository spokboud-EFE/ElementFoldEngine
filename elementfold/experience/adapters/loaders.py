# ElementFold · loaders.py
# One place to build DataLoaders for text / image / audio / synthetic streams.
# Think: “give me a folder and a kind; I get a DataLoader I can iterate.”
#
# Goals:
#   • Small + explicit: one LoaderSpec, four builders (text/image/audio/synthetic).
#   • Portable: no hard dependency on torchvision; datasets handle fallbacks.
#   • Consistent UX: same defaults as the rest of ElementFold (seq_len=128, vocab=256, …).
#
# Import note:
#   You indicated that in this file the working import style for text is:
#       from engine.elementfold.datasets.text_files import <…>
#   We mirror that style for image/audio for consistency.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os

import torch
from torch.utils.data import DataLoader

# — Text (absolute import path as requested) —
from elementfold.datasets.text_files import (
    make_text_loader,
    TextChunkDataset,
    TextLineDataset,
    pad_collate,
)

# — Image (absolute import for consistency) —
from elementfold.datasets.image_folder import (
    make_image_loader,
    ImageFolderDataset,
)

# — Audio (absolute import for consistency) —
from elementfold.datasets.audio_folder import AudioFolderDataset

# — Synthetic (local tiny pulse stream) —
from elementfold.core.data import PulseDataset


# ———————————————————————————————————————————————————————————
# Spec & tiny URI parser
# ———————————————————————————————————————————————————————————

@dataclass
class LoaderSpec:
    # Kind & path
    kind: str                        # 'text' | 'image' | 'audio' | 'synthetic'
    path: Optional[str] = None       # folder path (None for synthetic)

    # Common knobs
    batch: int = 32
    workers: int = 0

    # Text‑specific
    mode: str = "chunks"             # 'chunks' | 'lines'
    seq_len: int = 128
    stride: Optional[int] = None
    recursive: bool = True

    # Image‑specific
    size: int = 64
    to_float: bool = False
    normalize: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]] = None
    augment: bool = False

    # Audio‑specific
    # (no extra knobs; yields (seq_len,) tokens like text)

    # Synthetic‑specific
    vocab: int = 256
    n: int = 10000                   # synthetic sample count (approx; DL can repeat)

def parse_uri(uri_or_kind: str) -> tuple[str, Optional[str]]:
    """
    Accept either:
      • 'text', 'image', 'audio', 'synthetic'  (no path),
      • 'text:/path/to/folder'                 (kind:path),
      • '/path/to/folder'                      (auto‑infer kind from contents).
    Returns (kind, path|None).

    Heuristics for folder → kind:
      • any *.wav      → 'audio'
      • any common img → 'image'
      • otherwise      → 'text'
    """
    s = str(uri_or_kind).strip()
    if ":" in s:
        # kind:/abs/or/rel/path
        kind, path = s.split(":", 1)
        return kind.strip().lower(), (path or None)

    # If it's an existing directory, try to guess the kind.
    if os.path.isdir(s):
        try:
            names = [n.lower() for n in os.listdir(s)[:50]]
        except Exception:
            names = []
        if any(n.endswith((".wav",)) for n in names):
            return "audio", s
        if any(n.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp")) for n in names):
            return "image", s
        return "text", s  # default to text
    return s.lower(), None  # bare kind


# ———————————————————————————————————————————————————————————
# Builders per kind
# ———————————————————————————————————————————————————————————

def make_text(spec: LoaderSpec) -> DataLoader:
    """
    Build a text DataLoader from a folder using either:
      • mode='chunks' → fixed (B, seq_len) blocks (LM‑style),
      • mode='lines'  → padded (B, T) batches.
    """
    return make_text_loader(
        root=spec.path or ".",
        mode=spec.mode,
        seq_len=spec.seq_len,
        stride=spec.stride,
        batch=spec.batch,
        shuffle_files=False,
        workers=spec.workers,
        recursive=spec.recursive,
    )

def make_image(spec: LoaderSpec) -> DataLoader:
    """
    Build an image DataLoader streaming (B,C,H,W) tensors.
    dtype/normalization is controlled by dataset kwargs (to_float/normalize).
    """
    return make_image_loader(
        root=spec.path or ".",
        size=spec.size,
        batch=spec.batch,
        workers=spec.workers,
        recursive=spec.recursive,
        to_float=spec.to_float,
        normalize=spec.normalize,
        augment=spec.augment,
    )

def make_audio(spec: LoaderSpec) -> DataLoader:
    """
    Build an audio DataLoader that yields (seq_len,) int64 token ids in [0..255].
    """
    ds = AudioFolderDataset(root=spec.path or ".", seq_len=spec.seq_len)
    return DataLoader(ds, batch_size=spec.batch, shuffle=False, num_workers=spec.workers)

def make_synthetic(spec: LoaderSpec) -> DataLoader:
    """
    Build a simple synthetic token stream for quick smoke tests (PulseDataset).
    """
    ds = PulseDataset(n=spec.n, seq_len=spec.seq_len, vocab=spec.vocab)
    return torch.utils.data.DataLoader(ds, batch_size=spec.batch, shuffle=True, drop_last=True)


# ———————————————————————————————————————————————————————————
# Unified front doors
# ———————————————————————————————————————————————————————————

def make_loader(kind_or_uri: str, **kw) -> DataLoader:
    """
    Explicit loader: you give a kind or URI + kwargs; we return a DataLoader.
      • kind_or_uri='text:/data/books' (plus seq_len=…, batch=…)
      • kind_or_uri='image:/data/imagenet'
      • kind_or_uri='audio:/data/wavs'
      • kind_or_uri='synthetic'       (uses seq_len, vocab, n, batch)
    """
    kind, path = parse_uri(kind_or_uri)
    spec = LoaderSpec(kind=kind, path=path, **kw)

    if spec.kind == "text":
        return make_text(spec)
    if spec.kind == "image":
        return make_image(spec)
    if spec.kind == "audio":
        return make_audio(spec)
    if spec.kind == "synthetic":
        return make_synthetic(spec)

    raise ValueError(f"unknown loader kind: {spec.kind!r}")

def auto_loader(source: str | None, **kw) -> DataLoader:
    """
    Convenience:
      • None or 'synthetic' → synthetic tokens,
      • '/some/folder'      → guess kind from contents,
      • 'kind:/folder'      → explicit.
    """
    kind, path = parse_uri(source or "synthetic")
    return make_loader(f"{kind}:{path}" if path else kind, **kw)


__all__ = [
    "LoaderSpec",
    "parse_uri",
    "make_loader",
    "auto_loader",
    "make_text",
    "make_image",
    "make_audio",
    "make_synthetic",
    # Re‑export a few dataset classes for callers that want lower‑level control:
    "TextChunkDataset",
    "TextLineDataset",
    "pad_collate",
    "ImageFolderDataset",
    "AudioFolderDataset",
]
