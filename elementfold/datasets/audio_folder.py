# ElementFold · datasets/audio_folder.py
# WAV files → mono waveform → 8‑bit tokens (linear or μ‑law) with graceful fallbacks.
#
# Why this design:
#   • Pure stdlib (wave/struct) so it runs anywhere; no heavy deps required.
#   • Robust to odd files: if decode fails, we yield synthetic tokens so pipelines keep running.
#   • Returns int64 token ids in [0,255] of fixed length (seq_len) ready for the model.
#
# What you get:
#   • iter_wav_paths()       — walk a folder (optionally recursive) for *.wav files.
#   • pcm_to_tokens()        — linear or μ‑law companding to 8‑bit ids.
#   • read_wav_mono()        — decode WAV → mono float32 waveform in [-1,1] (best‑effort).
#   • AudioFolderDataset     — Dataset(root, seq_len, mode='mulaw'|'linear').
#   • make_audio_loader()    — convenience DataLoader builder.

from __future__ import annotations
import os, glob, wave, struct, math           # ✴ stdlib decode + basic math
from typing import Iterator, Optional, List
import torch                                  # ✴ tensors + DataLoader
from torch.utils.data import Dataset, DataLoader


# ———————————————————————————————————————————————————————————
# Folder walking
# ———————————————————————————————————————————————————————————

def iter_wav_paths(root: str, recursive: bool = True) -> Iterator[str]:
    """
    Yield absolute paths to .wav files under `root`.
    """
    root = os.path.abspath(root)
    pat = "**/*.wav" if recursive else "*.wav"
    for p in sorted(glob.glob(os.path.join(root, pat), recursive=bool(recursive))):
        if os.path.isfile(p):
            yield p


# ———————————————————————————————————————————————————————————
# WAV decode → mono float32 in [-1,1]
# ———————————————————————————————————————————————————————————

def _decode_frames(frames: bytes, sampwidth: int) -> torch.Tensor:
    """
    Best‑effort PCM decode without numpy:
      • 8‑bit  : unsigned → shift to signed
      • 16‑bit : little‑endian signed
      • 24/32  : fallback to byte tensor, then scale as if 8‑bit to keep pipeline alive
    Returns int tensor (not scaled).
    """
    if sampwidth == 1:
        # Unsigned bytes [0,255] → signed centered at 0
        x = torch.tensor(list(frames), dtype=torch.int16) - 128
        return x
    if sampwidth == 2:
        it = struct.iter_unpack("<h", frames)                  # little‑endian int16
        x = torch.tensor([v[0] for v in it], dtype=torch.int16)
        return x
    # Fallback path for 24/32‑bit or unknown widths: keep bytes and treat like 8‑bit signed.
    xb = torch.tensor(list(frames), dtype=torch.int16)         # degrade gracefully
    return xb - 128


def read_wav_mono(path: str, max_samples: int | None = None) -> torch.Tensor:
    """
    Decode a WAV file to mono float32 in [-1,1]. If decode fails, raise.
    We average channels for multi‑channel input (crude but effective).
    """
    with wave.open(path, "rb") as w:
        nchan, sampwidth, fr, nframes, _, _ = w.getparams()
        # Cap frames for memory safety; if max_samples is None, read all.
        count = int(nframes if max_samples is None else min(nframes, max_samples))
        frames = w.readframes(count)

    x = _decode_frames(frames, int(sampwidth)).to(torch.int16)  # raw signed-ish ints
    if nchan > 1:
        # If channel count divides the length, average across channels; else truncate.
        length = (x.numel() // nchan) * nchan
        if length > 0:
            xm = x[:length].view(-1, nchan).to(torch.float32).mean(dim=1)
        else:
            xm = x.to(torch.float32)
    else:
        xm = x.to(torch.float32)

    # Normalize to [-1,1] using observed peak; avoid division by zero.
    peak = xm.abs().max().clamp(min=1.0)
    xf = (xm / peak).to(torch.float32)
    return xf


# ———————————————————————————————————————————————————————————
# Waveform → 8‑bit tokens
# ———————————————————————————————————————————————————————————

def _mulaw_encode(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """
    μ‑law companding (ITU‑T G.711 variant):
        f(x) = sign(x) * ln(1 + μ|x|) / ln(1 + μ) ,  x ∈ [-1,1], μ=255
    Then map to integers in [0, μ].
    """
    mu = int(mu)
    x = x.clamp(-1.0, 1.0)
    y = torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)
    # Map [-1,1] → [0,μ]
    ids = ((y + 1.0) * 0.5 * mu).round().clamp(0, mu).to(torch.long)
    return ids


def _linear_8bit(x: torch.Tensor) -> torch.Tensor:
    """
    Linear 8‑bit quantization: [-1,1] → [0,255].
    """
    x = x.clamp(-1.0, 1.0)
    ids = (((x + 1.0) * 0.5) * 255.0).round().clamp(0, 255).to(torch.long)
    return ids


def pcm_to_tokens(x: torch.Tensor, seq_len: int, mode: str = "mulaw") -> torch.Tensor:
    """
    Convert mono float32 waveform in [-1,1] to token ids in [0,255], fixed length seq_len.
    mode ∈ {'mulaw','linear'} controls the companding curve.
    """
    if x.numel() == 0:
        return torch.randint(0, 256, (seq_len,), dtype=torch.long)

    if mode == "mulaw":
        ids = _mulaw_encode(x, mu=255)
    else:
        ids = _linear_8bit(x)

    # Pad or trim to seq_len
    N = ids.numel()
    if N < seq_len:
        out = torch.zeros(seq_len, dtype=torch.long)
        out[:N] = ids
        return out
    return ids[:seq_len]


# ———————————————————————————————————————————————————————————
# Dataset + loader
# ———————————————————————————————————————————————————————————

class AudioFolderDataset(Dataset):
    """
    Stream token sequences from a folder of WAV files.

    Args:
        root:     folder with .wav files (searched recursively)
        seq_len:  fixed token length to return
        mode:     'mulaw' (default) or 'linear' companding
        max_wav:  cap samples per file for decoding safety (None = all)
    """
    def __init__(self, root: str, seq_len: int = 128, mode: str = "mulaw", max_wav: int | None = 10_000):
        self.root = str(root)
        self.seq_len = int(seq_len)
        self.mode = "mulaw" if str(mode).lower() == "mulaw" else "linear"
        self.max_wav = None if max_wav is None else int(max_wav)
        self.paths: List[str] = list(iter_wav_paths(self.root, recursive=True))

    def __len__(self):
        # Return at least 1 so DataLoader with empty folder still yields synthetic data.
        return max(1, len(self.paths))

    def __getitem__(self, idx):
        if len(self.paths) == 0:
            return torch.randint(0, 256, (self.seq_len,), dtype=torch.long)
        path = self.paths[idx % len(self.paths)]
        try:
            x = read_wav_mono(path, max_samples=self.max_wav)
            ids = pcm_to_tokens(x, self.seq_len, mode=self.mode)
        except Exception:
            ids = torch.randint(0, 256, (self.seq_len,), dtype=torch.long)
        return ids


def make_audio_loader(
    root: str,
    seq_len: int = 128,
    batch: int = 32,
    workers: int = 0,
    **kw,
) -> DataLoader:
    """
    Build a DataLoader over AudioFolderDataset that yields (B, T) int64 token batches.
    Extra kwargs (**kw) are forwarded to AudioFolderDataset.
    """
    ds = AudioFolderDataset(root=root, seq_len=seq_len, **kw)
    return DataLoader(ds, batch_size=int(batch), shuffle=True, num_workers=int(workers))


__all__ = [
    "iter_wav_paths",
    "read_wav_mono",
    "pcm_to_tokens",
    "AudioFolderDataset",
    "make_audio_loader",
]
