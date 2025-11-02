# ElementFold · datasets/grid_pulse_synth.py
# Click‑structured synthetic tokens for quick, controllable training.
#
# Why this exists:
#   • Real data is great, but you often need a fast, deterministic signal to
#     validate that the engine (Fold–Gate–Norm + rotary click) locks to δ⋆.
#   • These generators build sequences whose latent ledger X follows the
#     click law exactly (up to small noise), then map them to tokens.
#
# What you get:
#   • make_pulse(...)           → (N, T) LongTensor tokens (back‑compat name).
#   • ClickPulseDataset         → indexable Dataset (synthesizes per sample).
#   • make_pulse_loader(...)    → DataLoader for the dataset.
#
# Signal family (mode):
#   • 'click-sine'   : a_t = cos(2π X/δ⋆)                           (smooth)
#   • 'click-square' : a_t = sign(cos(2π X/δ⋆))                     (binary)
#   • 'click-step'   : a_t = stair case over seats (0..C−1)         (ramp)
# These modes share the same ledger X; they differ only in how amplitude a_t
# is derived from phase(X). Tokens are produced by scaling a_t to [0..vocab−1].

from __future__ import annotations
import math                                   # ✴ π for cos/sin
from typing import Iterable, Optional, Tuple  # ✴ light hints
import torch                                  # ✴ tensors + Dataset/DataLoader
from torch.utils.data import Dataset, DataLoader


# ———————————————————————————————————————————————————————————
# Defaults (kept visible so other modules can import if needed)
# ———————————————————————————————————————————————————————————

DEFAULT_DELTA = 0.030908106561043047          # δ⋆ coherence step (from the paper)
DEFAULT_CAPS  = (2, 6, 10, 14)                # canonical capacities (s,p,d,f)


# ———————————————————————————————————————————————————————————
# Helpers: map amplitudes ↦ tokens, synth ledger X, synth one sample
# ———————————————————————————————————————————————————————————

def _amp_to_tokens(a: torch.Tensor, vocab: int) -> torch.Tensor:
    """
    Map amplitude a ∈ [-1,1] (or 0..1 for steps) into integer tokens 0..vocab-1.
    We clamp to keep the mapping safe under small numeric noise.
    """
    if a.dtype.is_floating_point:
        # If a looks like [-1,1], convert to [0,1]; if already [0,1], this still works.
        a01 = (a + 1.0) * 0.5
        a01 = a01.clamp(0.0, 1.0)
    else:
        a01 = a.float().clamp(0.0, 1.0)
    ids = (a01 * (vocab - 1)).round().to(torch.long)  # nearest integer bin
    return ids


def _synth_ledger_X(
    T: int,
    delta: float,
    C: int,
    k0: int,
    s0: int,
    noise: float,
    g: torch.Generator | None = None,
) -> torch.Tensor:
    """
    Build a ledger sequence X_t that advances by one seat per time step:
        step = δ⋆ / C
        X_t = k0·δ⋆ + ((s0 + t) mod C) · step + ε_t
    with small noise ε_t bounded safely inside the half‑click margin.

    Args:
      T:     sequence length
      delta: δ⋆
      C:     seat capacity for this sample
      k0:    starting rung
      s0:    starting seat
      noise: Gaussian std as a *fraction of step*
      g:     optional torch.Generator for reproducibility
    """
    step = float(delta) / float(C)                         # seat spacing δ⋆/C
    t = torch.arange(T, dtype=torch.long)                  # time indices 0..T−1
    seats = (s0 + t) % int(C)                              # seat index per step
    X = k0 * float(delta) + seats.to(torch.float32) * step # base click‑perfect ledger
    if noise > 0.0:
        # Keep ε well within the half‑click: use σ = noise·step and clamp at 0.25·δ⋆.
        eps = torch.normal(
            mean=0.0,
            std=noise * step,
            size=(T,),
            generator=g,
            dtype=torch.float32,
        )
        bound = 0.25 * float(delta)                        # strict bound < δ⋆/2
        eps = eps.clamp(min=-bound, max=bound)
        X = X + eps
    return X.to(torch.float32)


def _from_ledger_to_amplitude(
    X: torch.Tensor,
    delta: float,
    mode: str,
    C: int,
) -> torch.Tensor:
    """
    Convert ledger positions X into an amplitude a_t according to a display mode.
    The phase is θ = 2π X / δ⋆. We derive different shapes from θ or the seat index.
    """
    theta = (2.0 * math.pi / float(delta)) * X          # phase angle θ_t
    if mode == "click-sine":
        a = torch.cos(theta)                            # smooth cosine around the circle
    elif mode == "click-square":
        a = torch.sign(torch.cos(theta))                # binary beat (−1 or +1); sign(0)=0 is rare and harmless
        a[a == 0] = 1.0                                 # remove rare zeros so map stays {−1,+1}
    elif mode == "click-step":
        # Use seat index (0..C−1) as a 0..1 ramp, then rescale to [−1,1] for the common mapper.
        step = float(delta) / float(C)
        seat = torch.remainder(X, float(delta)) / step  # fractional seat
        a01 = (seat / float(C - 1)).clamp(0.0, 1.0)     # 0..1 stair
        a = a01 * 2.0 - 1.0                             # map to [−1,1] to reuse the same token mapper
    else:
        raise ValueError(f"unknown mode: {mode!r}")
    return a.to(torch.float32)


def _synth_one(
    T: int,
    vocab: int,
    delta: float,
    capacities: Tuple[int, ...],
    mode: str,
    noise: float,
    seed: int,
    index: int,
) -> torch.Tensor:
    """
    Synthesize one token sequence of length T under the chosen settings.
    We keep a per‑index RNG so each item is reproducible and independent.
    """
    # Per‑item RNG seeded by (seed ^ index) so dataloader workers produce the same item for a given idx.
    g = torch.Generator()
    g.manual_seed((int(seed) & 0xFFFFFFFF) ^ int(index))

    # Sample a capacity C from the canonical set (s,p,d,f); this changes the “octave.”
    C = int(capacities[torch.randint(0, len(capacities), (1,), generator=g).item()])
    # Sample starting rung k0 and seat s0 so sequences are not all aligned.
    k0 = int(torch.randint(0, 8, (1,), generator=g).item())          # a few rungs up
    s0 = int(torch.randint(0, C, (1,), generator=g).item())          # any seat inside the block

    # 1) Ledger X with click‑perfect spacing (plus small noise within the half‑click margin).
    X = _synth_ledger_X(T=T, delta=delta, C=C, k0=k0, s0=s0, noise=noise, g=g)

    # 2) Amplitude a_t from phase(X) according to the mode (sine/square/step).
    a = _from_ledger_to_amplitude(X=X, delta=delta, mode=mode, C=C)

    # 3) Tokens by scaling amplitude to bins 0..vocab−1.
    ids = _amp_to_tokens(a, vocab=vocab)
    return ids  # LongTensor (T,)


# ———————————————————————————————————————————————————————————
# Public: back‑compat function name returning a full tensor batch
# ———————————————————————————————————————————————————————————

def make_pulse(
    n: int = 1024,                 # number of sequences
    seq_len: int = 128,            # tokens per sequence (T)
    vocab: int = 256,              # token bins
    delta: float = DEFAULT_DELTA,  # δ⋆
    capacities: Tuple[int, ...] = DEFAULT_CAPS,  # canonical C choices
    mode: str = "click-sine",      # signal family ('click-sine'|'click-square'|'click-step')
    noise: float = 0.05,           # noise as a fraction of seat step (safe default)
    seed: int = 0,                 # RNG seed for reproducibility
) -> torch.Tensor:
    """
    Generate an (n, seq_len) LongTensor of tokens with a clean δ⋆ rhythm.
    This preserves the original function name for notebooks that relied on it.
    """
    n = int(n); T = int(seq_len); vocab = int(vocab)
    out = torch.empty(n, T, dtype=torch.long)
    for i in range(n):
        out[i] = _synth_one(
            T=T, vocab=vocab, delta=delta, capacities=capacities,
            mode=mode, noise=float(noise), seed=int(seed), index=i
        )
    return out


# ———————————————————————————————————————————————————————————
# Dataset + DataLoader: indexable, reproducible synthetic data
# ———————————————————————————————————————————————————————————

class ClickPulseDataset(Dataset):
    """
    Indexable synthetic dataset that generates one sequence per __getitem__.
    Useful when you want a stable stream over many epochs without storing tensors.

    Args mirror make_pulse(); each index i uses RNG seed (seed ^ i) to keep items stable.
    """
    def __init__(
        self,
        n: int = 1024,
        seq_len: int = 128,
        vocab: int = 256,
        delta: float = DEFAULT_DELTA,
        capacities: Tuple[int, ...] = DEFAULT_CAPS,
        mode: str = "click-sine",
        noise: float = 0.05,
        seed: int = 0,
    ):
        super().__init__()
        self.n = int(n)
        self.T = int(seq_len)
        self.vocab = int(vocab)
        self.delta = float(delta)
        self.capacities = tuple(int(c) for c in capacities)
        self.mode = str(mode)
        self.noise = float(noise)
        self.seed = int(seed)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> torch.Tensor:
        return _synth_one(
            T=self.T, vocab=self.vocab, delta=self.delta, capacities=self.capacities,
            mode=self.mode, noise=self.noise, seed=self.seed, index=int(idx)
        )


def make_pulse_loader(
    n: int = 1024,
    seq_len: int = 128,
    vocab: int = 256,
    batch: int = 32,
    workers: int = 0,
    **kw,
) -> DataLoader:
    """
    Convenience wrapper that builds a DataLoader over ClickPulseDataset.
    Extra kwargs (**kw) are forwarded to ClickPulseDataset (delta, capacities, mode, noise, seed).
    """
    ds = ClickPulseDataset(n=n, seq_len=seq_len, vocab=vocab, **kw)
    return DataLoader(ds, batch_size=int(batch), shuffle=True, num_workers=int(workers))


__all__ = [
    "DEFAULT_DELTA",
    "DEFAULT_CAPS",
    "make_pulse",
    "ClickPulseDataset",
    "make_pulse_loader",
]
