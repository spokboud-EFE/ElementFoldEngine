# ElementFold · data.py
# ============================================================
# A tiny data pipeline that yields token batches for training.
#
# Design goals:
#   • Simple structure with a gentle, learnable rhythm.
#   • Deterministic by default: same index → same sample (debug‑friendly).
#   • Aligned with the ledger idea: default “periods” mirror capacities.
#
# Components:
#   • PulseDataset — synthesizes sequences with a periodic pulse + light noise.
#   • DataLoaderBuilder — assembles a tuned DataLoader with safe defaults.
#
# Output contract:
#   __getitem__ returns a 1‑D LongTensor of length T (dtype = int64).
#   DataLoader yields (B, T) int64 batches suitable for the training loop.
# ============================================================

from __future__ import annotations

import os
import random
from typing import Iterable, Optional

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Dataset
# ============================================================

class PulseDataset(Dataset):
    """
    Synthetic token sequences with gentle regularity:

      1) Start from uniform random tokens in [0, vocab).
      2) Pick a period L from a small set (defaults: 2,6,10,14).
      3) Choose an anchor token and stamp it every L steps
         starting at a random offset o ∈ {0,…,L−1}.
      4) Flip a fraction of positions to random tokens with probability p_noise.

    Determinism:
      • Each dataset index uses its own Torch Generator (seed = base_seed + idx).
      • __getitem__ is deterministic per index, independent of worker count/order.

    Args
    ----
    n        : number of samples
    seq_len  : sequence length T
    vocab    : token vocabulary size (ids in [0, vocab))
    seed     : base RNG seed (None → 0, still deterministic)
    p_noise  : probability to replace a position by a random token
    periods  : iterable of legal periods (defaults to (2,6,10,14))
    """

    def __init__(
        self,
        n: int = 10_000,
        seq_len: int = 128,
        vocab: int = 256,
        seed: int | None = 1234,
        p_noise: float = 0.15,
        periods: Optional[Iterable[int]] = None,
    ) -> None:
        self.n = int(n)
        self.seq_len = max(1, int(seq_len))
        self.vocab = max(2, int(vocab))  # need ≥2 so noise can differ from anchor
        self.base_seed = int(seed if seed is not None else 0)
        self.p_noise = float(max(0.0, p_noise))

        # Mirror ledger capacities unless overridden
        base = (2, 6, 10, 14) if periods is None else tuple(int(p) for p in periods)
        self._periods = tuple(p for p in base if p > 0) or (2, 6, 10, 14)

    def __len__(self) -> int:
        return self.n

    def _rng(self, idx: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(self.base_seed + int(idx))
        return g

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Produce one token sequence x ∈ ℕ^{T}, dtype=int64.
        """
        g = self._rng(idx)

        # (1) random tokens in [0, vocab)
        x = torch.randint(0, self.vocab, (self.seq_len,), generator=g, dtype=torch.long)

        # (2) choose a legal period L, clamp to [1, T]
        if len(self._periods) == 1:
            L = int(self._periods[0])
        else:
            j = int(torch.randint(0, len(self._periods), (1,), generator=g).item())
            L = int(self._periods[j])
        L = max(1, min(L, self.seq_len))

        # (3) choose an anchor token and stamp it every L steps from a random offset
        anchor = int(torch.randint(0, self.vocab, (1,), generator=g).item())
        if L == 1:
            x.fill_(anchor)
        else:
            offset = int(torch.randint(0, L, (1,), generator=g).item())
            x[offset::L] = anchor

        # (4) inject light noise via an independent Bernoulli mask
        if self.p_noise > 0.0:
            mask = torch.rand(self.seq_len, generator=g) < self.p_noise
            if mask.any():
                noise = torch.randint(0, self.vocab, (self.seq_len,), generator=g, dtype=torch.long)
                x = torch.where(mask, noise, x)

        return x  # (T,) int64


# ============================================================
# DataLoader builder
# ============================================================

class DataLoaderBuilder:
    """
    Convenience factory for a tuned DataLoader:

      • drop_last=True — fixed (B,T) batch shapes throughout training.
      • num_workers — defaults to CPU_count−1 (capped at 8).
      • pin_memory — True on CUDA for faster H2D copies; harmless on CPU.
      • persistent_workers — keep workers alive across epochs when workers>0.
      • worker_init_fn — deterministic seeding per worker.

    Typical use:
        dl = DataLoaderBuilder(seq_len=128, vocab=256, batch=32).make()
        for x in dl:   # x is (B,T) int64
            ...
    """

    def __init__(
        self,
        seq_len: int = 128,
        vocab: int = 256,
        batch: int = 32,
        n: int = 10_000,
        seed: int | None = 1234,
        workers: int | None = None,
        pin_memory: bool | None = None,
        persistent_workers: bool | None = None,
        p_noise: float = 0.15,
        periods: Optional[Iterable[int]] = None,  # defaults to (2,6,10,14) when None
        drop_last: bool = True,
        prefetch_factor: int | None = None,       # let PyTorch decide when None
    ) -> None:
        self.seq_len = int(seq_len)
        self.vocab = int(vocab)
        self.batch = int(batch)
        self.n = int(n)
        self.seed = seed
        self.p_noise = float(max(0.0, p_noise))
        self.periods = tuple(int(x) for x in (periods if periods is not None else (2, 6, 10, 14)))
        self.drop_last = bool(drop_last)
        self.prefetch_factor = prefetch_factor

        # Workers: keep 1 CPU free; cap at 8 (dataset is light)
        if workers is None:
            cpu = os.cpu_count() or 2
            workers = max(0, min(8, cpu - 1))
        self.workers = int(workers)

        # Pin memory speeds up H2D when CUDA is present
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self.pin_memory = bool(pin_memory)

        # Persist workers only when >0 workers are used
        if persistent_workers is None:
            persistent_workers = (self.workers > 0)
        self.persistent_workers = bool(persistent_workers) and (self.workers > 0)

    def _worker_init(self, worker_id: int) -> None:
        """
        Deterministic worker seeding: gives each worker a distinct RNG stream.
        """
        if self.seed is not None:
            worker_seed = int(self.seed) + int(worker_id)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    def make(self) -> DataLoader:
        """
        Return a ready DataLoader that yields (B,T) int64 batches.
        """
        ds = PulseDataset(
            n=self.n,
            seq_len=self.seq_len,
            vocab=self.vocab,
            seed=self.seed,
            p_noise=self.p_noise,
            periods=self.periods,
        )

        # Optional deterministic shuffling via a dedicated Generator
        g = None
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(int(self.seed))

        kwargs = dict(
            dataset=ds,
            batch_size=self.batch,
            shuffle=True,                           # reshuffle each epoch
            drop_last=self.drop_last,               # fixed (B,T) shapes
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            worker_init_fn=self._worker_init,
            generator=g,                            # deterministic shuffle when seeded
        )
        if self.prefetch_factor is not None and self.workers > 0:
            kwargs["prefetch_factor"] = int(self.prefetch_factor)

        return DataLoader(**kwargs)
