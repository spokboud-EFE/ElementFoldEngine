# ElementFold · data.py
# A tiny, friendly data pipeline that yields token batches for training.
# We keep it dependency‑free and readable:
#   • PulseDataset — synthesizes sequences with simple “pulse” structure
#                    aligned to the ledger capacities (2, 6, 10, 14).
#   • DataLoaderBuilder — returns a tuned DataLoader (workers, pinning, seeding).

import os, math, random                         # stdlib: OS info, math, RNG for reproducibility
import torch                                    # tensors and data utils
from torch.utils.data import Dataset, DataLoader  # tiny Dataset/DataLoader primitives


class PulseDataset(Dataset):
    """
    Synthetic token sequences with gentle regularity:
      • Start from uniform random tokens in [0, vocab).
      • Choose a period L from {2, 6, 10, 14} (mirrors ledger capacities).
      • Overlay a “pulse” token every L steps (x[0], x[L], x[2L], …).
      • Add small random noise flips with probability p_noise.

    This gives the model an easy-to-learn rhythm while remaining varied.
    """
    def __init__(self, n: int = 10_000, seq_len: int = 128, vocab: int = 256,
                 seed: int | None = 1234, p_noise: float = 0.15):
        self.n = int(n)                           # how many examples the dataset exposes
        self.seq_len = int(seq_len)               # length T of each token sequence
        self.vocab = int(vocab)                   # token IDs range in [0, vocab)
        self.p_noise = float(p_noise)             # probability to flip a token to random noise
        # Fix a base seed so each index deterministically maps to a sample (useful for debugging).
        self.base_seed = int(seed if seed is not None else 0)

        # The “capacities” mirror ledger block sizes; using them as periods creates friendly structure.
        self._capacities = (2, 6, 10, 14)

    def __len__(self) -> int:
        return self.n                              # PyTorch asks “how many samples?”

    def _rng(self, idx: int) -> torch.Generator:
        """
        Build a per‑item RNG so __getitem__ is deterministic for each idx.
        """
        g = torch.Generator()                      # independent RNG for this item
        g.manual_seed(self.base_seed + int(idx))   # seed derived from base + index
        return g

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Return one token sequence x ∈ ℕ^{T}, where each entry is in [0, vocab).
        """
        g = self._rng(idx)                         # deterministic RNG
        # 1) Start from uniform random tokens.
        x = torch.randint(0, self.vocab, (self.seq_len,), generator=g)

        # 2) Choose a period L from the ledger‑like set and an anchor token to “pulse”.
        L = int(self._capacities[torch.randint(0, len(self._capacities), (1,), generator=g).item()])
        anchor = int(torch.randint(0, self.vocab, (1,), generator=g).item())

        # 3) Overlay pulses: place the anchor every L steps (simple regularity signal).
        x[::L] = anchor

        # 4) Add light noise so the task isn’t trivial and the model learns to denoise.
        if self.p_noise > 0.0:
            mask = torch.rand(self.seq_len, generator=g) < self.p_noise  # True where we inject noise
            noise = torch.randint(0, self.vocab, (self.seq_len,), generator=g)
            x = torch.where(mask, noise, x)

        return x                                          # shape (T,)


class DataLoaderBuilder:
    """
    Convenience builder for a tuned DataLoader:
      • sensible defaults for num_workers, pin_memory, and persistence,
      • worker seeding so each worker has a distinct RNG stream,
      • drop_last=True so training sees fixed batch shapes.
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
    ):
        self.seq_len = int(seq_len)                     # expose same knobs as the dataset
        self.vocab = int(vocab)
        self.batch = int(batch)
        self.n = int(n)
        self.seed = seed
        self.p_noise = float(p_noise)

        # Choose a reasonable default for number of workers (CPU threads).
        if workers is None:
            cpu = os.cpu_count() or 2
            # Keep 1–2 CPUs free; cap at 8 because dataset is light.
            workers = max(0, min(8, cpu - 1))
        self.workers = int(workers)

        # Pin memory helps host→GPU transfer when CUDA is available; harmless on CPU.
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()
        self.pin_memory = bool(pin_memory)

        # Persistent workers avoid process re‑spawns when iterating many times; requires workers>0.
        if persistent_workers is None:
            persistent_workers = (self.workers > 0)
        self.persistent_workers = bool(persistent_workers) and (self.workers > 0)

    def _worker_init(self, worker_id: int):
        """
        Seed each worker deterministically so random ops inside __getitem__
        produce distinct streams across workers and epochs.
        """
        if self.seed is not None:
            worker_seed = self.seed + worker_id
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

    def make(self) -> DataLoader:
        """
        Build and return a ready DataLoader that yields (B,T) int64 token tensors.
        """
        ds = PulseDataset(
            n=self.n,
            seq_len=self.seq_len,
            vocab=self.vocab,
            seed=self.seed,
            p_noise=self.p_noise,
        )

        # Use a dedicated Generator to make shuffling deterministic if a seed is given.
        g = None
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed)

        return DataLoader(
            ds,
            batch_size=self.batch,
            shuffle=True,                        # reshuffle each epoch
            drop_last=True,                      # fixed shapes for training
            num_workers=self.workers,            # parallel data loading
            pin_memory=self.pin_memory,          # faster H2D copies on CUDA
            persistent_workers=self.persistent_workers,  # keep workers alive across epochs
            worker_init_fn=self._worker_init,    # per‑worker seed setup
            generator=g,                         # deterministic shuffling if seeded
        )
