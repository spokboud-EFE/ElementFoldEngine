# ElementFold · datasets/text_files.py
# Streaming text → token ids for training.
# Two iterable datasets:
#   • TextLineDataset   — yields variable‑length token tensors (one per line).
#   • TextChunkDataset  — yields fixed‑length chunks for LM training (seq_len, with stride).
# Plus helpers:
#   • iter_text_paths() — walk a folder (optionally recursive) filtering by extensions.
#   • make_text_loader()— build a DataLoader for lines or chunks.
#   • load_folder()     — simple generator (back‑compat): yield tokenized lines.

from __future__ import annotations
import os, io, random                                  # ✴ file I/O and a dab of RNG for shuffling
from typing import Iterable, Iterator, List, Optional   # ✴ readable type hints
import torch                                            # ✴ tensors + DataLoader
from torch.utils.data import IterableDataset, DataLoader
from ..tokenizer import SimpleTokenizer                 # ✴ tiny byte tokenizer (vocab≈256)


# ———————————————————————————————————————————————————————————
# File walking (filter by extension; optional recursion)
# ———————————————————————————————————————————————————————————

def iter_text_paths(
    root: str,
    recursive: bool = True,
    allowed_exts: Optional[set[str]] = None,
    follow_symlinks: bool = False,
) -> Iterator[str]:
    """
    Yield text file paths under `root`. We keep it permissive by default and
    filter obvious binary types out via extensions.
    """
    # Reasonable default whitelist for plain‑text sources
    if allowed_exts is None:
        allowed_exts = {
            ".txt", ".md", ".log", ".jsonl", ".json",
            ".csv", ".tsv", ".yml", ".yaml", ".py",
        }
    root = os.path.abspath(root)
    if not recursive:
        for name in sorted(os.listdir(root)):
            p = os.path.join(root, name)
            if not os.path.isfile(p):
                continue
            if os.path.splitext(p)[1].lower() in allowed_exts:
                yield p
        return

    # Recursive walk with a simple extension check
    for dirpath, dirnames, filenames in os.walk(root, followlinks=bool(follow_symlinks)):
        for name in sorted(filenames):
            p = os.path.join(dirpath, name)
            if os.path.splitext(p)[1].lower() in allowed_exts:
                yield p


# ———————————————————————————————————————————————————————————
# Dataset 1: lines → variable‑length tensors
# ———————————————————————————————————————————————————————————

class TextLineDataset(IterableDataset):
    """
    Stream lines from a folder of text files and tokenize each line.
    Yields: 1‑D LongTensor of token ids per line (variable length).
    """

    def __init__(
        self,
        root: str,
        max_lines: Optional[int] = None,
        recursive: bool = True,
        vocab: int = 256,
        encoding: str = "utf-8",
        errors: str = "ignore",
        shuffle_files: bool = False,
        seed: int = 0,
        keep_newline: bool = True,
    ):
        super().__init__()
        self.root = str(root)
        self.max_lines = max_lines if (max_lines is None) else int(max_lines)
        self.recursive = bool(recursive)
        self.vocab = int(vocab)
        self.encoding = str(encoding)
        self.errors = str(errors)
        self.shuffle_files = bool(shuffle_files)
        self.seed = int(seed)
        self.keep_newline = bool(keep_newline)
        self.tok = SimpleTokenizer(vocab=self.vocab)  # ✴ byte‑level tokenizer

    def _paths(self) -> List[str]:
        paths = list(iter_text_paths(self.root, recursive=self.recursive))
        if self.shuffle_files:
            rng = random.Random(self.seed)
            rng.shuffle(paths)
        return paths

    def _iter_lines(self) -> Iterator[torch.Tensor]:
        count = 0
        for p in self._paths():
            with io.open(p, "r", encoding=self.encoding, errors=self.errors) as f:
                for line in f:
                    if (self.max_lines is not None) and (count >= self.max_lines):
                        return
                    s = line.rstrip("\n\r")
                    ids = self.tok.encode(s)
                    if self.keep_newline:
                        # Add a newline byte (10) as a simple sentence separator.
                        if len(ids) == 0 or ids[-1] != 10:
                            ids = ids + [10]
                    if not ids:
                        ids = [0]  # ensure we never yield an empty tensor
                    yield torch.tensor(ids, dtype=torch.long)
                    count += 1

    def __iter__(self) -> Iterator[torch.Tensor]:
        # IterableDataset requires all state inside the iterator to be re‑created per worker.
        return self._iter_lines()


# ———————————————————————————————————————————————————————————
# Dataset 2: fixed‑length chunks (for LM training)
# ———————————————————————————————————————————————————————————

class TextChunkDataset(IterableDataset):
    """
    Stream token ids from files and pack them into fixed‑length chunks.
    This is typically what you want for language‑model training.

    Yields: LongTensor of shape (seq_len,)
    """

    def __init__(
        self,
        root: str,
        seq_len: int = 128,
        stride: Optional[int] = None,
        max_lines: Optional[int] = None,
        recursive: bool = True,
        vocab: int = 256,
        encoding: str = "utf-8",
        errors: str = "ignore",
        shuffle_files: bool = False,
        seed: int = 0,
        keep_newline: bool = True,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.stride = int(stride) if stride is not None else int(seq_len)
        self.lines = TextLineDataset(
            root=root,
            max_lines=max_lines,
            recursive=recursive,
            vocab=vocab,
            encoding=encoding,
            errors=errors,
            shuffle_files=shuffle_files,
            seed=seed,
            keep_newline=keep_newline,
        )

    def __iter__(self) -> Iterator[torch.Tensor]:
        buf: List[int] = []
        for t in self.lines:
            # Accumulate into a Python list for cheap appends, then slice into tensors on output.
            buf.extend(int(x) for x in t.tolist())
            # Emit as many full chunks as we can
            while len(buf) >= self.seq_len:
                out = torch.tensor(buf[: self.seq_len], dtype=torch.long)
                yield out
                del buf[: self.stride]  # advance by stride


# ———————————————————————————————————————————————————————————
# Collation and convenience DataLoader
# ———————————————————————————————————————————————————————————

def pad_collate(batch: List[torch.Tensor], pad_id: int = 0, max_len: Optional[int] = None) -> torch.Tensor:
    """
    Pad a list of 1‑D LongTensors (variable length) into (B, T_max) with pad_id.
    Useful with TextLineDataset. For TextChunkDataset (fixed length), vanilla collate is fine.
    """
    if not batch:
        return torch.empty(0, 0, dtype=torch.long)
    T = max(len(x) for x in batch)
    if max_len is not None:
        T = min(T, int(max_len))
    out = torch.full((len(batch), T), int(pad_id), dtype=torch.long)
    for i, x in enumerate(batch):
        n = min(T, x.numel())
        out[i, :n] = x[:n]
    return out


def make_text_loader(
    root: str,
    mode: str = "chunks",              # 'chunks' or 'lines'
    seq_len: int = 128,
    stride: Optional[int] = None,
    batch: int = 32,
    shuffle_files: bool = False,
    workers: int = 0,
    **kw,
) -> DataLoader:
    """
    Build a DataLoader for a folder of text files.
      • mode='chunks' returns (B, seq_len) fixed blocks (best for LM training).
      • mode='lines'  returns (B, T) padded batches (best for quick inspection).
    Extra kwargs (kw) are forwarded to the dataset constructors.
    """
    if mode == "chunks":
        ds = TextChunkDataset(root=root, seq_len=seq_len, stride=stride, shuffle_files=shuffle_files, **kw)
        return DataLoader(ds, batch_size=int(batch), shuffle=False, num_workers=int(workers))
    elif mode == "lines":
        ds = TextLineDataset(root=root, shuffle_files=shuffle_files, **kw)
        return DataLoader(ds, batch_size=int(batch), shuffle=False, num_workers=int(workers),
                          collate_fn=lambda b: pad_collate(b, pad_id=0, max_len=seq_len))
    else:
        raise ValueError(f"unknown mode: {mode!r}")


# ———————————————————————————————————————————————————————————
# Back‑compat generator (simple, line‑by‑line)
# ———————————————————————————————————————————————————————————

def load_folder(path: str, max_lines: Optional[int] = None, recursive: bool = False) -> Iterable[torch.Tensor]:
    """
    Legacy/simple interface kept for compatibility with earlier notebooks:
        for ids in load_folder("data/text"):
            ...
    Yields 1‑D LongTensor tokenized per line.
    """
    ds = TextLineDataset(root=path, max_lines=max_lines, recursive=bool(recursive))
    yield from ds  # delegate to the iterable dataset


__all__ = [
    "iter_text_paths",
    "TextLineDataset",
    "TextChunkDataset",
    "pad_collate",
    "make_text_loader",
    "load_folder",
]
