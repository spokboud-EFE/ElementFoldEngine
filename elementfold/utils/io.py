# ElementFold · utils/io.py
# ============================================================
# Portable, dependency-free I/O helpers.
#
# Purpose:
#   • Handle files safely and atomically.
#   • Offer human-readable JSON and binary checkpoint saving.
#   • Work on any POSIX or modern Windows system without extra deps.
#
# Every function is designed with the same “relaxation” philosophy as
# the rest of ElementFold: state changes must be deliberate, reversible,
# and complete.  No partial writes; every save either succeeds fully
# or leaves the previous file untouched.
# ============================================================

from __future__ import annotations
import os
import io
import json
import gzip
import hashlib
import tempfile
from typing import Any, Iterable, Iterator, Optional

# ============================================================
# 1. PATHS AND DIRECTORY HELPERS
# ============================================================

def ensure_dir(path: str) -> str:
    """
    Make sure the parent directory of `path` exists.
    Returns the absolute parent path.
    Plain words:
        Before we can save a file, the folder must exist.
        This helper quietly creates it if missing.
    """
    parent = os.path.abspath(os.path.dirname(path) or ".")
    os.makedirs(parent, exist_ok=True)
    return parent


def ls_files(root: str,
             exts: Optional[Iterable[str]] = None,
             recursive: bool = True) -> list[str]:
    """
    List files under `root`.

    Parameters
    ----------
    exts : iterable of extensions (e.g. {'.py','.json'}) or None
        If provided, only files with matching (lowercased) extensions are kept.
    recursive : bool
        Whether to traverse subdirectories.

    Plain words:
        Think of this as "find all files here", optionally filtered by type.
    """
    root = os.path.abspath(root)
    extset = None if exts is None else set(map(str.lower, exts))
    out: list[str] = []

    if not recursive:
        # One-level listing.
        names = sorted(os.listdir(root)) if os.path.isdir(root) else []
        for n in names:
            p = os.path.join(root, n)
            if os.path.isfile(p):
                if extset is None or os.path.splitext(n)[1].lower() in extset:
                    out.append(p)
        return out

    # Recursive walk.
    for dp, _, files in os.walk(root):
        for n in sorted(files):
            p = os.path.join(dp, n)
            if extset is None or os.path.splitext(n)[1].lower() in extset:
                out.append(p)
    return out


# ============================================================
# 2. ATOMIC WRITES (BYTES AND TEXT)
# ============================================================
# Atomicity means: the file on disk is *either* the old version
# or the new one—never a half-written mixture. We achieve this
# by writing to a temp file in the same directory and using
# os.replace(), which is atomic on modern OSes.

def _atomic_write_bytes(path: str, data: bytes, *, do_fsync: bool = True) -> None:
    """
    Write bytes atomically:
      1. Write into a temporary file in the same directory.
      2. Flush and fsync to ensure bits are on disk.
      3. Atomically replace the destination.
    """
    ensure_dir(path)
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=folder)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            if do_fsync:
                f.flush()
                os.fsync(f.fileno())  # force to disk
        os.replace(tmp, path)       # atomic swap
    except Exception:
        # Cleanup in case of failure.
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def write_bytes(path: str, data: bytes, *, atomic: bool = True) -> None:
    """Save raw bytes to disk (atomic by default)."""
    if atomic:
        _atomic_write_bytes(path, data)
    else:
        ensure_dir(path)
        with open(path, "wb") as f:
            f.write(data)


def read_bytes(path: str) -> bytes:
    """Read raw bytes from a file."""
    with open(path, "rb") as f:
        return f.read()


def write_text(path: str,
               text: str,
               *,
               encoding: str = "utf-8",
               atomic: bool = True) -> None:
    """
    Write text to a file (UTF-8 by default).
    Uses the atomic write mechanism to avoid corruption.
    """
    b = text.encode(encoding)
    if atomic:
        _atomic_write_bytes(path, b)
    else:
        ensure_dir(path)
        with open(path, "wb") as f:
            f.write(b)


def read_text(path: str,
              *,
              encoding: str = "utf-8",
              errors: str = "ignore") -> str:
    """Read text (forgiving UTF-8 decoder)."""
    with open(path, "rb") as f:
        return f.read().decode(encoding, errors=errors)


def iter_lines(path: str,
               *,
               encoding: str = "utf-8",
               errors: str = "ignore",
               strip: bool = True) -> Iterator[str]:
    """
    Stream lines from a text file efficiently.

    If `strip` is True, removes newline characters.
    Useful for large logs: reads one line at a time.
    """
    with open(path, "r", encoding=encoding, errors=errors) as f:
        for line in f:
            yield line.rstrip("\r\n") if strip else line


# ============================================================
# 3. JSON (OPTIONALLY GZIPPED)
# ============================================================
# JSON is our default human-readable serialization format.
# These helpers optionally gzip the file based on suffix.

def write_json(path: str,
               obj: Any,
               *,
               sort_keys: bool = True,
               indent: int = 2,
               atomic: bool = True,
               use_gzip: Optional[bool] = None) -> None:
    """
    Serialize `obj` as JSON and save it.

    If `use_gzip` is None, gzip automatically when filename ends with '.gz'.
    """
    gzip_on = bool(use_gzip) if use_gzip is not None else str(path).lower().endswith(".gz")
    text = json.dumps(obj, sort_keys=sort_keys, indent=indent, ensure_ascii=False)

    if not gzip_on:
        write_text(path, text, atomic=atomic)
        return

    # Gzipped path: write compressed into temp file, then replace.
    ensure_dir(path)
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=folder)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as gzf:
                gzf.write(text.encode("utf-8"))
            if atomic:
                raw.flush()
                os.fsync(raw.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def read_json(path: str, *, use_gzip: Optional[bool] = None) -> Any:
    """
    Load JSON data. Auto-detect gzip from '.gz' extension if needed.
    """
    gzip_on = bool(use_gzip) if use_gzip is not None else str(path).lower().endswith(".gz")
    if not gzip_on:
        return json.loads(read_text(path))
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


# ============================================================
# 4. CHECKPOINT UTILITIES (TORCH SAVE/LOAD)
# ============================================================
# For model snapshots: save a dictionary safely and atomically.
# This mirrors the same integrity guarantee as write_bytes().

def save_checkpoint(path: str, payload: dict) -> None:
    """
    Save a checkpoint dict atomically.
    Serializes to an in-memory buffer, then swaps in the finished file.
    """
    import torch
    bio = io.BytesIO()
    torch.save(payload, bio)
    _atomic_write_bytes(path, bio.getvalue())


def load_checkpoint(path: str, map_location: str | dict | None = "cpu") -> dict:
    """
    Load a checkpoint previously saved by `save_checkpoint`.
    By default it loads on CPU to avoid device mismatch.
    """
    import torch
    return torch.load(path, map_location=map_location)


# ============================================================
# 5. MISCELLANEOUS UTILITIES
# ============================================================

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """
    Compute the SHA-256 hash of a file by reading it in chunks.
    Used to verify integrity of large datasets or checkpoints.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def touch(path: str) -> None:
    """
    Create an empty file or update its modification time (mtime).
    Equivalent to the UNIX `touch` command.
    """
    ensure_dir(path)
    with open(path, "ab"):
        os.utime(path, None)


# ============================================================
# 6. EXPORTS
# ============================================================
# Explicit __all__ ensures only the public interface is imported.

__all__ = [
    "ensure_dir", "ls_files",
    "write_bytes", "read_bytes",
    "write_text", "read_text", "iter_lines",
    "write_json", "read_json",
    "save_checkpoint", "load_checkpoint",
    "sha256_file", "touch",
]
