# ElementFold · utils/io.py
# Small, portable I/O helpers with atomic saves, fsync, and zero non-stdlib deps.
# Backward-compatible superset of the previous version.

from __future__ import annotations
import os, io, json, gzip, hashlib, tempfile
from typing import Any, Iterable, Iterator, Optional

import gz


# ———————————————————————————————————————————————————————————
# Paths & directory helpers
# ———————————————————————————————————————————————————————————

def ensure_dir(path: str) -> str:
    """
    Ensure the parent directory of `path` exists; return the absolute parent path.
    Useful before saving files.
    """
    parent = os.path.abspath(os.path.dirname(path) or ".")
    os.makedirs(parent, exist_ok=True)
    return parent


def ls_files(root: str, exts: Optional[Iterable[str]] = None, recursive: bool = True) -> list[str]:
    """
    List files under `root`. If `exts` is given, keep files whose lower-cased
    extension is in that set. When `recursive=True`, walk subfolders.
    """
    root = os.path.abspath(root)
    if not recursive:
        names = sorted(os.listdir(root)) if os.path.isdir(root) else []
        out = []
        extset = None if exts is None else set(map(str.lower, exts))
        for n in names:
            p = os.path.join(root, n)
            if os.path.isfile(p):
                if extset is None or os.path.splitext(n)[1].lower() in extset:
                    out.append(p)
        return out

    out = []
    extset = None if exts is None else set(map(str.lower, exts))
    for dp, _, files in os.walk(root):
        for n in sorted(files):
            p = os.path.join(dp, n)
            if extset is None or os.path.splitext(n)[1].lower() in extset:
                out.append(p)
    return out


# ———————————————————————————————————————————————————————————
# Atomic writes (bytes / text / JSON)
# ———————————————————————————————————————————————————————————

def _atomic_write_bytes(path: str, data: bytes, do_fsync: bool = True) -> None:
    """
    Write `data` to `path` atomically by writing into a temporary file in the
    same directory, flushing & fsync'ing, then os.replace() (atomic on POSIX and modern Windows).
    """
    ensure_dir(path)
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=folder)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            if do_fsync:
                f.flush()
                os.fsync(f.fileno())
        os.replace(tmp, path)  # atomic swap
    except Exception:
        # Best effort cleanup on failure
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def write_bytes(path: str, data: bytes, atomic: bool = True) -> None:
    """Save raw bytes to `path` (atomic by default)."""
    if atomic:
        _atomic_write_bytes(path, data)
    else:
        ensure_dir(path)
        with open(path, "wb") as f:
            f.write(data)


def read_bytes(path: str) -> bytes:
    """Read raw bytes from `path`."""
    with open(path, "rb") as f:
        return f.read()


def write_text(path: str, text: str, encoding: str = "utf-8", atomic: bool = True) -> None:
    """Save text to `path`. Default is atomic save."""
    b = text.encode(encoding)
    if atomic:
        _atomic_write_bytes(path, b)
    else:
        ensure_dir(path)
        with open(path, "wb") as f:
            f.write(b)


def read_text(path: str, encoding: str = "utf-8", errors: str = "ignore") -> str:
    """Read text from `path` (UTF‑8 by default; forgiving decode)."""
    with open(path, "rb") as f:
        return f.read().decode(encoding, errors=errors)


def iter_lines(path: str, encoding: str = "utf-8", errors: str = "ignore", strip: bool = True) -> Iterator[str]:
    """
    Stream lines from a text file (low memory).
    If `strip` is True, trailing newlines are stripped.
    """
    with open(path, "r", encoding=encoding, errors=errors) as f:
        for line in f:
            yield line.rstrip("\r\n") if strip else line


# ———————————————————————————————————————————————————————————
# JSON (optionally gzipped by extension)
# ———————————————————————————————————————————————————————————

def write_json(
    path: str,
    obj: Any,
    *,
    sort_keys: bool = True,
    indent: int = 2,
    atomic: bool = True,
    gz: Optional[bool] = None,
) -> None:
    """
    Serialize `obj` as JSON into `path` with pretty defaults.
    If `gz` is None, we auto-enable gzip when the filename ends with '.gz'.
    """
    use_gz = (gz if gz is not None) else str(path).lower().endswith(".gz")
    text = json.dumps(obj, sort_keys=sort_keys, indent=indent, ensure_ascii=False)
    if not use_gz:
        write_text(path, text, atomic=atomic)
        return

    # Gzipped path
    ensure_dir(path)
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=folder)
    try:
        with os.fdopen(fd, "wb") as f:
            with gzip.GzipFile(fileobj=f, mode="wb") as gzf:
                gzf.write(text.encode("utf-8"))
            if atomic:
                f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def read_json(path: str, *, gz: Optional[bool] = None) -> Any:
    """
    Load JSON from `path`. If `gz` is None, auto-detect by '.gz' suffix.
    """
    use_gz = (gz if gz is not None) else str(path).lower().endswith(".gz")
    if not use_gz:
        return json.loads(read_text(path))
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


# ———————————————————————————————————————————————————————————
# Checkpoint utilities (thin wrappers over torch.save/load)
# ———————————————————————————————————————————————————————————

def save_checkpoint(path: str, payload: dict) -> None:
    """
    Save a checkpoint dict with atomic replace.
    The dict is torch‑saveable; we serialize to an in‑memory buffer first, then atomically replace.
    """
    import torch, io as _io
    bio = _io.BytesIO()
    torch.save(payload, bio)
    _atomic_write_bytes(path, bio.getvalue())


def load_checkpoint(path: str, map_location: str | dict | None = "cpu") -> dict:
    """Load a checkpoint saved by `save_checkpoint`."""
    import torch
    return torch.load(path, map_location=map_location)


# ———————————————————————————————————————————————————————————
# Misc helpers
# ———————————————————————————————————————————————————————————

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Compute SHA‑256 of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def touch(path: str) -> None:
    """Create an empty file or update its mtime."""
    ensure_dir(path)
    with open(path, "ab"):
        os.utime(path, None)


__all__ = [
    "ensure_dir", "ls_files",
    "write_bytes", "read_bytes",
    "write_text", "read_text", "iter_lines",
    "write_json", "read_json",
    "save_checkpoint", "load_checkpoint",
    "sha256_file", "touch",
]
