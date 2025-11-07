# ElementFold · utils/io.py
# ============================================================
# Lightweight, atomic file I/O for ElementFold simulations
# ------------------------------------------------------------
# Goals:
#   • Atomic writes — either old or new file exists, never half-written.
#   • Human-friendly JSON for configs and telemetry.
#   • Binary NPY/NPZ for field arrays.
#   • Works on all major OSes with no external dependencies.
# ============================================================

from __future__ import annotations
import os, io, json, gzip, hashlib, tempfile
from typing import Any, Iterable, Iterator, Optional
import numpy as np


# ============================================================
# 1. Directory and listing helpers
# ============================================================

def ensure_dir(path: str) -> str:
    """Ensure parent folder exists; return its absolute path."""
    folder = os.path.abspath(os.path.dirname(path) or ".")
    os.makedirs(folder, exist_ok=True)
    return folder


def ls_files(root: str,
             exts: Optional[Iterable[str]] = None,
             recursive: bool = True) -> list[str]:
    """List all files under `root` filtered by optional extension set."""
    root = os.path.abspath(root)
    extset = None if exts is None else set(map(str.lower, exts))
    out: list[str] = []
    if not recursive:
        for n in sorted(os.listdir(root)):
            p = os.path.join(root, n)
            if os.path.isfile(p):
                if extset is None or os.path.splitext(n)[1].lower() in extset:
                    out.append(p)
        return out
    for dp, _, files in os.walk(root):
        for n in sorted(files):
            p = os.path.join(dp, n)
            if extset is None or os.path.splitext(n)[1].lower() in extset:
                out.append(p)
    return out


# ============================================================
# 2. Atomic write primitives
# ============================================================

def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Write bytes atomically (tmp + replace)."""
    ensure_dir(path)
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=folder)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def write_bytes(path: str, data: bytes, atomic: bool = True) -> None:
    if atomic:
        _atomic_write_bytes(path, data)
    else:
        ensure_dir(path)
        with open(path, "wb") as f:
            f.write(data)


def read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def write_text(path: str, text: str, encoding: str = "utf-8", atomic: bool = True) -> None:
    b = text.encode(encoding)
    if atomic:
        _atomic_write_bytes(path, b)
    else:
        ensure_dir(path)
        with open(path, "wb") as f:
            f.write(b)


def read_text(path: str, encoding: str = "utf-8", errors: str = "ignore") -> str:
    with open(path, "rb") as f:
        return f.read().decode(encoding, errors=errors)


def iter_lines(path: str, encoding: str = "utf-8", errors: str = "ignore", strip: bool = True) -> Iterator[str]:
    """Stream lines from a file; strip newlines if requested."""
    with open(path, "r", encoding=encoding, errors=errors) as f:
        for line in f:
            yield line.rstrip("\r\n") if strip else line


# ============================================================
# 3. JSON (optionally gzipped)
# ============================================================

def write_json(path: str,
               obj: Any,
               indent: int = 2,
               sort_keys: bool = True,
               use_gzip: Optional[bool] = None) -> None:
    """Write a JSON or gzipped JSON file atomically."""
    gzip_on = bool(use_gzip) if use_gzip is not None else str(path).lower().endswith(".gz")
    text = json.dumps(obj, indent=indent, sort_keys=sort_keys, ensure_ascii=False)
    if not gzip_on:
        write_text(path, text)
        return
    ensure_dir(path)
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=folder)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb") as gzf:
                gzf.write(text.encode("utf-8"))
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


def read_json(path: str, use_gzip: Optional[bool] = None) -> Any:
    """Read JSON or gzipped JSON automatically."""
    gzip_on = bool(use_gzip) if use_gzip is not None else str(path).lower().endswith(".gz")
    if not gzip_on:
        return json.loads(read_text(path))
    with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


# ============================================================
# 4. NumPy checkpoint helpers
# ============================================================

def save_npy(path: str, array: np.ndarray) -> None:
    """Atomic save for a NumPy array."""
    ensure_dir(path)
    fd, tmp = tempfile.mkstemp(prefix=".tmp.", dir=os.path.dirname(path) or ".")
    try:
        with os.fdopen(fd, "wb") as f:
            np.save(f, array)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def load_npy(path: str) -> np.ndarray:
    """Load a NumPy array."""
    return np.load(path, allow_pickle=False)


def save_npz(path: str, **arrays: np.ndarray) -> None:
    """Save multiple arrays into one compressed archive."""
    ensure_dir(path)
    np.savez_compressed(path, **arrays)


def load_npz(path: str) -> dict[str, np.ndarray]:
    """Load arrays from an NPZ archive."""
    data = np.load(path, allow_pickle=False)
    return {k: data[k] for k in data.files}


# ============================================================
# 5. Integrity and filesystem utilities
# ============================================================

def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    """Return SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def touch(path: str) -> None:
    """Create an empty file or update its modification time."""
    ensure_dir(path)
    with open(path, "ab"):
        os.utime(path, None)


# ============================================================
# Exports
# ============================================================

__all__ = [
    "ensure_dir", "ls_files",
    "write_bytes", "read_bytes",
    "write_text", "read_text", "iter_lines",
    "write_json", "read_json",
    "save_npy", "load_npy", "save_npz", "load_npz",
    "sha256_file", "touch",
]
