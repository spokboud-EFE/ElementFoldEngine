# elementfold/registry.py
# ──────────────────────────────────────────────────────────────────────────────
# A tiny, friendly *model registry* so callers can pick architectures by name
# without importing constructors directly.
#
# Public API (unchanged):
#   • register(name, factory)   — add/override a model family by key
#   • ensure(name, factory)     — get existing factory or atomically set a default
#   • build(name='default', **) — instantiate via its factory(**kw)
#   • get(name='default', **)   — alias for build (back‑compat)
#   • names()                   — list available model keys
#   • has(name)                 — quick membership check
#
# Design notes:
#   1) Factories are callables returning model instances (e.g., nn.Module).
#   2) We pre‑register sensible presets that build elementfold.model.Model.
#   3) Presets are simple kwarg templates (e.g., 'tiny', 'base', 'large').
#   4) Unknown keys raise KeyError with helpful alternatives.
#   5) Thread‑safe: a lock protects concurrent register/lookup.
#   6) No external deps; stdlib only.
from __future__ import annotations

from typing import Any, Callable, Dict, Tuple
import threading
import difflib

from .model import Model  # 🧱 canonical ElementFold model

# Type alias for readability
Factory = Callable[..., Any]

# Internal map: name → factory(**kw) → model instance
_MODELS: Dict[str, Factory] = {}
_LOCK = threading.RLock()  # re‑entrant to avoid deadlocks in nested uses


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def _key(name: str) -> str:
    """Normalize registry keys."""
    return str(name).strip()


def _suggest(name: str, universe: Tuple[str, ...]) -> str:
    """Return a human‑friendly 'did you mean' string (or empty)."""
    candidates = difflib.get_close_matches(name, universe, n=3, cutoff=0.4)
    return f" Did you mean: {', '.join(candidates)}?" if candidates else ""


# ──────────────────────────────────────────────────────────────────────────────
# Core API
# ──────────────────────────────────────────────────────────────────────────────
def register(name: str, factory: Factory) -> None:
    """
    Add or override a model family under a string key.
    Why: separates *selection* (by name) from *construction* (by kwargs).
    """
    if not callable(factory):
        raise TypeError(f"factory for '{name}' must be callable, got {type(factory).__name__}")
    k = _key(name)
    with _LOCK:
        _MODELS[k] = factory


def ensure(name: str, default_factory: Factory) -> Factory:
    """
    Get the factory registered under `name`, or atomically register `default_factory`
    if it was missing. Returns the resulting factory.
    """
    if not callable(default_factory):
        raise TypeError(f"default_factory for '{name}' must be callable")
    k = _key(name)
    with _LOCK:
        return _MODELS.setdefault(k, default_factory)


def has(name: str) -> bool:
    """Quick membership check."""
    with _LOCK:
        return _key(name) in _MODELS


def build(name: str = "default", **kw) -> Any:
    """
    Instantiate a model by name with keyword overrides.

    Example:
        m = build('default', d=256, layers=8)

    Raises:
        KeyError if the name is unknown (lists available keys and suggestions).
        Whatever the underlying factory raises when construction fails.
    """
    k = _key(name)
    with _LOCK:
        factory = _MODELS.get(k)
        if factory is None:
            available = tuple(sorted(_MODELS.keys()))
            hint = _suggest(k, available)
            also = f" (available: {', '.join(available)})" if available else " (no models registered)"
            raise KeyError(f"unknown model '{k}'{also}.{hint}")
    # Call outside the lock to keep critical sections short
    return factory(**kw)


# Back‑compat alias used by early scripts: get(...) == build(...)
def get(name: str = "default", **kw) -> Any:
    return build(name, **kw)


def names() -> tuple[str, ...]:
    """Return a sorted tuple of all registered model keys (for help/CLI)."""
    with _LOCK:
        return tuple(sorted(_MODELS.keys()))


# ──────────────────────────────────────────────────────────────────────────────
# Built‑in families (simple, opinionated presets)
# ──────────────────────────────────────────────────────────────────────────────
_DEFAULT_DELTA = 0.030908106561043047  # preserved constant


def _factory_default(**kw) -> Any:
    """
    Sensible default: delegate entirely to Model's own defaults.
    Callers can override any constructor kw via build('default', **overrides).
    """
    return Model(**kw)


def _factory_tiny(**kw) -> Any:
    """
    Small preset for smoke tests and CI: fewer layers/channels for fast runs.
    Users can still override any kw passed here.
    """
    cfg = dict(
        vocab=kw.get("vocab", 256),
        d=kw.get("d", 96),
        layers=kw.get("layers", 2),
        heads=kw.get("heads", 3),
        seq_len=kw.get("seq_len", 128),
        fold=kw.get("fold", "grid"),
        delta=kw.get("delta", _DEFAULT_DELTA),
    )
    # Allow explicit overrides to win:
    cfg.update({k: v for k, v in kw.items() if k not in cfg})
    return Model(**cfg)


def _factory_base(**kw) -> Any:
    """
    Baseline preset (a touch bigger than tiny; good for demos).
    """
    cfg = dict(
        vocab=kw.get("vocab", 256),
        d=kw.get("d", 192),
        layers=kw.get("layers", 6),
        heads=kw.get("heads", 6),
        seq_len=kw.get("seq_len", 256),
        fold=kw.get("fold", "grid"),
        delta=kw.get("delta", _DEFAULT_DELTA),
    )
    cfg.update({k: v for k, v in kw.items() if k not in cfg})
    return Model(**cfg)


def _factory_large(**kw) -> Any:
    """
    Larger preset for longer contexts / heavier demos (still modest by modern LMs).
    """
    cfg = dict(
        vocab=kw.get("vocab", 256),
        d=kw.get("d", 384),
        layers=kw.get("layers", 12),
        heads=kw.get("heads", 8),
        seq_len=kw.get("seq_len", 512),
        fold=kw.get("fold", "grid"),
        delta=kw.get("delta", _DEFAULT_DELTA),
    )
    cfg.update({k: v for k, v in kw.items() if k not in cfg})
    return Model(**cfg)


# Pre‑register built‑ins so users can call build('default'|'tiny'|'base'|'large')
register("default", _factory_default)
register("tiny", _factory_tiny)
register("base", _factory_base)
register("large", _factory_large)


__all__ = [
    "register",
    "ensure",
    "build",
    "get",
    "names",
    "has",
]
