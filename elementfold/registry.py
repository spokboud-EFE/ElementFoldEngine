# ElementFold · registry.py
# A tiny, friendly *model registry* so callers can pick architectures by name
# without importing constructors directly. Non‑experts can read this top‑to‑bottom:
#   • register(name, factory)   — add/override a model family by string key
#   • build(name='default', **) — instantiate a model from its factory
#   • get(name='default', **)   — alias for build (back‑compat with older stubs)
#   • names()                   — list available model keys
#
# Design notes:
#   1) Factories are zero‑or‑keyword‑arg callables returning nn.Module instances.
#   2) We pre‑register a sensible 'default' that builds elementfold.model.Model.
#   3) You can layer presets (e.g., 'tiny', 'base', 'large') that just tweak kwargs.
#   4) Silent, explicit: unknown keys raise a clear KeyError listing alternatives.

from __future__ import annotations                   # ↻ future‑proof annotations
from typing import Callable, Dict, Any               # ✴ light typing to help readers
from .model import Model                             # 🧱 canonical ElementFold model

# ————————————————————————————————————————————————————————————
# Internal map: name → factory(**kw) → nn.Module
# ————————————————————————————————————————————————————————————
_MODELS: Dict[str, Callable[..., Any]] = {}          # 🗂 registry backing store


# ————————————————————————————————————————————————————————————
# Core API
# ————————————————————————————————————————————————————————————
def register(name: str, factory: Callable[..., Any]) -> None:
    """
    Add or override a model family under a string key.
    Why: separates *selection* (by name) from *construction* (by kwargs).
    """
    _MODELS[str(name)] = factory                      # store under normalized string key


def build(name: str = "default", **kw) -> Any:
    """
    Instantiate a model by name with keyword overrides.

    Example:
        m = build('default', d=256, layers=8)

    Raises:
        KeyError if the name is unknown (lists available keys).
    """
    key = str(name)
    if key not in _MODELS:
        avail = ", ".join(sorted(_MODELS.keys())) or "∅"
        raise KeyError(f"unknown model '{key}' (available: {avail})")
    return _MODELS[key](**kw)                         # call the factory to get a model instance


# Back‑compat alias used by early scripts: get(...) == build(...)
def get(name: str = "default", **kw) -> Any:
    return build(name, **kw)


def names() -> tuple[str, ...]:
    """Return a sorted tuple of all registered model keys (for help/CLI)."""
    return tuple(sorted(_MODELS.keys()))


# ————————————————————————————————————————————————————————————
# Built‑in families (simple, opinionated presets)
# ————————————————————————————————————————————————————————————
def _factory_default(**kw) -> Any:
    """
    Sensible default: medium width/depth, rotary click enabled, FGN stack.
    Callers can override any constructor kw via build('default', **overrides).
    """
    return Model(**kw)


def _factory_tiny(**kw) -> Any:
    """
    Small preset for smoke tests and CI: fewer layers/channels for fast runs.
    Users can still override any kw passed here.
    """
    cfg = dict(vocab=kw.get("vocab", 256),
               d=kw.get("d", 96),
               layers=kw.get("layers", 2),
               heads=kw.get("heads", 3),
               seq_len=kw.get("seq_len", 128),
               fold=kw.get("fold", "grid"),
               delta=kw.get("delta", 0.030908106561043047))
    cfg.update({k: v for k, v in kw.items() if k not in cfg})  # allow explicit overrides
    return Model(**cfg)


def _factory_base(**kw) -> Any:
    """
    Baseline preset (a touch bigger than default tiny; good for demos).
    """
    cfg = dict(vocab=kw.get("vocab", 256),
               d=kw.get("d", 192),
               layers=kw.get("layers", 6),
               heads=kw.get("heads", 6),
               seq_len=kw.get("seq_len", 256),
               fold=kw.get("fold", "grid"),
               delta=kw.get("delta", 0.030908106561043047))
    cfg.update({k: v for k, v in kw.items() if k not in cfg})
    return Model(**cfg)


# Pre‑register built‑ins so users can call build('default'|'tiny'|'base')
register("default", _factory_default)
register("tiny", _factory_tiny)
register("base", _factory_base)


__all__ = [
    "register",
    "build",
    "get",
    "names",
]
