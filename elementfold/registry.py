# ElementFold · registry.py
# ============================================================
# Model Registry — declarative construction hub.
#
# Motivation
# ----------
# Lets any script or CLI build models by name instead of hard-importing classes.
# Keeps configuration clear and central:
#     model = build('base', d=256, layers=8)
#
# Contract
# --------
#   register(name, factory)   → add or override model family
#   ensure(name, factory)     → get or atomically set default
#   build(name='default', **) → instantiate model
#   get(...)                  → alias for build()
#   names()                   → list registered keys
#   has(name)                 → membership test
#
# Thread-safe, dependency-free, friendly.
# ============================================================

from __future__ import annotations
import threading, difflib
from typing import Any, Callable, Dict, Tuple

from .model import Model  # canonical ElementFold model

Factory = Callable[..., Any]
_MODELS: Dict[str, Factory] = {}
_LOCK = threading.RLock()   # allows nested re-entry safely

# ============================================================
# Helpers
# ============================================================

def _key(name:str)->str:
    """Normalize registry key (strip spaces)."""
    return str(name).strip()

def _suggest(name:str, universe:Tuple[str,...])->str:
    """Return a small 'Did you mean ...?' hint for typos."""
    c=difflib.get_close_matches(name,universe,n=3,cutoff=0.4)
    return f" Did you mean: {', '.join(c)}?" if c else ""

# ============================================================
# Core API
# ============================================================

def register(name:str, factory:Factory)->None:
    """Add or override a model family under a key."""
    if not callable(factory):
        raise TypeError(f"Factory for '{name}' must be callable.")
    with _LOCK: _MODELS[_key(name)] = factory

def ensure(name:str, default_factory:Factory)->Factory:
    """Return factory if present, else atomically set a default."""
    if not callable(default_factory):
        raise TypeError(f"default_factory for '{name}' must be callable.")
    k=_key(name)
    with _LOCK: return _MODELS.setdefault(k,default_factory)

def has(name:str)->bool:
    """Check if registry has this model."""
    with _LOCK: return _key(name) in _MODELS

def build(name:str="default",**kw)->Any:
    """Instantiate a registered model by name (with kwargs overrides)."""
    k=_key(name)
    with _LOCK:
        factory=_MODELS.get(k)
        if factory is None:
            avail=tuple(sorted(_MODELS.keys()))
            hint=_suggest(k,avail)
            also=f" (available: {', '.join(avail)})" if avail else " (no models registered)"
            raise KeyError(f"Unknown model '{k}'{also}.{hint}")
    return factory(**kw)  # call outside lock

# Back-compat alias
def get(name:str="default",**kw)->Any: return build(name,**kw)

def names()->tuple[str,...]:
    """Return sorted tuple of registered keys."""
    with _LOCK: return tuple(sorted(_MODELS.keys()))

# ============================================================
# Built-in presets
# ============================================================

_DEFAULT_DELTA = 0.030908106561043047  # canonical δ★

def _factory_default(**kw)->Any:
    """Plain Model with whatever kwargs are provided."""
    return Model(**kw)

def _factory_tiny(**kw)->Any:
    """Small, fast preset for CI / smoke tests."""
    cfg=dict(
        vocab=256, d=96, layers=2, heads=3, seq_len=128,
        fold="grid", delta=_DEFAULT_DELTA)
    cfg.update(kw)
    return Model(**cfg)

def _factory_base(**kw)->Any:
    """Baseline demo model."""
    cfg=dict(
        vocab=256, d=192, layers=6, heads=6, seq_len=256,
        fold="grid", delta=_DEFAULT_DELTA)
    cfg.update(kw)
    return Model(**cfg)

def _factory_large(**kw)->Any:
    """Larger context model (still lightweight)."""
    cfg=dict(
        vocab=256, d=384, layers=12, heads=8, seq_len=512,
        fold="grid", delta=_DEFAULT_DELTA)
    cfg.update(kw)
    return Model(**cfg)

# Register presets
register("default", _factory_default)
register("tiny",    _factory_tiny)
register("base",    _factory_base)
register("large",   _factory_large)

__all__=["register","ensure","build","get","names","has"]
