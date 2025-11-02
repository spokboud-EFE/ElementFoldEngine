# ElementFold · domains/__init__.py
# “Domains” describe the *geometry* where a Fold–Gate–Norm block lives.
# Examples:
#   • 'grid'  — Euclidean grids (1‑D sequences, images, volumes),
#   • 'graph' — irregular graphs (message passing via Laplacian filters),
#   • 'manifold' (future) — smooth surfaces with heat kernels.
#
# This file gives you a tiny, friendly registry:
#   register_fold(name, factory)    → add a new domain fold
#   get_fold(name, **kw)            → build a fold operator for a domain
#   available_folds()               → list registered names
#
# We keep everything dependency‑free and fail‑safe: if an optional domain module
# is missing, we simply don’t register it and explain why at lookup time.

from __future__ import annotations                      # ↻ forward annotations on older Python
from typing import Callable, Dict, Any                  # ✴ light typing for clarity

# ————————————————————————————————————————————————————————————
# Registry (name → factory). A “factory” is any callable that
# returns a fold operator (nn.Module or callable) for that domain.
# ————————————————————————————————————————————————————————————
FOLD_REALIZATIONS: Dict[str, Callable[..., Any]] = {}   # 🗂 global map of domain → factory


def register_fold(name: str, factory: Callable[..., Any]) -> None:
    """
    Add or replace a domain fold factory.
    Why: lets you plug custom folds without editing the core package.
    """
    FOLD_REALIZATIONS[str(name)] = factory              # store under a normalized string key


def get_fold(name: str, **kwargs) -> Any:
    """
    Build a fold operator for the requested domain.
    If the name is unknown, we raise a clear, actionable error.
    """
    key = str(name)
    if key not in FOLD_REALIZATIONS:
        # Explain what is available and how to enable more.
        avail = ", ".join(sorted(FOLD_REALIZATIONS.keys())) or "∅"
        raise KeyError(f"fold domain '{key}' is not registered (available: {avail})")
    return FOLD_REALIZATIONS[key](**kwargs)             # call the factory with the user’s kwargs


def available_folds() -> tuple[str, ...]:
    """
    Return a sorted tuple of all registered domain names.
    Handy for CLI help and sanity checks.
    """
    return tuple(sorted(FOLD_REALIZATIONS.keys()))


# ————————————————————————————————————————————————————————————
# Best‑effort auto‑registration of built‑in domains.
# We import lazily and guard with try/except so missing files do not
# break the package. Each submodule exposes a `make_fold(**kw)` factory.
# ————————————————————————————————————————————————————————————
try:
    from .grid import make_fold as _make_grid_fold      # Euclidean grids
    register_fold("grid", _make_grid_fold)              # register under name 'grid'
except Exception:
    # If this import fails, users can still register their own grid fold later.
    pass

try:
    from .graph import make_fold as _make_graph_fold    # Irregular graphs
    register_fold("graph", _make_graph_fold)            # register under name 'graph'
except Exception:
    # Graph domain is optional; skip silently if deps are not present.
    pass


__all__ = [
    "register_fold",
    "get_fold",
    "available_folds",
    "FOLD_REALIZATIONS",
]
