# ElementFold · experience/adapters/base.py
# ──────────────────────────────────────────────────────────────────────────────
# Adapters are small “bridges” from a model to a user modality (language, audio,
# photonics, …). The registry below maps a *name* → *factory*, where a factory
# is a zero‑arg callable returning a *runner*. A runner has the signature:
#
#     runner(model, prompt, style) -> Any
#
# Public API (tiny, stdlib‑only):
#   • AdapterRegistry.register(name, factory, *, overwrite=True)
#   • AdapterRegistry.get(name)          → factory (KeyError on miss, with hints)
#   • AdapterRegistry.ensure(name, default) → factory (install default if absent)
#   • AdapterRegistry.names()            → sorted tuple of names
#   • AdapterRegistry.has(name) / "name" in AdapterRegistry
#   • AdapterRegistry.register_fn("name")  decorator form
#   • AdapterRegistry.unregister(name)   → remove a factory (testing/dev only)
#   • AdapterRegistry.clear()            → drop all factories (testing/dev only)
#
# Design notes:
#   • Names are normalized to lowercase (ASCII) for stable CLI UX.
#   • Errors include close‑match suggestions (handy at the REPL).
#   • Thread‑safe writes via a simple lock; reads are uncontended.
#   • Backward‑compatible: existing calls keep working as‑is.
#
from __future__ import annotations

from typing import Callable, Dict, Tuple
import threading
import warnings
import difflib


def _norm(name: str) -> str:
    """Normalize external names to a stable, case‑insensitive key."""
    return str(name).strip().lower()


class AdapterRegistry:
    """
    A tiny, thread‑safe mapping from adapter name → adapter factory.

    Definitions:
      • name:    string key like "language", "resonator", "interferometer".
      • factory: zero‑arg callable returning a *runner* callable.
                 Runner signature: runner(model, prompt, style) -> Any
    """

    _reg: Dict[str, Callable[[], Callable]] = {}
    _lock = threading.Lock()

    # ────────────────────────────────────────────────────────────────────────
    # Core operations
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def register(
        cls,
        name: str,
        factory: Callable[[], Callable],
        *,
        overwrite: bool = True,
    ) -> None:
        """
        Register an adapter factory under `name`.

        Args:
            name:     registry key (case‑insensitive; stored lowercase).
            factory:  zero‑arg callable producing a runner.
            overwrite: if False and name exists, raise KeyError; otherwise replace
                       and emit a gentle warning so accidental shadowing is visible.

        Raises:
            TypeError: if `factory` is not callable.
            KeyError:  if `overwrite=False` and `name` already exists.
        """
        if not callable(factory):
            raise TypeError(f"adapter factory for {name!r} must be callable, got {type(factory).__name__}")
        key = _norm(name)
        with cls._lock:
            if key in cls._reg and not overwrite:
                raise KeyError(f"adapter:{key} already registered; set overwrite=True to replace")
            if key in cls._reg and overwrite:
                prev = cls._reg[key]
                warnings.warn(
                    f"Adapter '{key}' overwritten (was {getattr(prev, '__name__', type(prev).__name__)}).",
                    RuntimeWarning,
                    stacklevel=2,
                )
            cls._reg[key] = factory

    @classmethod
    def get(cls, name: str) -> Callable[[], Callable]:
        """
        Fetch the adapter factory for `name`.

        Returns:
            The registered zero‑arg factory.

        Raises:
            KeyError with a friendly message and close‑match suggestions.
        """
        key = _norm(name)
        try:
            return cls._reg[key]
        except KeyError:
            avail = sorted(cls._reg.keys())
            hint = ""
            if avail:
                # Suggest up to 3 close matches (threshold tuned for short names)
                matches = difflib.get_close_matches(key, avail, n=3, cutoff=0.6)
                if matches:
                    hint = f"  did you mean: {', '.join(matches)}?"
            available = ", ".join(avail) if avail else "∅ (none registered)"
            raise KeyError(f"adapter:{key} not found; available = [{available}].{hint}") from None

    @classmethod
    def ensure(cls, name: str, default: Callable[[], Callable]) -> Callable[[], Callable]:
        """
        Get existing factory for `name`, or register `default` if absent.

        Returns:
            The existing or newly registered factory.

        Raises:
            TypeError if `default` is not callable.
        """
        if not callable(default):
            raise TypeError(f"default factory for {name!r} must be callable, got {type(default).__name__}")
        key = _norm(name)
        with cls._lock:
            if key not in cls._reg:
                cls._reg[key] = default
            return cls._reg[key]

    # ────────────────────────────────────────────────────────────────────────
    # Convenience and UX
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def names(cls) -> Tuple[str, ...]:
        """Return a sorted, immutable tuple of registered adapter names."""
        return tuple(sorted(cls._reg.keys()))

    @classmethod
    def has(cls, name: str) -> bool:
        """Quick membership test (same as `name in AdapterRegistry`)."""
        return _norm(name) in cls._reg

    def __contains__(self, name: str) -> bool:  # Allows: "language" in AdapterRegistry
        return self.has(name)

    def __len__(self) -> int:                   # How many adapters are registered?
        return len(self._reg)

    # ────────────────────────────────────────────────────────────────────────
    # Decorator form for concise registration
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def register_fn(cls, name: str) -> Callable[[Callable[[], Callable]], Callable[[], Callable]]:
        """
        Decorator to register a zero‑arg *factory* under `name`.

            @AdapterRegistry.register_fn("language")
            def make_language_adapter():
                def run(model, prompt, style):
                    ...
                return run

        Returns:
            The original factory unchanged (normal Python semantics).
        """
        def _decorator(factory: Callable[[], Callable]) -> Callable[[], Callable]:
            cls.register(name, factory)  # uses normalization + locking
            return factory
        return _decorator

    # ────────────────────────────────────────────────────────────────────────
    # Testing / maintenance helpers (no external callers in core flow)
    # ────────────────────────────────────────────────────────────────────────

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a factory by name. Intended for tests or dynamic reload scenarios."""
        key = _norm(name)
        with cls._lock:
            cls._reg.pop(key, None)

    @classmethod
    def clear(cls) -> None:
        """Drop all registered adapters. Intended for tests or process teardown."""
        with cls._lock:
            cls._reg.clear()
