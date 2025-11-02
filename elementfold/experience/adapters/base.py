# ElementFold · experience/adapters/base.py
# Adapters are small “bridges” from a model to a user modality (language, vision, audio, …).
# This registry:
#   • register(name, factory)   — add an adapter factory under a string key,
#   • get(name)                 — fetch the factory (raises with a helpful message if missing),
#   • names()                   — list all registered names (sorted for stable UX),
#   • ensure(name, default)     — get existing or set a default factory,
#   • has(name) / __contains__  — quick membership checks,
#   • decorator form            — @AdapterRegistry.register_fn("name") for concise definitions.
# We keep the contract used elsewhere in the project:
#     factory = AdapterRegistry.get("language")
#     runner  = factory()                     # factory produces a callable
#     out     = runner(model, prompt, style)  # runner executes the modality

from __future__ import annotations                   # Allow forward annotations on older Python
from typing import Callable, Dict, Tuple             # Light typing for clarity
import threading                                     # Simple lock for thread‑safe updates


class AdapterRegistry:
    """
    A tiny, thread‑safe mapping from modality name → adapter factory.

    Definitions:
      • name:    str key like "language", "vision", "audio".
      • factory: zero‑arg callable that returns a *runner* callable.
                 The runner has signature runner(model, prompt, style) → Any.

    We store factories (not runners) so that adapters can capture configuration
    each time they’re constructed (e.g., lazy imports, device picks, etc.).
    """

    _reg: Dict[str, Callable[[], Callable]] = {}      # Internal map: name → factory (zero‑arg callable)
    _lock = threading.Lock()                          # Protect concurrent register/ensure calls

    # ————————————————————————————————————————————————
    # Core operations
    # ————————————————————————————————————————————————

    @classmethod
    def register(cls, name: str, factory: Callable[[], Callable]) -> None:
        """
        Register an adapter factory under `name`.
        We validate that the factory is callable so errors are caught early.
        """
        key = str(name)                                # Normalize key to string
        if not callable(factory):                      # Guard: enforce callable
            raise TypeError(f"adapter factory for {key!r} must be callable, got {type(factory).__name__}")
        with cls._lock:                                # Thread‑safe write
            cls._reg[key] = factory                    # Install/replace the factory

    @classmethod
    def get(cls, name: str) -> Callable[[], Callable]:
        """
        Fetch the adapter factory for `name`. Raises KeyError if missing,
        and includes a helpful list of available names.
        """
        key = str(name)                                # Normalize lookup key
        try:
            return cls._reg[key]                       # Fast path: found
        except KeyError:
            available = ", ".join(sorted(cls._reg.keys())) or "∅ (none registered)"
            raise KeyError(f"adapter:{key} not found; available = [{available}]") from None

    @classmethod
    def ensure(cls, name: str, default: Callable[[], Callable]) -> Callable[[], Callable]:
        """
        Get existing factory for `name`, or register `default` if absent.
        Useful for optional adapters that should have a fallback.
        """
        if not callable(default):                      # Validate fallback
            raise TypeError(f"default factory for {name!r} must be callable, got {type(default).__name__}")
        key = str(name)
        with cls._lock:                                # Thread‑safe read‑then‑write
            return cls._reg.setdefault(key, default)   # Return existing or install default

    # ————————————————————————————————————————————————
    # Convenience and UX
    # ————————————————————————————————————————————————

    @classmethod
    def names(cls) -> Tuple[str, ...]:
        """
        Return a sorted, immutable tuple of registered adapter names.
        Stable order makes CLI and logs deterministic.
        """
        return tuple(sorted(cls._reg.keys()))

    @classmethod
    def has(cls, name: str) -> bool:
        """
        Quick membership test (same as `name in AdapterRegistry`).
        """
        return str(name) in cls._reg

    def __contains__(self, name: str) -> bool:         # Allows:  "language" in AdapterRegistry
        return self.has(name)

    def __len__(self) -> int:                          # How many adapters are registered?
        return len(self._reg)

    # ————————————————————————————————————————————————
    # Decorator form for concise registration
    # ————————————————————————————————————————————————

    @classmethod
    def register_fn(cls, name: str) -> Callable[[Callable[[], Callable]], Callable[[], Callable]]:
        """
        Decorator to register a factory:

            @AdapterRegistry.register_fn("language")
            def make_language_adapter():
                def run(model, prompt, style):
                    ...
                return run

        Returns the original factory unchanged for normal Python semantics.
        """
        def _decorator(factory: Callable[[], Callable]) -> Callable[[], Callable]:
            cls.register(name, factory)                # Reuse validation and locking
            return factory
        return _decorator
