# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ElementFold · experience/adapters/base.py                                    ║
# ║──────────────────────────────────────────────────────────────────────────────║
# ║  Adapters are “bridges” between the relaxation model and a user modality.    ║
# ║  Each adapter can observe the physics core, infer from signals, or emit     ║
# ║  outputs.  They respect the global forcing / shaping mode and produce short ║
# ║  Unicode narratives for the Studio telemetry.                               ║
# ║                                                                              ║
# ║  Public API (summary):                                                       ║
# ║    AdapterRegistry.register(name, factory, *, overwrite=True)               ║
# ║    AdapterRegistry.get(name) → factory                                       ║
# ║    AdapterRegistry.instantiate(name) → Adapter or runner                     ║
# ║    AdapterRegistry.names(), AdapterRegistry.active()                         ║
# ║    AdapterRegistry.describe() → list of dicts for Studio menu                ║
# ║                                                                              ║
# ║  New in this version:                                                        ║
# ║    • Base class `Adapter` with `infer()`, `observe()`, `reset()`             ║
# ║    • Integration with core.control.get_mode()                                ║
# ║    • Optional .narrate(state) hook returning Unicode feedback                ║
# ║    • Thread-safe tracking of active instances                                ║
# ║                                                                              ║
# ║  Stdlib-only, NumPy-friendly (torch optional).                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations
import threading, difflib, warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, List

# Optional torch for dtype names; otherwise ignored
try:
    import torch
except Exception:
    torch = None  # type: ignore

# ────────────────────────────────────────────────────────────────────────────────
# Optional imports from core (for mode awareness)
# ────────────────────────────────────────────────────────────────────────────────
try:
    from elementfold.core import control
except Exception:
    control = None  # type: ignore

# ────────────────────────────────────────────────────────────────────────────────
# TensorSpec / AdapterSpec / AdapterMeta remain as friendly, declarative objects
# ────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TensorSpec:
    shape: Tuple[Any, ...] = field(default_factory=tuple)
    dtype: Any = None
    device: str = "any"
    doc: str = ""

    def shape_str(self) -> str:
        def _p(x: Any) -> str:
            return "?" if x in (None, -1) else str(x)
        return "(" + ", ".join(_p(x) for x in self.shape) + ")"

    def dtype_str(self) -> str:
        if self.dtype is None:
            return "∗"
        try:
            if torch is not None and isinstance(self.dtype, torch.dtype):
                return str(self.dtype).replace("torch.", "")
        except Exception:
            pass
        return str(self.dtype)

    def device_str(self) -> str:
        return self.device or "any"


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    description: str = ""
    expects: Dict[str, TensorSpec] = field(default_factory=dict)
    predicts: Dict[str, str] = field(default_factory=dict)
    wait: str = "allow_sim"

    def pretty(self) -> str:
        lines = [f"{self.name}: wait={self.wait}"]
        if not self.expects:
            lines.append("  expects ∅ (none)")
        for k, s in self.expects.items():
            lines.append(f"  • {k}: {s.shape_str()}  dtype={s.dtype_str()}  device={s.device_str()}  {s.doc}")
        if self.predicts:
            lines.append("  ↳ predicts:")
            for k, doc in self.predicts.items():
                lines.append(f"     • {k}: {doc}")
        return "\n".join(lines)


@dataclass(frozen=True)
class AdapterMeta:
    kind: str = "generic"
    what: str = ""
    why: str = ""
    actions: Tuple[str, ...] = ()
    params: Dict[str, str] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────────────────────
# Decorators for attaching specs / metadata
# ────────────────────────────────────────────────────────────────────────────────
def with_spec(spec: AdapterSpec):
    def _decor(factory: Callable[[], Any]):
        setattr(factory, "__adapter_spec__", spec)
        if spec.description and not getattr(factory, "DESCRIPTION", None):
            try:
                setattr(factory, "DESCRIPTION", spec.description)
            except Exception:
                pass
        return factory
    return _decor


def with_meta(meta: AdapterMeta):
    def _decor(factory: Callable[[], Any]):
        setattr(factory, "__adapter_meta__", meta)
        try: setattr(factory, "KIND", meta.kind)
        except Exception: pass
        desc = meta.what or meta.why
        try: setattr(factory, "DESCRIPTION", desc)
        except Exception: pass
        if not getattr(factory, "description", None):
            try: setattr(factory, "description", desc)
            except Exception: pass
        return factory
    return _decor


# ────────────────────────────────────────────────────────────────────────────────
# Base class for all adapters
# ────────────────────────────────────────────────────────────────────────────────
class Adapter:
    """
    Base adapter: exposes a common control interface.
    Each adapter should implement `infer()` (active computation)
    and may override `observe()` (passive listening).
    """

    def __init__(self, name: str):
        self.name = name
        self.mode = control.get_mode() if control else "shaping"
        self.last_state: Optional[dict] = None

    # Core methods to override
    def infer(self, model, data, **kw) -> Any:
        raise NotImplementedError(f"{self.name}.infer() not implemented")

    def observe(self, telemetry: Mapping[str, Any]) -> None:
        """Optional passive hook called with every telemetry update."""
        self.last_state = dict(telemetry)

    def reset(self) -> None:
        """Optional state reset."""
        self.last_state = None

    def narrate(self, state: Optional[Mapping[str, Any]] = None) -> str:
        """
        Return a short Unicode string describing current adapter state.
        Default: just report the global mode.
        """
        return f"🧭 {self.name}: operating in {self.mode} mode — coherence stable."


# ────────────────────────────────────────────────────────────────────────────────
# Registry
# ────────────────────────────────────────────────────────────────────────────────
def _norm(name: str) -> str:
    return str(name).strip().lower()


class AdapterRegistry:
    """Thread-safe registry of adapter factories and active instances."""

    _reg: Dict[str, Callable[[], Any]] = {}
    _lock = threading.Lock()
    _active: Dict[str, Adapter] = {}

    # ── Core registration ───────────────────────────────────────────────────────
    @classmethod
    def register(cls, name: str, factory: Callable[[], Any], *, overwrite: bool = True) -> None:
        if not callable(factory):
            raise TypeError(f"adapter factory for {name!r} must be callable")
        key = _norm(name)
        with cls._lock:
            if key in cls._reg and not overwrite:
                raise KeyError(f"adapter:{key} already exists (set overwrite=True)")
            if key in cls._reg and overwrite:
                warnings.warn(f"Adapter '{key}' overwritten.", RuntimeWarning)
            cls._reg[key] = factory

    @classmethod
    def get(cls, name: str) -> Callable[[], Any]:
        key = _norm(name)
        if key not in cls._reg:
            avail = sorted(cls._reg.keys())
            near = difflib.get_close_matches(key, avail, n=3)
            hint = f" did you mean {', '.join(near)}?" if near else ""
            raise KeyError(f"adapter:{key} not found; available={avail}{hint}")
        return cls._reg[key]

    @classmethod
    def names(cls) -> Tuple[str, ...]:
        return tuple(sorted(cls._reg.keys()))

    # ── Instantiate and track ──────────────────────────────────────────────────
    @classmethod
    def instantiate(cls, name: str) -> Any:
        factory = cls.get(name)
        obj = factory()
        # attach spec/meta
        for attr in ("__adapter_spec__", "__adapter_meta__"):
            try:
                setattr(obj, attr, getattr(factory, attr, None))
            except Exception:
                pass
        # track active adapters if they subclass Adapter
        if isinstance(obj, Adapter):
            with cls._lock:
                cls._active[name] = obj
        return obj

    @classmethod
    def active(cls) -> Tuple[str, ...]:
        """Return names of currently instantiated adapters."""
        with cls._lock:
            return tuple(sorted(cls._active.keys()))

    @classmethod
    def describe(cls) -> List[Dict[str, Any]]:
        """
        Return summary suitable for /adapters endpoint and Studio menu.
        Includes narrative snippets if available.
        """
        result = []
        for name in cls.names():
            meta = getattr(cls.get(name), "__adapter_meta__", None)
            kind = getattr(meta, "kind", "generic") if meta else "generic"
            desc = getattr(meta, "what", "") if meta else ""
            active = name in cls._active
            adapter_obj = cls._active.get(name)
            mode = getattr(adapter_obj, "mode", control.get_mode() if control else "shaping")
            narrative = (
                adapter_obj.narrate(adapter_obj.last_state)
                if isinstance(adapter_obj, Adapter)
                else f"{name}: {mode} mode"
            )
            result.append({
                "name": name,
                "kind": kind,
                "active": active,
                "mode": mode,
                "description": desc,
                "narrative": narrative
            })
        return result

    # ── Maintenance ────────────────────────────────────────────────────────────
    @classmethod
    def unregister(cls, name: str) -> None:
        key = _norm(name)
        with cls._lock:
            cls._reg.pop(key, None)
            cls._active.pop(name, None)

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._reg.clear()
            cls._active.clear()
